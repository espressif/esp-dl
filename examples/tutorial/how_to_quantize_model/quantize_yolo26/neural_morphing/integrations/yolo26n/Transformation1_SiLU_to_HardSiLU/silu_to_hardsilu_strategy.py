"""
SiLU → HardSiLU PPQ-Native Strategy (Self-contained)
=====================================================
Replaces Swish ops with HardSiluPie8 + learnable scale directly
in the PPQ graph — no PyTorch module, no ONNX tracing.

Result: 2 clean PPQ ops (HardSiluPie8 + Mul) instead of 4-5
from ONNX-traced ScaledHardSwish.

HardSiluPie8:  y = x * clamp(x/8 + 0.5, 0, 1)
Scale (Mul):   y = y * scale  (folds into successor Conv)

NOTE: HardSiluPie8 executor registration is handled by
      custom_ops_patch.apply_patch_hardsilu() — must be called
      before running the engine.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

from esp_ppq.IR import BaseGraph, Operation
from esp_ppq.IR.base.graph import Variable

from neural_morphing.interface import BaseReplacementStrategy


# ==========================================
# PPQ-NATIVE STRATEGY
# ==========================================
class SiLUToHardSiLUPPQStrategy(BaseReplacementStrategy):
    """SiLU → HardSiLU replacement using PPQ-native ops.

    Builds the replacement subgraph directly in PPQ IR:
        Swish → [HardSiluPie8 + Mul(scale)]

    No ONNX round-trip, no PyTorch module tracing.
    The scale parameter is trainable via the engine's distillation loop.
    """

    def __init__(self,
                 lr: float = 1e-3,
                 wd: float = 1e-4,
                 steps: int = 200,
                 patience: int = 5,
                 min_cosine: float = 0.98,
                 scale_bounds: tuple = (0.90, 1.10),
                 skip_prefixes: list = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._lr = lr
        self._wd = wd
        self._steps = steps
        self._patience = patience
        self._min_cosine = min_cosine
        self._scale_bounds = scale_bounds
        self._skip_prefixes = skip_prefixes or []

    @property
    def learning_rate(self) -> float: return self._lr

    @property
    def weight_decay(self) -> float: return self._wd

    @property
    def steps(self) -> int: return self._steps

    @property
    def patience(self) -> int: return self._patience

    @property
    def requires_predecessor(self) -> bool:
        return True

    @property
    def compensate_on_reject(self) -> bool:
        return True

    def select_target(self, op: Operation, block: Any, graph: BaseGraph) -> bool:
        if op.type != "Swish":
            return False
        if any(op.name.startswith(p) for p in self._skip_prefixes):
            return False
        # Skip activations after depthwise convolutions
        if op.inputs:
            predecessor = op.inputs[0].source_op
            if predecessor is not None and predecessor.type == "Conv":
                group = predecessor.attributes.get("group", 1)
                if group > 1:
                    return False
        return True

    def build_replacement(self, op: Operation, block: Any, graph: BaseGraph) -> Tuple[nn.Module, list]:
        """Not used — build_replacement_subgraph overrides the pipeline."""
        raise NotImplementedError("PPQ-native strategy uses build_replacement_subgraph directly")

    def build_replacement_subgraph(self, op: Operation, block: Any, graph: BaseGraph,
                                   module_to_subgraph_fn) -> Tuple[BaseGraph, list]:
        """Build HardSiluPie8 + Scale subgraph directly in PPQ IR."""
        return _build_hardsilu_subgraph(op, graph)

    def calculate_samples(self, block, block_ops, dataloader_size: int) -> int:
        return dataloader_size

    def evaluate_validation(self, executor, block_ops, input_wire, output_wire,
                            val_inputs, val_outputs, executing_device):
        cos_sims, scale_ratios = [], []
        with torch.no_grad():
            for val_in, val_out in zip(val_inputs, val_outputs):
                feed_dict = {input_wire: val_in.to(executing_device)}
                Y_stud = executor.partial_graph_forward(
                    operations=block_ops, feed_dict=feed_dict, output_names=[output_wire]
                )[0]
                val_out_dev = val_out.to(executing_device)
                cos_sims.append(
                    F.cosine_similarity(Y_stud, val_out_dev, dim=1).mean().item()
                )
                scale_ratios.append(
                    (torch.norm(Y_stud, p=2) / torch.norm(val_out_dev, p=2)).item()
                )

        mean_cos   = sum(cos_sims)     / len(cos_sims)
        mean_scale = sum(scale_ratios) / len(scale_ratios)
        cos_ok   = mean_cos >= self._min_cosine
        scale_ok = self._scale_bounds[0] <= mean_scale <= self._scale_bounds[1]
        accepted = cos_ok and scale_ok

        metrics = {
            'mean_cosine':    mean_cos,
            'min_cosine_req': self._min_cosine,
            'cosine_pass':    cos_ok,
            'mean_scale':     mean_scale,
            'scale_bounds':   self._scale_bounds,
            'scale_pass':     scale_ok,
        }
        return accepted, metrics

    def get_criterion(self) -> nn.Module:
        # return MSECosineLoss(lambda_cos=5.0)
        # Huber: robust to activation spikes in FFN/deep layers
        return HuberCosineLoss(lambda_cos=2.0, delta=1.0)

    def get_scheduler(self, optimizer, steps) -> Optional[Any]:
        # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
        # Warm restarts: LR jolts back up every T_0 steps, helping escape plateaus
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-6
        )


# ─────────────────────────────────────────────
# SUBGRAPH BUILDER
# ─────────────────────────────────────────────
def _build_hardsilu_subgraph(op: Operation, graph: BaseGraph) -> Tuple[BaseGraph, list]:
    """Build a PPQ subgraph: single HardSiluPie8 with embedded scale.

    y = scale * HardSiLU(x)

    The scale lives as op._scale_param (nn.Parameter) directly on the
    HardSiluPie8 operation — no separate Mul node in the graph.

    During training:
        scale = RoundSTE(sigmoid(_scale_param) * 256) / 256
        → fake-quantized, gradients flow via STE

    At export:
        _freeze_hardsilu_scales() converts _scale_param → scale_int attribute
        → serialized to FlatBuffers, applied as 4 extra PIE instructions

    Returns:
        (sub_graph, input_shape) ready for splice().
    """
    input_var = op.inputs[0]
    assert input_var.shape is not None, (
        f"Input shape is None for op {op.name} — run tracing_operation_meta first"
    )
    shape = list(input_var.shape)
    sub = BaseGraph(name=f'{op.name}_hardsilu_subgraph')

    # ── Variables ──
    v_input = Variable(name='input', value=None, is_parameter=False, shape=shape)
    sub._variables['input'] = v_input

    v_output = Variable(name='output', value=None, is_parameter=False, shape=shape)
    sub._variables['output'] = v_output

    # ── Single op: HardSiluPie8(input) → output ──
    op1 = Operation(name='HardSiluPie8', op_type='HardSiluPie8', attributes={},
                    inputs=[v_input], outputs=[v_output])
    sub._operations['HardSiluPie8'] = op1
    v_input._dest_ops.append(op1)
    v_output._source_op = op1 

    # Embed trainable scale as private attribute on the op.
    op1.attributes['scale_param'] = nn.Parameter(torch.tensor(1.0))
    op1._extra_params = [op1.attributes['scale_param']]

    # ── Boundaries ──
    sub.inputs[v_input.name] = v_input
    sub.outputs[v_output.name] = v_output

    return sub, shape


# ─────────────────────────────────────────────
# DISTILLATION LOSS
# ─────────────────────────────────────────────
class MSECosineLoss(nn.Module):
    """Combined MSE + Cosine Similarity loss."""
    def __init__(self, lambda_cos: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_cos = lambda_cos

    def forward(self, y_pred, y_real):
        mse_loss = self.mse(y_pred, y_real)
        cos_sim = F.cosine_similarity(
            y_pred.flatten(1), y_real.flatten(1), dim=1
        ).mean()
        return mse_loss + self.lambda_cos * (1.0 - cos_sim)

class HuberCosineLoss(nn.Module):
    """Huber + Cosine Similarity loss. Robust to activation outliers."""
    def __init__(self, lambda_cos: float = 2.0, delta: float = 1.0):
        super().__init__()
        self.lambda_cos = lambda_cos
        self.delta = delta

    def forward(self, y_pred, y_real):
        huber_loss = F.huber_loss(y_pred, y_real, delta=self.delta)
        cos_sim = F.cosine_similarity(
            y_pred.flatten(1), y_real.flatten(1), dim=1
        ).mean()
        return huber_loss + self.lambda_cos * (1.0 - cos_sim)
