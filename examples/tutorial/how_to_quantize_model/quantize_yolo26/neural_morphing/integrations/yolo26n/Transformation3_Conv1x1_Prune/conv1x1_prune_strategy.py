"""
Conv1x1 Input-Channel Pruning Strategy (Self-contained)
=======================================================
Zeroes out the N least important input channels (by L1-norm)
in 1x1 Conv layers. Uses the engine's on_step_end hook to re-apply
the binary mask after each optimizer step, guaranteeing the pruned
channels remain exactly zero throughout training.

Weight shape stays unchanged -- this is soft pruning. To get
actual compute savings, a follow-up pass must structurally remove
the zeroed channels from both producer and consumer ops.
"""

from typing import Any, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from esp_ppq.IR import BaseGraph, Operation
from neural_morphing.interface import BaseReplacementStrategy


# ==========================================
# CHANNEL SELECTION
# ==========================================
def select_channels_to_prune(weight: torch.Tensor, n: int) -> List[int]:
    """Select the N least important input channels by L1-norm.

    Args:
        weight: Conv weight tensor [C_out, C_in, kH, kW]
        n: number of input channels to prune

    Returns:
        Sorted list of channel indices to prune.
    """
    importance = weight.abs().sum(dim=(0, 2, 3))  # [C_in]
    _, indices_sorted = importance.sort()
    return sorted(indices_sorted[:n].tolist())


# ==========================================
# DISTILLATION LOSS
# ==========================================
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


# ==========================================
# STRATEGY
# ==========================================
class Conv1x1PruneStrategy(BaseReplacementStrategy):
    """Prune N input channels from 1x1 Conv layers via masked distillation.

    Architecture: same Conv1x1(C_in->C_out) but with N input channels
    forced to zero. The engine distills the remaining weights to
    compensate for the removed channels.

    The on_step_end hook re-applies the mask after every optimizer step.
    """

    def __init__(self,
                 n_prune: int = 16,
                 lr: float = 1e-3,
                 wd: float = 1e-4,
                 steps: int = 2000,
                 patience: int = 2,
                 min_cosine: float = 0.9908,
                 scale_bounds: tuple = (0.985, 1.015),
                 skip_prefixes: list = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_prune = n_prune
        self._lr = lr
        self._wd = wd
        self._steps = steps
        self._patience = patience
        self._min_cosine = min_cosine
        self._scale_bounds = scale_bounds
        self._skip_prefixes = skip_prefixes or []

        # Current block's mask (cleared per build_replacement call)
        self._masks = {}

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

    # ── Target selection ──────────────────────────────────────────
    def select_target(self, op: Operation, block: Any, graph: BaseGraph) -> bool:
        """Target all 1x1 Conv ops except those matching skip_prefixes."""
        if op.type != 'Conv':
            return False
        if op.attributes.get("kernel_shape") != [1, 1]:
            return False
        if any(op.name.startswith(p) for p in self._skip_prefixes):
            return False
        return True

    # ── Build replacement ─────────────────────────────────────────
    def build_replacement(self, op: Operation, block: Any, graph: BaseGraph) -> Tuple[nn.Module, list]:
        """Build a Conv1x1 with pruned channels zeroed + register mask."""
        # Clear masks from previous blocks to avoid shape collisions
        self._masks.clear()

        input_var  = op.inputs[0]
        output_var = op.outputs[0]

        in_c  = int(input_var.shape[1])  if input_var.shape  is not None else 128
        out_c = int(output_var.shape[1]) if output_var.shape is not None else 128
        h     = int(input_var.shape[2])  if input_var.shape  is not None else 32
        w     = int(input_var.shape[3])  if input_var.shape  is not None else 32

        # Extract original weight [C_out, C_in, 1, 1]
        W = op.inputs[1].value
        W = torch.as_tensor(W).float() if not isinstance(W, torch.Tensor) else W.float()

        # Extract original bias
        bias = None
        if len(op.inputs) >= 3 and op.inputs[2].value is not None:
            b = op.inputs[2].value
            bias = torch.as_tensor(b).float() if not isinstance(b, torch.Tensor) else b.float()

        # Select channels to prune by L1-norm importance
        channels_to_prune = select_channels_to_prune(W, self.n_prune)

        # Build mask [C_out, C_in, 1, 1] -- zero at pruned input channels
        mask = torch.ones_like(W)
        mask[:, channels_to_prune, :, :] = 0.0

        # Apply mask to initial weights
        W_init = W * mask

        # Build replacement module (same structure, masked weights)
        module = nn.Conv2d(in_c, out_c, 1, bias=bias is not None)
        module.weight.data.copy_(W_init)
        if bias is not None:
            module.bias.data.copy_(bias)

        # Store mask for on_step_end enforcement
        self._masks[op.name] = {
            'mask': mask,
            'channels': channels_to_prune,
            'shape': tuple(W.shape),
        }

        importance = W.abs().sum(dim=(0, 2, 3))
        total_energy = importance.sum().item()
        pruned_energy = importance[channels_to_prune].sum().item()
        print(f"       [Prune] {op.name}: zeroing {self.n_prune}/{in_c} channels "
              f"({pruned_energy/total_energy:.1%} energy), indices={channels_to_prune}")

        return module, [1, in_c, h, w]

    # ── Per-step mask enforcement ─────────────────────────────────
    def on_step_end(self, trainable_params: List[torch.Tensor], step: int, **kwargs):
        """Re-apply binary mask to pruned channels after each optimizer step."""
        for param in trainable_params:
            param_shape = tuple(param.shape)
            for op_name, info in self._masks.items():
                if info['shape'] == param_shape:
                    mask = info['mask'].to(param.device)
                    with torch.no_grad():
                        param.mul_(mask)
                    break

    # ── Validation ────────────────────────────────────────────────
    def calculate_samples(self, block, block_ops, dataloader_size: int) -> int:
        return dataloader_size

    def evaluate_validation(self, executor, block_ops, input_wire, output_wire,
                            val_inputs, val_outputs, executing_device):
        """Evaluate over ALL validation batches."""
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
        return HuberCosineLoss(lambda_cos=2.0, delta=1.0)

    def get_scheduler(self, optimizer, steps) -> Optional[Any]:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-6
        )
