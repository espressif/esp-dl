"""
Conv3x3 -> DW+PW Decomposition Strategy (Self-contained)
=========================================================
Replaces standard 3x3 convolutions with depthwise-separable form:
    DW(3x3) + PW(1x1) + 1x1 skip (center pixel)

SVD-based residual initialization:
  1. Extract center pixel -> PW_skip (40-60% of weight energy)
  2. Remove center pixel -> W_residual
  3. SVD rank-k on residual -> DW + PW_main

Architecture:
    x --> DW(3x3) -> PW_main(1x1) --> Add --> output
      |-> PW_skip(1x1, stride) --------|
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

from esp_ppq.IR import BaseGraph, Operation
from neural_morphing.interface import BaseReplacementStrategy


# ==========================================
# SVD INITIALIZATION
# ==========================================
@torch.no_grad()
def init_residual_dw_pw(W, rank=4):
    """Decompose Conv3x3 into DW+PW (spatial residual) + PW_skip (center pixel).

    1. Extract center pixel W[:,:,1,1] -> PW_skip initialization
    2. Remove center pixel from W -> W_residual
    3. SVD rank-k on W_residual -> DW + PW_main
    """
    C_out, C_in, kH, kW = W.shape
    max_rank = min(rank, C_out, kH * kW)
    center_h, center_w = kH // 2, kW // 2

    # PW_skip: center pixel weights [C_out, C_in]
    skip_init = W[:, :, center_h, center_w].clone()

    # Remove center pixel -> residual
    W_res = W.clone()
    W_res[:, :, center_h, center_w] = 0

    # SVD on residual (per-channel)
    dw = torch.zeros(C_in * max_rank, 1, kH, kW)
    pw = torch.zeros(C_out, C_in * max_rank, 1, 1)
    energy_ratios = []
    for c_in in range(C_in):
        M = W_res[:, c_in, :, :].reshape(C_out, kH * kW)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        total_energy = (S ** 2).sum()
        rank_energy = (S[:max_rank] ** 2).sum()
        if total_energy > 0:
            energy_ratios.append((rank_energy / total_energy).item())
        for i in range(max_rank):
            idx = c_in * max_rank + i
            dw[idx, 0]        = (S[i].sqrt() * Vt[i, :]).reshape(kH, kW)
            pw[:, idx, 0, 0]  = S[i].sqrt() * U[:, i]

    if energy_ratios:
        mean_e = sum(energy_ratios) / len(energy_ratios)
        min_e  = min(energy_ratios)
        print(f"       [SVD] residual rank={max_rank}, energy: mean={mean_e:.1%}, min={min_e:.1%} "
              f"({C_in}->{C_in*max_rank}->{C_out}, skip=1x1)")

    return dw, pw, skip_init


# ==========================================
# REPLACEMENT MODULE
# ==========================================
class ResidualDecomposedConv(nn.Module):
    """DW+PW with 1x1 skip connection for center pixel.

    Architecture:
        x --> DW(3x3) -> PW_main(1x1) --> Add --> output
          |-> PW_skip(1x1, stride) --------|

    PW_skip handles the center pixel (40-60% of weight energy).
    DW+PW handles the spatial residual (lower rank -> better SVD fit).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, rank=4,
                 original_weight=None, original_bias=None):
        super().__init__()
        mid_channels = in_channels * rank

        self.dw = nn.Conv2d(in_channels, mid_channels, kernel_size,
                            stride=stride, padding=padding,
                            groups=in_channels, bias=False)
        self.pw_main = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.pw_skip = nn.Conv2d(in_channels, out_channels, 1,
                                 stride=stride, bias=original_bias is not None)

        if original_weight is not None:
            dw_init, pw_init, skip_init = init_residual_dw_pw(
                original_weight.detach().cpu(), rank=rank)
            self.dw.weight.data.copy_(dw_init)
            self.pw_main.weight.data.copy_(pw_init)
            self.pw_skip.weight.data.copy_(skip_init.unsqueeze(-1).unsqueeze(-1))

        if original_bias is not None:
            self.pw_skip.bias.data.copy_(original_bias.detach().cpu())

    def forward(self, x):
        return self.pw_main(self.dw(x)) + self.pw_skip(x)


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
class ConvDecompStrategy(BaseReplacementStrategy):
    """Conv3x3 -> DW+PW decomposition strategy.

    Decomposes standard 3x3 convolutions into depthwise-separable form
    using SVD-based weight initialization + per-channel correction scaler.
    Uses Huber+Cosine loss (robust to activation spikes) and
    CosineAnnealingWarmRestarts scheduler.
    """

    def __init__(self,
                 lr: float = 5e-4,
                 wd: float = 1e-4,
                 steps: int = 3000,
                 patience: int = 10,
                 min_cosine: float = 0.985,
                 scale_bounds: tuple = (0.90, 1.10),
                 skip_prefixes: list = None,
                 skip_depthwise: bool = True,
                 skip_1x1: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._lr = lr
        self._wd = wd
        self._steps = steps
        self._patience = patience
        self._min_cosine = min_cosine
        self._scale_bounds = scale_bounds
        self._skip_prefixes = skip_prefixes or []
        self._skip_depthwise = skip_depthwise
        self._skip_1x1 = skip_1x1

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
        return False

    def select_target(self, op: Operation, block: Any, graph: BaseGraph) -> bool:
        """Replace standard 3x3 Convs only. Skip DW, 1x1, and frozen prefixes."""
        if op.type != "Conv":
            return False
        if op.attributes.get("kernel_shape") != [3, 3]:
            return False
        if any(op.name.startswith(p) for p in self._skip_prefixes):
            return False
        if self._skip_depthwise:
            group = op.attributes.get("group", 1)
            in_c = op.inputs[0].shape[1] if op.inputs[0].shape is not None else 1
            if group == in_c and group > 1:
                return False
        return True

    def build_replacement(self, op: Operation, block: Any, graph: BaseGraph) -> Tuple[nn.Module, list]:
        """Build a ResidualDecomposedConv with SVD initialization."""
        input_var  = op.inputs[0]
        output_var = op.outputs[0]

        in_c    = int(input_var.shape[1])  if input_var.shape  is not None else 64
        out_c   = int(output_var.shape[1]) if output_var.shape is not None else 64
        stride  = op.attributes.get("strides", [1, 1])
        padding = op.attributes.get("pads", [0, 0, 0, 0])
        h       = int(input_var.shape[2]) if input_var.shape is not None else 32
        w       = int(input_var.shape[3]) if input_var.shape is not None else 32

        original_weight = None
        original_bias = None
        if len(op.inputs) >= 2 and op.inputs[1].value is not None:
            wt = op.inputs[1].value
            original_weight = torch.as_tensor(wt).float() if not isinstance(wt, torch.Tensor) else wt.float()
        if len(op.inputs) >= 3 and op.inputs[2].value is not None:
            b = op.inputs[2].value
            original_bias = torch.as_tensor(b).float() if not isinstance(b, torch.Tensor) else b.float()

        module = ResidualDecomposedConv(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=int(stride[0]),
            padding=int(padding[0]),
            rank=3,
            original_weight=original_weight,
            original_bias=original_bias,
        )
        return module, [1, in_c, h, w]

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
            optimizer, T_0=1000, T_mult=1, eta_min=1e-6
        )
