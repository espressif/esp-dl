"""Example user_quant.py contract for a torch model (MobileNet-V2 / ImageNet).

Copy this file to your project, rename to ``user_quant.py``, and fill in the parts marked
``USER:``. The skill never edits this file — only ``QuantizationSettingFactory.espdl_setting()``
in iteration JSONs.

Key contract requirements:
- Define ``QUANT_CONFIG`` dict with the keys below.
- Define ``create_calib_dataloader()`` returning a torch DataLoader.
- Define ``get_torch_model()`` returning the eval()-mode nn.Module.
- Define ``evaluate(quant_graph) -> dict`` whose dict contains the ``primary_metric`` key.
- Optionally define ``evaluate_fast(quant_graph) -> dict`` for fast intermediate iterations.
- Optionally define ``collate_fn(batch)`` for esp-ppq's calibration path.

See references/contract.md for the full contract spec.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


# ---------------------------------------------------------------------------
# QUANT_CONFIG — knobs the skill reads but never modifies.
# ---------------------------------------------------------------------------
QUANT_CONFIG = {
    "model_type": "torch",
    "input_shape": [3, 224, 224],
    "batch_size": 32,
    "target": "esp32s3",  # esp32s3 / esp32p4 / c
    "num_of_bits": 8,
    "device": "cpu",
    "calib_steps": 32,
    "primary_metric": "top1",
    "metric_direction": "max",
    "target_metric": 65.0,  # stop iterating when top1 reaches this
    "analyse_steps": 8,
    "top_k_layers": 20,
}

# Paths are resolved relative to this file's directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(_HERE, "imagenet", "calib")
EVAL_DIR = os.path.join(
    _HERE, "imagenet", "calib"
)  # USER: replace with full validation set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calib_collate(samples) -> torch.Tensor:
    """Stack ImageFolder samples into a single tensor (drop labels)."""
    return torch.cat([sample[0].unsqueeze(0) for sample in samples], dim=0)


def _convert_relu6_to_relu(model: nn.Module) -> nn.Module:
    """MobileNet-V2 uses ReLU6 which interacts badly with weight equalization;
    replace with ReLU before quantization. The skill suggests this when it
    sees the layer-wise equalization rule firing on a depthwise conv chain.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            _convert_relu6_to_relu(child)
    return model


# ---------------------------------------------------------------------------
# Required exports
# ---------------------------------------------------------------------------


def get_torch_model() -> nn.Module:
    model = torchvision.models.mobilenet.mobilenet_v2(
        weights=MobileNet_V2_Weights.IMAGENET1K_V1
    )
    # USER: keep this if you plan to enable equalization; remove otherwise.
    model = _convert_relu6_to_relu(model)
    model.eval()
    return model.to(QUANT_CONFIG["device"])


def create_calib_dataloader() -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(CALIB_DIR, transform)
    dataset = Subset(dataset, indices=list(range(min(1024, len(dataset)))))
    return DataLoader(
        dataset=dataset,
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=_calib_collate,
    )


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """Sent into espdl_quantize_*; just move to the configured device."""
    return batch.to(QUANT_CONFIG["device"])


def _accuracy(output, target, topk: Tuple[int, ...]):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res


def _evaluate_loop(quant_graph, batch_limit: int) -> dict:
    """Shared eval body. ``batch_limit`` caps batches when called from evaluate_fast."""
    from esp_ppq.executor.torch import TorchExecutor

    eval_loader = DataLoader(
        dataset=datasets.ImageFolder(
            EVAL_DIR,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    executor = TorchExecutor(graph=quant_graph, device=QUANT_CONFIG["device"])
    top1_acc, top5_acc, n = 0.0, 0.0, 0
    for idx, (batch_input, batch_label) in enumerate(eval_loader):
        batch_input = batch_input.to(QUANT_CONFIG["device"])
        batch_label = batch_label.to(QUANT_CONFIG["device"])
        outputs = executor(*[batch_input])
        pred = outputs[0]
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        prec1, prec5 = _accuracy(pred.cpu(), batch_label.cpu(), topk=(1, 5))
        top1_acc += prec1.item()
        top5_acc += prec5.item()
        n += 1
        if batch_limit is not None and idx + 1 >= batch_limit:
            break
    if n == 0:
        return {"top1": float("nan"), "top5": float("nan")}
    return {"top1": top1_acc / n, "top5": top5_acc / n, "batches": n}


def evaluate(quant_graph) -> dict:
    """Full evaluation. Called once on the chosen best iteration."""
    return _evaluate_loop(quant_graph, batch_limit=None)


def evaluate_fast(quant_graph) -> dict:
    """Fast evaluation used during iteration. 4 batches ≈ 30s on CPU."""
    return _evaluate_loop(quant_graph, batch_limit=4)
