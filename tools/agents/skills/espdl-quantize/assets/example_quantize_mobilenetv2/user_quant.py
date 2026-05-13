"""User contract module for the espdl-quantize skill — MobileNet-V2 / ImageNet.

Self-contained example: downloads MobileNet-V2 weights via torchvision and the 1024-image
ImageNet calibration subset from dl.espressif.com on first run. After that, calibration
and evaluation reuse the cached files.

The skill harness (`.cursor/skills/espdl-quantize/scripts/run_iteration.py`) imports this
module and calls:

  - QUANT_CONFIG          — knobs the skill reads but NEVER modifies
  - get_torch_model()     — returns the eval()-mode nn.Module
  - create_calib_dataloader() — returns a DataLoader of input tensors
  - collate_fn(batch)     — used during esp-ppq calibration
  - evaluate(quant_graph) — full evaluation, runs once on the chosen best iteration
  - evaluate_fast(quant_graph) — fast eval used during iteration

The skill ONLY tunes QuantizationSettingFactory.espdl_setting() — it doesn't touch
this file. See ``.cursor/skills/espdl-quantize/references/contract.md`` for the full spec.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


# ---------------------------------------------------------------------------
# QUANT_CONFIG — read by the skill, never modified.
# ---------------------------------------------------------------------------
QUANT_CONFIG = {
    "model_type": "torch",
    "input_shape": [3, 224, 224],
    "batch_size": 32,
    "target": "esp32s3",  # esp32s3 / esp32p4 / c
    "num_of_bits": 8,
    "device": "cuda",
    "calib_steps": 32,
    "primary_metric": "top1",
    "metric_direction": "max",
    "target_metric": 71.878,  # stop iterating when top1 reaches this
    "analyse_steps": 8,
    "top_k_layers": 20,
}

# All paths are resolved relative to this file. The harness runs in the current Python
# interpreter and resolves any relative path in QUANT_CONFIG against this file's directory,
# so relative paths here continue to work regardless of the user's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGENET_DIR = os.path.join(_HERE, "imagenet")
_CALIB_DIR = os.path.join(_IMAGENET_DIR, "calib")
_CALIB_ZIP = os.path.join(_HERE, "imagenet_calib.zip")
_CALIB_URL = "https://dl.espressif.com/public/imagenet_calib.zip"


# ---------------------------------------------------------------------------
# One-time data download helper. Idempotent.
# ---------------------------------------------------------------------------
def _ensure_calib_dataset() -> str:
    """Download and extract the 1024-image ImageNet calibration subset on first call."""
    if os.path.isdir(_CALIB_DIR) and os.listdir(_CALIB_DIR):
        return _CALIB_DIR

    os.makedirs(_IMAGENET_DIR, exist_ok=True)

    if not os.path.exists(_CALIB_ZIP):
        print(f"[user_quant] Downloading {_CALIB_URL} -> {_CALIB_ZIP}")

        def _hook(blocknum, blocksize, total):
            if total <= 0:
                return
            pct = (blocknum * blocksize / total) * 100
            print(f"\r[user_quant] download progress: {pct:5.1f}%", end="")

        urllib.request.urlretrieve(_CALIB_URL, _CALIB_ZIP, reporthook=_hook)
        print()

    print(f"[user_quant] Extracting {_CALIB_ZIP} -> {_IMAGENET_DIR}")
    with zipfile.ZipFile(_CALIB_ZIP, "r") as z:
        z.extractall(_IMAGENET_DIR)
    return _CALIB_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calib_collate(samples) -> torch.Tensor:
    """ImageFolder yields (img, label); calibration only needs the image tensor."""
    return torch.cat([sample[0].unsqueeze(0) for sample in samples], dim=0)


def _convert_relu6_to_relu(model: nn.Module) -> nn.Module:
    """MobileNet-V2 uses ReLU6, which interacts badly with weight equalization. Swapping
    in ReLU before quantization keeps the model behaviour identical (ReLU6 is just
    clip(x, 0, 6); ImageNet activations rarely exceed 6 after the trained BN/Conv).
    The decision_playbook tells the agent to enable equalization on the Conv→Conv pairs
    in this network — that pass works much better with plain ReLU.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            _convert_relu6_to_relu(child)
    return model


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# Required exports.
# ---------------------------------------------------------------------------
def get_torch_model() -> nn.Module:
    model = torchvision.models.mobilenet.mobilenet_v2(
        weights=MobileNet_V2_Weights.IMAGENET1K_V1
    )
    # Replace ReLU6 with ReLU for better weight-equalization behaviour. Safe pre-quant
    # transform — accuracy on float ImageNet validation barely changes (<0.2%).
    model = _convert_relu6_to_relu(model)
    model.eval()
    return model.to(QUANT_CONFIG["device"])


def create_calib_dataloader() -> DataLoader:
    calib_dir = _ensure_calib_dataset()
    dataset = datasets.ImageFolder(calib_dir, _eval_transform())
    # Subset to 1024 images so calibration finishes fast.
    dataset = Subset(dataset, indices=list(range(min(1024, len(dataset)))))
    return DataLoader(
        dataset=dataset,
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=_calib_collate,
    )


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """Sent into espdl_quantize_torch — move tensors to the configured device."""
    return batch.to(QUANT_CONFIG["device"])


def evaluate(quant_graph) -> dict:
    """Full evaluation. Delegates to the shared ImageNet evaluator shipped with the
    esp-dl tutorial (``datasets/imagenet_util.py``). That helper wires a
    :class:`esp_ppq.executor.torch.TorchExecutor` around the quantized PPQ graph and
    walks the validation directory reporting top-1 / top-5 accuracy. We wrap the
    ``(dataframe, top1, top5)`` triple it returns into the ``dict`` shape required by
    the espdl-quantize skill harness — it must at minimum contain the key named in
    ``QUANT_CONFIG["primary_metric"]`` (``"top1"`` here).
    """
    # Imported lazily so the heavy esp_ppq / pandas / tqdm deps only load when the
    # harness actually runs an evaluation. The harness puts this file's directory on
    # sys.path, which makes the local ``datasets`` package importable.
    from datasets.imagenet_util import evaluate_ppq_module_with_imagenet

    eval_dir = (
        _ensure_calib_dataset()
    )  # reuse calib subset as the eval set in this demo
    _, top1, top5 = evaluate_ppq_module_with_imagenet(
        model=quant_graph,
        imagenet_validation_dir=eval_dir,
        batchsize=QUANT_CONFIG["batch_size"],
        device=QUANT_CONFIG["device"],
        verbose=False,
    )
    return {"top1": float(top1), "top5": float(top5)}


# ---------------------------------------------------------------------------
# Standalone smoke check: `python user_quant.py` validates the contract without
# importing esp-ppq itself. Useful for a fast sanity check before invoking the
# skill harness.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("QUANT_CONFIG:", QUANT_CONFIG)
    print("Loading MobileNet-V2 ...")
    model = get_torch_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e6:.2f}M")
    print("Building calibration dataloader (this may download data the first time) ...")
    loader = create_calib_dataloader()
    sample = next(iter(loader))
    print(f"  one batch: shape={tuple(sample.shape)} dtype={sample.dtype}")
    print(
        "Contract OK. Now run the skill harness, e.g.:\n"
        "    python .cursor/skills/espdl-quantize/scripts/run_iteration.py \\\n"
        "        --user-quant example_quantize_mobilenetv2/user_quant.py \\\n"
        "        --output-dir example_quantize_mobilenetv2/outputs/iter_0 --baseline\n"
        "or just ask the agent to use the espdl-quantize skill."
    )
