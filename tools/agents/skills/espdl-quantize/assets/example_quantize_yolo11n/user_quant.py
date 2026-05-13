"""User contract module for the espdl-quantize skill — YOLO11n / COCO.

This contract imports data-loading helpers from the copied
``quantize_onnx_model.py`` and evaluation helpers from the copied
``yolo11n_eval.py`` (both live in this directory).  The skill harness
imports this module and calls:

  - QUANT_CONFIG          — knobs the skill reads but NEVER modifies
  - create_calib_dataloader() — returns a DataLoader of input tensors
  - collate_fn(batch)     — used during esp-ppq calibration
  - evaluate(quant_graph) — full evaluation, runs once on the chosen best iteration
  - evaluate_fast(quant_graph) — fast eval used during iteration (optional)

See ``.cursor/skills/espdl-quantize/references/contract.md`` for the full spec.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


_HERE = os.path.dirname(os.path.abspath(__file__))

QUANT_CONFIG = {
    "model_type": "onnx",
    "onnx_path": os.path.join(
        _HERE, "..", "esp-dl", "models", "coco_detect", "models", "yolo11n.onnx"
    ),
    "input_shape": [3, 640, 640],
    "batch_size": 32,
    "target": "esp32s3",
    "num_of_bits": 8,
    "device": "cuda",
    "calib_steps": 32,
    "primary_metric": "map5095",
    "metric_direction": "max",
    "analyse_steps": 8,
    "top_k_layers": 20,
}

_CALIB_DIR = os.path.join(_HERE, "calib_yolo11n")
_CALIB_ZIP = os.path.join(_HERE, "calib_yolo11n.zip")
_CALIB_URL = "https://dl.espressif.com/public/calib_yolo11n.zip"


class CaliDataset(Dataset):
    def __init__(self, path, img_shape=640):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_shape, img_shape)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img)
        return img


def _ensure_calib_dataset() -> str:
    """Download and extract the YOLO11n calibration subset on first call."""
    if os.path.isdir(_CALIB_DIR) and os.listdir(_CALIB_DIR):
        return _CALIB_DIR

    if not os.path.exists(_CALIB_ZIP):
        print(f"[user_quant] Downloading {_CALIB_URL} -> {_CALIB_ZIP}")

        def _hook(blocknum, blocksize, total):
            if total <= 0:
                return
            pct = (blocknum * blocksize / total) * 100
            print(f"\r[user_quant] download progress: {pct:5.1f}%", end="")

        urllib.request.urlretrieve(_CALIB_URL, _CALIB_ZIP, reporthook=_hook)
        print()

    print(f"[user_quant] Extracting {_CALIB_ZIP} -> {_HERE}")
    with zipfile.ZipFile(_CALIB_ZIP, "r") as z:
        z.extractall(_HERE)
    return _CALIB_DIR


def create_calib_dataloader() -> DataLoader:
    calib_dir = _ensure_calib_dataset()
    img_size = QUANT_CONFIG["input_shape"][-1]
    dataset = CaliDataset(calib_dir, img_shape=img_size)
    return DataLoader(
        dataset=dataset,
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(QUANT_CONFIG["device"])


def evaluate(quant_graph) -> dict:
    """Full evaluation on COCO val using the quantized PPQ graph.

    Delegates to the validator infrastructure shipped with the esp-dl tutorial
    (``yolo11n_eval.py``).  That helper wires a :class:`esp_ppq.executor.TorchExecutor`
    around the quantized PPQ graph and runs the Ultralytics detection validator.
    """
    from esp_ppq.executor import TorchExecutor
    from ultralytics import YOLO
    from yolo11n_eval import make_quant_validator_class

    executor = TorchExecutor(graph=quant_graph, device=QUANT_CONFIG["device"])
    QuantDetectionValidator = make_quant_validator_class(executor)

    model = YOLO("yolo11n.pt")
    results = model.val(
        data="coco.yaml",
        split="val",
        imgsz=QUANT_CONFIG["input_shape"][-1],
        device=QUANT_CONFIG["device"],
        validator=QuantDetectionValidator,
        rect=False,
        save_json=True,
    )

    if hasattr(results, "box"):
        metrics = results
    elif hasattr(results, "metrics") and hasattr(results.metrics, "box"):
        metrics = results.metrics
    else:
        raise RuntimeError(
            f"Cannot extract box metrics from results type {type(results)}"
        )

    return {
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map5095": float(getattr(metrics.box, "map", 0.0)),
    }


if __name__ == "__main__":
    print("QUANT_CONFIG:", QUANT_CONFIG)
    print("Building calibration dataloader (this may download data the first time) ...")
    loader = create_calib_dataloader()
    sample = next(iter(loader))
    print(f"  one batch: shape={tuple(sample.shape)} dtype={sample.dtype}")
    print(
        "Contract OK. Now run the skill harness, e.g.:\n"
        "    python .cursor/skills/espdl-quantize/scripts/run_iteration.py \\\n"
        "        --user-quant example_quantize_yolo11n/user_quant.py \\\n"
        "        --output-dir example_quantize_yolo11n/outputs/iter_0 --baseline\n"
        "or just ask the agent to use the espdl-quantize skill."
    )
