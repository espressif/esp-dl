"""Example user_quant.py contract for an ONNX detection model (YOLO11n / COCO).

Copy this file to your project, rename to ``user_quant.py``, and fill in the parts marked
``USER:``. The skill never edits this file — only ``QuantizationSettingFactory.espdl_setting()``
in iteration JSONs.

This template is suitable for object-detection / pose / segmentation models exported as
ONNX. The skill expects ``evaluate(quant_graph)`` to return a dict that includes a key
matching ``QUANT_CONFIG['primary_metric']`` (``map50`` here, but use whatever your eval
returns — mAP@0.5, mAP@0.5:0.95, F1, etc.).

See references/contract.md for the full contract spec.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# QUANT_CONFIG
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

QUANT_CONFIG = {
    "model_type": "onnx",
    "onnx_path": os.path.join(_HERE, "models", "yolo11n.onnx"),  # USER: edit
    "input_shape": [3, 640, 640],
    "batch_size": 16,  # USER: drop to 8 if RAM-tight
    "target": "esp32p4",
    "num_of_bits": 8,
    "device": "cpu",
    "calib_steps": 32,
    "primary_metric": "map50",  # USER: pick what your eval returns
    "metric_direction": "max",
    "analyse_steps": 8,
    "top_k_layers": 20,
}

CALIB_DIR = os.path.join(
    _HERE, "calib_yolo11n"
)  # USER: directory of calibration images
EVAL_DATASET_PATH = os.path.join(_HERE, "datasets", "coco_val")  # USER: validation set


# ---------------------------------------------------------------------------
# Calibration dataloader
# ---------------------------------------------------------------------------
class CalibDataset(Dataset):
    def __init__(self, root: str, img_size: int) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )
        self.paths = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def create_calib_dataloader() -> DataLoader:
    img_size = QUANT_CONFIG["input_shape"][-1]
    return DataLoader(
        dataset=CalibDataset(CALIB_DIR, img_size=img_size),
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
    )


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(QUANT_CONFIG["device"])


# ---------------------------------------------------------------------------
# Evaluation — USER: replace the body with your real eval
# ---------------------------------------------------------------------------
def evaluate(quant_graph) -> dict:
    """Run the user's full validation loop.

    USER: this is the only function that depends on your task. Below is a sketch — fill
    in with whatever runs your model end-to-end (YOLO post-processing, COCO mAP, etc.).
    The returned dict must contain ``QUANT_CONFIG['primary_metric']``.
    """
    from esp_ppq.executor.torch import TorchExecutor

    executor = TorchExecutor(graph=quant_graph, device=QUANT_CONFIG["device"])

    # --- USER: load your validation set, run forward, post-process, compute metrics ---
    # Pseudocode:
    # for image, gt in coco_val_loader:
    #     preds_raw = executor(*[image.to(QUANT_CONFIG["device"])])
    #     boxes = postprocess_yolo(preds_raw)   # NMS, scale to original image size
    #     update_coco_evaluator(image_id, boxes, gt)
    # map50 = coco_evaluator.compute("map_50")
    # map5095 = coco_evaluator.compute("map_50_95")
    # return {"map50": map50, "map5095": map5095}
    raise NotImplementedError(
        "Replace evaluate() body with your real validation loop. "
        "Return dict including QUANT_CONFIG['primary_metric']."
    )


def evaluate_fast(quant_graph) -> dict:
    """Optional fast eval — same metric keys, smaller subset.

    USER: e.g. evaluate on 200 images instead of 5000. Used between iterations.
    """
    return evaluate(quant_graph)  # USER: replace with a faster version
