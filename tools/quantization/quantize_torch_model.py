import os
import subprocess
from typing import Iterable

import torch
import torchvision
from datasets.imagenet_util import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)
from ppq.api import espdl_quantize_torch, get_target_platform
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

BATCH_SIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = (
    "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
)
TARGET = "esp32p4"
NUM_OF_BITS = 8
ESPDL_MODLE_PATH = "models/torch/mobilenet_v2.espdl"
CALIB_DIR = "./imagenet/calib"

# Download mobilenet_v2 from torchvision and dataset
model = torchvision.models.mobilenet.mobilenet_v2(
    weights=MobileNet_V2_Weights.IMAGENET1K_V1
)
model = model.to(DEVICE)
imagenet_url = "https://dl.espressif.com/public/imagenet_calib.zip"
subprocess.run(["wget", imagenet_url])
subprocess.run(["unzip", "imagenet_calib.zip", "-d", CALIB_DIR])
CALIB_DIR = os.path.join(CALIB_DIR, "calib")

# -------------------------------------------
# Prepare Calibration Dataset
# --------------------------------------------
if os.path.exists(CALIB_DIR):
    print(f"load imagenet calibration dataset from directory: {CALIB_DIR}")
    dataloader = load_imagenet_from_directory(
        directory=CALIB_DIR,
        batchsize=BATCH_SIZE,
        shuffle=False,
        subset=1024,
        require_label=False,
        num_of_workers=4,
    )
else:
    # Random calibration dataset only for debug
    print("load random calibration dataset")

    def load_random_calibration_dataset() -> Iterable:
        return [torch.rand(size=INPUT_SHAPE) for _ in range(BATCH_SIZE)]

    # Load training data for creating a calibration dataloader.
    dataloader = DataLoader(
        dataset=load_random_calibration_dataset(), batch_size=BATCH_SIZE, shuffle=False
    )


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


# -------------------------------------------
# Quantize Torch Model.
# --------------------------------------------
# These layers have large errors in 8-bit quantization, dispatching to 16-bit quantization.
# You can remove or add layers according to your needs.
dispatching_override = {
    "/features/features.1/conv/conv.0/conv.0.0/Conv": get_target_platform(TARGET, 16),
    "/features/features.1/conv/conv.0/conv.0.2/Clip": get_target_platform(TARGET, 16),
}
quant_ppq_graph, _ = espdl_quantize_torch(
    model=model,
    espdl_export_file=ESPDL_MODLE_PATH,
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=[1] + INPUT_SHAPE,
    target=TARGET,
    num_of_bits=NUM_OF_BITS,
    collate_fn=collate_fn,
    dispatching_override=dispatching_override,
    device=DEVICE,
    error_report=True,
    skip_export=False,
    export_test_values=False,
    verbose=1,
)

# -------------------------------------------
# Evaluate Quantized Model.
# --------------------------------------------
evaluate_ppq_module_with_imagenet(
    model=quant_ppq_graph,
    imagenet_validation_dir=CALIB_DIR,
    batchsize=BATCH_SIZE,
    device=DEVICE,
    verbose=1,
)
