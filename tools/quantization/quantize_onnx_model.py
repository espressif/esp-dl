import os
import subprocess
from typing import Iterable

import torch
from datasets.imagenet_util import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)
from ppq import QuantizationSettingFactory
from ppq.api import espdl_quantize_onnx, get_target_platform
from torch.utils.data import DataLoader

BATCH_SIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = (
    "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
)
TARGET = "esp32p4"
NUM_OF_BITS = 8
ONNX_PATH = "./models/onnx/mobilenet_v2.onnx"  #'models/onnx/mobilenet_v2.onnx'
ESPDL_MODLE_PATH = "models/onnx/mobilenet_v2.espdl"
CALIB_DIR = "./imagenet"

# Download mobilenet_v2 model from onnx models and dataset
model_url = "https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx"
imagenet_url = "https://dl.espressif.com/public/imagenet_calib.zip"
os.makedirs(ONNX_PATH, exist_ok=True)
os.makedirs(CALIB_DIR, exist_ok=True)
subprocess.run(["wget", model_url, "-O", ONNX_PATH])
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
# Quantize ONNX Model.
# --------------------------------------------
# These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
# You can remove or add layers according to your needs.

# create a setting for quantizing your network with ESPDL.
quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.dispatching_table.append(
    "/features/features.1/conv/conv.0/conv.0.0/Conv", get_target_platform(TARGET, 16)
)
quant_setting.dispatching_table.append(
    "/features/features.1/conv/conv.0/conv.0.2/Clip", get_target_platform(TARGET, 16)
)

quant_ppq_graph, _ = espdl_quantize_onnx(
    onnx_import_file=ONNX_PATH,
    espdl_export_file=ESPDL_MODLE_PATH,
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=[1] + INPUT_SHAPE,
    target=TARGET,
    num_of_bits=NUM_OF_BITS,
    collate_fn=collate_fn,
    setting=quant_setting,
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
