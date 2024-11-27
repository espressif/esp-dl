import os
import subprocess
from typing import Iterable

import torch
import torchvision
from datasets.imagenet_util import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)
from ppq import QuantizationSettingFactory, QuantizationSetting
from ppq.api import espdl_quantize_torch, get_target_platform
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torch.nn as nn


def convert_relu6_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_relu6_to_relu(child)
    return model


def quant_setting_mobilenet_v2(
    model: nn.Module,
    optim_quant_method: list[str] = None,
) -> tuple[QuantizationSetting, nn.Module]:
    """Quantize torch model with optim_quant_method.

    Args:
        optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                        the layers to 16-bit quantization. You can remove or add layers according to your
                                        needs.
        -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                Since ReLU6 exists in MobilenetV2, convert ReLU6 to ReLU for better precision.

    Returns:
        [tuple]: [QuantizationSetting, nn.Module]
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if optim_quant_method is not None:
        if "MixedPrecision_quantization" in optim_quant_method:
            # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
            # You can remove or add layers according to your needs.
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.0/Conv",
                get_target_platform(TARGET, 16),
            )
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.2/Clip",
                get_target_platform(TARGET, 16),
            )
        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            # layerwise equalization
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = 4
            quant_setting.equalization_setting.value_threshold = 0.4
            quant_setting.equalization_setting.opt_level = 2
            quant_setting.equalization_setting.interested_layers = None
            # replace ReLU6 with ReLU
            model = convert_relu6_to_relu(model)
        else:
            raise ValueError(
                "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
            )

    return quant_setting, model


BATCH_SIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = (
    "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
)
TARGET = "esp32p4"
NUM_OF_BITS = 8
ESPDL_MODLE_PATH = "models/torch/mobilenet_v2.espdl"
CALIB_DIR = "./imagenet/calib"

if not os.path.exists(CALIB_DIR):
    os.makedirs(CALIB_DIR)
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

# create a setting for quantizing your network with ESPDL.
# if you don't need to optimize quantization, set the input 1 of the quant_setting_mobilenet_v2 function None
# Example: Using LayerwiseEqualization_quantization
quant_setting, model = quant_setting_mobilenet_v2(
    model, ["LayerwiseEqualization_quantization"]
)

quant_ppq_graph = espdl_quantize_torch(
    model=model,
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
