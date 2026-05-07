# please replace this script to your customized evaluation script
import os
from quantize_mobilenetv2.datasets.imagenet_util import (
    evaluate_ppq_module_with_imagenet,
)

from auto_quant.config import runtime_config


def evaluation(quant_graph):
    CALIB_DIR = "./quantize_mobilenetv2/imagenet"
    CALIB_DIR = os.path.join(CALIB_DIR, "calib")

    _, top1, top5 = evaluate_ppq_module_with_imagenet(
        model=quant_graph,
        imagenet_validation_dir=CALIB_DIR,
        batchsize=32,
        device=runtime_config["device"],
        verbose=0,
    )
    return {"top1": float(top1), "top5": float(top5)}
