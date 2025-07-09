import ppq.lib as PFL
import torch
from ppq.core import QuantizationVisibility, TargetPlatform
from ppq.executor import TorchExecutor
from ppq.quantization.optim import *

from trainer import Trainer, CaliDataset
from ppq.api import get_target_platform
from torch.utils.data import DataLoader
from PIL import Image
from ppq.api.interface import load_onnx_graph
from ultralytics.data.dataset import YOLODataset
import yaml
import re
import os
import random
import numpy as np
import zipfile

imgsz = 640


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataset(batchsz=32, imgsz=640, num_workers=8):
    # train dataset
    dataset = YOLODataset
    yaml_path = "coco-pose.yaml"
    with open(yaml_path, errors="ignore", encoding="utf-8") as f:
        s = f.read()
        s = re.sub(
            r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
            "",
            s,
        )
    data = yaml.safe_load(s)

    data["nc"] = len(data["names"])

    train_dataset = dataset(
        img_path="coco-pose/images/train2017",
        imgsz=imgsz,
        batch_size=batchsz,
        augment=False,
        rect=False,
        cache=None,
        task="pose",
        data=data,
    )

    g = torch.Generator()
    g.manual_seed(1234)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsz,
        shuffle=True,
        num_workers=min(
            num_workers, os.cpu_count() // max(torch.cuda.device_count(), 1)
        ),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_init_fn,
        generator=g,  # shuffle seed
    )

    return train_loader


def qat():
    seed = 1234
    seed_everything(seed)
    CFG_BATCHSIZE = 128
    CFG_PLATFORM = get_target_platform("esp32p4", 8)
    cali_path = "calib_yolo11n-pose"
    calib_steps = 32
    EPOCH = 6

    with zipfile.ZipFile("calib_yolo11n-pose.zip", "r") as zip_file:
        zip_file.extractall("./")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(
        script_dir, "../../../../models/coco_pose/models/yolo11n-pose.onnx"
    )

    graph = load_onnx_graph(onnx_import_file=ONNX_PATH)

    # quant and clibration
    quantizer = PFL.Quantizer(platform=CFG_PLATFORM, graph=graph)

    dispatching_table = PFL.Dispatcher(graph=graph, method="conservative").dispatch(
        quantizer.quant_operation_types
    )
    dispatching_override = None
    # override dispatching result
    if dispatching_override is not None:
        for opname, platform in dispatching_override.items():
            if opname not in graph.operations:
                continue
            assert isinstance(platform, int) or isinstance(platform, TargetPlatform), (
                f"Your dispatching_override table contains a invalid setting of operation {opname}, "
                "All platform setting given in dispatching_override table is expected given as int or TargetPlatform, "
                f"however {type(platform)} was given."
            )
            dispatching_table[opname] = TargetPlatform(platform)

    for opname, platform in dispatching_table.items():
        if platform == TargetPlatform.UNSPECIFIED:
            dispatching_table[opname] = TargetPlatform(quantizer.target_platform)

    # init quant information
    for op in graph.operations.values():
        quantizer.quantize_operation(
            op_name=op.name, platform=dispatching_table[op.name]
        )
    executor = TorchExecutor(graph=graph)
    executor.tracing_operation_meta(
        inputs=torch.zeros([1, 3, imgsz, imgsz]).cuda()
    )  # .cuda()

    # calibration dataset
    calibration_dataset = CaliDataset(cali_path)
    cali_iter = DataLoader(
        dataset=calibration_dataset, batch_size=CFG_BATCHSIZE, shuffle=False
    )

    train_loader = get_dataset()

    pipeline = PFL.Pipeline(
        [
            QuantizeSimplifyPass(),
            QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
            ParameterQuantizePass(),
            RuntimeCalibrationPass(method="kl"),
            PassiveParameterQuantizePass(
                clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE
            ),
            QuantAlignmentPass(elementwise_alignment="Align to Output"),
        ]
    )

    pipeline.optimize(
        calib_steps=calib_steps,
        collate_fn=(lambda x: x.type(torch.float).to("cuda")),
        graph=graph,
        dataloader=cali_iter,
        executor=executor,
    )
    print(
        f"Calibrate images number: {len(cali_iter.dataset)}, len(Calibrate iter): {len(cali_iter)}"
    )

    # Start QAT train, only support single GPU

    # ------------------------------------------------------------
    trainer = Trainer(graph=graph)

    current_pose_mAP50_95 = trainer.eval()  # before training
    best_pose_mAP50_95 = 0
    for epoch in range(EPOCH):
        trainer.epoch(train_loader)
        if not os.path.exists(str(epoch)):
            os.mkdir(str(epoch))
        qat_graph = trainer.save(
            os.path.join(str(epoch), "coco_pose_yolo11n_pose_s8_p4_v2.espdl"),
            os.path.join(str(epoch), "coco_pose_yolo11n_pose_s8_p4_v2.native"),
        )
        # evaluate trained quantized model per epoch
        current_pose_mAP50_95 = trainer.eval()
        print(f"Epoch: {epoch}, mAP: {current_pose_mAP50_95}")

        if current_pose_mAP50_95 > best_pose_mAP50_95:
            trainer.save("Best_p4.espdl", "Best_p4.native")
            best_pose_mAP50_95 = current_pose_mAP50_95
    return qat_graph


if __name__ == "__main__":
    qat_graph = qat()
