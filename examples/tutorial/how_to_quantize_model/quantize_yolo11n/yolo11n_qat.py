import ppq.lib as PFL
import torch
from ppq.core import QuantizationVisibility, TargetPlatform
from ppq.executor import TorchExecutor
from ppq.quantization.optim import *
from trainer import Trainer, CaliDataset, TrainDataset
from ppq.api import get_target_platform
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from ppq.api.interface import load_onnx_graph
import os
import onnxruntime as ort
import zipfile
import urllib.request


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def qat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CFG_BATCHSIZE = 128
    CFG_TRAIN_DIR = "COCO2017/images/train2017"  # change to your own path
    CFG_PLATFORM = get_target_platform("esp32p4", 8)
    EPOCH = 10  # please set a reasonable epoch number

    # calibration setting
    calib_steps = 32
    cali_path = "calib_yolo11n"
    yolo11n_caib_url = "https://dl.espressif.com/public/calib_yolo11n.zip"

    urllib.request.urlretrieve(
        yolo11n_caib_url, "calib_yolo11n.zip", reporthook=report_hook
    )

    with zipfile.ZipFile("calib_yolo11n.zip", "r") as zip_file:
        zip_file.extractall("./")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(
        script_dir, "../../../../models/coco_detect/models/yolo11n.onnx"
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
    executor = TorchExecutor(graph=graph, device=device)
    executor.tracing_operation_meta(inputs=torch.zeros([1, 3, 640, 640]).to(device))

    train_set = TrainDataset(CFG_TRAIN_DIR)
    training_dataloader = DataLoader(train_set, batch_size=CFG_BATCHSIZE, shuffle=True)

    calibration_dataset = CaliDataset(cali_path)
    cali_iter = DataLoader(
        dataset=calibration_dataset, batch_size=CFG_BATCHSIZE, shuffle=False
    )

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
        collate_fn=(lambda x: x.type(torch.float).to(device)),
        graph=graph,
        dataloader=cali_iter,
        executor=executor,
    )
    print(
        f"Calibrate images number: {len(cali_iter.dataset)}, len(Calibrate iter): {len(cali_iter)}"
    )

    # Start QAT train, only support single GPU when using cuda
    # ------------------------------------------------------------
    trainer = Trainer(graph=graph, onnx_path=ONNX_PATH, device=device)
    best_mAP50_95 = 0
    for epoch in range(EPOCH):
        trainer.epoch(training_dataloader)
        if not os.path.exists(str(epoch)):
            os.mkdir(str(epoch))
        qat_graph = trainer.save(
            os.path.join(str(epoch), "coco_detect_yolo11n_s8_v3.espdl"),
            os.path.join(str(epoch), "coco_detect_yolo11n_s8_v3.native"),
        )
        # evaluate trained quantized model per epoch
        current_mAP50_95 = trainer.eval()
        print(f"Epoch: {epoch}, mAP: {current_mAP50_95}")

        if current_mAP50_95 > best_mAP50_95:
            trainer.save("Best.espdl", "Best.native")

    return qat_graph


if __name__ == "__main__":
    qat_graph = qat()
