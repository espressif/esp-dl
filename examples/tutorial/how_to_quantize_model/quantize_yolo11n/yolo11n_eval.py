from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.head import Detect
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    de_parallel,
    select_device,
    smart_inference_mode,
)
import json
import torch
from ppq.api import load_native_graph
from ppq.executor import TorchExecutor
from quantize_onnx_model import quant_yolo11n


class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, executor=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning(
                    "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP."
                )
            callbacks.add_integration_callbacks(self)

            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )

            # self.model = model
            self.device = model.device  # update device

            self.args.half = model.fp16  # update half

            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get(
                    "batch", 1
                )  # export.py models default to batch-size 1
                LOGGER.info(
                    f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})"
                )

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(
                        f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                    )
                )

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )

            if not pt:
                self.args.rect = False  # set to false

            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
            )  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                preds = ppq_graph_inference(executor, "detect", batch["img"], "cpu")
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats


def make_quant_validator_class(executor):
    class QuantDetectionValidator(DetectionValidator):
        def __init__(
            self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
        ):
            super().__init__(dataloader, save_dir, pbar, args, _callbacks)
            self.executor = executor

        def __call__(self, trainer=None, model=None):
            return QuantizedModelValidator.__call__(self, trainer, model, self.executor)

    return QuantDetectionValidator


def ppq_graph_init(quant_func, imgsz, device, native_path=None):
    """
    Init ppq graph inference.
        # case 1: PTQ graph validation: ppq_graph = quant_func()
        # case 2: QAT graph validation:
                    utilize .native to load the graph
                    while training, the .native model is saved along with .espdl model
    """
    if native_path is not None:
        ppq_graph = load_native_graph(native_path)
    else:
        ppq_graph = quant_func(imgsz)

    executor = TorchExecutor(graph=ppq_graph, device=device)
    return executor


def ppq_graph_inference(executor, task, inputs, device):
    """ppq graph inference"""
    graph_outputs = executor(inputs)
    if task == "detect":
        x = [
            torch.cat((graph_outputs[i], graph_outputs[i + 1]), 1)
            for i in range(0, 6, 2)
        ]
        detect_model = Detect(nc=80, ch=[32, 64, 128])
        detect_model.stride = [8.0, 16.0, 32.0]
        detect_model.to(device)

        y = detect_model._inference(x)

        return y
    else:
        raise NotImplementedError(f"{task} is not supported.")


if __name__ == "__main__":
    executor = ppq_graph_init(quant_yolo11n, 640, "cpu")
    QuantDetectionValidator = make_quant_validator_class(executor)
    # eval quantized yolo11n model
    model = YOLO("yolo11n.pt")
    results = model.val(
        data="coco.yaml",
        split="val",
        imgsz=640,
        device="cpu",
        validator=QuantDetectionValidator,
        rect=False,
        save_json=True,
    )
