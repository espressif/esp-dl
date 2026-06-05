from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.head import Detect
from ultralytics.data.utils import (
    check_cls_dataset,
    check_det_dataset,
    convert_ndjson_to_yolo_if_needed,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import (
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    callbacks,
    colorstr,
    emojis,
)
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    attempt_compile,
    select_device,
    smart_inference_mode,
    torch_distributed_zero_first,
    unwrap_model,
)
import json
import torch
import torch.distributed as dist
from esp_ppq.api import load_native_graph
from esp_ppq.executor import TorchExecutor
from quantize_yolo11n.quantize_onnx_model import quant_yolo11n


class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, executor=None):
        """Execute validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = (
                    model._orig_mod
                )  # validate non-compiled original model to avoid issues
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning(
                    "validating an untrained model YAML will result in 0 mAP."
                )
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(
                        max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms
                    )
            with torch_distributed_zero_first(LOCAL_RANK):
                self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)
            model = AutoBackend(
                model=model or self.args.model,
                device=(
                    select_device(self.args.device)
                    if RANK == -1
                    else torch.device("cuda", RANK)
                ),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(
                model, "dynamic", False
            ):
                self.args.batch = model.metadata.get(
                    "batch", 1
                )  # export.py models default to batch-size 1
                LOGGER.info(
                    f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})"
                )

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
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
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(
                imgsz=(
                    1 if pt else self.args.batch,
                    self.data["channels"],
                    imgsz,
                    imgsz,
                )
            )  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
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
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(
                zip(
                    self.speed.keys(),
                    (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
                )
            )
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            # Reduce loss across all GPUs
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {
                **stats,
                **trainer.label_loss_items(
                    loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(
                    str(self.save_dir / "predictions.json"), "w", encoding="utf-8"
                ) as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


def make_quant_validator_class(executor):
    class QuantDetectionValidator(DetectionValidator):
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            super().__init__(dataloader, save_dir, args, _callbacks)
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
        bs = graph_outputs[0].shape[0]
        boxes = torch.cat(
            [
                graph_outputs[2 * i].view(
                    bs,
                    -1,
                    graph_outputs[2 * i].shape[2] * graph_outputs[2 * i].shape[3],
                )
                for i in range(3)
            ],
            dim=-1,
        )
        scores = torch.cat(
            [
                graph_outputs[2 * i + 1].view(
                    bs,
                    -1,
                    graph_outputs[2 * i + 1].shape[2]
                    * graph_outputs[2 * i + 1].shape[3],
                )
                for i in range(3)
            ],
            dim=-1,
        )
        feats = [
            torch.cat((graph_outputs[0], graph_outputs[1]), dim=1),
            torch.cat((graph_outputs[2], graph_outputs[3]), dim=1),
            torch.cat((graph_outputs[4], graph_outputs[5]), dim=1),
        ]
        detect_model = Detect(nc=80, reg_max=16, end2end=False, ch=[32, 64, 128])
        detect_model.stride = [8.0, 16.0, 32.0]
        detect_model.to(device)

        pred_dict = {
            "boxes": boxes,  # [B, 64, N] for reg_max=16
            "scores": scores,  # [B, 80, N]
            "feats": feats,  # list of 3 feature maps
        }
        y = detect_model._inference(pred_dict)

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
