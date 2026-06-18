import json
import os
import torch
import torch.distributed as dist
from esp_ppq.api import load_native_graph
from esp_ppq.executor import TorchExecutor
from ultralytics import YOLO
from ultralytics.data.utils import (
    check_cls_dataset,
    check_det_dataset,
    convert_ndjson_to_yolo_if_needed,
)
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.head import Detect
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

from quantize_yolo11n_seg.quantize_onnx_model import quant_yolo11n_seg

# yolo11n-seg head config (width-scaled channels, same as quantize_yolo11n detect eval)
NUM_SCALES = 3
NC = 80
NM = 32
REG_MAX = 16
HEAD_CH = (32, 64, 128)
STRIDES = (8.0, 16.0, 32.0)

# ESP ONNX export order:
# box0, score0, box1, score1, box2, score2, mc0, mc1, mc2, p
BOX_SCORE_OFFSET = 0
MC_OFFSET = 6
PROTO_INDEX = 9
script_dir = os.path.dirname(os.path.abspath(__file__))


def _reshape_scale_outputs(graph_outputs, start_index):
    """Flatten per-scale head outputs to (B, C, N)."""
    bs = graph_outputs[start_index].shape[0]
    return torch.cat(
        [
            graph_outputs[start_index + 2 * i].view(
                bs,
                -1,
                graph_outputs[start_index + 2 * i].shape[2]
                * graph_outputs[start_index + 2 * i].shape[3],
            )
            for i in range(NUM_SCALES)
        ],
        dim=-1,
    )


def _build_detect_feats(graph_outputs):
    """Rebuild 3-scale feature maps for anchor generation."""
    return [
        torch.cat(
            (
                graph_outputs[2 * i + BOX_SCORE_OFFSET],
                graph_outputs[2 * i + 1 + BOX_SCORE_OFFSET],
            ),
            dim=1,
        )
        for i in range(NUM_SCALES)
    ]


def ppq_graph_inference(executor, task, inputs, device):
    """Run PPQ graph and decode outputs with Ultralytics heads."""
    graph_outputs = executor(inputs)

    if task == "detect":
        boxes = _reshape_scale_outputs(graph_outputs, BOX_SCORE_OFFSET)
        scores = _reshape_scale_outputs(graph_outputs, BOX_SCORE_OFFSET + 1)
        feats = _build_detect_feats(graph_outputs)

        detect_model = Detect(nc=NC, reg_max=REG_MAX, end2end=False, ch=HEAD_CH)
        detect_model.stride = torch.tensor(STRIDES)
        detect_model.to(device)

        pred_dict = {"boxes": boxes, "scores": scores, "feats": feats}
        return detect_model._inference(pred_dict)

    if task == "segment":
        boxes = _reshape_scale_outputs(graph_outputs, BOX_SCORE_OFFSET)
        scores = _reshape_scale_outputs(graph_outputs, BOX_SCORE_OFFSET + 1)
        feats = _build_detect_feats(graph_outputs)

        bs = graph_outputs[0].shape[0]
        mask_coefficient = torch.cat(
            [
                graph_outputs[MC_OFFSET + i].view(
                    bs,
                    graph_outputs[MC_OFFSET + i].shape[1],
                    graph_outputs[MC_OFFSET + i].shape[2]
                    * graph_outputs[MC_OFFSET + i].shape[3],
                )
                for i in range(NUM_SCALES)
            ],
            dim=-1,
        )
        proto = graph_outputs[PROTO_INDEX]

        detect_model = Detect(nc=NC, reg_max=REG_MAX, end2end=False, ch=HEAD_CH)
        detect_model.stride = torch.tensor(STRIDES)
        detect_model.to(device)

        pred_dict = {"boxes": boxes, "scores": scores, "feats": feats}
        detections = detect_model._inference(pred_dict)
        detections = torch.cat([detections, mask_coefficient], dim=1)
        return detections, proto

    raise NotImplementedError(f"{task} is not supported.")


def ppq_graph_init(quant_func, imgsz, device, native_path=None):
    """
    Init PPQ graph inference.
      - PTQ validation: ppq_graph = quant_func(imgsz)
      - QAT validation: load .native graph saved during training
    """
    ppq_graph = load_native_graph(native_path) if native_path else quant_func(imgsz)
    return TorchExecutor(graph=ppq_graph, device=device)


class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, executor=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
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
            self.device = model.device
            self.args.half = model.fp16
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if fmt not in {"pt", "torchscript"} and not getattr(
                model, "dynamic", False
            ):
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(
                    f"Setting batch={self.args.batch} input of shape "
                    f"({self.args.batch}, 3, {imgsz}, {imgsz})"
                )

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(
                        f"Dataset '{self.args.data}' for task={self.args.task} not found"
                    )
                )

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False

            self.stride = model.stride
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
            )

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []

        infer_task = "segment" if self.args.task == "segment" else "detect"

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            with dt[0]:
                batch = self.preprocess(batch)

            with dt[1]:
                preds = ppq_graph_inference(executor, infer_task, batch["img"], "cpu")

            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

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
            return {k: round(float(v), 5) for k, v in results.items()}

        if RANK > 0:
            return stats

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, "
            "{:.1f}ms postprocess per image".format(*tuple(self.speed.values()))
        )
        if self.args.save_json and self.jdict:
            pred_json = self.save_dir / "predictions.json"
            with open(str(pred_json), "w", encoding="utf-8") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats


def make_quant_validator_class(executor):
    class QuantSegmentationValidator(SegmentationValidator):
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            super().__init__(dataloader, save_dir, args, _callbacks)
            self.executor = executor

        def __call__(self, trainer=None, model=None):
            return QuantizedModelValidator.__call__(self, trainer, model, self.executor)

    return QuantSegmentationValidator


if __name__ == "__main__":
    executor = ppq_graph_init(
        quant_yolo11n_seg,
        640,
        "cuda",
        os.path.join(
            script_dir,
            "../../../../models/coco_seg/models/p4/coco_seg_yolo11n_seg_s8_v1.native",
        ),
    )
    QuantSegmentationValidator = make_quant_validator_class(executor)

    model = YOLO("yolo11n-seg.pt")
    results = model.val(
        data="coco.yaml",
        split="val",
        imgsz=640,
        device="cuda",
        validator=QuantSegmentationValidator,
        rect=False,
        save_json=True,
    )
