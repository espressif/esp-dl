from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.model import Model
from ultralytics.nn.modules.head import Detect
from ultralytics import YOLO
from ppq.api import load_native_graph
import ppq.lib as PFL
from ppq.executor import TorchExecutor
from quantize_onnx_model import quant_yolo11n
import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics.cfg import get_cfg, get_save_dir
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
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.engine.model import Model
import os
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors


class QuantDetect(Detect):
    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # init self.stride for quantized model
        self.stride = [8.0, 16.0, 32.0]
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in {
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        }:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides,
                self.anchors.unsqueeze(0) * self.strides,
                xywh=False,
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = (
                self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))
                * self.strides
            )

        return torch.cat((dbox, cls.sigmoid()), 1)


class QuantModel(Model):
    def val(
        self,
        validator=None,
        **kwargs,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): An instance of a custom validator class for
                validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        # since ppq_graph inference need a fixed input shape, change to rect = False
        custom = {"rect": False}
        args = {
            **self.overrides,
            **custom,
            **kwargs,
            "mode": "val",
        }  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(
            args=args, _callbacks=self.callbacks
        )
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics


class YOLO11n(QuantModel):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {
            ".pt",
            ".yaml",
            ".yml",
        }:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.QuantDetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
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

        """
        # case 1: PTQ graph validation: ppq_graph = quant_yolo11n()
        # case 2: QAT graph validation:
                    utilize .native to load the graph
                    while training, the .native model is saved along with .espdl model
        """
        native_path = "coco_detect_yolo11n_s8_v3.native"  # change to your own .native file path after qat or set to None
        if native_path is not None:  # qat graph
            ppq_graph = load_native_graph(native_path)
        else:
            ppq_graph = quant_yolo11n()  # ptq graph

        executor = TorchExecutor(
            graph=ppq_graph, device=self.device.type
        )  # torch.device

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                graph_outputs = executor(batch["img"])
                box0, score0, box1, score1, box2, score2 = graph_outputs[:6]

                x = []
                x.append(torch.cat((box0, score0), 1))
                x.append(torch.cat((box1, score1), 1))
                x.append(torch.cat((box2, score2), 1))

                detect_model = QuantDetect(nc=80, ch=[32, 64, 128])
                detect_model.to(self.device)
                preds = detect_model._inference(x)

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


class QuantDetectionValidator(QuantizedModelValidator):
    """
    A class extending the QuantizedModelValidator class for validation based on a detection model.
    """

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor(
                (width, height, width, height), device=self.device
            )
            self.lb = [
                torch.cat(
                    [
                        batch["cls"][batch["batch_idx"] == i],
                        bboxes[batch["batch_idx"] == i],
                    ],
                    dim=-1,
                )
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (
                val.endswith(f"{os.sep}val2017.txt")
                or val.endswith(f"{os.sep}test-dev2017.txt")
            )
        )  # is COCO
        self.is_lvis = (
            isinstance(val, str) and "lvis" in val and not self.is_coco
        )  # is LVIS
        self.class_map = (
            converter.coco80_to_coco91_class()
            if self.is_coco
            else list(range(len(model.names)))
        )
        self.args.save_json |= (
            self.args.val and (self.is_coco or self.is_lvis) and not self.training
        )  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = (
                ops.xywh2xyxy(bbox)
                * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            )  # target boxes
            ops.scale_boxes(
                imgsz, bbox, ori_shape, ratio_pad=ratio_pad
            )  # native-space labels
        return {
            "cls": cls,
            "bbox": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"],
            predn[:, :4],
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bbox, gt_cls=cls
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {
            k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()
        }  # to numpy
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )
        self.nt_per_image = np.bincount(
            stats["target_img"].astype(int), minlength=self.nc
        )
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf
            % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.nt_per_image[c],
                        self.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=self.stride
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1
        )  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / (
                    "instances_val2017.json"
                    if self.is_coco
                    else f"lvis_v1_{self.args.split}.json"
                )
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements(
                    "pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3"
                )
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(
                        str(pred_json)
                    )  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(
                        str(pred_json)
                    )  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [
                    int(Path(x).stem) for x in self.dataloader.dataset.im_files
                ]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2]
                    if self.is_coco
                    else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats


if __name__ == "__main__":
    # load yolo11n.pt to enter val method and thereby run BaseGraph inference
    model = YOLO11n("yolo11n.pt")
    device = "cpu"
    model.to(device)
    results = model.val(
        data="coco.yaml", imgsz=640, device=device, validator=QuantDetectionValidator
    )  # replace DetectionValidator
