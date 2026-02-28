from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import unwrap_model, smart_inference_mode, select_device
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.utils import check_det_dataset
import torch
import json
from config import QATConfig

def ppq_graph_inference(executor, task, inputs, device, model_meta):
    """PPQ graph inference with manual decoding for YOLOv26 using dynamic metadata."""
    graph_outputs = executor(inputs)
    
    # Extract meta
    nc = model_meta['nc']
    reg_max = model_meta['reg_max']
    stride = model_meta['stride']
    ch = model_meta['ch']
    
    if task == "detect":
        # Manually decode raw QAT outputs for Detect._inference
        raw_x = [t.to(device) for t in graph_outputs[0:3]]
        box_list = []
        score_list = []
        
        # Calculate split point dynamically
        # Legacy/v8: reg_max=16 -> 4*16 = 64. 64+NC = 144
        # YOLOv26: reg_max=1 -> 4*1 = 4. 4+NC = 84
        split_point = 4 * reg_max
        
        for t in raw_x:
            # t: [B, C, H, W] 
            b, c, h, w = t.shape
            
            # Use dynamic split
            box_part = t[:, :split_point, :, :]
            score_part = t[:, split_point:, :, :]
            
            # Flatten H,W into A (Anchors) -> [B, C, A]
            box_list.append(box_part.flatten(2))
            score_list.append(score_part.flatten(2))
            
        # Concatenate all scales along the anchor dimension (dim 2)
        cat_boxes = torch.cat(box_list, dim=2)   # [B, 4*reg_max, TotalAnchors]
        cat_scores = torch.cat(score_list, dim=2) # [B, NC, TotalAnchors]
            
        preds = {
            "feats": raw_x,
            "boxes": cat_boxes, 
            "scores": cat_scores
        }

        # Initialize a Detect head for post-processing with dynamic params
        detect_model = Detect(nc=nc, reg_max=reg_max, ch=ch)
        detect_model.stride = stride
        detect_model.to(device)

        y = detect_model._inference(preds)
        return y
    else:
        raise NotImplementedError(f"{task} is not supported.")

class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, executor=None):
        """Executes validation process."""
        self.training = trainer is not None
        if self.training:
            self.device = select_device(trainer._device)
            if hasattr(trainer, 'data') and trainer.data is not None:
                self.data = trainer.data
            elif hasattr(trainer.model, 'args'):
                if isinstance(trainer.model.args, dict):
                    self.data = trainer.model.args.get('data')
                else:
                    self.data = getattr(trainer.model.args, 'data', None)
            else:
                self.data = None
            
            if isinstance(self.data, str):
                 self.data = check_det_dataset(self.data)
            
            # Standard setup
            model = trainer.model
            # Force FP32 or maintain precision
            self.loss = torch.zeros(3, device=trainer._device) # Mock loss
            self.args.plots &= (trainer._epoch == QATConfig.EPOCHS - 1)
            model.eval()
        else:
            # Eval mode setup
            if hasattr(self.args, 'data'):
                if isinstance(self.args.data, str):
                    self.data = check_det_dataset(self.args.data)
                elif isinstance(self.args, dict):
                     self.data = check_det_dataset(self.args.get('data'))
                else:
                     self.data = check_det_dataset(getattr(self.args, 'data'))
            self.device = select_device(self.args.device, self.args.batch)

        # Check/Init DataLoader
        if self.dataloader is None:
             self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        self.run_callbacks("on_val_start")
        dt = (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.end2end = False 
        self.jdict = []

        # Get meta from trainer if available
        model_meta = getattr(trainer, 'model_meta', None)
        # Get meta from self (validator) or trainer
        model_meta = getattr(self, 'model_meta', None)
        if not model_meta and trainer:
            model_meta = getattr(trainer, 'model_meta', None)
            
        if not model_meta:
             raise ValueError("Trainer/Validator missing model_meta for validation")

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                preds = ppq_graph_inference(executor, "detect", batch["img"], self.device, model_meta)
            # Loss (skipped)
            
            # Postprocess
            with dt[2]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < QATConfig.VAL_PLOT_MAX_BATCHES:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
            
        stats = self.get_stats()
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        return stats

def make_quant_validator_class(executor, model_meta):
    class QuantDetectionValidator(DetectionValidator):
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            super().__init__(dataloader, save_dir, args, _callbacks)
            self.executor = executor
            self.model_meta = model_meta

        def __call__(self, trainer=None, model=None):
            # Inject meta into trainer just in case, though validator usually calls back
            if trainer:
                trainer.model_meta = self.model_meta
            return QuantizedModelValidator.__call__(self, trainer, model, self.executor)

    return QuantDetectionValidator
