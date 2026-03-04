import torch
import os
from esp_ppq.executor import TorchExecutor
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.data.utils import check_det_dataset
from config import QATConfig
from validator import make_quant_validator_class

class QATTrainer:
    """
    Stripped-down Trainer used strictly as an Evaluation Wrapper for PTQ mAP assessment.
    All training loops, optimizers, and backpropagation logic have been explicitly removed.
    """
    def __init__(self, graph, model_meta, device=QATConfig.DEVICE):
        self._device = device
        self._executor = TorchExecutor(graph, device=self._device)
        self.graph = graph
        self.model_meta = model_meta

        # -- YOLOv26n Specific Model Init for Loss Calculation --
        tmp_model = YOLO(QATConfig.PT_FILE)
        self.model = tmp_model.model.to(self._device)
        
        # Ensure correct args in model for Loss compatibility
        if not hasattr(self.model, 'args'):
             overrides = {
                "data": QATConfig.DATA_YAML_FILE,
                "model": QATConfig.MODEL_NAME,
                "task": "detect",
                "device": self._device,
             }
             cfg = {**DEFAULT_CFG_DICT, **overrides}
             if isinstance(self.model.args, dict):
                 self.model.args = cfg
             else:
                 self.model.args = IterableSimpleNamespace(**cfg)

        # Make dataset available directly to validator (bypass model args legacy path)
        self.data = check_det_dataset(QATConfig.DATA_YAML_FILE)

        # -- Persistent Validator --
        Validator = make_quant_validator_class(self._executor, self.model_meta)
        
        # Build args for validator
        if isinstance(self.model.args, dict):
            args = self.model.args.copy()
        else:
            args = vars(self.model.args).copy()
        args.update({
             'mode': 'val',
             'data': QATConfig.DATA_YAML_FILE,
             'imgsz': QATConfig.IMG_SZ,
             'rect': False,
             'plots': True,
             'device': self._device,
             'batch': QATConfig.BATCH_SIZE,  # Changed from VAL_BATCH_SIZE to BATCH_SIZE dynamically
        })
        self.validator = Validator(args=IterableSimpleNamespace(**args))

    def eval(self) -> float:
        """Evaluate using persistent validator."""
        results = self.validator(trainer=self)
        map95 = results.get('metrics/mAP50-95(B)', 0.0)
        return map95
