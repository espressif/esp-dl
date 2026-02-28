import torch
import torch.optim as optim
from tqdm import tqdm
import os
from esp_ppq.IR import TrainableGraph
from esp_ppq.executor import TorchExecutor
from esp_ppq.parser import NativeExporter
from esp_ppq.api import load_native_graph
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from config import QATConfig
from validator import make_quant_validator_class
from ultralytics.data.utils import check_det_dataset

class QATTrainer:
    """
    Custom Trainer for YOLOv26n QAT.
    """
    def __init__(self, graph, model_meta, device=QATConfig.DEVICE):
        self._epoch = 0
        self._step = 0
        self._device = device
        self._executor = TorchExecutor(graph, device=self._device)
        self._training_graph = TrainableGraph(graph)
        self.graph = graph
        self.model_meta = model_meta

        # Initialize Optimizer
        # Ensure all parameters require grad
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True
            
        self._optimizer = optim.SGD(
            params=[{"params": self._training_graph.parameters()}],
            lr=QATConfig.OPTIMIZER_LR,
            momentum=QATConfig.OPTIMIZER_MOMENTUM,
            weight_decay=QATConfig.OPTIMIZER_WEIGHT_DECAY,
        )
        self._lr_scheduler = None

        # -- YOLOv26n Specific Model Init for Loss Calculation --
        print("Loading YOLOv26n model in Trainer to access correct Loss function...")
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

        self.data = check_det_dataset(QATConfig.DATA_YAML_FILE) # Ensure data is available on trainer dict or path

        # -- Persistent Validator --
        print("Initializing Persistent Validator (reusing dataloader)...")
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
             'batch': QATConfig.VAL_BATCH_SIZE,
        })
        # Note: Validator expects args namespace
        self.validator = Validator(args=IterableSimpleNamespace(**args))

    def decode_qat_output(self, feats_list):
        """
        Manually decode raw QAT feature maps for Loss Calculation.
        Using dynamic split based on model_meta.
        """
        if isinstance(feats_list, torch.Tensor):
             return feats_list
        
        pred_boxes = []
        pred_scores = []
        
        # Extract dynamic params
        reg_max = self.model_meta['reg_max']
        split_point = 4 * reg_max
        
        for x in feats_list:
            # x: [B, C, H, W]
            b, c, h, w = x.shape
            
            # Dynamic Split
            box_part = x[:, :split_point, :, :]
            cls_part = x[:, split_point:, :, :]
            
            # Flatten to [B, C, A] (A = H*W)
            pred_boxes.append(box_part.flatten(2))
            pred_scores.append(cls_part.flatten(2))
            
        # Concatenate across scales
        cat_boxes = torch.cat(pred_boxes, dim=2)   # [B, 4*reg_max, TotalAnchors]
        cat_scores = torch.cat(pred_scores, dim=2) # [B, NC, TotalAnchors]
        
        return {
            "boxes": cat_boxes,
            "scores": cat_scores,
            "feats": feats_list # Loss needs raw feats for anchor generation
        }

    def eval(self) -> float:
        """Evaluate using persistent validator."""
        results = self.validator(trainer=self)
        
        # results is a dict, keys include 'metrics/mAP50-95(B)', etc.
        # Fallback to 0.0 if key missing
        map95 = results.get('metrics/mAP50-95(B)', 0.0)
        return map95

    def epoch(self, dataloader) -> float:
        """Do one epoch Training."""
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {self._epoch}", total=len(dataloader))
        
        for bidx, batch in enumerate(pbar):
            _, loss = self.step(batch, True)
            epoch_loss += loss
            pbar.set_postfix(loss=f"{loss:.4f}")

        self._epoch += 1
        print(f"Epoch Loss: {epoch_loss / len(dataloader):.4f}")
        return epoch_loss
        
    def step(self, data, training: bool) -> tuple:
        """Custom step with manual output decoding."""
        if training:
            # Preprocess
            data["img"] = data["img"].to(self._device, non_blocking=True).float() / 255
            
            # Forward pass through QAT graph
            graph_outputs = self._executor.forward_with_gradient(data["img"])
            
            # Reconstruction of YOLOv26 output dictionary
            # graph_outputs: [P3, P4, P5 (one2many), P3, P4, P5 (one2one)]
            if len(graph_outputs) != 6:
                 print(f"WARNING: Expected 6 outputs, got {len(graph_outputs)}")
            
            preds = {
                "one2many": self.decode_qat_output(list(graph_outputs[0:3])),
                "one2one": self.decode_qat_output(list(graph_outputs[3:6]))
            }
            
            # Calculate loss
            loss = self.model.loss(data, preds)[0]
            if loss.numel() > 1:
                loss = loss.sum()
            
            # Backward
            loss.backward()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step(epoch=self._epoch)
            self._optimizer.step()
            self._training_graph.zero_grad()
            
            return preds, loss.item()
        
        
        return None, 0.0

    def save_graph(self, file_path):
        """Save the current graph state using NativeExporter."""
        # Ensure directory exists
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            
        exporter = NativeExporter()
        exporter.export(file_path=file_path, graph=self.graph)

    def load_graph(self, file_path):
        """Load a native graph and update the trainer's graph and executor."""
        self.graph = load_native_graph(file_path)
        
        # Re-initialize executor with the new graph
        self._executor = TorchExecutor(self.graph, device=self._device)
        self._training_graph = TrainableGraph(self.graph)
        return self.graph
