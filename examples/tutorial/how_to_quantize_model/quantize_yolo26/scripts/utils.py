import random
import numpy as np
import torch
import torch.nn as nn
from types import MethodType
from ultralytics.utils.loss import v8DetectionLoss
from config import QATConfig

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def register_mod_op():
    """Register 'Mod' operator if missing in ESP-PPQ."""
    try:
        from esp_ppq.executor import register_operation_handler
        from esp_ppq.core import TargetPlatform
        
        # Define implementation for Mod (modulo)
        def mod_impl(op, values, ctx):
            return values[0] % values[1]

        # Register for SOI (Shape/Meta inference) and FP32 (CPU/GPU execution)
        try:
            register_operation_handler(mod_impl, 'Mod', TargetPlatform.SOI)
        except Exception:
            pass 

        try:
            register_operation_handler(mod_impl, 'Mod', TargetPlatform.FP32)
        except Exception:
            pass

        print("Registered 'Mod' handler for PPQ.")

    except ImportError:
        print("Could not import register_operation_handler from esp_ppq.executor. Skipping Mod patch.")

def patch_v8_detection_loss():
    """Monkey Patch for v8DetectionLoss to ensure self.hyp is an object."""
    _original_loss_init = v8DetectionLoss.__init__

    def patched_loss_init(self, model, tal_topk=10, **kwargs):
        _original_loss_init(self, model, tal_topk=tal_topk, **kwargs)
        if isinstance(self.hyp, dict):
            # Inject default hyp params from Config if missing
            defaults = QATConfig.LOSS_DEFAULTS
            for k, v in defaults.items():
                self.hyp.setdefault(k, v)
                
            class HypWrapper:
                def __init__(self, d):
                    self.__dict__.update(d)
            
            
            self.hyp = HypWrapper(self.hyp)
            # print("Patched v8DetectionLoss.hyp with defaults and object access.")

    v8DetectionLoss.__init__ = patched_loss_init

def patch_detect_forward(model):
    """
    Monkey-patch the Detect head to return raw feature maps for BOTH branches
    (one2many and one2one) as a flat list.
    """
    # Define the new forward function
    def new_forward(self, x):
        # x is list of features from backbone/neck
        
        # 1. One-to-Many Branch (Standard YOLO)
        one2many = []
        for i in range(self.nl):
            one2many.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            
        # 2. One-to-One Branch (YOLOv10 specific)
        one2one = []
        for i in range(self.nl):
            one2one.append(torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1))
            
        # Return flat list of 6 tensors: [m0, m1, m2, o0, o1, o2]
        return one2many + one2one

    # Locate the Detect module
    detect_module = model.model.model[-1]
    print(f"Patching module: {type(detect_module)}")
    
    # Bind the new method
    detect_module.forward = MethodType(new_forward, detect_module)
    return model

def get_exclusive_ancestors(graph, target_outputs, excluded_outputs):
    """
    Finds operators that are ancestors of target_outputs but NOT ancestors of excluded_outputs.
    """
    def get_ancestors_set(output_vars):
        ancestors = set()
        stack = [v.source_op for v in output_vars if v.source_op is not None]
        while stack:
            op = stack.pop()
            if op in ancestors:
                continue
            ancestors.add(op)
            for inp in op.inputs:
                if inp.source_op is not None:
                    stack.append(inp.source_op)
        return ancestors

    target_vars = [graph.outputs[name] for name in target_outputs if name in graph.outputs]
    excluded_vars = [graph.outputs[name] for name in excluded_outputs if name in graph.outputs]
    
    target_ops = get_ancestors_set(target_vars)
    excluded_ops = get_ancestors_set(excluded_vars)
    
    return target_ops - excluded_ops
