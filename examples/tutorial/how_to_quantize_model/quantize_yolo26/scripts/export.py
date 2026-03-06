import torch
import torch.nn as nn
import onnx
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Attention
from ultralytics.engine.exporter import Exporter, try_export, arange_patch
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_requirements
try:
    from ultralytics.utils.torch_utils import get_latest_opset
except ImportError:
    def get_latest_opset():
        return 17 # Default fallback 2024+

from config import QATConfig

class ESP_Detect(Detect):
    """
    Custom Detect Head for QAT Preparation.
    It returns the raw feature maps for both branches (one2many and one2one)
    concatenated into a flat list, just like the original monkey patch.
    This ensures the QAT Trainer receives the expected 6 outputs.
    """
    def forward(self, x):
        # x is list of features from backbone/neck
        
        # 1. One-to-Many Branch (Standard YOLO)
        one2many = []
        for i in range(self.nl):
            one2many.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            
        # 2. One-to-One Branch (YOLOv10/26 specific)
        one2one = []
        for i in range(self.nl):
            one2one.append(torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1))
            
        # Return flat list of 6 tensors: [m0, m1, m2, o0, o1, o2]
        return one2many + one2one

class ESP_Attention(Attention):
    def forward(self, x):
        """
        Forward pass of the Attention module.
        Uses view(-1, ...) to support static shape export without breaking reshape ops.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            -1, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(-1, C, H, W) + self.pe(
            v.reshape(-1, C, H, W)
        )
        x = self.proj(x)
        return x

class ESP_Detect_Exporter(Exporter):
    """
    Adapted from ultralytics for detection task, robust to static/dynamic export.
    """
    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLO ONNX export."""
        requirements = ["onnx>=1.14.0"]
        if self.args.simplify:
            requirements += [
                "onnxsim",
                "onnxruntime" + ("-gpu" if torch.cuda.is_available() else ""),
            ]
        check_requirements(requirements)

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        
        f = QATConfig.ONNX_PATH
        
        # Custom output names for QAT (6 outputs)
        output_names = ["one2many_p3", "one2many_p4", "one2many_p5", 
                        "one2one_p3", "one2one_p4", "one2one_p5"]
        
        dynamic = self.args.dynamic
        if dynamic:
            dynamic_axes = {"images": {0: "batch"}}
            for name in output_names:
                dynamic_axes[name] = {0: "batch"}
        else:
            dynamic_axes = None

        with arange_patch(self.args):
            torch.onnx.export(
                self.model,
                self.im,
                f,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=False,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            
        # Checks
        model_onnx = onnx.load(f)

        # Simplify
        if self.args.simplify:
            try:
                import onnxsim
                LOGGER.info(f"{prefix} simplifying with onnxsim {onnxsim.__version__}...")
                model_onnx, _ = onnxsim.simplify(model_onnx)
            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        return f

def apply_export_patches(model):
    """Applies ESP_Attention and ESP_Detect patches to the model."""
    print("Applying ESP-DL patches for export...")
    
    # Patch Attention
    patch_count = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.forward = ESP_Attention.forward.__get__(m)
            patch_count += 1
    print(f"Patched {patch_count} Attention modules.")
    
    # Patch Detect (for QAT outputs)
    # We replace the forward method of the Detect instance directly
    detect_module = model.model.model[-1]
    detect_module.forward = ESP_Detect.forward.__get__(detect_module)
    print(f"Patched Detect module: {type(detect_module)}")
    
    # Break fuse to ensure raw heads are kept
    detect_module.fuse = lambda: print(">> Fuse method blocked! Keeping all heads.")
    
    return model

class ESP_YOLO(YOLO):
    """
    Subclass of YOLO that enforces usage of ESP_Detect_Exporter.
    """
    def export(self, **kwargs):
        """
        Custom export method.
        """
        self._check_is_pytorch_model()
        
        # Merge overrides and kwargs
        args = dict(self.overrides)
        args.update(kwargs)
        args["mode"] = "export"
        
        # Use our custom Exporter
        exporter = ESP_Detect_Exporter(overrides=args, _callbacks=self.callbacks)
        return exporter(model=self.model)
