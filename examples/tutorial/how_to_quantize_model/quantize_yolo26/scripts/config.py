import os
import torch
from esp_ppq.api import get_target_platform

class QATConfig:
    # Training Parameters
    EPOCHS = 10
    BATCH_SIZE = 14
    IMG_SZ = 640
    DATA_FRACTION = 1.0 # Use 0.5% of dataset for fast debugging
    SEED = 1234
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimizer
    OPTIMIZER_LR = 1e-6
    OPTIMIZER_MOMENTUM = 0.937
    OPTIMIZER_WEIGHT_DECAY = 5e-4
    
    # Data Settings
    DATA_YAML_FILE = "coco.yaml"
    DATA_FALLBACK_PATH = "coco2017/images/train2017"
    CALIB_MAX_IMAGES = 8192
    CALIB_VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Quantization Settings
    TARGET_PLATFORM = get_target_platform("esp32p4", 8)
    CALIB_STEPS = 32
    QUANT_CALIB_METHOD = "kl"
    QUANT_ALIGNMENT = "Align to Output"
    EXPORT_OPSET = 13
    EXPORT_DYNAMIC = False 
    
    # Loss Defaults (Standard YOLOv8/v10/v26 defaults)
    LOSS_DEFAULTS = {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
        'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
        'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
        'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0,
    }

    # Model Paths
    # Assuming current directory
    BASE_DIR = os.getcwd()
    MODEL_NAME = "yolo26n"
    PT_FILE = f"{MODEL_NAME}.pt"
    ONNX_FILE = f"{MODEL_NAME}_train.onnx"
    
    # Derived Paths
    ONNX_PATH = os.path.join(BASE_DIR, ONNX_FILE)
    ESPDL_OUTPUT_DIR = os.path.join(BASE_DIR, f"{MODEL_NAME}_qat_output")
    
    # Plotting
    VAL_PLOT_MAX_BATCHES = 3

    # Validation Batch Size
    VAL_BATCH_SIZE = 16
