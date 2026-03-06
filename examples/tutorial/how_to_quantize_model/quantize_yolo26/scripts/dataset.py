import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from utils import worker_init_fn
from config import QATConfig

class CaliDataset(Dataset):
    """Dataset for Calibration images."""
    def __init__(self, path):
        super().__init__()
        # Use simple transform for calibration
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((QATConfig.IMG_SZ, QATConfig.IMG_SZ)),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ])
        self.imgs_path = []
        self.path = path
        if os.path.exists(self.path):
            for img_name in os.listdir(self.path):
                img_path = os.path.join(self.path, img_name)
                self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])  # 0~255 hwc #RGB
        if img.mode == "L":
            img = img.convert("RGB")
        img = self.transform(img)
        return img

class SubsetCaliDataset(CaliDataset):
    """Subset of CaliDataset to limit size and filter extensions."""
    def __init__(self, path, max_images=QATConfig.CALIB_MAX_IMAGES): 
            super().__init__(path)
            # Filter out non-image files like .npy
            valid_extensions = QATConfig.CALIB_VALID_EXTENSIONS
            self.imgs_path = [p for p in self.imgs_path if p.lower().endswith(valid_extensions)]
            
            if len(self.imgs_path) > max_images:
                self.imgs_path = self.imgs_path[:max_images]

def get_calibration_loader(data_cfg):
    """Resolves calibration path and returns a DataLoader."""
    # Logic to find image path from data config (same as original script)
    train_path_entry = data_cfg['train']
    calibration_path = None
    
    if os.path.isdir(train_path_entry):
            calibration_path = train_path_entry
    else:
            dataset_root = data_cfg.get('path', '')
            if dataset_root:
                candidate = os.path.join(dataset_root, 'images', 'train2017')
                if os.path.isdir(candidate):
                    calibration_path = candidate
    
    if not calibration_path:
            calibration_path = QATConfig.DATA_FALLBACK_PATH
            print(f"Could not deduce exact image folder from {QATConfig.DATA_YAML_FILE}, defaulting to {calibration_path}")
    else:
            print(f"Using dataset at: {calibration_path}")
            
    if not os.path.exists(calibration_path):
         raise FileNotFoundError(f"Calibration images not found at {calibration_path}")

    dataset = SubsetCaliDataset(calibration_path)
    return DataLoader(dataset, batch_size=QATConfig.BATCH_SIZE, shuffle=False)

def get_train_loader(data_cfg):
    """Creates the YOLO training DataLoader."""
    dataset_class = YOLODataset
    batchsz = QATConfig.BATCH_SIZE
    imgsz = QATConfig.IMG_SZ
    fraction = QATConfig.DATA_FRACTION
    
    if "nc" not in data_cfg:
        data_cfg["nc"] = len(data_cfg["names"])
        
    train_path = data_cfg.get('train')
    
    train_dataset = dataset_class(
        img_path=train_path,
        imgsz=imgsz,
        batch_size=batchsz,
        augment=False,
        rect=False,
        cache=None,
        task="detect",
        data=data_cfg,
    )
    
    # Subsampling
    if fraction < 1.0:
        num_samples = max(1, int(len(train_dataset) * fraction))
        print(f"Subsampling dataset to {fraction:.1%}: {num_samples} samples")
        train_dataset.labels = train_dataset.labels[:num_samples]
        train_dataset.im_files = train_dataset.im_files[:num_samples]
        
    g = torch.Generator()
    g.manual_seed(QATConfig.SEED)

    # Worker logic
    if QATConfig.DEVICE == 'cpu':
        actual_workers = 0 
        print(f"Training on CPU detected. Setting num_workers={actual_workers}")
    else:
        actual_workers = 0 # Force 0 on Windows for safety
        
    return DataLoader(
        train_dataset,
        batch_size=batchsz,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=dataset_class.collate_fn,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
