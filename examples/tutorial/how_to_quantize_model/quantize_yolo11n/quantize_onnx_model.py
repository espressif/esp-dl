import os
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from onnxsim import simplify
import onnx
import zipfile
import urllib.request


class CaliDataset(Dataset):
    def __init__(self, path, img_shape=640):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_shape, img_shape)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img)
        return img


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def quant_yolo11n(imgsz):
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, imgsz, imgsz]
    DEVICE = "cpu"
    TARGET = "esp32p4"
    NUM_OF_BITS = 8
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(
        script_dir, "../../../../models/coco_detect/models/yolo11n.onnx"
    )
    ESPDL_MODLE_PATH = os.path.join(
        script_dir,
        "../../../../models/coco_detect/models/p4/coco_detect_yolo11n_s8_v1.espdl",
    )

    yolo11n_caib_url = "https://dl.espressif.com/public/calib_yolo11n.zip"
    CALIB_DIR = "calib_yolo11n"
    urllib.request.urlretrieve(
        yolo11n_caib_url, "calib_yolo11n.zip", reporthook=report_hook
    )

    with zipfile.ZipFile("calib_yolo11n.zip", "r") as zip_file:
        zip_file.extractall("./")

    model = onnx.load(ONNX_PATH)
    sim = True
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), ONNX_PATH)

    calibration_dataset = CaliDataset(CALIB_DIR, img_shape=imgsz)
    dataloader = DataLoader(
        dataset=calibration_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(DEVICE)

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.quantize_activation_setting.calib_algorithm = "percentile"

    quant_setting.equalization = False
    quant_setting.equalization_setting.iterations = 10
    quant_setting.equalization_setting.value_threshold = 0.1
    quant_setting.equalization_setting.opt_level = 2

    quant_setting.bias_correct = True
    quant_setting.bias_correct_setting.interested_layers = []
    quant_setting.bias_correct_setting.block_size = 4
    quant_setting.bias_correct_setting.steps = 32

    quant_setting.tqt_optimization = True
    tqt_setting = quant_setting.tqt_optimization_setting
    tqt_setting.lr = 1e-5
    tqt_setting.steps = 500
    tqt_setting.block_size = 4
    tqt_setting.is_scale_trainable = True
    tqt_setting.gamma = 0.0
    tqt_setting.int_lambda = 0.0
    tqt_setting.collecting_device = "cpu"

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph


if __name__ == "__main__":
    quant_yolo11n(imgsz=640)
