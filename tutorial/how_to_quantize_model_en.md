# Using ESP-PPQ for Model Quantization (PTQ)

In this tutorial, we will guide you through the process of quantizing a pre-trained model using ESP-PPQ and analyzing the quantization error. The quantization method used is Post Training Quantization (PTQ). ESP-PPQ builds upon [PPQ](https://github.com/OpenPPL/ppq) and adds Espressif-customized quantizers and exporters, allowing users to select quantization rules that match different chips and export them as standard model files that ESP-DL can directly load. ESP-PPQ is compatible with all PPQ APIs and quantization scripts. For more details, please refer to the [PPQ documentation and videos](https://github.com/OpenPPL/ppq).

## Prerequisites

### 1. Install ESP-PPQ. Note that PPQ needs to be uninstalled before installing ESP-PPQ to avoid conflicts:

```bash
pip uninstall ppq
pip install git+https://github.com/espressif/esp-ppq.git
```

### 2. Model Files
Currently, ESP-PPQ supports ONNX, PyTorch, and TensorFlow models. During the quantization process, PyTorch and TensorFlow models are first converted to ONNX models, so ensure that your model can be converted to an ONNX model. We provide the following quantization script templates to help users modify them according to their models:

For ONNX models, refer to the script [quantize_onnx_model.py](../tools/quantization/quantize_onnx_model.py)  
For PyTorch models, refer to the script [quantize_pytorch_model.py](../tools/quantization/quantize_torch_model.py)  
For TensorFlow models, refer to the script [quantize_tf_model.py](../tools/quantization/quantize_tf_model.py)

## Model Quantization Example

We will use the [MobileNet_v2](https://arxiv.org/abs/1801.04381) model as an example to demonstrate how to use the [quantize_torch_model.py](../tools/quantization/quantize_torch_model.py) script to quantize the model.

### 1. Prepare the Pre-trained Model
Load the pre-trained MobileNet_v2 model from torchvision. You can also download it from [ONNX models](https://github.com/onnx/models) or [TensorFlow models](https://github.com/tensorflow/models):
```python
import torchvision
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

model = torchvision.models.mobilenet.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
```

### 2. Prepare the Calibration Dataset

The calibration dataset needs to match the input format of your model. The calibration dataset should cover all possible input scenarios to better quantize the model. Here, we use the ImageNet dataset as an example to demonstrate how to prepare the calibration dataset.

- Load the ImageNet dataset using torchvision:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

calib_dataset = datasets.ImageNet(root=CALIB_DIR, split='val', transform=transform)
dataloader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=false)
```

- Use the provided [imagenet_util.py](../tools/quantization/datasets/imagenet_util.py) script and the [ImageNet calibration dataset](https://dl.espressif.com/public/imagenet_calib.zip) to quickly download and test.

```python
# Load
from datasets.imagenet_util import load_imagenet_from_directory
dataloader = load_imagenet_from_directory(
        directory=CALIB_DIR,
        batchsize=BATCH_SIZE,
        shuffle=False,
        subset=1024,
        require_label=False,
        num_of_workers=4,
    )
```

### 3. Quantize the Model and Export the ESPDL Model

Use the `espdl_quantize_torch` API to quantize the model and export the ESPDL model file. After quantization, three files will be exported:
```
**.espdl: The ESPDL model binary file, which can be directly used for inference on the chip.
**.info:  The ESPDL model text file, used for debugging and verifying that the ESPDL model was correctly exported.
**.json:  The quantization information file, used for saving and loading quantization information.
```

The function parameters are described as follows:
```
from ppq.api import espdl_quantize_torch

def espdl_quantize_torch(
    model: torch.nn.Module,
    espdl_export_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[Any],
    inputs: Union[dict, list, torch.Tensor, None] = None,
    target:str = "esp32p4",
    num_of_bits:int = 8,
    collate_fn: Callable = None,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "conservative",
    device: str = "cpu",
    error_report: bool = True,
    test_output_names: List[str] = None,
    skip_export: bool = False,
    export_config: bool = True,
    verbose: int = 0,
) -> Tuple[BaseGraph, TorchExecutor]:

    """Quantize ONNX model and return quantized ppq graph and executor .
    
    Args:
        model (torch.nn.Module): torch model
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4" and "esp32s3"
        num_of_bits: the number of quantizer bits, 8 or 16
        collate_fn (Callable): batch collate func for preprocessing
        dispatching_override: override dispatching result.
        dispatching_method: Refer to https://github.com/espressif/esp-ppq/blob/master/ppq/scheduler/__init__.py#L8
        device (str, optional):  execution device, defaults to 'cpu'.
        error_report (bool, optional): whether to print error report, defaults to True.
        test_output_names (List[str], optional): tensor names of the model want to test, defaults to None.
        skip_export (bool, optional): whether to export the quantized model, defaults to False.
        export_config (bool, optional): whether to export the quantization configuration, defaults to True.
        verbose (int, optional): whether to print details, defaults to 0.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
        TorchExecutor:  PPQ Graph Executor 
    """
```


#### Quantization Test1

- **Quantization Settings:**
```
target="esp32p4"
num_of_bits=8
batch_size=32
dispatching_override=None
```

- **Quantization Results:**

```
Analysing Graphwise Quantization Error::
Layer                                            | NOISE:SIGNAL POWER RATIO
/features/features.16/conv/conv.2/Conv:          | ████████████████████ | 48.831%
/features/features.15/conv/conv.2/Conv:          | ███████████████████  | 45.268%
/features/features.17/conv/conv.2/Conv:          | ██████████████████   | 43.112%
/features/features.18/features.18.0/Conv:        | █████████████████    | 41.586%
/features/features.14/conv/conv.2/Conv:          | █████████████████    | 41.135%
/features/features.13/conv/conv.2/Conv:          | ██████████████       | 35.090%
/features/features.17/conv/conv.0/conv.0.0/Conv: | █████████████        | 32.895%
/features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 29.226%
/features/features.12/conv/conv.2/Conv:          | ████████████         | 28.895%
/features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 27.808%
/features/features.7/conv/conv.2/Conv:           | ███████████          | 27.675%
/features/features.10/conv/conv.2/Conv:          | ███████████          | 26.292%
/features/features.11/conv/conv.2/Conv:          | ███████████          | 26.085%
/features/features.6/conv/conv.2/Conv:           | ███████████          | 25.892%
/classifier/classifier.1/Gemm:                   | ██████████           | 25.591%
/features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 25.323%
/features/features.4/conv/conv.2/Conv:           | ██████████           | 24.787%
/features/features.15/conv/conv.1/conv.1.0/Conv: | ██████████           | 24.354%
/features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 20.207%
/features/features.9/conv/conv.2/Conv:           | ████████             | 19.808%
/features/features.14/conv/conv.0/conv.0.0/Conv: | ████████             | 18.465%
/features/features.5/conv/conv.2/Conv:           | ███████              | 17.868%
/features/features.12/conv/conv.1/conv.1.0/Conv: | ███████              | 16.589%
/features/features.13/conv/conv.1/conv.1.0/Conv: | ███████              | 16.143%
/features/features.11/conv/conv.1/conv.1.0/Conv: | ██████               | 15.382%
/features/features.3/conv/conv.2/Conv:           | ██████               | 15.105%
/features/features.13/conv/conv.0/conv.0.0/Conv: | ██████               | 15.029%
/features/features.10/conv/conv.1/conv.1.0/Conv: | ██████               | 14.875%
/features/features.2/conv/conv.2/Conv:           | ██████               | 14.869%
/features/features.11/conv/conv.0/conv.0.0/Conv: | ██████               | 14.552%
/features/features.9/conv/conv.1/conv.1.0/Conv:  | ██████               | 14.050%
/features/features.8/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.929%
/features/features.8/conv/conv.2/Conv:           | ██████               | 13.833%
/features/features.12/conv/conv.0/conv.0.0/Conv: | ██████               | 13.684%
/features/features.7/conv/conv.0/conv.0.0/Conv:  | █████                | 12.942%
/features/features.6/conv/conv.1/conv.1.0/Conv:  | █████                | 12.765%
/features/features.10/conv/conv.0/conv.0.0/Conv: | █████                | 12.251%
/features/features.5/conv/conv.1/conv.1.0/Conv:  | █████                | 11.186%
/features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 11.070%
/features/features.9/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.371%
/features/features.4/conv/conv.1/conv.1.0/Conv:  | ████                 | 10.356%
/features/features.6/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.149%
/features/features.4/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.472%
/features/features.8/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.232%
/features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 9.187%
/features/features.1/conv/conv.1/Conv:           | ████                 | 8.770%
/features/features.5/conv/conv.0/conv.0.0/Conv:  | ███                  | 8.408%
/features/features.7/conv/conv.1/conv.1.0/Conv:  | ███                  | 8.151%
/features/features.2/conv/conv.1/conv.1.0/Conv:  | ███                  | 7.156%
/features/features.3/conv/conv.0/conv.0.0/Conv:  | ███                  | 6.328%
/features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 5.392%
/features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.875%
/features/features.0/features.0.0/Conv:          |                      | 0.119%
Analysing Layerwise quantization error:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [08:44<00:00,  9.91s/it]
Layer                                            | NOISE:SIGNAL POWER RATIO 
/features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 14.303%
/features/features.0/features.0.0/Conv:          | █                    | 0.844%
/features/features.1/conv/conv.1/Conv:           | █                    | 0.667%
/features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.574%
/features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.419%
/features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.272%
/features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.238%
/features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.214%
/features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.180%
/features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.151%
/features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.148%
/features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.146%
/features/features.14/conv/conv.2/Conv:          |                      | 0.136%
/features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.105%
/features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.105%
/features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.083%
/features/features.7/conv/conv.2/Conv:           |                      | 0.076%
/features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.076%
/features/features.3/conv/conv.2/Conv:           |                      | 0.075%
/features/features.16/conv/conv.2/Conv:          |                      | 0.074%
/features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.072%
/features/features.15/conv/conv.2/Conv:          |                      | 0.066%
/features/features.4/conv/conv.2/Conv:           |                      | 0.065%
/features/features.11/conv/conv.2/Conv:          |                      | 0.063%
/classifier/classifier.1/Gemm:                   |                      | 0.063%
/features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.054%
/features/features.13/conv/conv.2/Conv:          |                      | 0.050%
/features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.042%
/features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.040%
/features/features.2/conv/conv.2/Conv:           |                      | 0.038%
/features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.034%
/features/features.17/conv/conv.2/Conv:          |                      | 0.030%
/features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.025%
/features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.024%
/features/features.10/conv/conv.2/Conv:          |                      | 0.022%
/features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.021%
/features/features.9/conv/conv.2/Conv:           |                      | 0.021%
/features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
/features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
/features/features.5/conv/conv.2/Conv:           |                      | 0.019%
/features/features.8/conv/conv.2/Conv:           |                      | 0.018%
/features/features.12/conv/conv.2/Conv:          |                      | 0.017%
/features/features.6/conv/conv.2/Conv:           |                      | 0.014%
/features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
/features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
/features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
/features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
/features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
/features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
/features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
/features/features.18/features.18.0/Conv:        |                      | 0.002%
/features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
/features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

Test: [0 / 125]	*Prec@1 60.500 Prec@5 83.275*
```

- **Quantization Error Analysis:**

The top-1 accuracy after quantization is only 60.5%, which is significantly lower than the accuracy of the float model (71.878%). The quantization model has a substantial loss in accuracy, with:

**Graphwise Error:**  
The last layer of the model is /classifier/classifier.1/Gemm, and the cumulative error for this layer is 25.591%. Generally, if the cumulative error of the last layer is less than 10%, the loss in accuracy of the quantized model is minimal.

**Layerwise Error:**  
Observing the Layerwise error, it is found that the errors for most layers are below 1%, indicating that the quantization errors for most layers are small. Only a few layers have larger errors, and we can choose to quantize these layers using int16.
Please refer to Test2 for details.

#### Quantization Test2

- **Quantization Settings:**
```
from ppq.api import get_target_platform
target="esp32p4"
num_of_bits=8
batch_size=32

# Quantize the following layers with 16-bits
dispatching_override={
    "/features/features.1/conv/conv.0/conv.0.0/Conv":get_target_platform(target, num_of_bits=16),
    "/features/features.1/conv/conv.0/conv.0.2/Clip":get_target_platform(target, num_of_bits=16),
}
```

- **Quantization Results:**

```
Layer                                            | NOISE:SIGNAL POWER RATIO 
/features/features.16/conv/conv.2/Conv:          | ████████████████████ | 31.585%
/features/features.15/conv/conv.2/Conv:          | ███████████████████  | 29.253%
/features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 25.077%
/features/features.14/conv/conv.2/Conv:          | ████████████████     | 24.819%
/features/features.17/conv/conv.2/Conv:          | ████████████         | 19.546%
/features/features.13/conv/conv.2/Conv:          | ████████████         | 19.283%
/features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 18.764%
/features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 18.596%
/features/features.18/features.18.0/Conv:        | ████████████         | 18.541%
/features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 15.633%
/features/features.12/conv/conv.2/Conv:          | █████████            | 14.784%
/features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 14.773%
/features/features.14/conv/conv.1/conv.1.0/Conv: | █████████            | 13.700%
/features/features.6/conv/conv.2/Conv:           | ████████             | 12.824%
/features/features.10/conv/conv.2/Conv:          | ███████              | 11.727%
/features/features.14/conv/conv.0/conv.0.0/Conv: | ███████              | 10.612%
/features/features.11/conv/conv.2/Conv:          | ██████               | 10.262%
/features/features.9/conv/conv.2/Conv:           | ██████               | 9.967%
/classifier/classifier.1/Gemm:                   | ██████               | 9.117%
/features/features.5/conv/conv.2/Conv:           | ██████               | 8.915%
/features/features.7/conv/conv.2/Conv:           | █████                | 8.690%
/features/features.3/conv/conv.2/Conv:           | █████                | 8.586%
/features/features.4/conv/conv.2/Conv:           | █████                | 7.525%
/features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 7.432%
/features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 7.317%
/features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.848%
/features/features.8/conv/conv.2/Conv:           | ████                 | 6.711%
/features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.100%
/features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.043%
/features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 5.962%
/features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.873%
/features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.833%
/features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 5.832%
/features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.736%
/features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.639%
/features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.017%
/features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 4.963%
/features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 4.870%
/features/features.3/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.655%
/features/features.2/conv/conv.2/Conv:           | ███                  | 4.650%
/features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.648%
/features/features.1/conv/conv.1/Conv:           | ███                  | 4.318%
/features/features.9/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.849%
/features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.712%
/features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.394%
/features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.391%
/features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.713%
/features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.637%
/features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.602%
/features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 2.397%
/features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.759%
/features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.433%
/features/features.0/features.0.0/Conv:          |                      | 0.119%
Analysing Layerwise quantization error:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [08:27<00:00,  9.58s/it]
* 
Layer                                            | NOISE:SIGNAL POWER RATIO 
/features/features.1/conv/conv.1/Conv:           | ████████████████████ | 1.096%
/features/features.0/features.0.0/Conv:          | ███████████████      | 0.844%
/features/features.2/conv/conv.1/conv.1.0/Conv:  | ██████████           | 0.574%
/features/features.3/conv/conv.1/conv.1.0/Conv:  | ████████             | 0.425%
/features/features.15/conv/conv.1/conv.1.0/Conv: | █████                | 0.272%
/features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 0.238%
/features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 0.214%
/features/features.4/conv/conv.1/conv.1.0/Conv:  | ███                  | 0.180%
/features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 0.151%
/features/features.12/conv/conv.1/conv.1.0/Conv: | ███                  | 0.148%
/features/features.16/conv/conv.1/conv.1.0/Conv: | ███                  | 0.146%
/features/features.14/conv/conv.2/Conv:          | ██                   | 0.136%
/features/features.13/conv/conv.1/conv.1.0/Conv: | ██                   | 0.105%
/features/features.6/conv/conv.1/conv.1.0/Conv:  | ██                   | 0.105%
/features/features.8/conv/conv.1/conv.1.0/Conv:  | █                    | 0.083%
/features/features.5/conv/conv.1/conv.1.0/Conv:  | █                    | 0.076%
/features/features.3/conv/conv.2/Conv:           | █                    | 0.075%
/features/features.16/conv/conv.2/Conv:          | █                    | 0.074%
/features/features.13/conv/conv.0/conv.0.0/Conv: | █                    | 0.072%
/features/features.7/conv/conv.2/Conv:           | █                    | 0.071%
/features/features.15/conv/conv.2/Conv:          | █                    | 0.066%
/features/features.4/conv/conv.2/Conv:           | █                    | 0.065%
/features/features.11/conv/conv.2/Conv:          | █                    | 0.063%
/classifier/classifier.1/Gemm:                   | █                    | 0.063%
/features/features.13/conv/conv.2/Conv:          | █                    | 0.059%
/features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.054%
/features/features.10/conv/conv.1/conv.1.0/Conv: | █                    | 0.042%
/features/features.17/conv/conv.0/conv.0.0/Conv: | █                    | 0.040%
/features/features.2/conv/conv.2/Conv:           | █                    | 0.038%
/features/features.4/conv/conv.0/conv.0.0/Conv:  | █                    | 0.034%
/features/features.17/conv/conv.2/Conv:          | █                    | 0.030%
/features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.025%
/features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.024%
/features/features.10/conv/conv.2/Conv:          |                      | 0.022%
/features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.021%
/features/features.9/conv/conv.2/Conv:           |                      | 0.021%
/features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
/features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
/features/features.5/conv/conv.2/Conv:           |                      | 0.019%
/features/features.8/conv/conv.2/Conv:           |                      | 0.018%
/features/features.12/conv/conv.2/Conv:          |                      | 0.017%
/features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.017%
/features/features.6/conv/conv.2/Conv:           |                      | 0.014%
/features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
/features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
/features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
/features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
/features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
/features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
/features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
/features/features.18/features.18.0/Conv:        |                      | 0.002%
/features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
/features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

Test: [0 / 125]	*Prec@1 69.550 Prec@5 88.450*
```

- **Quantization Error Analysis:**

After replacing the layer with the highest error with 16-bits quantization, a noticeable improvement in model accuracy can be observed. The top-1 accuracy after quantization is 69.550%, which is quite close to the accuracy of the float model (71.878%).

The graphwise error for the last layer of the model, /classifier/classifier.1/Gemm, is 9.117%.

If you wish to further reduce the quantization error, you can try using Quantization Aware Training (QAT). For specific methods, please refer to the [ppq QAT example](https://github.com/OpenPPL/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py).

 > Note: the model in [examples/mobilenet_v2](../examples/mobilenet_v2/) comes from Test1. The 16-bit conv operator is still under development。