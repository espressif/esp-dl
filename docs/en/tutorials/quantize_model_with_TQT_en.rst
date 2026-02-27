Quantizing Models with TQT
===========================

.. note::
   This document explains why and how to use Trained Quantization Thresholds (TQT) in ESP-PPQ for quantization to achieve higher quantized accuracy. Make sure your ESP-PPQ installation is updated to at least version 1.2.7.

---

Table of Contents
-----------------

- :ref:`Why Use TQT <why-tqt>`
  - :ref:`Limitations of Post-Training Quantization (PTQ) <ptq-limit>`
  - :ref:`Complexity of Quantization-Aware Training (QAT) <qat-complex>`
  - :ref:`What TQT Offers <what-tqt-offers>`
- :ref:`How to Use TQT <how-tqt>`
  - :ref:`Quick Start <quick-start>`
  - :ref:`TQTSetting Parameters <tqt-params>`
- :ref:`TQT Quantization Examples <tqt-examples>`
  - :ref:`YOLO26n Quantization <yolo26n-en>`
  - :ref:`MobileNetV2 Quantization <mobilenetv2-en>`
- :ref:`FAQ <faq>`

---

.. _why-tqt:

Why Use TQT
-----------

TQT (Trained Quantization Thresholds) comes from the paper `Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks <https://arxiv.org/abs/1903.08066>`_ (Sambhav R. Jain, Albert Gural, Michael Wu, Chris H. Dick), published at **MLSys 2020**. Since ESP32-series chips currently only support ``.espdl`` models produced by **Per-Tensor + Symmetric + Power-of-2** quantization, we have integrated the core idea of TQT into ESP-PPQ, jointly optimizing quantization thresholds (scale) and model weights via **standard backpropagation and gradient descent** to match hardware constraints.

In ESP-PPQ, TQT is implemented as **TrainedQuantizationThresholdPass**. We optimize scale in the log domain and adapt to ESP-DL's **Power-of-2** constraint (e.g. int_lambda, STE, etc.).

.. _ptq-limit:

Limitations of Post-Training Quantization (PTQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Method
     - Pros
     - Limitations
   * - PTQ
     - Simple, no training
     - Scale is statistics-based and may be suboptimal; larger errors on sensitive structures (e.g. YOLO26n head)
   * - TQT
     - Finetune scale/weights
     - Requires calibration data and compute

.. _qat-complex:

Complexity of Quantization-Aware Training (QAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantization-aware training (QAT) usually inserts fake quantization in full training or finetuning and can achieve good accuracy, but it requires:

- **Full labeled data and long training/finetuning** (even for models that do not use labels, QAT often needs long training/finetuning);
- **More compute and hyperparameter tuning**;
- Extra engineering to integrate with existing PTQ/calibration pipelines.

TQT, by contrast, **does not need labels**: the loss is MSE between floating-point output and quantized output; only a calibration set is needed. Under the Power-of-2 constraint it jointly finetunes weights and log2(scale), and often improves accuracy in relatively few steps.

.. _what-tqt-offers:

What TQT Offers
~~~~~~~~~~~~~~~

- **Learnable scale (still 2^k)**: Optimize alpha = log2(scale) in the log domain; numerically stable and naturally satisfies POWER_OF_2.
- **STE and range–precision tradeoff**: Using STE for threshold gradients and a sensible forward/backward design gives a better tradeoff between **representation range** and **accuracy**; in ESP-PPQ this is done via alpha_ste.
- **Optional learnable weights**: Block-wise finetuning of weights together with scale to further reduce quantization error.
- **Alignment with ESP-DL**: With the default alpha_ste forward, the forward uses integer exponents; with int_lambda the learned alpha is closer to an integer, which eases export and matches the chip side.
- **No labels**: Only a calibration set is needed for scale/weight finetuning.

---

.. _how-tqt:

How to Use TQT
--------------

.. _quick-start:

Quick Start
~~~~~~~~~~~

**1. Use ESP-DL default setting and enable TQT**

.. code-block:: python

   from esp_ppq import QuantizationSettingFactory
   from esp_ppq.api import espdl_quantize_onnx

   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.tqt_optimization = True
   quant_setting.tqt_optimization_setting.int_lambda = 0.25   # optional: make exponent closer to integer
   quant_setting.tqt_optimization_setting.steps = 500
   quant_setting.tqt_optimization_setting.lr = 1e-5
   quant_setting.tqt_optimization_setting.collecting_device = "cuda"  # or "cpu"
   quant_setting.tqt_optimization_setting.block_size = 2

**2. Pass the setting into quantization**

.. code-block:: python

   from esp_ppq.api import ENABLE_CUDA_KERNEL
   with ENABLE_CUDA_KERNEL():
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
           error_report=False,
           skip_export=False,
           export_test_values=True,
           verbose=0,
           inputs=None,
       )

**3. Export and return value**

- **Returned quant_ppq_graph**: scale is **TQT-finetuned 2^integer**, and can be used for model accuracy evaluation.
- **Exported .espdl file**: exponent comes from int(log2(config.scale)); **export and chip-side inference use the TQT-optimized scale**.

.. _tqt-params:

TQTSetting Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 12 10 56
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - interested_layers
     - List[str]
     - []
     - Names of ops to finetune; if empty, all eligible Conv/Gemm etc. are used
   * - steps
     - int
     - 500
     - Finetuning steps per block
   * - lr
     - float
     - 1e-5
     - Learning rate
   * - block_size
     - int
     - 4
     - Block size for graph splitting; affects stability and speed
   * - is_scale_trainable
     - bool
     - True
     - Whether to finetune scale (POWER_OF_2 + LINEAR + SYMMETRICAL). True: train scale and weights; False: train only weights
   * - gamma
     - float
     - 0.0
     - Regularization: MSE(floating-point output, quantized output)
   * - int_lambda
     - float
     - 0.0
     - Regularization: pull alpha toward round(alpha); range [0.0, 1.0]
   * - collecting_device
     - str
     - cpu
     - Device for caching calibration data; set to cuda if GPU is available

---

.. _tqt-examples:

TQT Quantization Examples
--------------------------

The following examples show how to enable TQT quantization for **YOLO26n** (detection) and **MobileNetV2** (classification) and export to ESP-DL. Adjust calibration data and model paths to your environment.

.. _yolo26n-en:

YOLO26n Quantization
~~~~~~~~~~~~~~~~~~~~~

Preparation
^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Item
     - Description
   * - Model name
     - YOLO26n (one2one branch end2end inference)
   * - Task
     - Object detection
   * - Input shape
     - [1, 3, 640, 640] (NCHW)
   * - ONNX
     - `yolo26n_o2o.onnx <https://dl.espressif.com/public/yolo26n_o2o.onnx>`_
   * - Calibration
     - `calib_yolo26n.zip <https://dl.espressif.com/public/calib_yolo26n.zip>`_
   * - Note
     - Non-end2end with one2many is also supported; ONNX: `yolo26n_o2m.onnx <https://dl.espressif.com/public/yolo26n_o2m.onnx>`_; quantization method is the same as end2end

Quantization script
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   from esp_ppq import QuantizationSettingFactory
   from esp_ppq.api import espdl_quantize_onnx
   from torch.utils.data import DataLoader
   import torch
   from torch.utils.data import Dataset
   from torchvision import transforms
   from PIL import Image
   from onnxsim import simplify
   import onnx
   import zipfile
   import urllib.request
   from esp_ppq.api import ENABLE_CUDA_KERNEL

   class CaliDataset(Dataset):
       def __init__(self, path, img_shape=640):
           super().__init__()
           self.transform = transforms.Compose(
               [
                   transforms.Resize((img_shape, img_shape)),
                   transforms.ToTensor(),
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
           if img.mode == 'L':
               img = img.convert('RGB')
           img = self.transform(img)
           return img


   def report_hook(blocknum, blocksize, total):
       downloaded = blocknum * blocksize
       percent = downloaded / total * 100
       print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


   def quant_yolo26n(imgsz):
       BATCH_SIZE = 32
       INPUT_SHAPE = [3, imgsz, imgsz]
       DEVICE = "cpu"
       TARGET = "esp32p4" # or "esp32s3"
       NUM_OF_BITS = 8

       yolo26n_onnx_url = "https://dl.espressif.com/public/yolo26n_o2o.onnx"
       ONNX_PATH = "yolo26n_o2o.onnx"
       urllib.request.urlretrieve(
           yolo26n_onnx_url, "yolo26n_o2o.onnx", reporthook=report_hook
       )

       ESPDL_MODLE_PATH = "yolo26n_o2o_ptq_fq_tqt_p4_640.espdl"

       yolo26n_caib_url = "https://dl.espressif.com/public/calib_yolo26n.zip"
       CALIB_DIR = "calib_yolo26n"
       urllib.request.urlretrieve(
           yolo26n_caib_url, "calib_yolo26n.zip", reporthook=report_hook
       )
       with zipfile.ZipFile("calib_yolo26n.zip", "r") as zip_file:
           zip_file.extractall("./")

       model = onnx.load(ONNX_PATH)
       sim = True
       if sim:
           model, check = simplify(model)
           assert check, "Simplified ONNX model could not be validated"
       onnx.save(onnx.shape_inference.infer_shapes(model), ONNX_PATH)

       calibration_dataset = CaliDataset(CALIB_DIR, img_shape=imgsz)
       dataloader = DataLoader(
           dataset=calibration_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
       )

       def collate_fn(batch: torch.Tensor) -> torch.Tensor:
           return batch.to(DEVICE)


       # default setting
       quant_setting = QuantizationSettingFactory.espdl_setting()
       # activation calibration algo
       quant_setting.quantize_activation_setting.calib_algorithm = "percentile"  # kl --> percentile to get better mAP
       # focused logits quantization
       quant_setting.quant_config_modify = True
       # o2o
       quant_setting.quant_config_modify_setting.custom_config = {
           "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv": 0.0625,
           "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv": 0.0625,
           "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv": 0.0625,
       }

       # o2m (alternative)
       # quant_setting.quant_config_modify_setting.custom_config = {
       #     "/model.23/cv3.0/cv3.0.2/Conv": 0.0625,
       #     "/model.23/cv3.1/cv3.1.2/Conv": 0.0625,
       #     "/model.23/cv3.2/cv3.2.2/Conv": 0.0625,
       # }

       # TQT
       quant_setting.tqt_optimization = True
       quant_setting.tqt_optimization_setting.collecting_device = "cpu"
       quant_setting.tqt_optimization_setting.steps = 200  #300 for o2m
       quant_setting.tqt_optimization_setting.block_size = 2
       quant_setting.tqt_optimization_setting.lr = 1e-5

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
       quant_yolo26n(imgsz=640)


Focused Logits Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In detection models, unlike backbone/neck, **detection head outputs** are very sensitive to quantization error. For models like YOLO26n, the classification branch outputs logits that are **inputs to Sigmoid**; after Sigmoid they become class confidence or scores. Using the same calibration-derived scale for these logits as for other layers often leads to **scale that is too large (quantization step too coarse)**, distorting class scores and reducing mAP.

- **Sigmoid has a limited effective input range**: σ(x) = 1/(1+exp(-x)) saturates quickly for large absolute x—output tends to 0 when x is very negative and to 1 when very positive; only the middle range (roughly -4 to 4) changes meaningfully. So the logits that actually discriminate are in this **limited range**; outside it they sit in saturation and barely change.

- **Saturation on both sides; detection cares more about positive samples**: In object detection we care more about **positive samples** (with object, high confidence), i.e. logits on the positive side and Sigmoid output near 1. On both sides, if logits are quantized too coarsely (scale too large, step too big), many different logits map to the same quantized level: near saturation they get "squashed" to 0 or 1, and in the effective middle range fine-grained distinction is lost, so scores are distorted and mAP drops.

So for **logits fed into Sigmoid** (the last Conv outputs of the three head branches one2one_cv3.0/1/2.2 in the example), we use **Focused logits quantization**: assign **finer scale** (e.g. 0.0625 = 2^-4) to these layers so that, while keeping Power-of-2, the quantization step is smaller, more levels are kept in Sigmoid's effective range, saturation is delayed, and discrimination on the positive side is preserved, stabilizing or improving mAP. For other detection models or different export structures, identify which layers feed Sigmoid and adjust custom_config accordingly.


Quantization accuracy on ESP32-P4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accuracy after quantizing yolo26n (size=640 pixels) on COCO val2017:

.. list-table::
   :widths: 20 18 18 44
   :header-rows: 1

   * - Config
     - mAP50-95(o2m)
     - mAP50-95(e2e)
     - Note
   * - PTQ (no TQT)
     - 0.342
     - 0.332
     - Calibration only
   * - PTQ + TQT
     - 0.371
     - 0.363
     - Calibration + TQT scale/weight finetuning

Results show that TQT significantly improves mAP for both end2end and non-end2end models.

.. note::
   For faster inference, you can use size=512 pixels. Experiments show that with size=512 pixels and end2end inference, the 8-bit PTQ model reaches mAP50-95 of 0.315. After TQT, mAP50-95 improves to 0.341, representing a gain of 2.6 percentage points.

.. _mobilenetv2-en:

MobileNetV2 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~

Quantization script
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   import subprocess
   from typing import Iterable, Tuple, List, Tuple

   import torch
   from datasets.imagenet_util import (
       evaluate_ppq_module_with_imagenet,
       load_imagenet_from_directory,
   )
   from esp_ppq import QuantizationSettingFactory, QuantizationSetting
   from esp_ppq.api import espdl_quantize_onnx, get_target_platform
   from torch.utils.data import DataLoader
   import torchvision.datasets as datasets
   import torchvision.transforms as transforms
   from torch.utils.data.dataset import Subset
   import urllib.request
   import zipfile


   def quant_setting_mobilenet_v2(
       onnx_path: str,
       optim_quant_method: List[str] = None,
   ) -> Tuple[QuantizationSetting, str]:
       '''
       Quantize onnx model with optim_quant_method.

       Args:
           optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
           -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                           the layers to 16-bit quantization. You can remove or add layers according to your
                                           needs.
           -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                   Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                   Since ReLU6 exists in MobilenetV2, convert ReLU6 to ReLU for better precision.

       Returns:
           [tuple]: [QuantizationSetting, str]
       '''
       quant_setting = QuantizationSettingFactory.espdl_setting()
       if optim_quant_method is not None:
           if "MixedPrecision_quantization" in optim_quant_method:
               # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
               # You can remove or add layers according to your needs.
               quant_setting.dispatching_table.append(
                   "/features/features.1/conv/conv.0/conv.0.0/Conv",
                   get_target_platform(TARGET, 16),
               )
               quant_setting.dispatching_table.append(
                   "/features/features.1/conv/conv.0/conv.0.2/Clip",
                   get_target_platform(TARGET, 16),
               )
           elif "LayerwiseEqualization_quantization" in optim_quant_method:
               # layerwise equalization
               quant_setting.equalization = True
               quant_setting.equalization_setting.iterations = 4
               quant_setting.equalization_setting.value_threshold = 0.4
               quant_setting.equalization_setting.opt_level = 2
               quant_setting.equalization_setting.interested_layers = None
               # replace ReLU6 with ReLU
               onnx_path = onnx_path.replace("mobilenet_v2.onnx", "mobilenet_v2_relu.onnx")
           else:
               raise ValueError(
                   "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
               )

       return quant_setting, onnx_path


   def collate_fn1(x: Tuple) -> torch.Tensor:
       return torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0)


   def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
       return batch.to(DEVICE)


   def report_hook(blocknum, blocksize, total):
       downloaded = blocknum * blocksize
       percent = downloaded / total * 100
       print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


   if __name__ == "__main__":
       BATCH_SIZE = 32
       INPUT_SHAPE = [3, 224, 224]
       DEVICE = "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
       TARGET = "esp32p4"  #  'c', 'esp32s3' or 'esp32p4'
       NUM_OF_BITS = 8
       ONNX_PATH = "./models/torch/mobilenet_v2.onnx"  #'models/onnx/mobilenet_v2.onnx'
       ESPDL_MODEL_PATH = "models/onnx/mobilenet_v2.espdl"
       CALIB_DIR = "./imagenet"

       # Download mobilenet_v2 model from onnx models and dataset
       imagenet_url = "https://dl.espressif.com/public/imagenet_calib.zip"
       os.makedirs(CALIB_DIR, exist_ok=True)
       if not os.path.exists("imagenet_calib.zip"):
           urllib.request.urlretrieve(
               imagenet_url, "imagenet_calib.zip", reporthook=report_hook
           )
       if not os.path.exists(os.path.join(CALIB_DIR, "calib")):
           with zipfile.ZipFile("imagenet_calib.zip", "r") as zip_file:
               zip_file.extractall(CALIB_DIR)
       CALIB_DIR = os.path.join(CALIB_DIR, "calib")

       # -------------------------------------------
       # Prepare Calibration Dataset
       # --------------------------------------------
       if os.path.exists(CALIB_DIR):
           print(f"load imagenet calibration dataset from directory: {CALIB_DIR}")
           dataset = datasets.ImageFolder(
               CALIB_DIR,
               transforms.Compose(
                   [
                       transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize(
                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                       ),
                   ]
               ),
           )
           dataset = Subset(dataset, indices=[_ for _ in range(0, 1024)])
           dataloader = DataLoader(
               dataset=dataset,
               batch_size=BATCH_SIZE,
               shuffle=False,
               num_workers=4,
               pin_memory=False,
               collate_fn=collate_fn1,
           )
       else:
           # Random calibration dataset only for debug
           print("load random calibration dataset")

           def load_random_calibration_dataset() -> Iterable:
               return [torch.rand(size=INPUT_SHAPE) for _ in range(BATCH_SIZE)]

           # Load training data for creating a calibration dataloader.
           dataloader = DataLoader(
               dataset=load_random_calibration_dataset(),
               batch_size=BATCH_SIZE,
               shuffle=False,
           )

       # -------------------------------------------
       # Quantize ONNX Model.
       # --------------------------------------------

       # create a setting for quantizing your network with ESPDL.
       # if you don't need to optimize quantization, set the input 1 of the quant_setting_mobilenet_v2 function None
       # Example: Using LayerwiseEqualization_quantization
       quant_setting, ONNX_PATH = quant_setting_mobilenet_v2(
           ONNX_PATH, None
       )

       # TQT
       quant_setting.tqt_optimization = True
       quant_setting.tqt_optimization_setting.collecting_device = "cpu"
       quant_setting.tqt_optimization_setting.steps = 500
       quant_setting.tqt_optimization_setting.block_size = 4
       quant_setting.tqt_optimization_setting.lr = 1e-4

       quant_ppq_graph = espdl_quantize_onnx(
           onnx_import_file=ONNX_PATH,
           espdl_export_file=ESPDL_MODEL_PATH,
           calib_dataloader=dataloader,
           calib_steps=32,
           input_shape=[1] + INPUT_SHAPE,
           target=TARGET,
           num_of_bits=NUM_OF_BITS,
           collate_fn=collate_fn2,
           setting=quant_setting,
           device=DEVICE,
           error_report=True,
           skip_export=False,
           export_test_values=False,
           verbose=1,
       )

       # -------------------------------------------
       # Evaluate Quantized Model.
       # --------------------------------------------

       evaluate_ppq_module_with_imagenet(
           model=quant_ppq_graph,
           imagenet_validation_dir=CALIB_DIR,
           batchsize=BATCH_SIZE,
           device=DEVICE,
           verbose=1,
       )


Quantization results and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using only the TQT strategy for 8-bit quantization, the model achieves a Top-1 accuracy of 71.525%, representing a 1.725 percentage point improvement over the weight-equalization-only quantized model, which attains 69.800%. The result is also much closer to the floating-point baseline accuracy of 71.878%. These results demonstrate that even under strict 8-bit quantization constraints, TQT effectively mitigates performance degradation caused by quantization error by learning quantization thresholds, enabling the quantized model to approach float32-level accuracy.

.. _faq:

FAQ
---

How to speed up TQT?
~~~~~~~~~~~~~~~~~~~~

With a GPU, put calibration data on the GPU.

First set collecting_device to cuda:

.. code-block:: python

   quant_setting.tqt_optimization_setting.collecting_device = "cuda"

Then run quantization inside ENABLE_CUDA_KERNEL:

.. code-block:: python

   with ENABLE_CUDA_KERNEL():
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
               error_report=False,
               skip_export=False,
               export_test_values=False,
               verbose=0,
               inputs=None,
           )

Increasing block_size (e.g. 2–4) reduces the number of blocks and total time, but too large can be unstable; try 4 first.

Can I train only model weights without modifying scale?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Yes. Set is_scale_trainable to False.

.. code-block:: python

   quant_setting.tqt_optimization_setting.is_scale_trainable = False
