使用 TQT 量化模型
=================

.. note::
   本文档说明为何以及如何在 ESP-PPQ 中使用 Trained Quantization Thresholds (TQT) 做量化，以获得更高的量化精度。注意ESP-PPQ 必须是 1.2.7 版本或更新版本。

---

目录
----

- :ref:`为何使用 TQT <tqt-why>`
  - :ref:`后量化（PTQ）的局限性 <ptq-limit>`
  - :ref:`量化感知训练（QAT）的复杂性 <qat-complex>`
  - :ref:`TQT 能带来什么 <tqt-benefit>`
- :ref:`如何使用 TQT <tqt-how>`
  - :ref:`快速开始 <tqt-quick>`
  - :ref:`TQTSetting常用参数 <tqt-params>`
- :ref:`TQT 量化示例 <tqt-examples>`
  - :ref:`YOLO26n 量化 <yolo26n>`
  - :ref:`MobileNetV2 量化 <mobilenetv2>`
- :ref:`常见问题 <tqt-faq>`

---

.. _tqt-why:

为何使用 TQT
------------

TQT（Trained Quantization Thresholds）出自论文 `Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks <https://arxiv.org/abs/1903.08066>`_ （Sambhav R. Jain, Albert Gural, Michael Wu, Chris H. Dick），发表于 :strong:`MLSys 2020`。由于ESP32系列芯片目前仅支持 :strong:`Per-Tensor + Symmetric + Power-of-2` 量化策略生成的 ``.espdl`` 模型，我们在ESP-PPQ中引入了TQT的核心思想，通过 :strong:`标准反向传播与梯度下降` 联合优化量化阈值（scale）和模型权重，以适配硬件约束。

在 ESP-PPQ 中，TQT 以 :strong:`TrainedQuantizationThresholdPass` 的形式实现，我们在 log 域对scale进行优化，并结合ESP-DL的 :strong:`Power-of-2` 约束做了适配（如 ``int_lambda`` ，STE等）。

.. _ptq-limit:

后量化（PTQ）的局限性
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - 方式
     - 优点
     - 局限
   * - PTQ
     - 实现简单、无需训练
     - scale 由统计得到，未必最优；对敏感结构（如YOLO26n head）误差较大
   * - TQT
     - 微调 scale/权重
     - 需要一定校准数据与算力

.. _qat-complex:

量化感知训练（QAT）的复杂性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

量化感知训练（QAT）通常在完整训练或微调阶段插入伪量化，能获得较好精度，但需要：

- :strong:`完整标注数据与较长训练、微调流程` (对于不使用标注数据进行训练的模型，QAT也往往花费较长的训练、微调流程)；
- :strong:`较大算力与调参成本`；
- 与现有 PTQ/校准工具链的衔接往往需要额外工程。

TQT 则 :strong:`无需标签`：损失为「浮点输出 vs 量化输出」的 MSE，只需校准集即可，操作简单，在保持 Power-of-2 约束下联合微调权重与 log2(scale)，通常用较少步数即可明显提升精度。

.. _tqt-benefit:

TQT 能带来什么
~~~~~~~~~~~~~~

- :strong:`可学习的 scale (仍为 2 的 k 次方)`：在 log 域优化 ``alpha = log2(scale)`` ，数值稳定且自然满足 POWER_OF_2。
- :strong:`STE 与 range-precision 折中`：对阈值梯度使用 STE 并合理设计前向/反向行为，可在 :strong:`表示范围` 与 :strong:`精度` 之间取得更好折中；ESP-PPQ 中通过 alpha_ste实现。
- :strong:`可选的可学习权重`：与 scale 一起做块级微调，进一步减小量化误差。
- :strong:`与 ESP-DL 对齐`：使用默认的 alpha_ste 前向时，前向用的就是整数 exponent；配合 ``int_lambda`` 可让学到的 alpha 更接近整数，便于导出与芯片端一致。
- :strong:`无需标签`：只需校准集即可做 scale/权重微调。

---

.. _tqt-how:

如何使用 TQT
------------

.. _tqt-quick:

快速开始
~~~~~~~~

:strong:`1. 使用 ESP-DL 默认setting并开启 TQT`

.. code-block:: python

   from esp_ppq import QuantizationSettingFactory
   from esp_ppq.api import espdl_quantize_onnx

   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.tqt_optimization = True
   quant_setting.tqt_optimization_setting.int_lambda = 0.25   # 可选：让 exponent 更接近整数
   quant_setting.tqt_optimization_setting.steps = 500
   quant_setting.tqt_optimization_setting.lr = 1e-5
   quant_setting.tqt_optimization_setting.collecting_device = 'cuda'  # or 'cpu'
   quant_setting.tqt_optimization_setting.block_size = 2

:strong:`2. 传入 setting 做量化`

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

:strong:`3. 导出与返回值`

- :strong:`返回的 quant_ppq_graph` 里 scale 为 :strong:`TQT 微调后的 2 的整数次方`，可用于模型精度评估。
- :strong:`导出的 .espdl 文件`：exponent 来自 int(log2(config.scale))，导出与芯片端推理会使用 TQT 优化后的 scale。

.. _tqt-params:

TQTSetting常用参数
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 12 10 56
   :header-rows: 1

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - interested_layers
     - List[str]
     - []
     - 指定要微调的 op 名称；空则对满足条件的 Conv/Gemm 等全部微调
   * - steps
     - int
     - 500
     - 每个 block 的微调步数
   * - lr
     - float
     - 1e-5
     - 学习率
   * - block_size
     - int
     - 4
     - 图划分 block 的大小，影响稳定性与速度
   * - is_scale_trainable
     - bool
     - True
     - 是否微调 scale（需 POWER_OF_2 + LINEAR + SYMMETRICAL）。True：同时训练 scale 与 weights；False：仅训练 weights
   * - gamma
     - float
     - 0.0
     - 正则：MSE(浮点输出, 量化输出)
   * - int_lambda
     - float
     - 0.0
     - 正则：让 alpha 靠近 round(alpha)，取值范围为[0.0, 1.0]
   * - collecting_device
     - str
     - cpu
     - 缓存校准数据的设备，有 GPU 可设为 cuda

---

.. _tqt-examples:

TQT 量化示例
------------

以下示例展示如何对 :strong:`YOLO26n` (检测) 和 :strong:`MobileNetV2` (分类) 模型启用 TQT 量化并导出 ESP-DL。校准数据与模型路径需按实际环境修改。

.. _yolo26n:

YOLO26n 量化
~~~~~~~~~~~~

准备
^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - 项目
     - 说明
   * - 模型名称
     - YOLO26n (one2one分支end2end推理)
   * - 任务
     - 目标检测
   * - 输入形状
     - [1, 3, 640, 640] (NCHW)
   * - ONNX
     - `yolo26n_o2o.onnx <https://dl.espressif.com/public/yolo26n_o2o.onnx>`_
   * - 校准集
     - `calib_yolo26n.zip <https://dl.espressif.com/public/calib_yolo26n.zip>`_
   * - 备注
     - 也可以通过one2many分支做非end2end推理，ONNX为 `yolo26n_o2m.onnx <https://dl.espressif.com/public/yolo26n_o2m.onnx>`_ ，量化方法与end2end模型一致

量化脚本
^^^^^^^^

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

       # o2m
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

目标检测模型中，与 backbone/neck 不同，:strong:`检测头 (head) 输出` 对量化误差非常敏感。同时，对于类似YOLO26n这样的检测模型，其 classification 分支的输出 logits 为 :strong:`Sigmoid 的输入`，经过Sigmoid可以得到类别置信度或分数。此时，若将logits与其它层使用同一套校准得到的 scale，容易得到 :strong:`过大的 scale (量化步长过粗)`，导致类别分数失真，mAP 下降。

- :strong:`Sigmoid 的有效输入范围有限`：σ(x) = 1/(1+exp(-x)) 在 x 的绝对值较大时很快饱和——x 很负时输出趋近 0，很正时趋近 1，只有中间一段（大致在 -4 到 4 内）变化明显。也就是说，对输出真正有区分度的 logits 只分布在这段 :strong:`有限范围` 内，超出部分进入饱和区、几乎不再变化。

- :strong:`两边都有饱和区，检测中更关注正样本`：在目标检测里，我们更关心 :strong:`正样本` (有目标、高置信度)，对应 logits 偏正、Sigmoid 输出接近 1 的一侧。但无论正负两侧，只要 logits 被量化得过粗 (scale 过大、步长太大)，多个不同的 logits 会被舍入到同一量化档位：在饱和区附近会成片地被「压」成 0 或 1，在中间有效区则丢失细粒度区分，导致分数失真，mAP 掉点。

因此，对 :strong:`送入 Sigmoid 的 logits` (即示例中 one2one_cv3.0/1/2.2 这三个 head 分支的最后一层 Conv 输出) 做 :strong:`Focused logits quantization`：为这几层 :strong:`单独指定更细的 scale` (如 0.0625 = 2 的 -4 次方)，在保持 Power-of-2 的前提下减小量化步长，在 Sigmoid 的有效输入范围内保留更多档位，避免过早进入饱和、并保留正样本侧的区分度，从而稳定或提升 mAP。若使用其他检测模型或不同导出结构，可根据「谁作为 Sigmoid 的输入」来定位对应层并调整 custom_config。


ESP32-P4上模型量化精度测试结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 COCO val2017 评估yolo26n(size=640 pixels)量化后的精度：

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - 配置
     - mAP50-95(o2m)
     - mAP50-95(e2e)
     - 备注
   * - PTQ (无 TQT)
     - 0.342
     - 0.332
     - 仅校准
   * - PTQ + TQT
     - 0.371
     - 0.363
     - 校准 + TQT 微调 scale/权重

测试结果表明，无论是对end2end推理模型，还是非end2end推理模型，TQT均可以显著提升量化后模型的mAP。

.. note::
   如果追求更快的模型推理速度，可以使用 size=512 pixels。实验结果表明，在 size=512 pixels 和 end2end 推理模式下，8-bit PTQ 量化模型的 mAP50-95 为 0.315，经过 TQT 后 mAP50-95 可提升至 0.341，精度提升了 2.6 个百分点。

.. _mobilenetv2:

MobileNetV2 量化
~~~~~~~~~~~~~~~~

量化脚本
^^^^^^^^

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


量化结果及分析
^^^^^^^^^^^^^^

在仅采用 TQT 策略对 8-bit 模型进行量化优化的情况下，模型的Top-1准确率为71.525%，相比仅使用 weight equalization 的量化模型（Top-1 准确率为69.800%），TQT 在 Top-1 精度上提升了 1.725 个百分点, 与float32模型的精度（71.878%）更加接近，表明即使在严格的 8-bit 量化约束下，仅通过对量化阈值进行可学习优化，TQT 也能够有效缓解量化误差带来的性能下降，使量化模型的精度接近浮点模型水平。

.. _tqt-faq:

常见问题
--------

如何加速 TQT？
~~~~~~~~~~~~~~

有 GPU 时，把校准数据放到 GPU 上可加快速度。

先将 collecting_device 设为 cuda：

.. code-block:: python

   quant_setting.tqt_optimization_setting.collecting_device = "cuda"

再在 ENABLE_CUDA_KERNEL 中执行量化：

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

同时适当增大 block_size（如 2～4）可减少块数、缩短总时间，但过大可能不稳定，建议先试 4。

可以不修改scale只训练模型权重吗？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以。将 is_scale_trainable 设置为 False 即可。

.. code-block:: text

   quant_setting.tqt_optimization_setting.is_scale_trainable = False
