如何部署 MobileNetV2
=========================

:link_to_translation:`en:[English]`

在本教程中，我们介绍如何使用 ESP-PPQ 对预训练的 MobileNetV2 模型进行量化，并使用 ESP-DL 部署量化后的 MobileNetV2 模型。

.. contents::
  :local:
  :depth: 2

准备工作
-----------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_mobilenetv2:

模型量化
-----------

.. note::

   - **ESP32P4**：对 Conv 和 Gemm 算子的权重采用 **per-channel** 量化策略，其余算子仍使用 **per-tensor**。
   - **ESP32S3 及其他芯片**：因指令集限制，所有算子均统一采用 **per-tensor** 量化策略。

   由于 **per-channel** 量化在细节保留上通常优于 **per-tensor**，因此相同模型在 ESP32P4 上的量化精度往往高于 ESP32S3。

   为演示不同的量化方法，下文的示例主要以 ESP32S3 为目标平台进行说明。同时，在 ``8bit 默认配置量化`` 章节中也提供了 ESP32P4 的量化结果，以供开发者对比参考。

:project:`量化脚本 <examples/tutorial/how_to_quantize_model/quantize_mobilenetv2>`

预训练模型
^^^^^^^^^^^

从 torchvision 加载 MobileNet_v2 的预训练模型，你也可以从 `ONNX models <https://github.com/onnx/models>`__ 或 `TensorFlow models <https://github.com/tensorflow/models>`__ 下载：

.. code-block:: python

   import torchvision
   from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

   model = torchvision.models.mobilenet.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

校准数据集
^^^^^^^^^^^^

校准数据集需要和你的模型输入格式一致，校准数据集需要尽可能覆盖你的模型输入的所有可能情况，以便更好地量化模型。这里以 ImageNet 数据集为例，演示如何准备校准数据集。

使用 torchvision 加载 ImageNet 数据集：

.. code-block:: python
   
   import torchvision.datasets as datasets
   from torch.utils.data.dataset import Subset
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

8bit 默认配置量化
^^^^^^^^^^^^^^^^^^

ESP32P4 量化
""""""""""""""

**量化设置**

.. code-block:: python

   import torch.nn as nn

   def convert_relu6_to_relu(model):
      for child_name, child in model.named_children():
         if isinstance(child, nn.ReLU6):
               setattr(model, child_name, nn.ReLU())
         else:
               convert_relu6_to_relu(child)
      return model

   target="esp32p4"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting
   # Replace ReLU6 with ReLU. When testing per-channel, 
   # using ReLU results in higher precision after quantization.
   model = convert_relu6_to_relu(model)

**量化结果**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 28.455%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 24.981%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ███████████████      | 21.120%
   /features/features.14/conv/conv.2/Conv:          | ██████████████       | 20.162%
   /features/features.17/conv/conv.2/Conv:          | █████████████        | 18.303%
   /features/features.18/features.18.0/Conv:        | ████████████         | 17.422%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 16.590%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 15.886%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███████████          | 15.033%
   /features/features.12/conv/conv.2/Conv:          | ██████████           | 13.968%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | █████████            | 12.973%
   /features/features.6/conv/conv.2/Conv:           | █████████            | 12.624%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 12.366%
   /features/features.10/conv/conv.2/Conv:          | █████████            | 12.207%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ███████              | 10.319%
   /features/features.11/conv/conv.2/Conv:          | ███████              | 10.139%
   /features/features.5/conv/conv.2/Conv:           | ███████              | 9.430%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.324%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 9.298%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ██████               | 8.819%
   /features/features.7/conv/conv.2/Conv:           | ██████               | 8.649%
   /features/features.1/conv/conv.1/Conv:           | ██████               | 8.423%
   /features/features.4/conv/conv.2/Conv:           | ██████               | 7.972%
   /classifier/classifier.1/Gemm:                   | █████                | 7.204%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 6.975%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 6.867%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | █████                | 6.626%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.296%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 6.113%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.068%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.044%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.839%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.814%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 5.790%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.580%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.575%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.476%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ████                 | 5.182%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.157%
   /features/features.2/conv/conv.2/Conv:           | ████                 | 5.115%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 5.003%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.470%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 4.412%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.133%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.012%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.524%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.370%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.004%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.680%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.617%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.962%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.477%
   /features/features.0/features.0.0/Conv:          |                      | 0.114%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:33<00:00, 10.81s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.0/features.0.0/Conv:          | ████████████████████ | 0.879%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████         | 0.526%
   /features/features.16/conv/conv.2/Conv:          | █████                | 0.215%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 0.165%
   /features/features.15/conv/conv.2/Conv:          | ███                  | 0.127%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.110%
   /classifier/classifier.1/Gemm:                   | █                    | 0.061%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.055%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.048%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.043%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.039%
   /features/features.17/conv/conv.2/Conv:          | █                    | 0.037%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.035%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.034%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.033%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.027%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.017%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.015%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.013%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.011%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.010%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.007%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.007%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.006%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.006%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.005%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.004%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.004%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.004%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.003%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.003%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.002%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.001%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.001%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.000%

   * Prec@1 71.150 Prec@5 89.350

ESP32S3 量化
""""""""""""""

**量化设置**

.. code-block:: python

   target="esp32s3"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting

**量化结果**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 49.069%
   /features/features.17/conv/conv.2/Conv:          | ███████████████████  | 45.605%
   /features/features.15/conv/conv.2/Conv:          | ███████████████████  | 45.446%
   /features/features.18/features.18.0/Conv:        | ██████████████████   | 43.709%
   /features/features.14/conv/conv.2/Conv:          | █████████████████    | 41.199%
   /features/features.13/conv/conv.2/Conv:          | ██████████████       | 35.330%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █████████████        | 33.151%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 29.433%
   /features/features.12/conv/conv.2/Conv:          | ████████████         | 29.049%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 27.952%
   /features/features.7/conv/conv.2/Conv:           | ███████████          | 27.420%
   /classifier/classifier.1/Gemm:                   | ███████████          | 27.097%
   /features/features.11/conv/conv.2/Conv:          | ███████████          | 26.299%
   /features/features.10/conv/conv.2/Conv:          | ███████████          | 26.026%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 25.428%
   /features/features.6/conv/conv.2/Conv:           | ██████████           | 25.363%
   /features/features.4/conv/conv.2/Conv:           | ██████████           | 24.751%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | ██████████           | 24.485%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 20.296%
   /features/features.9/conv/conv.2/Conv:           | ████████             | 19.854%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ████████             | 18.552%
   /features/features.5/conv/conv.2/Conv:           | ███████              | 17.704%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███████              | 16.861%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ███████              | 16.398%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ██████               | 15.579%
   /features/features.2/conv/conv.2/Conv:           | ██████               | 15.369%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ██████               | 15.193%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 14.844%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ██████               | 14.811%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ██████               | 14.660%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.995%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ██████               | 13.825%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.750%
   /features/features.8/conv/conv.2/Conv:           | █████                | 13.557%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | █████                | 12.814%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | █████                | 12.610%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | █████                | 12.173%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | █████                | 11.607%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █████                | 11.136%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.368%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ████                 | 10.333%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.038%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.632%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 9.282%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 8.970%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ████                 | 8.947%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | ███                  | 8.389%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ███                  | 8.067%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ███                  | 7.462%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | ███                  | 6.513%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 5.537%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.875%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [34:02<00:00, 38.53s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 14.306%
   /features/features.0/features.0.0/Conv:          | █                    | 0.843%
   /features/features.1/conv/conv.1/Conv:           | █                    | 0.667%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.425%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.271%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.237%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.146%
   /features/features.14/conv/conv.2/Conv:          |                      | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.104%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.083%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.076%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.076%
   /features/features.3/conv/conv.2/Conv:           |                      | 0.075%
   /features/features.16/conv/conv.2/Conv:          |                      | 0.073%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.072%
   /features/features.15/conv/conv.2/Conv:          |                      | 0.066%
   /features/features.4/conv/conv.2/Conv:           |                      | 0.064%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.063%
   /classifier/classifier.1/Gemm:                   |                      | 0.063%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.054%
   /features/features.13/conv/conv.2/Conv:          |                      | 0.050%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.040%
   /features/features.2/conv/conv.2/Conv:           |                      | 0.039%
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

   * Prec@1 60.325 Prec@5 83.100

**量化误差分析**

实验结果表明：在 ESP32P4 上采用 per-channel 量化时，量化模型 Prec@1 为 71.150%，与浮点模型（71.878%）接近；而在 ESP32S3 上采用 per-tensor 量化时，Prec@1 仅为 60.5%，相对同一浮点基准差距明显，精度损失较大，有必要结合其他量化手段进一步改进。以下从量化误差角度对 ESP32S3 作简要分析：

- **累计误差 (Graphwise Error)**

  该模型的最后一层为 /classifier/classifier.1/Gemm，该层的累计误差为 27.097%。经验来说最后一层的累计误差小于 10%，量化模型的精度损失较小。

- **逐层误差 (Layerwise error)**

  观察 Layerwise error，发现大部分层的误差都在 1% 以下，说明大部分层的量化误差较小，只有少数几层误差较大，我们可以选择将误差较大的层使用 int16 进行量化。具体请看混合精度量化。


.. _mixed_precision_quantization_label:

混合精度量化
^^^^^^^^^^^^^^

**量化设置**

.. code-block:: python

   from esp_ppq.api import get_target_platform
   target="esp32s3"
   num_of_bits=8
   batch_size=32

   # 以下层使用int16进行量化
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.0/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.2/Clip", get_target_platform(TARGET, 16))

**量化结果**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 31.995%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 29.554%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 25.400%
   /features/features.14/conv/conv.2/Conv:          | ████████████████     | 25.125%
   /features/features.17/conv/conv.2/Conv:          | █████████████        | 21.358%
   /features/features.18/features.18.0/Conv:        | █████████████        | 20.177%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 19.204%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 18.988%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 18.846%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 15.831%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 14.938%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 14.918%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | █████████            | 13.865%
   /features/features.6/conv/conv.2/Conv:           | ████████             | 12.909%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 11.843%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ███████              | 10.648%
   /features/features.11/conv/conv.2/Conv:          | ██████               | 10.391%
   /classifier/classifier.1/Gemm:                   | ██████               | 10.289%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.940%
   /features/features.5/conv/conv.2/Conv:           | ██████               | 8.902%
   /features/features.7/conv/conv.2/Conv:           | █████                | 8.745%
   /features/features.3/conv/conv.2/Conv:           | █████                | 8.614%
   /features/features.4/conv/conv.2/Conv:           | █████                | 7.645%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 7.501%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 7.428%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.915%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.488%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.148%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 6.042%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.955%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.904%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.894%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 5.847%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.793%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.737%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 5.293%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.022%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 5.004%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 4.670%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.662%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.643%
   /features/features.1/conv/conv.1/Conv:           | ███                  | 4.320%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.878%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.740%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.420%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.306%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.734%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.639%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.603%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 2.404%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.765%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.433%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:29<00:00, 10.74s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.1/Conv:           | ████████████████████ | 1.099%
   /features/features.0/features.0.0/Conv:          | ███████████████      | 0.843%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██████████           | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████████             | 0.425%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████                | 0.271%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 0.237%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ███                  | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███                  | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███                  | 0.146%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ██                   | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ██                   | 0.104%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | █                    | 0.083%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █                    | 0.076%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.075%
   /features/features.16/conv/conv.2/Conv:          | █                    | 0.073%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | █                    | 0.072%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.071%
   /features/features.15/conv/conv.2/Conv:          | █                    | 0.066%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.064%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.063%
   /classifier/classifier.1/Gemm:                   | █                    | 0.063%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.054%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | █                    | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █                    | 0.040%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.039%
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
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.017%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.017%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.014%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

   * Prec@1 69.225 Prec@5 88.700

**量化误差分析**

将之前误差最大的层替换为 16 位量化后，可以观察到模型准确度明显提升，量化后的 top1 准确率为 69.225%，和 float 模型的准确率 (71.878%) 比较接近。该模型的最后一层 ``/classifier/classifier.1/Gemm`` 的累计误差为 10.289%。

.. _layerwise_equalization_quantization_label:

层间均衡量化
^^^^^^^^^^^^^^^^

该方法在论文 `Data-Free Quantization Through Weight Equalization and Bias Correction <https://arxiv.org/abs/1906.04721>`_ 中提出。使用此方法时，需要将 MobilenetV2 模型中原来的 ReLU6 替换为 ReLU。

**量化设置**

.. code-block:: python

   import torch.nn as nn

   target="esp32s3"
   num_of_bits=8
   batch_size=32

   def convert_relu6_to_relu(model):
      for child_name, child in model.named_children():
         if isinstance(child, nn.ReLU6):
               setattr(model, child_name, nn.ReLU())
         else:
               convert_relu6_to_relu(child)
      return model

   # Replace ReLU6 with ReLU
   model = convert_relu6_to_relu(model)
   # Use layerwise equalization
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.equalization = True
   quant_setting.equalization_setting.iterations = 6
   quant_setting.equalization_setting.value_threshold = 0.5
   quant_setting.equalization_setting.opt_level = 2
   quant_setting.equalization_setting.interested_layers = None

**量化结果**

.. code-block::

   Analysing Graphwise Quantization Error
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 34.197%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 30.936%
   /features/features.14/conv/conv.2/Conv:          | ████████████████     | 27.411%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 27.235%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 21.350%
   /features/features.17/conv/conv.2/Conv:          | ████████████         | 20.880%
   /features/features.18/features.18.0/Conv:        | ████████████         | 19.900%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███████████          | 19.559%
   /features/features.13/conv/conv.2/Conv:          | ███████████          | 19.325%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 17.006%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 15.475%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 14.920%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 13.397%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 12.634%
   /features/features.6/conv/conv.2/Conv:           | ███████              | 11.711%
   /classifier/classifier.1/Gemm:                   | ██████               | 10.808%
   /features/features.11/conv/conv.2/Conv:          | ██████               | 10.234%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ██████               | 10.109%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.563%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 9.448%
   /features/features.5/conv/conv.2/Conv:           | █████                | 8.716%
   /features/features.7/conv/conv.2/Conv:           | █████                | 8.040%
   /features/features.4/conv/conv.2/Conv:           | ████                 | 7.647%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ████                 | 7.205%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ████                 | 7.021%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.244%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.205%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 6.036%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ███                  | 5.824%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.452%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 5.408%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 5.324%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.187%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ███                  | 4.646%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 4.576%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.493%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.364%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 4.305%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ██                   | 4.160%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ██                   | 4.031%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.667%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.905%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.730%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | █                    | 2.202%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | █                    | 2.170%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | █                    | 2.150%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | █                    | 2.117%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 2.101%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 1.506%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 1.238%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 0.965%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.357%
   /features/features.0/features.0.0/Conv:          |                      | 0.025%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:22<00:00, 10.61s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 1.770%
   /features/features.0/features.0.0/Conv:          | ██████████           | 0.844%
   /features/features.16/conv/conv.2/Conv:          | ██                   | 0.180%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.176%
   /features/features.1/conv/conv.1/Conv:           | ██                   | 0.153%
   /features/features.15/conv/conv.2/Conv:          | █                    | 0.109%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.101%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.092%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.081%
   /classifier/classifier.1/Gemm:                   | █                    | 0.062%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | █                    | 0.061%
   /features/features.12/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.048%
   /features/features.17/conv/conv.2/Conv:          |                      | 0.043%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.041%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.038%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.035%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.033%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.032%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  |                      | 0.030%
   /features/features.3/conv/conv.2/Conv:           |                      | 0.029%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.028%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.026%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.026%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.025%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.022%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.020%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.019%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.018%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.014%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.014%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.012%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.012%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.011%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.009%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.009%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.008%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.007%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.006%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.006%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.004%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.004%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.003%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.002%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.18/features.18.0/Conv:        |                      | 0.001%

   * Prec@1 69.800 Prec@5 88.400

**量化误差分析**

注意到对 8bit 量化应用层间均衡有助于降低量化损失，量化后的top1准确率为69.80%，和float模型的准确率(71.878%)更加接近，比混合精度量化的量化精度更高。

.. note::
   
   如果想进一步降低量化误差，可以尝试使用 QAT (Auantization Aware Training)。具体方法请参考 `PPQ QAT example <https://github.com/OpenPPL/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py>`__。


模型部署
------------

:project:`参考示例 <examples/mobilenetv2_cls>`

图像分类基类
^^^^^^^^^^^^^^^^^

- :project_file:`dl_cls_base.hpp <esp-dl/vision/classification/dl_cls_base.hpp>`
- :project_file:`dl_cls_base.cpp <esp-dl/vision/classification/dl_cls_base.cpp>`

前处理
^^^^^^^^^

``ImagePreprocessor`` 类中封装了常用的图像前处理流程，包括 ``color conversion``, ``crop``, ``resize``, ``normalization``, ``quantize``。

- :project_file:`dl_image_preprocessor.hpp <esp-dl/vision/image/dl_image_preprocessor.hpp>`
- :project_file:`dl_image_preprocessor.cpp <esp-dl/vision/image/dl_image_preprocessor.cpp>`

后处理
^^^^^^^^^

- :project_file:`dl_cls_postprocessor.hpp <esp-dl/vision/classification/dl_cls_postprocessor.hpp>`
- :project_file:`dl_cls_postprocessor.cpp <esp-dl/vision/classification/dl_cls_postprocessor.cpp>`
- :project_file:`imagenet_cls_postprocessor.hpp <esp-dl/vision/classification/imagenet_cls_postprocessor.hpp>`
- :project_file:`imagenet_cls_postprocessor.cpp <esp-dl/vision/classification/imagenet_cls_postprocessor.cpp>`
