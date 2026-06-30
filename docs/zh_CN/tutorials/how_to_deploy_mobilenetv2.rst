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

   由于 **per-channel** 量化在细节保留上通常优于 **per-tensor**，因此相同模型在 ESP32-P4 上的量化精度往往高于 ESP32-S3。

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

8bit 后量化
^^^^^^^^^^^^^^^^^^

下面的量化设置通过AutoQuant搜索得到。 要使用AutoQuant，请更新esp-ppq为最新版本并参考 `教程 <https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/auto_quantization/how_to_use_AutoQuant.html#>`__。

**ESP32-P4 量化设置**

.. code-block:: python

   # default: Replace ReLU6 with ReLU.

   quant_setting = QuantizationSettingFactory.espdl_setting() 
   quant_setting.quantize_activation_setting.calib_algorithm = 'kl'

   quant_setting.equalization = True
   quant_setting.equalization_setting.iterations = 4
   quant_setting.equalization_setting.value_threshold = 0.1
   quant_setting.equalization_setting.opt_level = 2
   quant_setting.equalization_setting.interested_layers = None

   quant_setting.tqt_optimization = True
   tqt_setting = quant_setting.tqt_optimization_setting
   tqt_setting.lr = 5e-5
   tqt_setting.steps = 800
   tqt_setting.block_size = 4
   tqt_setting.is_scale_trainable = True
   tqt_setting.gamma = 0.0
   tqt_setting.int_lambda = 0.0
   tqt_setting.collecting_device = 'cuda'

**ESP32-P4 量化误差**

.. code-block::

   Analysing Graphwise Quantization Error(Phrase 1):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.45it/s]
   Analysing Graphwise Quantization Error(Phrase 2):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:07<00:00,  1.13it/s]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████████ | 16.879%
   /features/features.16/conv/conv.2/Conv:          | ███████████████████  | 15.993%
   /features/features.15/conv/conv.2/Conv:          | ██████████████       | 11.915%
   /features/features.1/conv/conv.1/Conv:           | █████████████        | 10.603%
   /features/features.17/conv/conv.2/Conv:          | ████████████         | 10.027%
   /features/features.14/conv/conv.2/Conv:          | ████████████         | 9.947%
   /features/features.18/features.18.0/Conv:        | ███████████          | 9.698%
   /features/features.6/conv/conv.2/Conv:           | ██████████           | 8.669%
   /features/features.13/conv/conv.2/Conv:          | ██████████           | 8.486%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 7.475%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | █████████            | 7.409%
   /features/features.3/conv/conv.2/Conv:           | ████████             | 7.102%
   /features/features.5/conv/conv.2/Conv:           | ████████             | 6.808%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████             | 6.629%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 6.346%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | ███████              | 6.103%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ███████              | 5.871%
   /features/features.11/conv/conv.2/Conv:          | ███████              | 5.859%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 5.553%
   /features/features.4/conv/conv.2/Conv:           | █████                | 4.576%
   /features/features.7/conv/conv.2/Conv:           | █████                | 4.275%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | █████                | 3.975%
   /features/features.2/conv/conv.2/Conv:           | ████                 | 3.867%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ████                 | 3.843%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 3.688%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ████                 | 3.620%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ████                 | 3.545%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ████                 | 3.508%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ████                 | 3.423%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 3.422%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 3.398%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 3.311%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 3.184%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ███                  | 2.923%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 2.919%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ███                  | 2.899%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ███                  | 2.801%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ███                  | 2.687%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 2.534%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ███                  | 2.532%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 2.404%
   /classifier/classifier.1/Gemm:                   | ███                  | 2.323%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ███                  | 2.314%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 2.218%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.154%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.094%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 1.591%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | █                    | 1.313%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | █                    | 1.147%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | █                    | 1.126%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 0.995%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 0.810%
   /features/features.0/features.0.0/Conv:          |                      | 0.109%
   Analysing Layerwise quantization error:: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [05:01<00:00,  5.69s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 27.681%
   /features/features.1/conv/conv.1/Conv:           | ███████████████████  | 26.101%
   /features/features.0/features.0.0/Conv:          | █                    | 1.908%
   /features/features.14/conv/conv.2/Conv:          | █                    | 1.741%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 1.665%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 1.123%
   /features/features.16/conv/conv.2/Conv:          | █                    | 0.902%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.872%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.806%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.735%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.529%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.510%
   /features/features.15/conv/conv.2/Conv:          |                      | 0.409%
   /features/features.4/conv/conv.2/Conv:           |                      | 0.352%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.340%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.339%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.333%
   /classifier/classifier.1/Gemm:                   |                      | 0.295%
   /features/features.17/conv/conv.2/Conv:          |                      | 0.244%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.231%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.214%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.211%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.206%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.194%
   /features/features.13/conv/conv.2/Conv:          |                      | 0.179%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.160%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.137%
   /features/features.18/features.18.0/Conv:        |                      | 0.134%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.131%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.122%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.106%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.103%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.098%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.090%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.083%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.080%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.072%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.070%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.069%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.067%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.052%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.049%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.044%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.036%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.033%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.029%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.029%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.024%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.024%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.023%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.019%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.019%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.016%
   
   * Prec@1 72.025 Prec@5 89.425

**ESP32-P4 量化结果**

实验结果表明：在 ESP32P4 上采用 PTQ 量化时，量化模型 Prec@1 为 72.025%，与浮点模型（71.878%）接近。

**ESP32-S3 量化设置**

.. code-block:: python

   # default: Replace ReLU6 with ReLU.

   quant_setting = QuantizationSettingFactory.espdl_setting() 
   quant_setting.quantize_activation_setting.calib_algorithm = 'kl'

   quant_setting.equalization = True
   quant_setting.equalization_setting.iterations = 4
   quant_setting.equalization_setting.value_threshold = 0.1
   quant_setting.equalization_setting.opt_level = 2
   quant_setting.equalization_setting.interested_layers = None

   quant_setting.tqt_optimization = True
   tqt_setting = quant_setting.tqt_optimization_setting
   tqt_setting.lr = 1e-5
   tqt_setting.steps = 1000
   tqt_setting.block_size = 4
   tqt_setting.is_scale_trainable = True
   tqt_setting.gamma = 0.0
   tqt_setting.int_lambda = 0.0
   tqt_setting.collecting_device = 'cuda'

**ESP32-S3 量化误差**

.. code-block::
   
   Analysing Graphwise Quantization Error(Phrase 1):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.34it/s]
   Analysing Graphwise Quantization Error(Phrase 2):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:07<00:00,  1.01it/s]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████████ | 34.420%
   /features/features.16/conv/conv.2/Conv:          | ██████████████       | 24.168%
   /features/features.15/conv/conv.2/Conv:          | ██████████           | 16.545%
   /features/features.14/conv/conv.2/Conv:          | ████████             | 13.160%
   /features/features.1/conv/conv.1/Conv:           | ████████             | 12.979%
   /features/features.17/conv/conv.2/Conv:          | ██████               | 11.054%
   /features/features.18/features.18.0/Conv:        | ██████               | 10.546%
   /features/features.13/conv/conv.2/Conv:          | ██████               | 10.353%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ██████               | 9.501%
   /features/features.6/conv/conv.2/Conv:           | █████                | 9.038%
   /features/features.12/conv/conv.2/Conv:          | █████                | 8.405%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | █████                | 8.007%
   /features/features.3/conv/conv.2/Conv:           | ████                 | 7.629%
   /features/features.10/conv/conv.2/Conv:          | ████                 | 7.459%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ████                 | 7.418%
   /features/features.5/conv/conv.2/Conv:           | ████                 | 6.852%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | ████                 | 6.680%
   /features/features.11/conv/conv.2/Conv:          | ████                 | 6.381%
   /features/features.9/conv/conv.2/Conv:           | ███                  | 5.971%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ███                  | 5.716%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ███                  | 5.159%
   /features/features.4/conv/conv.2/Conv:           | ███                  | 5.090%
   /features/features.7/conv/conv.2/Conv:           | ███                  | 4.734%
   /features/features.2/conv/conv.2/Conv:           | ██                   | 4.244%
   /features/features.8/conv/conv.2/Conv:           | ██                   | 4.103%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 4.070%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ██                   | 3.780%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ██                   | 3.755%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.752%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ██                   | 3.726%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.653%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.541%
   /classifier/classifier.1/Gemm:                   | ██                   | 3.391%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ██                   | 3.327%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.197%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ██                   | 3.168%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.083%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.063%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.819%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ██                   | 2.757%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.656%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | █                    | 2.573%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | █                    | 2.570%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | █                    | 2.513%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | █                    | 2.172%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | █                    | 1.688%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | █                    | 1.399%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | █                    | 1.272%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | █                    | 1.250%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 1.083%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.813%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.763%
   /features/features.0/features.0.0/Conv:          |                      | 0.049%
   Analysing Layerwise quantization error:: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [04:31<00:00,  5.11s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 10.635%
   /features/features.1/conv/conv.1/Conv:           | █████████████████    | 9.070%
   /features/features.14/conv/conv.2/Conv:          | ███                  | 1.356%
   /features/features.0/features.0.0/Conv:          | ██                   | 1.057%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ██                   | 0.905%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.649%
   /features/features.17/conv/conv.2/Conv:          | █                    | 0.631%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.605%
   /features/features.16/conv/conv.2/Conv:          | █                    | 0.572%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.529%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.505%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | █                    | 0.285%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.284%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.277%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.244%
   /features/features.15/conv/conv.2/Conv:          |                      | 0.242%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.235%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.147%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.133%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.126%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.105%
   /classifier/classifier.1/Gemm:                   |                      | 0.104%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.102%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.100%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.099%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.091%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.090%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.079%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.079%
   /features/features.13/conv/conv.2/Conv:          |                      | 0.072%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.068%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.064%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.062%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.061%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.060%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.056%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.051%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.047%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.040%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.040%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.039%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.038%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.034%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.033%
   /features/features.18/features.18.0/Conv:        |                      | 0.029%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.025%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.023%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.022%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.021%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.017%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.011%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.007%

   * Prec@1 70.543 Prec@5 89.525

**ESP32-S3 量化结果**

实验结果表明：在 ESP32-S3 上采用 PTQ 量化时，量化模型 Prec@1 为 70.543%，略高于在ESP32-P4上的量化精度。

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
