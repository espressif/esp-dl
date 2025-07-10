如何部署 YOLO11n
====================

:link_to_translation:`en:[English]`

在本教程中，我们介绍如何使用 ESP-PPQ 对预训练的 YOLO11n 模型进行量化，并使用 ESP-DL 部署量化后的 YOLO11n 模型。

.. contents::
  :local:
  :depth: 2

准备工作
--------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_yolo11n:

模型量化
--------

预训练模型
^^^^^^^^^^^^

你可以从 `Ultralytics release <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>`__ 下载预训练的 yolo11n 模型。

目前ESP-PPQ支持 ONNX、PyTorch、TensorFlow 模型。在量化过程中，PyTorch 和 TensorFlow 会先转化为 ONNX 模型，因此将与训练的 yolo11n 转化成ONNX模型。

具体来说，参考脚本： :project_file:`export_onnx.py <models/coco_detect/models/export_onnx.py>` 将预训练的 yolo11n 模型转换为 ONNX 模型。

在该脚本中，我们重载了 Detect 类的 forward 方法，具有以下优势：

- 更快的推理速度。 与原始的 yolo11n 模型相比, 将推理过程中 Detect 里与解码边界框相关的操作移至后处理中完成, 从而显著减少了推理延迟。一方面，``Conv``， ``Transpose``， ``Slice``， ``Split`` 和 ``Concat`` 操作在推理过程中运行是非常耗时的。另一方面，在后处理阶段，模型推理的输出首先进行置信度筛选，然后再解码边界框，这大大减少了计算量，从而加快了整体推理速度。

- 更低的量化误差。 ESP-PPQ中的 ``Concat`` 和 ``Add`` 操作采用了联合量化。为了减少量化误差，由于 box 和 score 的范围差异较大，它们通过不同的分支输出，而不是拼接在一起。类似地，由于 ``Add`` 和 ``Sub`` 的输入的范围差异较大，相关计算被移到了后处理中进行，避免被量化。


校准数据集
^^^^^^^^^^^^

校准数据集需要和模型输入格式一致，同时尽可能覆盖模型输入的所有可能情况，以便更好地量化模型。本示例中，我们使用的校准集为 `calib_yolo11n <https://dl.espressif.com/public/calib_yolo11n.zip>`__ 。

8bit 默认配置量化
^^^^^^^^^^^^^^^^^^^

**量化设置**

.. code-block:: python

   target="esp32p4"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting

**量化结果**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 36.008%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ████████████████     | 28.705%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████        | 22.865%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 21.718%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | ████████████         | 21.624%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ████████████         | 21.392%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ████████████         | 21.224%
   /model.22/m.0/cv2/conv/Conv:                 | ███████████          | 19.763%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 19.436%
   /model.22/m.0/cv3/conv/Conv:                 | ███████████          | 19.378%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ██████████           | 18.913%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ██████████           | 18.645%
   /model.22/cv2/conv/Conv:                     | ██████████           | 18.628%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ██████████           | 17.980%
   /model.8/m.0/cv2/conv/Conv:                  | █████████            | 16.247%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █████████            | 15.602%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ████████             | 14.666%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ████████             | 14.556%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████             | 14.302%
   /model.22/cv1/conv/Conv:                     | ████████             | 13.921%
   /model.10/m/m.0/attn/MatMul_1:               | ████████             | 13.905%
   /model.10/cv1/conv/Conv:                     | ███████              | 13.494%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 11.800%
   /model.19/m.0/cv2/conv/Conv:                 | ██████               | 11.515%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████               | 11.286%
   /model.20/conv/Conv:                         | ██████               | 10.930%
   /model.13/m.0/cv2/conv/Conv:                 | ██████               | 10.882%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████               | 10.692%
   /model.23/cv2.2/cv2.2.2/Conv:                | ██████               | 10.113%
   /model.10/cv2/conv/Conv:                     | █████                | 9.720%
   /model.8/cv2/conv/Conv:                      | █████                | 9.598%
   /model.8/m.0/cv1/conv/Conv:                  | █████                | 9.470%
   /model.19/cv2/conv/Conv:                     | █████                | 9.314%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████                | 9.068%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 9.065%
   /model.8/cv1/conv/Conv:                      | █████                | 9.051%
   /model.8/m.0/cv3/conv/Conv:                  | █████                | 9.044%
   /model.6/m.0/cv2/conv/Conv:                  | █████                | 8.811%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████                | 8.781%
   /model.13/cv2/conv/Conv:                     | █████                | 8.687%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | █████                | 8.503%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 8.470%
   /model.19/cv1/conv/Conv:                     | ████                 | 8.199%
   /model.10/m/m.0/attn/MatMul:                 | ████                 | 8.117%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 7.964%
   /model.13/cv1/conv/Conv:                     | ████                 | 7.734%
   /model.19/m.0/cv1/conv/Conv:                 | ████                 | 7.661%
   /model.22/m.0/cv1/conv/Conv:                 | ████                 | 7.490%
   /model.13/m.0/cv1/conv/Conv:                 | ████                 | 7.162%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ████                 | 7.145%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 7.041%
   /model.23/cv2.1/cv2.1.2/Conv:                | ████                 | 6.917%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 6.778%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ████                 | 6.641%
   /model.17/conv/Conv:                         | ███                  | 6.125%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 5.937%
   /model.6/cv2/conv/Conv:                      | ███                  | 5.838%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 5.832%
   /model.6/cv1/conv/Conv:                      | ███                  | 5.688%
   /model.7/conv/Conv:                          | ███                  | 5.612%
   /model.9/cv2/conv/Conv:                      | ███                  | 5.367%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ███                  | 5.158%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 5.143%
   /model.16/m.0/cv1/conv/Conv:                 | ███                  | 5.137%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ███                  | 5.087%
   /model.16/cv2/conv/Conv:                     | ███                  | 4.989%
   /model.2/cv2/conv/Conv:                      | ██                   | 4.547%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 4.441%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██                   | 4.343%
   /model.3/conv/Conv:                          | ██                   | 4.304%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 4.006%
   /model.5/conv/Conv:                          | ██                   | 3.932%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 3.837%
   /model.4/cv1/conv/Conv:                      | ██                   | 3.687%
   /model.2/cv1/conv/Conv:                      | ██                   | 3.565%
   /model.4/cv2/conv/Conv:                      | ██                   | 3.559%
   /model.16/cv1/conv/Conv:                     | ██                   | 3.107%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 2.882%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | █                    | 2.758%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 2.564%
   /model.9/cv1/conv/Conv:                      | █                    | 2.017%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.785%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 1.327%
   /model.1/conv/Conv:                          | █                    | 1.313%
   /model.23/cv3.2/cv3.2.2/Conv:                | █                    | 1.155%
   /model.2/m.0/cv1/conv/Conv:                  |                      | 0.727%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.493%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.282%
   /model.0/conv/Conv:                          |                      | 0.159%
   Analysing Layerwise quantization error:: 100%|██████████| 89/89 [03:39<00:00,  2.46s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.1/conv/Conv:                          | ████████████████████ | 0.384%
   /model.22/cv1/conv/Conv:                     | █████████████        | 0.247%
   /model.4/cv2/conv/Conv:                      | ████████████         | 0.233%
   /model.2/cv2/conv/Conv:                      | ██████████           | 0.201%
   /model.0/conv/Conv:                          | ██████████           | 0.192%
   /model.9/cv2/conv/Conv:                      | ████████             | 0.156%
   /model.10/cv1/conv/Conv:                     | ███████              | 0.132%
   /model.3/conv/Conv:                          | ██████               | 0.108%
   /model.4/cv1/conv/Conv:                      | ████                 | 0.074%
   /model.16/cv1/conv/Conv:                     | ███                  | 0.066%
   /model.2/cv1/conv/Conv:                      | ███                  | 0.060%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███                  | 0.052%
   /model.2/m.0/cv1/conv/Conv:                  | ██                   | 0.044%
   /model.6/cv1/conv/Conv:                      | ██                   | 0.033%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██                   | 0.029%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.028%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █                    | 0.023%
   /model.16/cv2/conv/Conv:                     | █                    | 0.021%
   /model.16/m.0/cv2/conv/Conv:                 | █                    | 0.020%
   /model.19/m.0/cv1/conv/Conv:                 | █                    | 0.020%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.018%
   /model.19/cv2/conv/Conv:                     | █                    | 0.017%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.016%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.016%
   /model.19/cv1/conv/Conv:                     | █                    | 0.015%
   /model.13/cv2/conv/Conv:                     | █                    | 0.015%
   /model.8/cv1/conv/Conv:                      | █                    | 0.013%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | █                    | 0.013%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | █                    | 0.012%
   /model.13/cv1/conv/Conv:                     | █                    | 0.012%
   /model.10/cv2/conv/Conv:                     | █                    | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 | █                    | 0.011%
   /model.6/cv2/conv/Conv:                      | █                    | 0.011%
   /model.13/m.0/cv2/conv/Conv:                 | █                    | 0.010%
   /model.5/conv/Conv:                          |                      | 0.010%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.009%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.009%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.008%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.008%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.008%
   /model.9/cv1/conv/Conv:                      |                      | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.007%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.007%
   /model.17/conv/Conv:                         |                      | 0.007%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.007%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.007%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.006%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.006%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.005%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.005%
   /model.7/conv/Conv:                          |                      | 0.005%
   /model.8/cv2/conv/Conv:                      |                      | 0.004%
   /model.22/cv2/conv/Conv:                     |                      | 0.004%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.004%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.004%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.003%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.003%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.003%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.003%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.002%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.002%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.001%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.20/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%

**量化误差分析**

在相同输入下，量化后的模型在 COCO val2017 上的 mAP50:95 仅为 30.7%，低于浮点模型，存在一定的精度损失：

- **累计误差 (Graphwise Error)**

   模型的输出层是 /model.23/cv3.2/cv3.2.2/Conv，/model.23/cv2.2/cv2.2.2/Conv，/model.23/cv3.1/cv3.1.2/Conv，/model.23/cv2.1/cv2.1.2/Conv，/model.23/cv3.0/cv3.0.2/Conv 和 /model.23/cv2.0/cv2.0.2/Conv，累计误差分别为 1.155%，10.113%，0.493%，6.917%，0.282% 和 6.778% 。通常，如果输出层的累计误差小于 10%，则量化模型的精度损失较小。

- **逐层误差 (Layerwise error)**

   观察逐层误差发现，所有层的误差均低于 1%，这表明所有层的量化误差都很小。

我们注意到，虽然所有层的逐层误差都很小，但是一些层的累计误差却较大。这可能与 yolo11n 模型中复杂的CSP结构有关，模型中 ``Concat`` 或 ``Add`` 层的输入可能具有不同的分布或尺度。我们可以选择使用int16对某些层进行量化，并采用算子分裂过程优化量化效果。有关详细信息，请参阅混合精度+算子分裂过程量化测试。

.. _horizontal_layer_split_label:

混合精度+算子分裂量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**量化设置**

.. code-block:: python

   from ppq.api import get_target_platform
   target="esp32p4"
   num_of_bits=8
   batch_size=32

   # Quantize the following layers with 16-bits
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.dispatching_table.append("/model.2/cv2/conv/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/model.3/conv/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/model.4/cv2/conv/Conv", get_target_platform(TARGET, 16))

   # Horizontal Layer Split Pass
   quant_setting.weight_split = True
   quant_setting.weight_split_setting.method = 'balance'
   quant_setting.weight_split_setting.value_threshold = 1.5
   quant_setting.weight_split_setting.interested_layers = ['/model.0/conv/Conv', '/model.1/conv/Conv']
    

**量化结果**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 24.835%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ███████████████      | 18.632%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ██████████████       | 17.908%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | ██████████████       | 16.922%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | █████████████        | 16.754%
   /model.22/m.0/cv3/conv/Conv:                 | ████████████         | 15.404%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ████████████         | 15.042%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ████████████         | 14.948%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████████         | 14.702%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | ███████████          | 13.683%
   /model.22/cv2/conv/Conv:                     | ███████████          | 13.654%
   /model.22/m.0/cv2/conv/Conv:                 | ███████████          | 13.514%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ██████████           | 12.885%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | █████████            | 10.865%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ████████             | 9.875%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████             | 9.658%
   /model.22/cv1/conv/Conv:                     | ███████              | 8.917%
   /model.10/m/m.0/attn/MatMul_1:               | ███████              | 8.368%
   /model.23/cv2.2/cv2.2.2/Conv:                | ███████              | 8.156%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████               | 8.056%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ██████               | 7.948%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 7.824%
   /model.13/m.0/cv2/conv/Conv:                 | ██████               | 7.504%
   /model.19/m.0/cv2/conv/Conv:                 | ██████               | 7.290%
   /model.20/conv/Conv:                         | ██████               | 6.986%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██████               | 6.926%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 6.771%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | █████                | 6.756%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████                | 6.465%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████                | 6.274%
   /model.19/cv2/conv/Conv:                     | █████                | 6.116%
   /model.10/cv1/conv/Conv:                     | █████                | 5.868%
   /model.13/cv2/conv/Conv:                     | █████                | 5.815%
   /model.10/cv2/conv/Conv:                     | ████                 | 5.664%
   /model.19/cv1/conv/Conv:                     | ████                 | 5.178%
   /model.8/m.0/cv2/conv/Conv:                  | ████                 | 4.970%
   /model.19/m.0/cv1/conv/Conv:                 | ████                 | 4.919%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ████                 | 4.864%
   /model.22/m.0/cv1/conv/Conv:                 | ████                 | 4.844%
   /model.10/m/m.0/attn/MatMul:                 | ████                 | 4.650%
   /model.13/cv1/conv/Conv:                     | ████                 | 4.564%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███                  | 4.389%
   /model.13/m.0/cv1/conv/Conv:                 | ███                  | 4.243%
   /model.23/cv2.0/cv2.0.2/Conv:                | ███                  | 4.232%
   /model.23/cv2.1/cv2.1.2/Conv:                | ███                  | 4.222%
   /model.6/m.0/cv2/conv/Conv:                  | ███                  | 4.023%
   /model.17/conv/Conv:                         | ███                  | 3.754%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 3.511%
   /model.8/m.0/cv1/conv/Conv:                  | ███                  | 3.277%
   /model.16/m.0/cv1/conv/Conv:                 | ██                   | 3.158%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██                   | 3.155%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██                   | 3.152%
   /model.8/cv2/conv/Conv:                      | ██                   | 3.119%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.106%
   /model.8/m.0/cv3/conv/Conv:                  | ██                   | 3.083%
   /model.6/m.0/cv3/conv/Conv:                  | ██                   | 3.068%
   /model.8/cv1/conv/Conv:                      | ██                   | 3.035%
   /model.16/cv2/conv/Conv:                     | ██                   | 3.002%
   /model.2/cv2/conv/Conv:                      | ██                   | 2.992%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 2.971%
   /model.6/cv1/conv/Conv:                      | ██                   | 2.819%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.809%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██                   | 2.760%
   /model.2/cv1/conv/Conv:                      | ██                   | 2.683%
   /model.6/cv2/conv/Conv:                      | ██                   | 2.630%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.615%
   /model.9/cv2/conv/Conv:                      | ██                   | 2.540%
   /model.3/conv/Conv:                          | ██                   | 2.503%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 2.474%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.273%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 2.246%
   /model.4/cv2/conv/Conv:                      | ██                   | 2.141%
   /model.7/conv/Conv:                          | ██                   | 2.120%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 2.069%
   /model.5/conv/Conv:                          | ██                   | 2.015%
   /model.16/cv1/conv/Conv:                     | █                    | 1.894%
   /model.4/cv1/conv/Conv:                      | █                    | 1.793%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 1.776%
   /model.6/m.0/cv1/conv/Conv:                  | █                    | 1.731%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | █                    | 1.550%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.257%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 0.886%
   /model.1/conv/Conv:                          | █                    | 0.775%
   /model.23/cv3.2/cv3.2.2/Conv:                | █                    | 0.771%
   PPQ_Operation_2:                             |                      | 0.696%
   /model.9/cv1/conv/Conv:                      |                      | 0.695%
   /model.2/m.0/cv1/conv/Conv:                  |                      | 0.534%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.339%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.190%
   PPQ_Operation_0:                             |                      | 0.110%
   /model.0/conv/Conv:                          |                      | 0.099%
   Analysing Layerwise quantization error:: 100%|██████████| 91/91 [04:13<00:00,  2.79s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/cv1/conv/Conv:                     | ████████████████████ | 0.244%
   /model.9/cv2/conv/Conv:                      | █████████████        | 0.156%
   /model.10/cv1/conv/Conv:                     | ███████████          | 0.132%
   /model.1/conv/Conv:                          | ██████               | 0.077%
   /model.4/cv1/conv/Conv:                      | ██████               | 0.074%
   /model.16/cv1/conv/Conv:                     | █████                | 0.066%
   /model.0/conv/Conv:                          | █████                | 0.061%
   /model.2/cv1/conv/Conv:                      | █████                | 0.060%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 0.052%
   PPQ_Operation_0:                             | ████                 | 0.047%
   /model.2/m.0/cv1/conv/Conv:                  | ████                 | 0.045%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██                   | 0.029%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 0.029%
   /model.10/m/m.0/attn/MatMul:                 | ██                   | 0.025%
   /model.6/cv1/conv/Conv:                      | ██                   | 0.025%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | ██                   | 0.023%
   /model.16/cv2/conv/Conv:                     | ██                   | 0.021%
   /model.16/m.0/cv2/conv/Conv:                 | ██                   | 0.020%
   /model.19/m.0/cv1/conv/Conv:                 | ██                   | 0.020%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.018%
   /model.19/cv2/conv/Conv:                     | █                    | 0.017%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.016%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.016%
   /model.19/cv1/conv/Conv:                     | █                    | 0.015%
   /model.13/cv2/conv/Conv:                     | █                    | 0.015%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | █                    | 0.013%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | █                    | 0.012%
   /model.13/cv1/conv/Conv:                     | █                    | 0.012%
   /model.6/cv2/conv/Conv:                      | █                    | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 | █                    | 0.011%
   /model.8/cv1/conv/Conv:                      | █                    | 0.010%
   /model.13/m.0/cv2/conv/Conv:                 | █                    | 0.010%
   /model.5/conv/Conv:                          | █                    | 0.010%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | █                    | 0.009%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █                    | 0.008%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | █                    | 0.008%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | █                    | 0.008%
   /model.19/m.0/cv2/conv/Conv:                 | █                    | 0.008%
   /model.8/cv2/conv/Conv:                      | █                    | 0.008%
   /model.9/cv1/conv/Conv:                      | █                    | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █                    | 0.007%
   /model.16/m.0/cv1/conv/Conv:                 | █                    | 0.007%
   /model.17/conv/Conv:                         | █                    | 0.007%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | █                    | 0.007%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | █                    | 0.007%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.006%
   /model.10/cv2/conv/Conv:                     |                      | 0.006%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.006%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.005%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.005%
   /model.22/cv2/conv/Conv:                     |                      | 0.005%
   /model.7/conv/Conv:                          |                      | 0.004%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.004%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.004%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.003%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.003%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.003%
   PPQ_Operation_2:                             |                      | 0.003%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.003%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.002%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.002%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.002%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.001%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.001%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.001%
   /model.2/cv2/conv/Conv:                      |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.20/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%
   /model.3/conv/Conv:                          |                      | 0.000%
   /model.4/cv2/conv/Conv:                      |                      | 0.000%


**量化误差分析**

在对逐层误差较高的层使用16-bit量化，并采用算子分裂过程后，在相同输入下，量化后的模型在 COCO val2017 上的 mAP50:95 提升至33.4%；同时可以观察到输出层的累计误差明显减少。

模型的输出层/model.23/cv3.2/cv3.2.2/Conv， /model.23/cv2.2/cv2.2.2/Conv， /model.23/cv3.1/cv3.1.2/Conv， /model.23/cv2.1/cv2.1.2/Conv， /model.23/cv3.0/cv3.0.2/Conv和/model.23/cv2.0/cv2.0.2/Conv的累计误差分别为0.771%，8.156%，0.339%，4.222%，0.190%和4.232%。

.. _quantization_aware_label:

量化感知训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了进一步提高量化模型的精度，可以采用量化感知训练。本示例基于8-bit量化方式进行量化感知训练。

**量化设置**

- :project_file:`yolo11n_qat.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n/yolo11n_qat.py>`
- :project_file:`trainer.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n/trainer.py>`

**量化结果**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 29.837%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ████████████████     | 23.397%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██████████           | 15.253%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ██████████           | 14.819%
   /model.10/m/m.0/attn/MatMul_1:               | ██████████           | 14.725%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ██████████           | 14.315%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | █████████            | 14.212%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████            | 14.187%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █████████            | 13.797%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | █████████            | 13.721%
   /model.22/m.0/cv2/conv/Conv:                 | █████████            | 13.540%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████            | 13.408%
   /model.8/m.0/cv2/conv/Conv:                  | █████████            | 12.809%
   /model.22/m.0/cv3/conv/Conv:                 | ████████             | 12.623%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ████████             | 12.472%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████             | 12.177%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████             | 11.719%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ████████             | 11.711%
   /model.10/cv1/conv/Conv:                     | ████████             | 11.589%
   /model.22/cv2/conv/Conv:                     | ████████             | 11.551%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ████████             | 11.505%
   /model.10/m/m.0/attn/MatMul:                 | ████████             | 11.346%
   /model.22/cv1/conv/Conv:                     | ███████              | 10.201%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 9.710%
   /model.13/m.0/cv2/conv/Conv:                 | ██████               | 9.538%
   /model.20/conv/Conv:                         | ██████               | 8.870%
   /model.19/m.0/cv2/conv/Conv:                 | ██████               | 8.713%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 8.157%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | █████                | 8.005%
   /model.8/cv2/conv/Conv:                      | █████                | 7.952%
   /model.8/m.0/cv1/conv/Conv:                  | █████                | 7.697%
   /model.13/cv2/conv/Conv:                     | █████                | 7.557%
   /model.19/cv2/conv/Conv:                     | █████                | 7.443%
   /model.10/cv2/conv/Conv:                     | █████                | 7.403%
   /model.6/m.0/cv2/conv/Conv:                  | █████                | 7.099%
   /model.8/cv1/conv/Conv:                      | █████                | 6.996%
   /model.19/cv1/conv/Conv:                     | █████                | 6.912%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | █████                | 6.908%
   /model.8/m.0/cv3/conv/Conv:                  | ████                 | 6.755%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ████                 | 6.746%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 6.743%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 6.638%
   /model.13/cv1/conv/Conv:                     | ████                 | 6.361%
   /model.2/m.0/cv2/conv/Conv:                  | ████                 | 6.274%
   /model.13/m.0/cv1/conv/Conv:                 | ████                 | 6.261%
   /model.19/m.0/cv1/conv/Conv:                 | ████                 | 6.191%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | ████                 | 6.036%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 5.999%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | ████                 | 5.899%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 5.618%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ████                 | 5.560%
   /model.22/m.0/cv1/conv/Conv:                 | ███                  | 5.336%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 5.316%
   /model.17/conv/Conv:                         | ███                  | 5.113%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 5.103%
   /model.16/m.0/cv1/conv/Conv:                 | ███                  | 5.101%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ███                  | 5.052%
   /model.2/cv2/conv/Conv:                      | ███                  | 5.003%
   /model.6/cv2/conv/Conv:                      | ███                  | 4.968%
   /model.6/cv1/conv/Conv:                      | ███                  | 4.792%
   /model.23/cv2.1/cv2.1.2/Conv:                | ███                  | 4.543%
   /model.7/conv/Conv:                          | ███                  | 4.520%
   /model.3/conv/Conv:                          | ███                  | 4.362%
   /model.16/cv2/conv/Conv:                     | ███                  | 4.028%
   /model.23/cv2.0/cv2.0.2/Conv:                | ███                  | 4.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ███                  | 3.954%
   /model.9/cv2/conv/Conv:                      | ███                  | 3.901%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 3.891%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██                   | 3.791%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██                   | 3.711%
   /model.4/cv1/conv/Conv:                      | ██                   | 3.673%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 3.620%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.513%
   /model.4/cv2/conv/Conv:                      | ██                   | 3.421%
   /model.5/conv/Conv:                          | ██                   | 3.320%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 3.073%
   /model.2/cv1/conv/Conv:                      | ██                   | 3.021%
   /model.16/cv1/conv/Conv:                     | ██                   | 2.764%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.454%
   /model.4/m.0/cv1/conv/Conv:                  | ██                   | 2.408%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.689%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.602%
   /model.9/cv1/conv/Conv:                      | █                    | 1.568%
   /model.1/conv/Conv:                          | █                    | 1.205%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 1.091%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.746%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.480%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.386%
   /model.0/conv/Conv:                          |                      | 0.163%
   Analysing Layerwise quantization error:: 100%|██████████| 89/89 [04:01<00:00,  2.72s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.2/cv2/conv/Conv:                      | ████████████████████ | 0.935%
   /model.9/cv2/conv/Conv:                      | ██████████████████   | 0.826%
   /model.2/m.0/cv1/conv/Conv:                  | ███████████████      | 0.698%
   /model.3/conv/Conv:                          | █████████████        | 0.611%
   /model.4/cv2/conv/Conv:                      | ██████████           | 0.491%
   /model.10/cv2/conv/Conv:                     | █████████            | 0.408%
   /model.23/cv2.2/cv2.2.2/Conv:                | ██████               | 0.283%
   /model.2/cv1/conv/Conv:                      | ██████               | 0.261%
   /model.4/cv1/conv/Conv:                      | █████                | 0.249%
   /model.1/conv/Conv:                          | █████                | 0.217%
   /model.22/cv1/conv/Conv:                     | ████                 | 0.201%
   /model.10/cv1/conv/Conv:                     | ███                  | 0.143%
   /model.5/conv/Conv:                          | ███                  | 0.136%
   /model.16/cv1/conv/Conv:                     | ███                  | 0.128%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ███                  | 0.120%
   /model.0/conv/Conv:                          | ███                  | 0.118%
   /model.16/m.0/cv1/conv/Conv:                 | ██                   | 0.105%
   /model.16/cv2/conv/Conv:                     | ██                   | 0.094%
   /model.16/m.0/cv2/conv/Conv:                 | ██                   | 0.092%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ██                   | 0.089%
   /model.4/m.0/cv1/conv/Conv:                  | ██                   | 0.071%
   /model.22/m.0/cv1/conv/Conv:                 | █                    | 0.067%
   /model.19/cv2/conv/Conv:                     | █                    | 0.063%
   /model.6/cv2/conv/Conv:                      | █                    | 0.061%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.059%
   /model.17/conv/Conv:                         | █                    | 0.054%
   /model.13/cv2/conv/Conv:                     | █                    | 0.053%
   /model.8/m.0/cv3/conv/Conv:                  | █                    | 0.051%
   /model.6/cv1/conv/Conv:                      | █                    | 0.047%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | █                    | 0.042%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █                    | 0.041%
   /model.13/cv1/conv/Conv:                     | █                    | 0.040%
   /model.7/conv/Conv:                          | █                    | 0.038%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.038%
   /model.13/m.0/cv1/conv/Conv:                 | █                    | 0.033%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | █                    | 0.031%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | █                    | 0.028%
   /model.19/m.0/cv2/conv/Conv:                 | █                    | 0.027%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | █                    | 0.026%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.026%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.022%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.021%
   /model.19/cv1/conv/Conv:                     |                      | 0.021%
   /model.9/cv1/conv/Conv:                      |                      | 0.016%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.016%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.015%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.015%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.014%
   /model.8/cv1/conv/Conv:                      |                      | 0.013%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.013%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.012%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.011%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.011%
   /model.8/cv2/conv/Conv:                      |                      | 0.011%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.010%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.010%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.008%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.007%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.007%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.007%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.006%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.005%
   /model.22/cv2/conv/Conv:                     |                      | 0.005%
   /model.20/conv/Conv:                         |                      | 0.005%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.005%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.005%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.003%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.003%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.003%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.003%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.002%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%


**量化误差分析**

在对8-bit量化应用量化感知训练后，在相同输入下，量化后的模型在 COCO val2017 上的 mAP50:95 提升至36.0%；同时输出层的累计误差大幅减少。相比前两种量化方式，量化感知训练后的8-bit量化模型可以在最快的推理速度下达到最高的量化精度。

模型的输出层/model.23/cv3.2/cv3.2.2/Conv， /model.23/cv2.2/cv2.2.2/Conv， /model.23/cv3.1/cv3.1.2/Conv， /model.23/cv2.1/cv2.1.2/Conv， /model.23/cv3.0/cv3.0.2/Conv和/model.23/cv2.0/cv2.0.2/Conv的累计误差分别为0.746%，5.999%，0.480%，4.543%，0.386%和4.001%。

.. note::
   
   如果想要更快的模型推理速度，并且可以接受一定程度的精度损失，可以考虑在量化YOLO11N的时候将输入大小设置为320x320。不同分辨率下的模型推理速度可以在 :project_file:`README.md <models/coco_detect/README.md>` 中找到。


模型部署
-----------

:project:`参考示例 <examples/yolo11_detect>`

目标检测基类
^^^^^^^^^^^^^^^^^

- :project_file:`dl_detect_base.hpp <esp-dl/vision/detect/dl_detect_base.hpp>`
- :project_file:`dl_detect_base.cpp <esp-dl/vision/detect/dl_detect_base.cpp>`

前处理
^^^^^^^^^

``ImagePreprocessor`` 类中封装了常用的图像前处理流程，包括 ``color conversion``, ``crop``, ``resize``, ``normalization``, ``quantize``。

- :project_file:`dl_image_preprocessor.hpp <esp-dl/vision/image/dl_image_preprocessor.hpp>`
- :project_file:`dl_image_preprocessor.cpp <esp-dl/vision/image/dl_image_preprocessor.cpp>`

后处理
^^^^^^^^^

- :project_file:`dl_detect_postprocessor.hpp <esp-dl/vision/detect/dl_detect_postprocessor.hpp>`
- :project_file:`dl_detect_postprocessor.cpp <esp-dl/vision/detect/dl_detect_postprocessor.cpp>`
- :project_file:`dl_detect_yolo11_postprocessor.hpp <esp-dl/vision/detect/dl_detect_yolo11_postprocessor.hpp>`
- :project_file:`dl_detect_yolo11_postprocessor.cpp <esp-dl/vision/detect/dl_detect_yolo11_postprocessor.cpp>`

