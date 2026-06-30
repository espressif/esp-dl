如何部署 YOLO11n-pose
==========================

:link_to_translation:`en:[English]`

在本教程中，我们介绍如何使用 ESP-PPQ 对预训练的 YOLO11n-pose 模型进行量化，并使用 ESP-DL 部署量化后的 YOLO11n-pose 模型。

.. contents::
  :local:
  :depth: 2

准备工作
--------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_yolo11n-pose:

模型量化
--------

预训练模型
^^^^^^^^^^^^

你可以从 `Ultralytics release <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt>`__ 下载预训练的 yolo11n-pose 模型。

目前ESP-PPQ支持 ONNX、PyTorch、TensorFlow 模型。在量化过程中，PyTorch 和 TensorFlow 会先转化为 ONNX 模型，因此将与训练的 yolo11n-pose 转化成ONNX模型。

具体来说，参考脚本： :project_file:`export_onnx.py <models/coco_pose/models/export_onnx.py>` 将预训练的 yolo11n-pose 模型转换为 ONNX 模型。

在该脚本中，我们重载了 Pose 类的 forward 方法，具有以下优势：

- 更快的推理速度。 与原始的 yolo11n-pose 模型相比, 将推理过程中 Pose 里与解码边界框相关的操作移至后处理中完成, 从而显著减少了推理延迟。一方面，``Conv``， ``Transpose``， ``Slice``， ``Split`` 和 ``Concat`` 操作在推理过程中运行是非常耗时的。另一方面，在后处理阶段，模型推理的输出首先进行置信度筛选，然后再解码边界框，这大大减少了计算量，从而加快了整体推理速度。

- 更低的量化误差。 ESP-PPQ中的 ``Concat`` 和 ``Add`` 操作采用了联合量化。为了减少量化误差，由于 box 和 score 的范围差异较大，它们通过不同的分支输出，而不是拼接在一起。类似地，由于 ``Add`` 和 ``Sub`` 的输入的范围差异较大，相关计算被移到了后处理中进行，避免被量化。


校准数据集
^^^^^^^^^^^^

校准数据集需要和模型输入格式一致，同时尽可能覆盖模型输入的所有可能情况，以便更好地量化模型。本示例中，我们使用的校准集为 `calib_yolo11n-pose <https://dl.espressif.com/public/calib_yolo11n-pose.zip>`__ 。

8bit 后量化
^^^^^^^^^^^^^^^^^^^

下面的量化设置通过AutoQuant搜索得到。 要使用AutoQuant，请更新esp-ppq为最新版本并参考 `教程 <https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/auto_quantization/how_to_use_AutoQuant.html#>`__。

**ESP32-P4 量化设置**

.. code-block:: python

   quant_setting = QuantizationSettingFactory.espdl_setting() 
   quant_setting.quantize_activation_setting.calib_algorithm = 'percentile'

   quant_setting.bias_correct = True
   quant_setting.bias_correct_setting.interested_layers = []
   quant_setting.bias_correct_setting.block_size = 2
   quant_setting.bias_correct_setting.steps = 32
   
**ESP32-P4 量化误差**

.. code-block::

   Analysing Graphwise Quantization Error(Phrase 1):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.03it/s]
   Analysing Graphwise Quantization Error(Phrase 2):: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.00s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 9.927%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | ███████████████████  | 9.484%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ██████████████████   | 9.145%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | █████████████████    | 8.567%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | █████████████████    | 8.372%
   /model.20/conv/Conv:                         | █████████████████    | 8.366%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ████████████████     | 7.941%
   /model.19/m.0/cv2/conv/Conv:                 | ████████████████     | 7.861%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | ████████████████     | 7.778%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ████████████████     | 7.773%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ████████████████     | 7.769%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ████████████████     | 7.749%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ████████████████     | 7.704%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ███████████████      | 7.641%
   /model.22/m.0/cv3/conv/Conv:                 | ███████████████      | 7.541%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ███████████████      | 7.432%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ███████████████      | 7.315%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 7.086%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | ██████████████       | 7.036%
   /model.22/cv1/conv/Conv:                     | █████████████        | 6.485%
   /model.19/cv2/conv/Conv:                     | █████████████        | 6.333%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | █████████████        | 6.296%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | █████████████        | 6.255%
   /model.22/cv2/conv/Conv:                     | ████████████         | 6.196%
   /model.17/conv/Conv:                         | ████████████         | 6.110%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 5.841%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | ████████████         | 5.763%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ████████████         | 5.740%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ███████████          | 5.657%
   /model.19/cv1/conv/Conv:                     | ███████████          | 5.583%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ███████████          | 5.552%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 5.254%
   /model.6/m.0/cv2/conv/Conv:                  | ███████████          | 5.245%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████           | 5.172%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 5.166%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████████           | 5.136%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ██████████           | 5.061%
   /model.8/m.0/cv2/conv/Conv:                  | ██████████           | 4.985%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ██████████           | 4.962%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 4.609%
   /model.23/cv4.2/cv4.2.2/Conv:                | █████████            | 4.572%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████████            | 4.417%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 4.411%
   /model.6/cv1/conv/Conv:                      | █████████            | 4.264%
   /model.23/cv4.1/cv4.1.2/Conv:                | █████████            | 4.264%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ████████             | 4.161%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████████             | 4.063%
   /model.3/conv/Conv:                          | ████████             | 3.764%
   /model.16/cv2/conv/Conv:                     | ███████              | 3.694%
   /model.8/cv1/conv/Conv:                      | ███████              | 3.689%
   /model.13/cv2/conv/Conv:                     | ███████              | 3.543%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 3.376%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 3.363%
   /model.10/cv1/conv/Conv:                     | ███████              | 3.357%
   /model.8/cv2/conv/Conv:                      | ███████              | 3.279%
   /model.6/m.0/cv3/conv/Conv:                  | ██████               | 3.254%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 3.219%
   /model.13/cv1/conv/Conv:                     | ██████               | 3.180%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██████               | 3.142%
   /model.13/m.0/cv1/conv/Conv:                 | ██████               | 3.129%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ██████               | 3.074%
   /model.16/m.0/cv1/conv/Conv:                 | ██████               | 3.061%
   /model.2/m.0/cv2/conv/Conv:                  | ██████               | 3.024%
   /model.4/cv1/conv/Conv:                      | ██████               | 2.990%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██████               | 2.844%
   /model.16/cv1/conv/Conv:                     | ██████               | 2.821%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ██████               | 2.807%
   /model.4/cv2/conv/Conv:                      | ██████               | 2.781%
   /model.4/m.0/cv1/conv/Conv:                  | █████                | 2.742%
   /model.10/cv2/conv/Conv:                     | █████                | 2.627%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █████                | 2.613%
   /model.2/cv2/conv/Conv:                      | █████                | 2.611%
   /model.6/cv2/conv/Conv:                      | █████                | 2.593%
   /model.8/m.0/cv1/conv/Conv:                  | █████                | 2.553%
   /model.10/m/m.0/attn/MatMul_1:               | █████                | 2.547%
   /model.7/conv/Conv:                          | █████                | 2.447%
   /model.5/conv/Conv:                          | █████                | 2.433%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 2.363%
   /model.23/cv2.1/cv2.1.2/Conv:                | █████                | 2.344%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | █████                | 2.305%
   /model.6/m.0/cv1/conv/Conv:                  | ████                 | 2.250%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 2.247%
   /model.2/cv1/conv/Conv:                      | ████                 | 2.080%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 2.070%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 1.977%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 1.927%
   /model.9/cv1/conv/Conv:                      | ████                 | 1.926%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 1.859%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 1.694%
   /model.9/cv2/conv/Conv:                      | ███                  | 1.672%
   /model.23/cv3.2/cv3.2.2/Conv:                | ███                  | 1.499%
   /model.4/m.0/cv2/conv/Conv:                  | ███                  | 1.491%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ███                  | 1.452%
   /model.2/m.0/cv1/conv/Conv:                  | ██                   | 1.093%
   /model.1/conv/Conv:                          | ██                   | 0.834%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 0.568%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.128%
   /model.0/conv/Conv:                          |                      | 0.046%
   Analysing Layerwise quantization error:: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [04:07<00:00,  2.53s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.0/conv/Conv:                          | ████████████████████ | 0.323%
   /model.2/cv1/conv/Conv:                      | ██████████           | 0.155%
   /model.1/conv/Conv:                          | ████████             | 0.128%
   /model.2/cv2/conv/Conv:                      | ██████               | 0.104%
   /model.8/cv1/conv/Conv:                      | ████                 | 0.070%
   /model.9/cv2/conv/Conv:                      | ███                  | 0.049%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | █                    | 0.016%
   /model.3/conv/Conv:                          | █                    | 0.014%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.014%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.014%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | █                    | 0.012%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | █                    | 0.011%
   /model.5/conv/Conv:                          | █                    | 0.010%
   /model.9/cv1/conv/Conv:                      | █                    | 0.010%
   /model.6/cv1/conv/Conv:                      | █                    | 0.010%
   /model.4/cv2/conv/Conv:                      | █                    | 0.009%
   /model.4/cv1/conv/Conv:                      | █                    | 0.009%
   /model.16/m.0/cv2/conv/Conv:                 | █                    | 0.009%
   /model.19/m.0/cv2/conv/Conv:                 | █                    | 0.009%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.008%
   /model.10/cv1/conv/Conv:                     |                      | 0.008%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           |                      | 0.008%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.008%
   /model.13/cv1/conv/Conv:                     |                      | 0.007%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.007%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.007%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.007%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.007%
   /model.22/cv2/conv/Conv:                     |                      | 0.007%
   /model.6/cv2/conv/Conv:                      |                      | 0.006%
   /model.23/cv4.2/cv4.2.2/Conv:                |                      | 0.006%
   /model.16/cv2/conv/Conv:                     |                      | 0.006%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           |                      | 0.006%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.006%
   /model.8/cv2/conv/Conv:                      |                      | 0.006%
   /model.16/cv1/conv/Conv:                     |                      | 0.006%
   /model.13/cv2/conv/Conv:                     |                      | 0.006%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.005%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.005%
   /model.7/conv/Conv:                          |                      | 0.005%
   /model.10/cv2/conv/Conv:                     |                      | 0.005%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.005%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.005%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.19/cv2/conv/Conv:                     |                      | 0.004%
   /model.4/m.0/cv2/conv/Conv:                  |                      | 0.004%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.004%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.004%
   /model.19/cv1/conv/Conv:                     |                      | 0.004%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.004%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.004%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.004%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.004%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.004%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.003%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.003%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.003%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.003%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.4/m.0/cv1/conv/Conv:                  |                      | 0.002%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.002%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.002%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.22/cv1/conv/Conv:                     |                      | 0.002%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.002%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.002%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.002%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.002%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.002%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.002%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.002%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.002%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.002%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.002%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.001%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.001%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.001%
   /model.17/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.20/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.000%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%

**ESP32-P4 量化结果**

在相同输入下，量化后的模型在 COCO 上的 Pose mAP50:95 为 43.9%，低于浮点模型（50.0%），存在一定的精度损失。


.. _quantization_aware_pose_label:

量化感知训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了进一步提高量化模型的精度，可以采用量化感知训练。本示例基于8-bit量化方式进行量化感知训练。

**量化设置**

- :project_file:`yolo11n-pose_qat.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/yolo11n_pose_qat.py>`
- :project_file:`trainer.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/trainer.py>`

**ESP32-P4 量化误差**

.. code-block::
   
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 27.739%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ███████████████████  | 26.872%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ███████████████████  | 26.229%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ██████████████████   | 25.300%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████████████████   | 24.625%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █████████████████    | 23.751%
   /model.20/conv/Conv:                         | █████████████████    | 23.320%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████████    | 22.901%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | ████████████████     | 22.516%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████     | 22.035%
   /model.19/m.0/cv2/conv/Conv:                 | ████████████████     | 21.569%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ███████████████      | 21.199%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ███████████████      | 20.785%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ███████████████      | 20.597%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ███████████████      | 20.329%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ███████████████      | 20.179%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 19.983%
   /model.22/m.0/cv3/conv/Conv:                 | ██████████████       | 19.919%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████████       | 19.424%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██████████████       | 18.893%
   /model.19/cv2/conv/Conv:                     | █████████████        | 18.055%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████████        | 17.915%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | █████████████        | 17.796%
   /model.22/cv1/conv/Conv:                     | █████████████        | 17.777%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | █████████████        | 17.573%
   /model.19/cv1/conv/Conv:                     | ████████████         | 17.116%
   /model.17/conv/Conv:                         | ████████████         | 16.869%
   /model.22/cv2/conv/Conv:                     | ████████████         | 16.750%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ████████████         | 16.540%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ████████████         | 16.491%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 16.421%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████████         | 16.205%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ████████████         | 16.116%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 15.400%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ███████████          | 15.251%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ███████████          | 14.851%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ███████████          | 14.659%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 14.289%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████████            | 13.038%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 12.941%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 12.791%
   /model.23/cv4.2/cv4.2.2/Conv:                | █████████            | 12.508%
   /model.23/cv4.1/cv4.1.2/Conv:                | █████████            | 12.226%
   /model.13/cv1/conv/Conv:                     | ████████             | 11.821%
   /model.13/cv2/conv/Conv:                     | ████████             | 11.612%
   /model.13/m.0/cv1/conv/Conv:                 | ████████             | 11.515%
   /model.10/m/m.0/attn/MatMul_1:               | ████████             | 11.303%
   /model.16/cv2/conv/Conv:                     | ████████             | 11.028%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ████████             | 10.951%
   /model.10/cv1/conv/Conv:                     | ████████             | 10.755%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████████             | 10.684%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 10.164%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ███████              | 9.968%
   /model.16/m.0/cv1/conv/Conv:                 | ███████              | 9.656%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 9.566%
   /model.8/m.0/cv2/conv/Conv:                  | ███████              | 9.521%
   /model.10/cv2/conv/Conv:                     | ██████               | 8.068%
   /model.16/cv1/conv/Conv:                     | ██████               | 7.989%
   /model.23/cv2.1/cv2.1.2/Conv:                | ██████               | 7.969%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 7.725%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █████                | 7.570%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 7.339%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | █████                | 7.283%
   /model.8/cv2/conv/Conv:                      | █████                | 7.092%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 6.654%
   /model.8/cv1/conv/Conv:                      | █████                | 6.492%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | █████                | 6.451%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 5.990%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 5.902%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 5.898%
   /model.6/m.0/cv2/conv/Conv:                  | ████                 | 5.881%
   /model.6/m.0/cv3/conv/Conv:                  | ████                 | 5.402%
   /model.8/m.0/cv1/conv/Conv:                  | ████                 | 5.210%
   /model.23/cv3.2/cv3.2.2/Conv:                | ████                 | 5.126%
   /model.6/cv1/conv/Conv:                      | ████                 | 4.983%
   /model.9/cv2/conv/Conv:                      | ███                  | 4.616%
   /model.9/cv1/conv/Conv:                      | ███                  | 3.934%
   /model.7/conv/Conv:                          | ███                  | 3.906%
   /model.3/conv/Conv:                          | ███                  | 3.654%
   /model.6/cv2/conv/Conv:                      | ██                   | 3.429%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.319%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.220%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.191%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.157%
   /model.4/cv1/conv/Conv:                      | ██                   | 2.893%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.792%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 2.761%
   /model.5/conv/Conv:                          | ██                   | 2.629%
   /model.4/cv2/conv/Conv:                      | ██                   | 2.298%
   /model.2/cv1/conv/Conv:                      | █                    | 2.107%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 2.095%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 2.069%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 1.744%
   /model.1/conv/Conv:                          | █                    | 1.631%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.583%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.126%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.535%
   /model.0/conv/Conv:                          |                      | 0.067%
   Analysing Layerwise quantization error:: 100%|██████████| 98/98 [10:49<00:00,  6.63s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.9/cv2/conv/Conv:                      | ████████████████████ | 2.976%
   /model.2/cv2/conv/Conv:                      | ███████████          | 1.610%
   /model.3/conv/Conv:                          | ██████               | 0.854%
   /model.2/cv1/conv/Conv:                      | ████                 | 0.543%
   /model.1/conv/Conv:                          | ███                  | 0.487%
   /model.8/cv1/conv/Conv:                      | ███                  | 0.414%
   /model.4/cv2/conv/Conv:                      | ███                  | 0.397%
   /model.0/conv/Conv:                          | ██                   | 0.364%
   /model.6/m.0/cv3/conv/Conv:                  | ██                   | 0.230%
   /model.5/conv/Conv:                          | █                    | 0.181%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.144%
   /model.13/cv2/conv/Conv:                     | █                    | 0.140%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.138%
   /model.4/cv1/conv/Conv:                      | █                    | 0.129%
   /model.16/cv2/conv/Conv:                     | █                    | 0.122%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | █                    | 0.120%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.107%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | █                    | 0.096%
   /model.19/cv2/conv/Conv:                     | █                    | 0.078%
   /model.23/cv2.2/cv2.2.2/Conv:                | █                    | 0.076%
   /model.4/m.0/cv2/conv/Conv:                  |                      | 0.071%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.071%
   /model.6/cv2/conv/Conv:                      |                      | 0.067%
   /model.6/cv1/conv/Conv:                      |                      | 0.066%
   /model.17/conv/Conv:                         |                      | 0.060%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.057%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.056%
   /model.16/cv1/conv/Conv:                     |                      | 0.051%
   /model.10/cv1/conv/Conv:                     |                      | 0.050%
   /model.23/cv4.2/cv4.2.2/Conv:                |                      | 0.046%
   /model.22/cv2/conv/Conv:                     |                      | 0.044%
   /model.7/conv/Conv:                          |                      | 0.043%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.043%
   /model.10/cv2/conv/Conv:                     |                      | 0.037%
   /model.19/cv1/conv/Conv:                     |                      | 0.037%
   /model.8/cv2/conv/Conv:                      |                      | 0.036%
   /model.13/cv1/conv/Conv:                     |                      | 0.036%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.033%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.031%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.027%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.026%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.025%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.025%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.024%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.023%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.023%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.021%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.021%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.020%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.020%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.019%
   /model.9/cv1/conv/Conv:                      |                      | 0.018%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.018%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.018%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.016%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.016%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           |                      | 0.016%
   /model.16/m.0/cv2/conv/Conv:                 |                      | 0.015%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.013%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.013%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.013%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.012%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.011%
   /model.20/conv/Conv:                         |                      | 0.011%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.011%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.011%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.010%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.009%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.009%
   /model.22/cv1/conv/Conv:                     |                      | 0.009%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.008%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.008%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.007%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.007%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.007%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.006%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.006%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.005%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.004%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.003%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.003%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.002%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.002%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.002%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.000%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%

**ESP32-P4 量化结果**

在对8-bit量化应用量化感知训练后，在相同输入下，量化后的模型在 COCO 上的 Pose mAP50:95 提升至44.2%；同时输出层的累计误差大幅减少。相比8-bit后量化方式，量化感知训练后的8-bit量化模型可以在相同的推理速度下达到最高的量化精度。

.. note::
   
   本文档提到的mAP计算结果均是基于ultralytics version 8.4.50得到的。

模型部署
-----------

:project:`example <examples/yolo11_pose>`

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
- :project_file:`dl_pose_yolo11_postprocessor.hpp <esp-dl/vision/detect/dl_pose_yolo11_postprocessor.hpp>`
- :project_file:`dl_pose_yolo11_postprocessor.cpp <esp-dl/vision/detect/dl_pose_yolo11_postprocessor.cpp>`

