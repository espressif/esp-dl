如何量化模型
===============

:link_to_translation:`en:[English]`

ESP-DL 必须使用专有格式 ``.espdl`` 进行模型部署。这是一种量化模型格式，支持 8bit 和 16bit。在本教程中，我们将以 :project:`quantize_sin_model <examples/tutorial/how_to_quantize_model/quantize_sin_model>` 为例，介绍如何使用 ESP-PPQ 量化并导出  ``.espdl`` 模型，量化方法为 Post Training Quantization (PTQ)。

.. contents::
  :local:
  :depth: 2

准备工作
---------

:ref:`安装 ESP_PPQ <requirements_esp_ppq>`

预训练模型
-----------

::

   python sin_model.py

执行 :project_file:`sin_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/sin_model.py>` 。该脚本会训练一个简单的 Pytorch 模型用于拟合 [0, 2pi] 范围内的 sin函数。训练结束会保存相应的 .pth 权重，并导出 ONNX 模型。

.. note::

   ESP-PPQ 提供了 ``espdl_quantize_onnx`` 和 ``espdl_quantize_torch`` 两种接口以支持 ONNX 模型和 PyTorch 模型。
   其他深度学习框架，如 TensorfFlow, PaddlePaddle 等都需要先将模型转换为 ONNX 。

   - TensorFlow 转 ONNX `tf2onnx <https://github.com/onnx/tensorflow-onnx>`__
   - TFLite 转 ONNX `tflite2onnx <https://github.com/zhenhuaw-me/tflite2onnx>`__
   - TFLite 转 TensorFlow `tflite2tensorflow <https://github.com/PINTO0309/tflite2tensorflow>`__
   - PaddlePaddle 转 ONNX `paddle2onnx <https://github.com/PaddlePaddle/Paddle2ONNX>`__

量化并导出  ``.espdl``
------------------------

参考 :project_file:`quantize_torch_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/quantize_torch_model.py>` 和 :project_file:`quantize_onnx_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/quantize_onnx_model.py>`, 了解如何使用 ``espdl_quantize_onnx`` 和 ``espdl_quantize_torch`` 接口量化并导出  ``.espdl`` 模型。

执行脚本后会导出三个文件，分别是：

- ``**.espdl``：ESPDL 模型二进制文件，可以直接用于芯片的推理。
- ``**.info``：ESPDL 模型文本文件，用于调试和确定  ``.espdl`` 模型是否被正确导出。包含了模型结构，量化完的模型权重，测试输入/输出等信息。
- ``**.json``：量化信息文件，用于量化信息的保存和加载。

.. note::

   1. 不同平台的 ``.espdl`` 模型不能混用，推理结果会有误差。

      - ``ESP32`` 使用的 ROUND 策略是 ``ROUND_HALF_UP``。

         - 使用 **ESP-PPQ** 量化 **ESP32** 平台模型时，需将 target 设置为 ``c``，因为在 ESP-DL 中，其算子实现采用 C 语言编写。
         - 使用 **ESP-DL** 部署 **ESP32** 平台模型时，项目编译 target 则设置为 ``esp32``。

      - ``ESP32S3`` 使用的 ROUND 策略是 ``ROUND_HALF_UP``。
      - ``ESP32P4`` 使用的则是 ``ROUND_HALF_EVEN``。

   2. 目前 ESP-DL 使用的量化策略是 对称量化 + POWER OF TWO。

.. _add_test_input_output:

添加测试输入/输出
^^^^^^^^^^^^^^^^^^^^

验证模型在板端的推理结果是否正确，首先需要记录PC端的一组测试输入/输出。 开启 api 中的 ``export_test_values`` 选项，就能将一组测试输入/输出固化在  ``.espdl`` 模型中。``input_shape`` 参数和 ``inputs`` 参数必须指定其中的一个，``input_shape`` 参数使用随机的测试输入，``inputs`` 则可以指定一个特定的测试输入。 ``.info`` 文件中可以查看测试输入/输出的值。搜索 ``test inputs value`` 和 ``test outputs value`` 查看它们。


量化模型推理 & 精度评估
^^^^^^^^^^^^^^^^^^^^^^^^^^

``espdl_quantize_onnx`` 和 ``espdl_quantize_torch`` API 会返回 ``BaseGraph``。使用 ``BaseGraph`` 构建相应的 ``TorchExecutor`` 就可以在 PC 端使用量化模型进行推理了。

.. code-block:: python

   executor = TorchExecutor(graph=quanted_graph, device=device)
   output = executor(input)

量化模型推理得到的输出可以用来计算各种精度指标。由于 ``esp-dl`` 板端推理的结果是能和 ``esp-ppq`` 对齐的，可以直接用该指标评估量化完模型的性能。

.. note::

   1. 当前 esp-dl 仅支持 batch_size 为 1，不支持 多batch 或者 动态batch。
   2. ``.info`` 文件中的测试输入/输出，以及量化完的模型权重都是16字节对齐的，也就是说如果不满16字节，会在后面填充0。


高级量化方法
----------------------------

如果你的模型使用默认的 8bit 量化方法无法达到满意的结果，我们也提供了如下量化方法可以进一步减少量化模型的性能损失：

训练后量化 (PTQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`混合精度量化 <mixed_precision_quantization_label>`
- :ref:`层间均衡量化 <layerwise_equalization_quantization_label>`
- :ref:`算子分裂量化 <horizontal_layer_split_label>`

量化感知训练 (QAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- :ref:`YOLO11n 量化感知训练 <quantization_aware_label>`
- :ref:`YOLO11n-pose 量化感知训练 <quantization_aware_pose_label>`