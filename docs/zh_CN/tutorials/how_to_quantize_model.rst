如何量化模型
===============

:link_to_translation:`en:[English]`

ESP-DL 必须使用专有格式 ``.espdl`` 进行模型部署。这是一种量化模型格式，支持 8bit (w8a8)、16bit (w16a16) 以及混合精度 (w8a16) 量化。这三种量化方式可在模型中任意组合使用，在保持模型精度的同时最大化推理效率。在本教程中，我们将以 :project:`quantize_sin_model <examples/tutorial/how_to_quantize_model/quantize_sin_model>` 为例，介绍如何使用 ESP-PPQ 量化并导出  ``.espdl`` 模型，量化方法为 Post Training Quantization (PTQ)。

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

- ``**.espdl``: ESPDL 模型二进制文件，可直接部署于芯片端执行推理，并支持通过 `Netron <https://netron.app>`__ 可视化查看模型结构。
- ``**.info``: ESPDL 模型文本文件，用于调试和确定  ``.espdl`` 模型是否被正确导出。包含了模型结构，量化完的模型权重，测试输入/输出等信息。
- ``**.json``: 量化信息文件，用于量化信息的保存和加载。

量化方案（``quant_type``）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通过 ``espdl_quantize_onnx`` / ``espdl_quantize_torch`` 的 ``quant_type`` 参数选择量化方案：

.. list-table::
   :header-rows: 1
   :widths: 18 16 16 50

   * - ``quant_type``
     - 权重
     - 激活
     - 典型用途
   * - ``w8a8``（默认）
     - INT8
     - INT8
     - 推理速度最快，大多数模型的默认选择
   * - ``w16a16``
     - INT16
     - INT16
     - 精度最高，计算和内存开销更大
   * - ``w8a16``
     - INT8
     - INT16
     - 混合精度：权重 8bit、激活 16bit。适用于带静态权重的 Conv / Gemm / MatMul（ESP-DL ≥ v3.3.11）
   * - ``none``
     - FP32
     - FP32
     - 跳过量化（仅用于调试 / 导出）

.. code-block:: python

   quant_ppq_graph = espdl_quantize_torch(
       model=model,
       espdl_export_file="sin_model.espdl",
       calib_dataloader=dataloader,
       calib_steps=32,
       input_shape=[1, 1],
       target="esp32s3",          # "esp32p4", "esp32s31", "esp32s3" 或 "c"
       quant_type="w8a8",         # "w8a8", "w8a16", "w16a16" 或 "none"
       collate_fn=collate_fn,
       device="cpu",
       error_report=True,
       export_test_values=True,
       verbose=1,
   )

.. note::

   - 请优先使用 ``quant_type``。当同时传入时，它会覆盖旧的 ``num_of_bits`` 参数。``num_of_bits=8`` 等价于 ``quant_type="w8a8"``；``num_of_bits=16`` 等价于 ``quant_type="w16a16"``。
   - ``w8a16`` 从 ESP-DL v3.3.11 开始全面支持。并非所有算子都有独立的 w8a16 kernel：带静态权重的 Conv / Gemm / MatMul 使用 INT8 权重和 INT16 激活；depthwise Conv（``group != 1``）及其它算子会回退到 INT16。详见 `operator_support_state.md <https://github.com/espressif/esp-dl/blob/master/operator_support_state.md>`__。
   - ``debug=True`` 会在 PPQ 中启用高精度 INT16 仿真，仅对 ``w16a16`` 生效（旧参数别名：``hi_precision=True``）。

.. _mixed_quantization_label:

混合量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以在同一模型中混合使用 ``w8a8``、``w8a16`` 和 ``w16a16``。先用 ``quant_type`` 设置默认方案，再通过 ``QuantizationSetting.dispatching_table`` 将指定算子调度到其它方案：

.. code-block:: python

   from esp_ppq import QuantizationSettingFactory
   from esp_ppq.api import espdl_quantize_onnx, get_target_platform

   TARGET = "esp32p4"
   quant_setting = QuantizationSettingFactory.espdl_setting()
   # 将精度敏感层提升为混合精度或全 INT16
   quant_setting.dispatching_table.append(
       "/features/features.1/conv/conv.0/conv.0.0/Conv",
       get_target_platform(TARGET, quant_type="w8a16"),
   )
   quant_setting.dispatching_table.append(
       "/features/features.1/conv/conv.0/conv.0.2/Clip",
       get_target_platform(TARGET, quant_type="w16a16"),
   )

   quant_ppq_graph = espdl_quantize_onnx(
       onnx_import_file="model.onnx",
       espdl_export_file="model.espdl",
       calib_dataloader=dataloader,
       calib_steps=32,
       input_shape=[1, 3, 224, 224],
       target=TARGET,
       quant_type="w8a8",
       setting=quant_setting,
       collate_fn=collate_fn,
       device="cpu",
   )

推荐使用 ``get_target_platform(target, quant_type="w8a8")``。旧写法 ``get_target_platform(target, 16)``（或 ``num_of_bits=16``）仍然可用，并映射为 ``w16a16``。

请通过 ``setting.dispatching_table`` 配置混合精度。``espdl_quantize_*`` 的 ``dispatching_override`` 和 ``dispatching_method`` 参数已废弃，将在后续版本中移除。

.. note::

   1. 不同平台的 ``.espdl`` 模型不能混用，推理结果会有误差。

      - ``ESP32`` 采用 ``Per-Tensor`` 量化策略, ROUND 策略为 ``ROUND_HALF_UP``。

         - 使用 **ESP-PPQ** 量化 **ESP32** 平台模型时，需将 target 设置为 ``c``，因为在 ESP-DL 中，其算子实现采用 C 语言编写。
         - 使用 **ESP-DL** 部署 **ESP32** 平台模型时，项目编译 target 则设置为 ``esp32``。

      - ``ESP32S3`` 采用 ``Per-Tensor`` 量化策略, ROUND 策略为 ``ROUND_HALF_UP``。
      - ``ESP32P4`` 和 ``ESP32S31`` 的 ``Conv, GEMM`` 算子采用 ``Per-Channel`` 量化策略，其它算子采用 ``Per-Tensor`` 量化策略, ROUND 策略为 ``ROUND_HALF_EVEN``。

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

如果你的模型使用默认的 w8a8 量化无法达到满意的结果，可以将整个模型切换为 ``w8a16`` / ``w16a16``，或按 :ref:`mixed_quantization_label` 混合使用多种方案，也可以使用如下量化方法进一步减少量化模型的性能损失：

训练后量化 (PTQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`混合量化 (w8a8 / w8a16 / w16a16) <mixed_quantization_label>`
- :ref:`8bit 后量化 <8bit_ptq_label>`

量化感知训练 (QAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`YOLO11n-pose 量化感知训练 <quantization_aware_pose_label>`
