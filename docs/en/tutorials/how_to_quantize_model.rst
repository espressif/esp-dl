How to quantize model
==============================

:link_to_translation:`zh_CN:[中文]`

ESP-DL must use a proprietary format ``.espdl`` for model deployment. This is a quantized model format that supports 8-bit (w8a8), 16-bit (w16a16), and mixed-precision (w8a16) quantization. These schemes can be freely combined within a model to preserve accuracy while maximizing inference efficiency. In this tutorial, we will take :project:`quantize_sin_model <examples/tutorial/how_to_quantize_model/quantize_sin_model>` as an example to show how to use ESP-PPQ to quantize and export a ``.espdl`` model. The quantization method is Post Training Quantization (PTQ).

.. contents::
  :local:
  :depth: 2

Preparation
-----------------

:ref:`Install ESP_PPQ <requirements_esp_ppq>`

Pre-trained model
-----------------------

::

   python sin_model.py

Run :project_file:`sin_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/sin_model.py>` . This script trains a simple Pytorch model to fit the sin function in the range [0, 2pi]. After training, the corresponding .pth weights will be saved and the ONNX model will be exported.

.. note::

   ESP-PPQ provides two interfaces, ``espdl_quantize_onnx`` and ``espdl_quantize_torch``, to support ONNX models and PyTorch models.
   Other deep learning frameworks, such as TensorfFlow, PaddlePaddle, etc., need to be converted to ONNX first.

   - Convert TensorFlow to ONNX `tf2onnx <https://github.com/onnx/tensorflow-onnx>`__
   - Convert TFLite to ONNX `tflite2onnx <https://github.com/zhenhuaw-me/tflite2onnx>`__
   - Convert TFLite to TensorFlow `tflite2tensorflow <https://github.com/PINTO0309/tflite2tensorflow>`__
   - Convert PaddlePaddle to ONNX `paddle2onnx <https://github.com/PaddlePaddle/Paddle2ONNX>`__

Quantize and export ``.espdl``
--------------------------------------

Reference :project_file:`quantize_torch_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/quantize_torch_model.py>` and :project_file:`quantize_onnx_model.py <examples/tutorial/how_to_quantize_model/quantize_sin_model/quantize_onnx_model.py>`, learn how to use the ``espdl_quantize_onnx`` and ``espdl_quantize_torch`` interfaces to quantize and export the ``.espdl`` model.

After executing the script, three files will be exported:

- ``**.espdl``: ESPDL model binary file, which can be deployed on chip for inference directly, and can be visualized with `Netron <https://netron.app>`__.
- ``**.info``: ESPDL model text file, used to debug and determine whether the ``.espdl`` model is exported correctly. Contains model structure, quantized model weights, test input/output and other information.
- ``**.json``: Quantization information file, used to save and load quantization information.

Quantization schemes (``quant_type``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``quant_type`` argument of ``espdl_quantize_onnx`` / ``espdl_quantize_torch`` to select the quantization scheme:

.. list-table::
   :header-rows: 1
   :widths: 18 16 16 50

   * - ``quant_type``
     - Weights
     - Activations
     - Typical use
   * - ``w8a8`` (default)
     - INT8
     - INT8
     - Highest inference speed; default for most models
   * - ``w16a16``
     - INT16
     - INT16
     - Highest accuracy; higher compute and memory cost
   * - ``w8a16``
     - INT8
     - INT16
     - Mixed precision: 8-bit weights with 16-bit activations. Applies to Conv / Gemm / MatMul with static weights (ESP-DL ≥ v3.3.11)
   * - ``none``
     - FP32
     - FP32
     - Skip quantization (debug / export only)

.. code-block:: python

   quant_ppq_graph = espdl_quantize_torch(
       model=model,
       espdl_export_file="sin_model.espdl",
       calib_dataloader=dataloader,
       calib_steps=32,
       input_shape=[1, 1],
       target="esp32s3",          # "esp32p4", "esp32s31", "esp32s3", or "c"
       quant_type="w8a8",         # "w8a8", "w8a16", "w16a16", or "none"
       collate_fn=collate_fn,
       device="cpu",
       error_report=True,
       export_test_values=True,
       verbose=1,
   )

.. note::

   - Prefer ``quant_type``. It overrides the legacy ``num_of_bits`` argument when both are set. ``num_of_bits=8`` is equivalent to ``quant_type="w8a8"``; ``num_of_bits=16`` is equivalent to ``quant_type="w16a16"``.
   - ``w8a16`` is fully supported from ESP-DL v3.3.11. Not every operator has a dedicated w8a16 kernel: Conv / Gemm / MatMul with static weights use INT8 weights and INT16 activations; depthwise Conv (``group != 1``) and other operators fall back to INT16. See `operator_support_state.md <https://github.com/espressif/esp-dl/blob/master/operator_support_state.md>`__.
   - ``debug=True`` enables high-precision INT16 simulation in PPQ. It only affects ``w16a16`` (legacy alias: ``hi_precision=True``).

.. _mixed_quantization_label:

Mixed quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can mix ``w8a8``, ``w8a16``, and ``w16a16`` in the same model. Set a default scheme with ``quant_type``, then dispatch selected operators to another scheme through ``QuantizationSetting.dispatching_table``:

.. code-block:: python

   from esp_ppq import QuantizationSettingFactory
   from esp_ppq.api import espdl_quantize_onnx, get_target_platform

   TARGET = "esp32p4"
   quant_setting = QuantizationSettingFactory.espdl_setting()
   # Promote accuracy-sensitive layers to mixed precision or full INT16
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

``get_target_platform(target, quant_type="w8a8")`` is the recommended API. The legacy form ``get_target_platform(target, 16)`` (or ``num_of_bits=16``) still works and maps to ``w16a16``.

Configure mixed precision through ``setting.dispatching_table``. The ``dispatching_override`` and ``dispatching_method`` arguments of ``espdl_quantize_*`` are deprecated and will be removed in a later version.

.. note::

   1. The ``.espdl`` models of different platforms cannot be mixed; inference results will be inaccurate.

      - ``ESP32`` uses a ``Per-Tensor`` quantization strategy; the rounding mode is ``ROUND_HALF_UP``.

         - When quantizing **ESP32** platform models with **ESP-PPQ**, set the target to ``c``, because ESP-DL implements those operators in C.
         - When deploying **ESP32** platform models with **ESP-DL**, set the project build target to ``esp32``.

      - ``ESP32S3`` uses a ``Per-Tensor`` quantization strategy; the rounding mode is ``ROUND_HALF_UP``.
      - On ``ESP32P4`` and ``ESP32S31``, ``Conv`` and ``GEMM`` use ``Per-Channel`` quantization; other operators use ``Per-Tensor``; the rounding mode is ``ROUND_HALF_EVEN``.

   2. The quantization strategy currently used by ESP-DL is symmetric quantization + POWER OF TWO.

.. _add_test_input_output:

Add test input/output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To verify whether the inference results of the model on the board are correct, you first need to record a set of test input/output on the PC. By turning on the ``export_test_values`` option in the api, a set of test input/output can be saved in the ``.espdl`` model. One of the ``input_shape`` and ``inputs`` parameters must be specified. The ``input_shape`` parameter uses a random test input, while ``inputs`` can use a specific test input. The values ​​of the test input/output can be viewed in the ``.info`` file. Search for ``test inputs value`` and ``test outputs value`` to view them.

Quantized model inference & accuracy evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``espdl_quantize_onnx`` and ``espdl_quantize_torch`` APIs will return ``BaseGraph``. Use ``BaseGraph`` to build the corresponding ``TorchExecutor`` to use the quantized model for inference on the PC side.

.. code-block:: python

   executor = TorchExecutor(graph=quanted_graph, device=device)
   output = executor(input)

The output obtained by quantized model inference can be used to calculate various accuracy metrics. Since the board-side ``esp-dl`` inference result can be aligned with ``esp-ppq``, these metrics can be used directly to evaluate the accuracy of the quantized model.

.. note::

   1. Currently esp-dl only supports batch_size of 1, and does not support multi-batch or dynamic batch.
   2. The test input/output and the quantized model weights in the ``.info`` file are all 16-byte aligned. If the length is less than 16 bytes, it will be padded with 0.


Advanced Quantization Methods
---------------------------------

If the default w8a8 quantization does not meet your accuracy needs, switch the whole model to ``w8a16`` / ``w16a16``, mix schemes as shown in :ref:`mixed_quantization_label`, or use the following methods to further reduce accuracy loss:

Post Training Quantization (PTQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`Mixed quantization (w8a8 / w8a16 / w16a16) <mixed_quantization_label>`
- :ref:`8bit 后量化 <8bit_ptq_label>`

Quantization Aware Training (QAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`YOLO11n-pose Quantization-Aware Training <quantization_aware_pose_label>`
