How to quantize model
==============================

:link_to_translation:`zh_CN:[中文]`

ESP-DL must use a proprietary format ``.espdl`` for model deployment. This is a quantized model format that supports 8bit and 16bit. In this tutorial, we will take :project:`quantize_sin_model <examples/tutorial/how_to_quantize_model/quantize_sin_model>` as an example to show how to use ESP-PPQ to quantize and export a ``.espdl`` model. The quantization method is Post Training Quantization (PTQ).

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

- ``**.espdl``: ESPDL model binary file, which can be directly used for chip reasoning.
- ``**.info``: ESPDL model text file, used to debug and determine whether the ``.espdl`` model is exported correctly. Contains model structure, quantized model weights, test input/output and other information.
- ``**.json``: Quantization information file, used to save and load quantization information.

.. note::

   1. The ``.espdl`` models of different platforms cannot be mixed, otherwise the inference results will be inaccurate.

      - The ``ESP32`` uses ``ROUND_HALF_UP`` as its rounding strategy.

         - When quantizing **ESP32** platform models using **ESP-PPQ**, set the target to ``c``. Because ESP-DL implements its operators in C.
         - When deploying **ESP32** platform models using **ESP-DL**, set the project compilation target to ``esp32``.

      - The ROUND strategy used by ``ESP32S3`` is ``ROUND_HALF_UP``.
      - The ROUND strategy used by ``ESP32P4`` is ``ROUND_HALF_EVEN``.

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

If you want to further improve the performance of the quantized model, please try the the following advanced quantization methods:

Post Training Quantization (PTQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`Mixed precision quantization <mixed_precision_quantization_label>`
- :ref:`Layerwise equalization quantization <layerwise_equalization_quantization_label>`
- :ref:`Horizontal Layer Split Quantization <horizontal_layer_split_label>`

Quantization Aware Training (QAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- :ref:`YOLO11n Quantization-Aware Training <quantization_aware_label>`
- :ref:`YOLO11n-pose Quantization-Aware Training <quantization_aware_pose_label>`