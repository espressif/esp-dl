****************
Getting Started
****************

:link_to_translation:`zh_CN:[中文]`

Hardware Requirements
-------------------------

.. list::

   - An ESP32-S3 or ESP32-P4 development board. Recommended: ESP32-S3-EYE or ESP32-P4-Function-EV-Board
   - PC (Linux)

.. note::

   - Some boards currently use Type C connectors. Make sure you use the right cable to connect the board!
   - ESP-DL also supports ESP32, but its operator implementations are written in C, so the execution speed on ESP32 will be significantly slower than on ESP32-S3 or ESP32-P4. If needed, you can manually add compilation configuration files to your project—the function interface calls in ESP-DL remain identical. Note:

      - When quantizing **ESP32** platform models using **ESP-PPQ**, set the target to ``c``.
      - When deploying **ESP32** platform models using **ESP-DL**, set the project compilation target to ``esp32``.

Software Requirements
----------------------------

.. _requirements_esp_idf:

ESP-IDF
^^^^^^^^^^^^^^^^

ESP-DL runs based on ESP-IDF. For detailed instructions on how to get ESP-IDF, see the `ESP-IDF Programming Guide <https://idf.espressif.com>`_.

.. note::

   Please use ``release/v5.3`` or higher version of `ESP-IDF <https://github.com/espressif/esp-idf>`_.

.. _requirements_esp_ppq:

ESP-PPQ
^^^^^^^^^^^^^^^^

ESP-PPQ is a quantization tool based on ppq. ESP-PPQ adds Espressif's customized quantizer and exporter based on `PPQ <https://github.com/OpenPPL/ppq>`__, which makes it convenient for users to select quantization rules that match ESP-DL according to different chip selections, and export them to standard model files that can be directly loaded by ESP-DL. ESP-PPQ is compatible with all PPQ APIs and quantization scripts. For more details, please refer to `PPQ documents and videos <https://github.com/OpenPPL/ppq>`__. If you want to quantize your own model, please install esp-ppq with the following command:

.. code-block:: bash

   pip uninstall ppq
   pip install git+https://github.com/espressif/esp-ppq.git

Quick Start
--------------

ESP-DL provides some out-of-the-box :project:`examples <examples>`

Example Compile & Flash
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   idf.py set-target [Soc]
   idf.py flash monitor -p [PORT]

Replace ``[Soc]`` with the specific chip, currently supports ``esp32s3`` and ``esp32p4``. The example does not yet include the model and compilation configuration files for ``esp32``.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^
::

   idf.py menuconfig

Some examples contain configurable options that can be configured using ``idf.py menuconfig`` after specifying the chip using ``idf.py set-target``.

Trouble shooting
^^^^^^^^^^^^^^^^^^^^^

Check ESP-IDF doc
""""""""""""""""""""""""
See `ESP-IDF DOC <https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html#>`_

Erase FLASH & Clear Example
""""""""""""""""""""""""""""""""""""""
::
  
   idf.py erase-flash -p [PORT]

Delete ``build/``, ``sdkconfig``, ``dependencies.lock``, ``managed_components/`` and try again.

Model Quantization
------------------------

First, please refer to :project_file:`operator_support_state.md` to ensure that the operators in your model are supported.

ESP-DL must use the proprietary format ``.espdl`` for model deployment. Deep learning models need to be quantized and converted to the format before they can be used. ESP-PPQ provides two interfaces, ``espdl_quantize_onnx`` and ``espdl_quantize_torch``, to support ONNX models and PyTorch models to be exported as ``.espdl`` models. Other deep learning frameworks, such as TensorfFlow, PaddlePaddle, etc., need to convert the model to ONNX first. So make sure your model can be converted to ONNX model. For more details, please refer to:

- :doc:`How to quantize model </tutorials/how_to_quantize_model>`
- :ref:`How to quantize MobileNetV2 <how_to_quantize_mobilenetv2>`
- :ref:`How to quantize YOLO11n <how_to_quantize_yolo11n>`
- :ref:`How to quantize YOLO11n-pose <how_to_quantize_yolo11n-pose>`
- :ref:`How to quantize streaming model <how_to_quantize_streaming_model>`

Model deployment
---------------------

ESP-DL provides a series of APIs to quickly load and run models. For more details, see:

- :doc:`How to load & test & profile model </tutorials/how_to_load_test_profile_model>`
- :doc:`How to run model </tutorials/how_to_run_model>`
- :ref:`How to deploy streaming model <how_to_deploy_streaming_model>`