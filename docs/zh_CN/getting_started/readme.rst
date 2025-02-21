****************
入门指南
****************

:link_to_translation:`en:[English]`

硬件要求
--------------------

.. list::

   - 一块 ESP32-S3 或 ESP32-P4 开发板。推荐使用：ESP32-S3-EYE 或 ESP32-P4-Function-EV-Board
   - 一台 PC（Linux 系统）

.. note::
   部分开发板目前采用 Type C 接口。请确保使用正确的线缆连接开发板！

软件要求
---------------------

.. _requirements_esp_idf:

ESP-IDF
^^^^^^^^^^^^^^^

ESP-DL 基于 ESP-IDF 运行。有关如何获取 ESP-IDF 的详细说明，请参阅 `ESP-IDF 编程指南 <https://idf.espressif.com>`_。

.. note::
   请使用 `ESP-IDF <https://github.com/espressif/esp-idf>`_ 的 ``release/v5.3`` 或更高版本。

.. _requirements_esp_ppq:

ESP-PPQ
^^^^^^^^^^^^^^^

ESP-PPQ 是基于 ppq 的量化工具。ESP-PPQ 在 `PPQ <https://github.com/OpenPPL/ppq>`__ 的基础上添加了乐鑫定制的 quantizer 和 exporter，方便用户根据不同的芯片选择和 ESP-DL 匹配的量化规则，并导出为 ESP-DL 可以直接加载的标准模型文件。ESP-PPQ 兼容 PPQ 所有的 API 和量化脚本。更多细节请参考 `PPQ 文档和视频 <https://github.com/OpenPPL/ppq>`__。如果您想量化自己的模型，请使用以下命令安装 esp-ppq：

.. code-block:: bash

   pip uninstall ppq
   pip install git+https://github.com/espressif/esp-ppq.git

快速开始
--------------

ESP-DL 提供了一些开箱即用的 :project:`示例 <examples>`

示例编译 & 烧录
^^^^^^^^^^^^^^^^^^
::

   idf.py set-target [Soc]
   idf.py flash monitor

使用具体的芯片替换 ``[Soc]``，目前支持 ``esp32s3`` 和 ``esp32p4``。

示例配置
^^^^^^^^^^^^
::

   idf.py menuconfig

一些示例包含可配置的选项，可以在使用 ``idf.py set-target`` 指定芯片之后使用 ``idf.py menuconfig`` 进行配置。

清空示例
^^^^^^^^^^^^

如果遇到了一些问题，可以尝试删除 ``build/``, ``sdkconfig``, ``dependencies.lock``, ``managed_components/`` 之后重试。

模型量化
------------------

首先，请参考 `ESP-DL 算子支持状态 <../../../operator_support_state.md>`_，确保您的模型中的算子已经得到支持。

ESP-DL 必须使用专有格式 ``.espdl`` 进行模型部署，深度学习模型需要进行量化和格式转换之后才能使用。ESP-PPQ 提供了 ``espdl_quantize_onnx`` 和 ``espdl_quantize_torch`` 两种接口以支持 ONNX 模型和 PyTorch 模型导出为 ``.espdl`` 模型。其他深度学习框架，如 TensorfFlow, PaddlePaddle 等都需要先将模型转换为 ONNX 。因此请确保您的模型可以转换为 ONNX 模型。更多详细信息，请参阅：

- :doc:`如何量化模型 </tutorials/how_to_quantize_model>`
- :ref:`如何量化 MobileNetV2 <how_to_quantize_mobilenetv2>`
- :ref:`如何量化 YOLO11n <how_to_quantize_yolo11n>`

模型部署
----------------

ESP-DL 提供了一系列 API 来快速加载和运行模型。更多详细信息，请参阅：

- :doc:`如何加载和测试模型 </tutorials/how_to_load_test_profile_model>`
- :doc:`如何进行模型推理 </tutorials/how_to_run_model>`