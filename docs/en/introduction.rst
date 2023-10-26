Introduction
============

:link_to_translation:`zh_CN:[中文]`

ESP-DL is a library for high-performance deep learning resources dedicated to `ESP32 <https://www.espressif.com/en/products/socs/esp32>`__, `ESP32-S2 <https://www.espressif.com/en/products/socs/esp32-s2>`__, `ESP32-S3 <https://www.espressif.com/en/products/socs/esp32-s3>`__ and `ESP32-C3 <https://www.espressif.com/en/products/socs/esp32-c3>`__.

Overview
--------

ESP-DL provides APIs for **Neural Network (NN) Inference**, **Image Processing**, **Math Operations** and some **Deep Learning Models**. With ESP-DL, you can use Espressif’s SoCs for AI applications easily and fast.

As ESP-DL does not need any peripherals, it can be used as a component of some projects. For example, you can use it as a component of `ESP-WHO <https://github.com/espressif/esp-who>`__, which contains several project-level examples of image application. The figure below shows what ESP-DL consists of and how ESP-DL is implemented as a component in a project.

.. figure:: ../_static/architecture_en.drawio.svg
    :align: center
    :scale: 90%
    :alt: Architecture Overview

    Architecture Overview

Get Started with ESP-DL
-----------------------

For setup instructions to get started with ESP-DL, please read :doc:`Getting Started with ESP-DL <../../get-started>`.

Please use the `ESP-IDF <https://github.com/espressif/esp-idf>`__ on version 5.0 or above.

Try Models in the Model Zoo
---------------------------

ESP-DL provides some model APIs in the :project:`model_zoo <include/model_zoo>`, such as Human Face Detection, Human Face Recognition, Cat Face Detection, etc. You can use these models in the table below out of box.

.. list-table::
    :header-rows: 1
    :widths: 40 60
    :align: center

    * - Name
      - API Example
    * - Human Face Detection
      - :example:`human_face_detect`
    * - Human Face Recognition
      - :example:`face_recognition`
    * - Cat Face Detection
      - :example:`cat_face_detect`

Deploy a Model
-----------------

To deploy a model, please proceed to :doc:`../../tutorials/index`, where the instructions with three runnable examples will quickly help you design your model.

When you read the instructions, the following materials might be helpful:

-  DL API

   -  :doc:`Variables and Constants <../../glossary>`: information about

      -  variable: tensors
      -  constants: filters, biases, and activations

   -  :doc:`Customizing Layers <../../tutorials/customizing-a-layer-step-by-step>`: instructions on how to customize a layer.

   -  :project:`API Documentation <include>`: guides to provided API about Layer, Neural Network (NN), Math and tools.

         For API documentation, please refer to annotations in header files for the moment.

-  Platform Conversion

   -  TVM(Recommended)： Use AI compiler TVM to deploy AI model，More information about TVM please refer to `TVM <https://tvm.apache.org/docs/>`__
   
   -  Quantization Toolkit: a tool for quantizing floating-point models and evaluating quantized models on ESP SoCs

      -  Toolkit: see :doc:`Quantization Toolkit Overview <../../tools/quantization-toolkit/quantization-toolkit-overview>`
      -  Toolkit API: see :doc:`Quantization Toolkit APIs <../../tools/quantization-toolkit/quantization-toolkit-api>`

   -  Convert Tool: the tool and configuration file for floating-point quantization on ``coefficient.npy``

      -  ``config.json``: see :doc:`Specification of config.json <../../tools/convert-tool/specification-of-config-json>`
      -  ``convert.py``: see :doc:`Usage of convert.py <../../tools/convert-tool/usage-of-convert-tool>`

         ``convert.py`` requires Python 3.7 or versions higher.

-  Software and Hardware Boost

   -  :doc:`Quantization Specification <../../tools/quantization-toolkit/quantization-specification>`: rules of floating-point quantization

Feedback
--------

For feature requests or bug reports, please submit an `issue <https://github.com/espressif/esp-dl/issues>`__. We will prioritize the most anticipated features.
