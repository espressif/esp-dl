Get Started
===========

:link_to_translation:`zh_CN:[中文]`

This document describes how to set up the environment for ESP-DL. You can use any ESP development board by `Espressif <https://www.espressif.com/en/products/devkits>`__ or other vendors.

Get ESP-IDF
-----------

ESP-DL runs based on ESP-IDF. For detailed instructions on how to get ESP-IDF, please see `ESP-IDF Programming Guide <https://idf.espressif.com/>`__.

Get ESP-DL and Run Example
--------------------------

1. Download ESP-DL using the following command:

   .. code:: shell

      git clone https://github.com/espressif/esp-dl.git

2. Open Terminal and go to the :project:`tutorial/convert_tool_example` directory:

   .. code:: shell

      cd ~/esp-dl/tutorial/convert_tool_example

   or to example projects in the :project:`examples` directory.

3. Set the SoC target by running:

   .. code:: shell

      idf.py set-target [SoC]

   Replace [SoC] with your SoC target, such as esp32, esp32s2, esp32s3 and esp32c3.

      Note that ESP32-C3 is aimed only at applications which do not require PSRAM.

4. Flash the firmware and print the result:

   .. code:: shell

      idf.py flash monitor

   If you go to the :project:`tutorial/convert_tool_example` directory in Step 2, and

   -  your SoC target is ESP32, then

      .. code:: shell

         MNIST::forward: 37294 μs
         Prediction Result: 9

   -  your SoC target is ESP32-S3, then

      .. code:: shell

         MNIST::forward: 6103 μs
         Prediction Result: 9

Use ESP-DL as Component
-----------------------

ESP-DL is a library that contains various deep-learning APIs. We recommend using ESP-DL as a component in a project.

For example, ESP-DL can be a submodule of the `ESP-WHO <https://github.com/espressif/esp-who>`__ repository if added to `esp-who/components/ <https://github.com/espressif/esp-who/tree/master/components>`__ directory.
