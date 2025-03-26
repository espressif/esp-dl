How to load & test & profile model
===========================================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will show you how to load, test, profile an espdl model. :project:`example <examples/tutorial/how_to_load_test_profile_model>`

.. contents::
  :local:
  :depth: 2

Preparation
------------------

1. :ref:`Install ESP_IDF <requirements_esp_idf>`
2. :doc:`how_to_quantize_model </tutorials/how_to_quantize_model>`

Load model from ``rodata``
----------------------------------

1. **Add model file in** ``CMakeLists.txt``

   If you want to put the ``.espdl`` model file into the ``.rodata`` section of the FLASH chip, you need to add the following code in ``CMakeLists.txt``. The first few lines should be placed before ``idf_component_register()`` and the last line should be placed after ``idf_component_register()``.

   .. code-block:: cmake

      idf_build_get_property(component_targets __COMPONENT_TARGETS)
      if ("___idf_espressif__esp-dl" IN_LIST component_targets)
      idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
      elseif("___idf_esp-dl" IN_LIST component_targets)
      idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
      endif()
      set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
      include(${cmake_dir}/utilities.cmake)
      set(embed_files your_model_path/model_name.espdl) idf_component_register(...)

      target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)

2. **Load the model in the program**

   .. code-block:: cpp

      // "_binary_model_espdl_start" is composed of three parts: the prefix "binary", the filename "model_espdl", and the suffix "_start".
      extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

      dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
      
      // Keep parameter in FLASH, saves PSRAM/internal RAM, lower performance.
      // dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);

.. note::

   1. When using `Load model from rodata`_, since the ``.rodata`` section belongs to the app partition, the model file will be flashed every time the code is modified. If the model file is large, you may need to adjust the size of the app partition. Using `Load model from partition`_ or `Load model from sdcard`_ can avoid repeatedly flashing the model, which helps to reduce the flashing time.
   2. When using `Load model from rodata`_ or `Load model from partition`_, turning off the param_copy option in the Model constructor can avoid copying the model weights in PSRAM or internal RAM to FLASH. This can reduce the use of PSRAM or internal RAM. However, since the frequency of PSRAM or internal RAM is higher than FLASH, the inference performance of the model will decrease.


Load model from ``partition``
-------------------------------------

1. **Add model information in** ``partition.csv``

About ``partition.csv``, please refer to the `partition table documentation <https://docs.espressif.com/projects/esp-idf/zh_CN/latest/esp32/api-guides/partition-tables.html>`_.

::

   # Name, Type, SubType, Offset, Size, Flags
   factory, app, factory, 0x010000, 4000K,
   model, data, spiffs, , 4000K,

The ``Name`` field of the model can be any meaningful name, but cannot exceed 16 bytes, including a null byte (the content after that will be truncated). The ``SubType`` field must be spiffs. The ``Offset`` can be left blank after other partitions and will be automatically calculated. ``Size`` must be larger than the size of the model file.

2. **Add model flashing information in** ``CMakeLists.txt``

.. code-block:: cmake

   idf_component_register(...)
   set(image_file your_model_path)
   esptool_py_flash_to_partition(flash "model" "${image_file}")

The second parameter in ``esptool_py_flash_to_partition`` must be consistent with the ``Name`` field in ``partition.csv``.

3. **Load the model in the program**

.. code-block:: cpp

   dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

   // Keep parameter in flash, saves PSRAM/internal RAM, lower performance.
   // dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY,
   // nullptr, false);

The first parameter of the constructor must be consistent with the ``Name`` field in ``partition.csv``.

.. note::

   Use ``idf.py app-flash`` instead of ``idf.py flash`` to flash only the app partition without flashing the model partition, this can reduce the flashing time.

Load model from ``sdcard``
---------------------------------

1. **Check if sdcard is in the correct format**

   Back up the data first, then try to mount it on the board. If the sdcard is not in the correct format, it will be automatically formatted to the correct format.

- If using `BSP(Board Support Package) <https://github.com/espressif/esp-bsp/tree/master/bsp>`__

  Enable ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` option in menuconfig.

  .. code-block:: cpp

     ESP_ERROR_CHECK(bsp_sdcard_mount());

- If not using `BSP(Board Support Package) <https://github.com/espressif/esp-bsp/tree/master/bsp>`__

  Set format_if_mount_failed in esp_vfs_fat_sdmmc_mount_config_t structure to true.

  .. code-block:: cpp
  
     esp_vfs_fat_sdmmc_mount_config_t mount_config = {
           .format_if_mount_failed = true,
           .max_files = 5,
           .allocation_unit_size = 16 * 1024
     };
     // Mount sdcard.

2. **Copy model to sdcard**

   Copy .espdl model to sdcard.

3. **Load the model in the program**

- If using `BSP(Board Support Package) <https://github.com/espressif/esp-bsp/tree/master/bsp>`__

  .. code-block:: cpp
  
     ESP_ERROR_CHECK(bsp_sdcard_mount());
     const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
     Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

- If not using `BSP(Board Support Package) <https://github.com/espressif/esp-bsp/tree/master/bsp>`__


  .. code-block:: cpp
  
     // Mount sdcard.
     const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
     Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

.. note::

   When using `load model from sdcard`_, the model loading process will take longer because the model data needs to be copied from sdcard to PSRAM or internal RAM. This method is useful if your FLASH is tight.


Test whether on-board model inference is correct
---------------------------------------------------------

In order to test whether on-board model inference is correct, the ``.espdl`` model needs to :ref:`add test input/output <add_test_input_output>` when exporting. When actually deploying, you can export a version without test input and output to reduce the size of the model file.

.. code-block:: cpp

   ESP_ERROR_CHECK(model->test());

Profile model memory usage
---------------------------------

.. code-block:: cpp

   model->profile_memory();

+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Name               | Explanation                                                                                                                                                                                                                   |
+====================+===============================================================================================================================================================================================================================+
| ``fbs_model``      | Flatbuffers model, contains a sub-item ``parameter``. In addition to the model parameters, the flatbuffers model also contains test input/output, test input and output, model parameter/variable shape, model structure etc. |
|     ``parameter``  |                                                                                                                                                                                                                               |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``parameter_copy`` | Copied Model parameter, when the flatbuffers model locates in FLASH, parameters are copied to PSRAM or internal RAM by default to improve inference performance.                                                              |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``variable``       | Memory allocated by the memory manager. Model input/output and intermediate calculation results will use this space.                                                                                                          |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``others``         | Space required for class member variables, extra part for alignment during ``heap_caps_aligned_alloc`` / ``heap_caps_aligned_calloc`` (very small).                                                                           |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Profile model inference latency
----------------------------------------

.. code-block:: cpp

   model->profile_module();

By default, the model modules are printed in ONNX topological sort. If you want to sort by the latency of each module, you can set the input parameter of ``profile_module`` to ``True``.

.. code-block:: cpp

   model->profile_module(true);