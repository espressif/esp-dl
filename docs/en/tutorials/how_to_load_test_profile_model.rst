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

This method embeds the model file directly into the application's ``.rodata`` section in FLASH. It's the simplest approach but has the drawback that the model gets re-flashed every time the application code changes.

1. **Add model file in** ``CMakeLists.txt``

   To embed the ``.espdl`` model file into the ``.rodata`` section, add the following code to your ``CMakeLists.txt``. The first few lines should be placed before ``idf_component_register()`` and the last line after ``idf_component_register()``.

   .. code-block:: cmake

      idf_build_get_property(component_targets __COMPONENT_TARGETS)
      if ("___idf_espressif__esp-dl" IN_LIST component_targets)
         idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
      elseif("___idf_esp-dl" IN_LIST component_targets)
         idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
      endif()
      set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
      include(${cmake_dir}/utilities.cmake)
      set(embed_files your_model_path/model_name.espdl)

      idf_component_register(...)

      target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)

2. **Load the model in the program**

   Include the header file:

   .. code-block:: cpp

      #include "dl_model_base.hpp"

   Declare the model symbol and create the model:

   .. code-block:: cpp

      // The symbol name is composed of three parts: prefix "_binary_", filename "model_espdl", and suffix "_start"
      extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

      // Basic usage - loads model with default parameters
      dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

      // Advanced usage with custom parameters:
      // - Keep parameters in FLASH (saves PSRAM/internal RAM, but lower performance)
      // - Limit internal RAM usage to 0 bytes (use PSRAM first)
      // - Use greedy memory manager
      // - No encryption key
      // - param_copy = false (keep parameters in FLASH)
      // dl::Model *model = new dl::Model((const char *)model_espdl,
      //                                  fbs::MODEL_LOCATION_IN_FLASH_RODATA,
      //                                  0,  // max_internal_size
      //                                  dl::MEMORY_MANAGER_GREEDY,
      //                                  nullptr,  // key
      //                                  false);   // param_copy

.. note::

   **Performance and Memory Trade-offs:**

   - **Flashing Time:** When using `Load model from rodata`_, the model file is embedded in the application binary and gets re-flashed every time you modify your code. For large models, this increases flashing time. Consider `Load model from partition`_ or `Load model from sdcard`_ to avoid this.

   - **Memory vs Performance:** The ``param_copy`` parameter controls whether model parameters are copied from FLASH to faster memory (PSRAM/internal RAM). Setting ``param_copy=false`` saves RAM but reduces inference performance since FLASH access is slower. Only disable parameter copying if RAM is extremely tight.

   - **App Partition Size:** Large models embedded in ``.rodata`` may require increasing the app partition size in ``partition.csv``.


Load model from ``partition``
-------------------------------------

This method stores the model in a separate FLASH partition, allowing you to update the model independently of the application code.

1. **Add model information in** ``partition.csv``

   Create or modify your ``partition.csv`` file to include a partition for the model. For details on partition tables, refer to the `ESP-IDF partition table documentation <https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/partition-tables.html>`_.

   .. code-block::

      # Name, Type, SubType, Offset, Size, Flags
      factory, app, factory, 0x010000, 4000K,
      model, data, spiffs, , 4000K,

   - **Name:** Any meaningful name (max 16 characters including null terminator)
   - **Type:** ``data``
   - **SubType:** ``spiffs`` (required for model storage)
   - **Offset:** Leave blank for automatic calculation
   - **Size:** Must be larger than the model file size

2. **Add model flashing information in** ``CMakeLists.txt``

   .. code-block:: cmake

      idf_component_register(...)
      set(image_file your_model_path/model_name.espdl)
      esptool_py_flash_to_partition(flash "model" "${image_file}")

   The second parameter in ``esptool_py_flash_to_partition`` must match the ``Name`` field in ``partition.csv``.

3. **Load the model in the program**

   Include the header file:

   .. code-block:: cpp

      #include "dl_model_base.hpp"

   Create the model instance:

   .. code-block:: cpp

      // Basic usage - loads model with default parameters
      dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

      // Advanced usage - keep parameters in FLASH to save RAM
      // dl::Model *model = new dl::Model("model",
      //                                  fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
      //                                  0,  // max_internal_size
      //                                  dl::MEMORY_MANAGER_GREEDY,
      //                                  nullptr,  // key
      //                                  false);   // param_copy

   The first parameter (partition label) must match the ``Name`` field in ``partition.csv``.

.. note::

   **Flashing Optimization:** Use ``idf.py app-flash`` instead of ``idf.py flash`` to flash only the application partition without re-flashing the model partition. This significantly reduces flashing time during development.

Load model from ``sdcard``
---------------------------------

This method loads the model from an SD card, which is useful when FLASH space is limited or when you need to update models frequently without re-flashing.

1. **Prepare the SD card**

   - **Format:** The SD card should be formatted as FAT32. If not, it will be automatically formatted when mounted (data will be lost).
   - **Backup:** Always backup SD card data before using it with ESP-DL.

2. **Mount the SD card**

   - **Using BSP (Board Support Package):**

     Enable ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` in menuconfig to allow automatic formatting.

     .. code-block:: cpp

        #include "bsp/esp-bsp.h"
        ESP_ERROR_CHECK(bsp_sdcard_mount());

   - **Without BSP:**

     Configure the mount options with ``format_if_mount_failed = true``.

     .. code-block:: cpp

        #include "esp_vfs_fat.h"
        #include "sdmmc_cmd.h"

        esp_vfs_fat_sdmmc_mount_config_t mount_config = {
            .format_if_mount_failed = true,
            .max_files = 5,
            .allocation_unit_size = 16 * 1024
        };
        // Mount SD card (implementation depends on your hardware)

3. **Copy model to SD card**

   Copy your ``.espdl`` model file to the SD card (e.g., to the root directory as ``model.espdl``).

4. **Load the model in the program**

   Include the header file:

   .. code-block:: cpp

      #include "dl_model_base.hpp"

   Create the model instance:

   .. code-block:: cpp

      // Basic usage with BSP
      ESP_ERROR_CHECK(bsp_sdcard_mount());
      dl::Model *model = new dl::Model("/sdcard/model.espdl", fbs::MODEL_LOCATION_IN_SDCARD);

      // Or with custom path
      // dl::Model *model = new dl::Model("/sdcard/models/my_model.espdl", fbs::MODEL_LOCATION_IN_SDCARD);

      // Don't forget to unmount when done
      // ESP_ERROR_CHECK(bsp_sdcard_unmount());

   For non-BSP usage, mount the SD card first, then create the model similarly.

.. note::

   **Performance Considerations:** Loading from SD card is slower than from FLASH because the model data must be copied from the SD card to RAM. However, this method saves FLASH space and allows easy model updates by swapping SD cards.


Test whether on-board model inference is correct
---------------------------------------------------------

The ``test()`` method verifies that the model produces correct inference results by comparing them against ground truth values embedded in the model file.

**Prerequisites:**

- The ``.espdl`` model must be exported with **test inputs and outputs** enabled in ESP-PPQ (use the ``export_test_values`` option).
- For deployment, you can export a version without test data to reduce model size.

**API:** ``esp_err_t dl::Model::test()``

**Returns:** ``ESP_OK`` if all tests pass, ``ESP_FAIL`` otherwise.

**Usage:**

.. code-block:: cpp

   #include "dl_model_base.hpp"

   // After creating the model...
   esp_err_t ret = model->test();
   if (ret == ESP_OK) {
       ESP_LOGI(TAG, "Model test passed!");
   } else {
       ESP_LOGE(TAG, "Model test failed!");
   }

   // Or using the convenience macro:
   ESP_ERROR_CHECK(model->test());

**How it works:**

1. Loads test input tensors embedded in the model
2. Runs inference through all model layers
3. Compares each output against the ground truth values (with tolerance for quantization errors)
4. Reports success or failure for each output

**Note for INT16 models:** Due to quantization rounding errors, INT16 models allow ±1 difference in comparison.

Profile model memory usage
---------------------------------

The ``profile_memory()`` method prints a detailed breakdown of memory usage across different memory types (internal RAM, PSRAM, FLASH).

**API:** ``void dl::Model::profile_memory()``

**Usage:**

.. code-block:: cpp

   #include "dl_model_base.hpp"

   // After creating and testing the model...
   model->profile_memory();

**Output includes:**

+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Name               | Explanation                                                                                                                                                       |
+====================+===================================================================================================================================================================+
| ``fbs_model``      | FlatBuffers model structure (includes model metadata, graph structure, tensor shapes, etc.)                                                                       |
|     ``parameter``  | Model parameters stored within the FlatBuffers model (sub-item of fbs_model)                                                                                      |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``parameter_copy`` | Parameters copied from FLASH to faster memory (PSRAM/internal RAM). Only present when ``param_copy=true`` (default). Improves inference performance.              |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``variable``       | Memory allocated for model inputs, outputs, and intermediate tensors by the memory manager.                                                                       |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``others``         | Miscellaneous memory usage (class member variables, alignment overhead, etc.). Usually very small.                                                                |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``total``          | Total memory usage across all categories.                                                                                                                         |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+

**Memory types shown:** Internal RAM, PSRAM, and FLASH usage for each category.

Profile model inference latency
----------------------------------------

The ``profile_module()`` method prints detailed latency information for each module (layer) in the model.

**API:** ``void dl::Model::profile_module(bool sort_module_by_latency = false)``

**Parameters:**
- ``sort_module_by_latency``: If ``true``, modules are sorted by latency (highest first). If ``false`` (default), modules are shown in ONNX topological order.

**Usage:**

.. code-block:: cpp

   // Default: topological order
   model->profile_module();

   // Sorted by latency (highest first)
   model->profile_module(true);

**Output includes:**
- Module name
- Module type (operation type)
- Inference latency in microseconds (or cycles if ``DL_LOG_LATENCY_UNIT`` is enabled)
- Total inference latency at the end

Combined profiling: profile() method
--------------------------------------------

The ``profile()`` method combines ``profile_memory()`` and ``profile_module()`` for comprehensive analysis.

**API:** ``void dl::Model::profile(bool sort_module_by_latency = false)``

**Usage:**

.. code-block:: cpp

   // Comprehensive profiling in topological order
   model->profile();

   // Comprehensive profiling sorted by latency
   model->profile(true);

This is the most convenient way to get both memory and performance analysis in one call.