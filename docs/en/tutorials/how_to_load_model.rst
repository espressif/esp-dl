Loading Models with ESP-DL
==========================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will guide you through the process of loading an ESP-DL model.

Prerequisites
-------------

Before you begin, ensure that you have the ESP-IDF development environment installed and your development board properly configured. Additionally, you need to have a pre-trained model file that has been quantized using `ESP-PPQ <https://github.com/espressif/esp-ppq>`__ and exported in the ``.espdl`` model format.

Method 1: Load Model from ``rodata``
----------------------------------------

1. **Add Model File in** ``CMakeLists.txt``

   To add the ``.espdl`` model file to the ``.rodata`` section of the chip flash, copy the `cmake folder <https://github.com/espressif/esp-dl/tree/master/models/human_face_detect/cmake>`__ to the component dir or main folder under your project, add the following code in your CMakeLists.txt, the first two lines should be placed before idf_component_register(), the last line should be placed after idf_component_register()

   .. code:: cmake

      include(${COMPONENT_DIR}/cmake/utilities.cmake)
      set(embed_files your_model_path/model_name.espdl)

      idf_component_register(...)

      target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)

2. **Load Model in Your Program**

   Use the following method to load the model:

   .. code:: cpp

      // "_binary_model_name_espdl_start" is composed of three parts: the prefix "binary", the filename "model_name_espdl", and the suffix "_start".
      extern const uint8_t espdl_model[] asm("_binary_model_name_espdl_start");

      Model *model = new Model((const char *)espdl_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA);


Method 2: Load Model from ``partition``
-------------------------------------------

1. **Add Model Information in** ``partition.csv``

   Add the model’s ``offset``, ``size``, and other information in the ``partition.csv`` file.

   ::

      # Name,   Type, SubType, Offset,  Size, Flags
      factory,  app,  factory,  0x010000,  4000K,
      model,   data,  spiffs,        ,  4000K,

2. **Add Automatic Loading Program in** ``CMakeLists.txt``

   Skip this step if you choose to manually flash.

   .. code:: cmake

      set(image_file your_model_path)
      partition_table_get_partition_info(size "--partition-name model" "size")
      if("${size}")
            esptool_py_flash_to_partition(flash "model" "${image_file}")
      else()

3. **Load Model in Your Program**

   There are two methods to load the model.

   -  Load the model using the constructor:

      .. code:: cpp

         // method1:
         Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

   -  First load the ``fbs_model``, then create the model using the ``fbs_model`` pointer:

      .. code:: cpp

         // method2:
         fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
         fbs::FbsModel *fbs_model = fbs_loader->load();
         Model *model2 = new Model(fbs_model);

Method 3: Load Model from ``sdcard``
-------------------------------------------

1. **Check if your sdcard is in the proper format**

   First backup your data in your sdcard.   

   - Work with `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      Trun on ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` in menuconfig, the following code will try to mount sdcard, if it is not in the proper format, it will be automatically formatted.

      .. code:: cpp
      
         ESP_ERROR_CHECK(bsp_sdcard_mount());
   
   - Work without `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      Set format_if_mount_failed to true in esp_vfs_fat_sdmmc_mount_config_t, Then try to mount the sdcard.

      .. code:: cpp
         
         esp_vfs_fat_sdmmc_mount_config_t mount_config = {
               .format_if_mount_failed = true,
               .max_files = 5,
               .allocation_unit_size = 16 * 1024
         };
         // your codes to mount sdcard.

2. **Copy model to your sdcard**
   
   Copy the .espdl model to your sdcard.

3. **Load Model in Your Program**

   Use the following method to load the model:  

   - Work with `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      .. code:: cpp

         ESP_ERROR_CHECK(bsp_sdcard_mount());
         const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
         Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);
   
   - Work without `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      .. code:: cpp

         // your code to mount sdcard.
         const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
         Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

More Information
-----------------

1. When using Method1 and Method2, if your psram size is tight, you can turn off the param_copy option in Model constructor. This option can avoid copy model parameters from flash to psram. It saves psram, but the model inference performance will drop because the frequency of psram is higher than flash.

- Method 1

   .. code:: cpp

      Model *model = new Model((const char *)espdl_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, MEMORY_MANAGER_GREEDY, nullptr, false);

- Method 2

   .. code:: cpp

      Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, false);

2. When using Method3, the model loading process will take longer time. We need to copy the model data from sdcard to psram. It is useful if your flash size is tight.

3. When using Method1, every time you modified your code, the model data is flashed. It is helpful to reduce the flash time by using Method2 and Method3.

- Method 2

   Use idf.py app-flash instead of idf.py flash to only flash the app partition without re-flash the model partition.

   .. code:: bash

      idf.py app-flash

By following the steps above, you can successfully load a pre-trained model using the ESP-DL library. We hope this tutorial is helpful to you! For more information, please refer to the code in :project_file:`fbs_loader.cpp <esp-dl/fbs_loader/src/fbs_loader.cpp>` and :project_file:`fbs_loader.hpp<esp-dl/fbs_loader/include/fbs_loader.hpp>`.
