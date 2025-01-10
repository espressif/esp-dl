使用 ESP-DL 加载模型
====================

:link_to_translation:`en:[English]`

在本教程中，我们将介绍如何加载一个 ESP-DL 的模型。

准备工作
--------

在开始之前，请确保您已经安装了 ESP-IDF 开发环境，并且已经配置好了您的开发板。此外，您需要有一个预训练的模型文件，并且已经使用 `ESP-PPQ <https://github.com/espressif/esp-ppq>`__ 量化完成并导出为 ``.espdl`` 模型格式。

方法 1：从 ``rodata`` 中加载模型
----------------------------------

1. **在** ``CMakeLists.txt`` **中添加模型文件**

   如果要将 ``.espdl`` 模型文件放到 flash 芯片的 ``.rodata`` 段，请将 `cmake folder <https://github.com/espressif/esp-dl/tree/master/models/human_face_detect/cmake>`__ 复制到项目下的组件目录或者main文件夹中。然后在 CMakeLists.txt 中添加以下代码。前两行应放在 idf_component_register() 之前，最后一行应放在 idf_component_register() 之后。

   .. code:: cmake

      include(${COMPONENT_DIR}/cmake/utilities.cmake)
      set(embed_files your_model_path/model_name.espdl)

      idf_component_register(...)

      target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)

2. **在程序中加载模型**

   使用以下方法加载模型：

   .. code:: cpp

      // "_binary_model_name_espdl_start" is composed of three parts: the prefix "binary", the filename "model_name_espdl", and the suffix "_start".
      extern const uint8_t espdl_model[] asm("_binary_model_name_espdl_start");

      Model *model = new Model((const char *)espdl_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA);


方法 2：从 ``partition`` 中加载模型
-------------------------------------

1. **在** ``partition.csv`` **中添加模型信息**

   在 ``partition.csv`` 文件中添加模型的 ``offset``、``size`` 等信息。

   ::

      # Name,   Type, SubType, Offset,  Size, Flags
      factory,  app,  factory,  0x010000,  4000K,
      model,   data,  spiffs,        ,  4000K,

2. **在** ``CMakeLists.txt`` **中添加自动加载程序**

   如果选择手动烧写，可以跳过此步骤。

   .. code:: cmake

      set(image_file your_model_path)
      partition_table_get_partition_info(size "--partition-name model" "size")
      if("${size}")
            esptool_py_flash_to_partition(flash "model" "${image_file}")
      else()

3. **在程序中加载模型**

   有两种方法可以加载模型。

   -  使用构造函数加载模型：

      .. code:: cpp

         // method1:
         Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

   -  首先加载 ``fbs_model``，然后使用 ``fbs_model`` 指针创建模型：

      .. code:: cpp

         // method2:
         fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
         fbs::FbsModel *fbs_model = fbs_loader->load();
         Model *model2 = new Model(fbs_model);

方法 3: 从 ``sdcard`` 中加载模型
-------------------------------------------

1. **检查 sdcard 是否是正确格式**

   首先备份 sdcard 中的数据.   

   - 如果使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      在 menuconfig 中打开 ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` 选项，以下代码将尝试挂载 sdcard，如果格式不正确，将自动格式化。

      .. code:: cpp
      
         ESP_ERROR_CHECK(bsp_sdcard_mount());
   
   - 如果不使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      将 esp_vfs_fat_sdmmc_mount_config_t 结构体中的 format_if_mount_failed 设置为 true，然后尝试挂载 sdcard。

      .. code:: cpp
         
         esp_vfs_fat_sdmmc_mount_config_t mount_config = {
               .format_if_mount_failed = true,
               .max_files = 5,
               .allocation_unit_size = 16 * 1024
         };
         // 挂载sdcard.

2. **将模型复制到 sdcard**
   
   将 .espdl 模型复制到 sdcard。

3. **在程序中加载模型**

   使用以下方法加载模型:  

   - 如果使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      .. code:: cpp

         ESP_ERROR_CHECK(bsp_sdcard_mount());
         const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
         Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);
   
   - 如果不使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

      .. code:: cpp

         // 挂载sdcard.
         const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
         Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

更多信息
-----------------

1. 使用方法1和方法2时，如果您的 PSRAM 空间紧张，可以关闭 Model 构造函数中的 param_copy 选项，该选项可以避免将模型参数从 flash 复制到 PSRAM，这会节省 PSRAM，但由于 PSRAM 的频率高于 flash，模型推理性能会下降。

- 方法1

   .. code:: cpp

      Model *model = new Model((const char *)espdl_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, MEMORY_MANAGER_GREEDY, nullptr, false);

- 方法2

   .. code:: cpp

      Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, false);

2. 使用方法3时，模型加载过程将花费更长的时间。我们需要将模型数据从 sdcard 复制到 PSRAM。如果你的 flash 空间紧张，这种方法很有用。

3. 使用方法1时，每次修改代码，模型数据都会被刷入。使用方法 2 和方法 3 可以不重复刷入模型，有助于减少刷入的时间。

- 方法 2

   使用 idf.py app-flash 代替 idf.py flash，只刷入 app 分区，而无需重新刷入模型分区。

   .. code:: bash

      idf.py app-flash

通过以上步骤，可以使用 ESP-DL 库成功加载一个预训练的模型。希望本教程对您有所帮助。更多信息请参考 :project_file:`fbs_loader.cpp <esp-dl/fbs_loader/src/fbs_loader.cpp>` 和 :project_file:`fbs_loader.hpp<esp-dl/fbs_loader/include/fbs_loader.hpp>`。
