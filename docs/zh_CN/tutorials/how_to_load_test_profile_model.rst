如何加载、测试和性能分析模型
==============================

:link_to_translation:`en:[English]`

在本教程中，我们将介绍如何加载、测试和分析一个 espdl 模型。:project:`参考例程 <examples/tutorial/how_to_load_test_profile_model>`

.. contents::
  :local:
  :depth: 2

准备工作
----------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :doc:`量化导出 espdl 模型 </tutorials/how_to_quantize_model>`

从 ``rodata`` 中加载模型
-------------------------

此方法将模型文件直接嵌入到应用程序 FLASH 的 ``.rodata`` 段中。这是最简单的方法，但缺点是每次应用程序代码更改时模型都会被重新烧录。

1. **在** ``CMakeLists.txt`` **中添加模型文件**

   要将 ``.espdl`` 模型文件嵌入到 ``.rodata`` 段，请在 ``CMakeLists.txt`` 中添加以下代码。前几行应放在 ``idf_component_register()`` 之前，最后一行放在 ``idf_component_register()`` 之后。

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

2. **在程序中加载模型**

   包含头文件：

   .. code-block:: cpp

      #include "dl_model_base.hpp"

   声明模型符号并创建模型：

   .. code-block:: cpp

      // 符号名由三部分组成：前缀 "_binary_"，文件名 "model_espdl"，后缀 "_start"
      extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

      // 基本用法 - 使用默认参数加载模型
      dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

      // 高级用法 - 自定义参数：
      // - 将参数保留在 FLASH 中（节省 PSRAM/内部 RAM，但性能较低）
      // - 限制内部 RAM 使用为 0 字节（优先使用 PSRAM）
      // - 使用贪婪内存管理器
      // - 无加密密钥
      // - param_copy = false（将参数保留在 FLASH 中）
      // dl::Model *model = new dl::Model((const char *)model_espdl,
      //                                  fbs::MODEL_LOCATION_IN_FLASH_RODATA,
      //                                  0,  // max_internal_size
      //                                  dl::MEMORY_MANAGER_GREEDY,
      //                                  nullptr,  // key
      //                                  false);   // param_copy

.. note::

   **性能与内存权衡：**

   - **烧录时间：** 使用 `从 rodata 中加载模型`_ 时，模型文件嵌入在应用程序二进制文件中，每次修改代码时都会重新烧录。对于大型模型，这会增加烧录时间。考虑使用 `从 partition 中加载模型`_ 或 `从 sdcard 中加载模型`_ 来避免此问题。

   - **内存 vs 性能：** ``param_copy`` 参数控制模型参数是否从 FLASH 复制到更快的内存（PSRAM/内部 RAM）。设置 ``param_copy=false`` 可以节省 RAM，但由于 FLASH 访问速度较慢，会降低推理性能。仅在 RAM 极其紧张时才禁用参数复制。

   - **应用程序分区大小：** 嵌入在 ``.rodata`` 中的大型模型可能需要增加 ``partition.csv`` 中的应用程序分区大小。


从 ``partition`` 中加载模型
----------------------------

此方法将模型存储在单独的 FLASH 分区中，允许您独立于应用程序代码更新模型。

1. **在** ``partition.csv`` **中添加模型信息**

   创建或修改您的 ``partition.csv`` 文件以包含模型分区。有关分区表的详细信息，请参阅 `ESP-IDF 分区表文档 <https://docs.espressif.com/projects/esp-idf/zh_CN/latest/esp32/api-guides/partition-tables.html>`_。

   .. code-block::

      # Name,   Type, SubType, Offset,  Size, Flags
      factory,  app,  factory,  0x010000,  4000K,
      model,   data,  spiffs,        ,  4000K,

   - **Name:** 任何有意义的名称（包括空终止符最多 16 个字符）
   - **Type:** ``data``
   - **SubType:** ``spiffs`` （模型存储必需）
   - **Offset:** 留空以自动计算
   - **Size:** 必须大于模型文件大小

2. **在** ``CMakeLists.txt`` **中添加模型烧录信息**

   .. code-block:: cmake

      idf_component_register(...)
      set(image_file your_model_path/model_name.espdl)
      esptool_py_flash_to_partition(flash "model" "${image_file}")

   ``esptool_py_flash_to_partition`` 中的第二个参数必须与 ``partition.csv`` 中的 ``Name`` 字段匹配。

3. **在程序中加载模型**

   包含头文件：

   .. code-block:: cpp

      #include "dl_model_base.hpp"

   创建模型实例：

   .. code-block:: cpp

      // 基本用法 - 使用默认参数加载模型
      dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

      // 高级用法 - 将参数保留在 FLASH 中以节省 RAM
      // dl::Model *model = new dl::Model("model",
      //                                  fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
      //                                  0,  // max_internal_size
      //                                  dl::MEMORY_MANAGER_GREEDY,
      //                                  nullptr,  // key
      //                                  false);   // param_copy

   第一个参数（分区标签）必须与 ``partition.csv`` 中的 ``Name`` 字段匹配。

.. note::

   **烧录优化：** 使用 ``idf.py app-flash`` 代替 ``idf.py flash``，可以仅烧录应用程序分区而不重新烧录模型分区。这显著减少了开发期间的烧录时间。

从 ``sdcard`` 中加载模型
--------------------------

此方法从 SD 卡加载模型，当 FLASH 空间有限或需要频繁更新模型而无需重新烧录时非常有用。

1. **准备 SD 卡**

   - **格式：** SD 卡应格式化为 FAT32。如果未格式化，挂载时将自动格式化（数据会丢失）。
   - **备份：** 在使用 ESP-DL 之前，请始终备份 SD 卡数据。

2. **挂载 SD 卡**

   - **使用 BSP（板级支持包）：**

     在 menuconfig 中启用 ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` 以允许自动格式化。

     .. code-block:: cpp

        #include "bsp/esp-bsp.h"
        ESP_ERROR_CHECK(bsp_sdcard_mount());

   - **不使用 BSP：**

     配置挂载选项，设置 ``format_if_mount_failed = true``。

     .. code-block:: cpp

        #include "esp_vfs_fat.h"
        #include "sdmmc_cmd.h"

        esp_vfs_fat_sdmmc_mount_config_t mount_config = {
            .format_if_mount_failed = true,
            .max_files = 5,
            .allocation_unit_size = 16 * 1024
        };
        // 挂载 SD 卡（具体实现取决于您的硬件）

3. **复制模型到 SD 卡**

   将您的 ``.espdl`` 模型文件复制到 SD 卡（例如，复制到根目录作为 ``model.espdl``）。

4. **在程序中加载模型**

   包含头文件：  

   .. code-block:: cpp

      #include "dl_model_base.hpp"
   
- 如果不使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

  .. code-block:: cpp
  
     // 挂载sdcard.
     const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
     Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

.. note::

   使用 `从 sdcard 中加载模型`_ 时，模型加载过程将花费更长的时间，因为模型数据需要从 sdcard 复制到 PSRAM 或者 internal RAM。如果你的 FLASH 空间紧张，这种方法很有用。

测试模型板端推理是否正确
-------------------------

``test()`` 方法通过将推理结果与模型文件中嵌入的基准真值进行比较，验证模型是否产生正确的推理结果。

**前提条件：**

- ``.espdl`` 模型必须在 ESP-PPQ 中导出时启用**测试输入和输出**（使用 ``export_test_values`` 选项）。
- 对于部署，您可以导出一个没有测试数据的版本以减小模型大小。

**API：** ``esp_err_t dl::Model::test()``

**返回值：** 如果所有测试通过则返回 ``ESP_OK``，否则返回 ``ESP_FAIL``。

**用法：**

.. code-block:: cpp

   #include "dl_model_base.hpp"

   // 创建模型后...
   esp_err_t ret = model->test();
   if (ret == ESP_OK) {
       ESP_LOGI(TAG, "模型测试通过！");
   } else {
       ESP_LOGE(TAG, "模型测试失败！");
   }

   // 或使用便捷宏：
   ESP_ERROR_CHECK(model->test());

**工作原理：**

1. 加载模型中嵌入的测试输入张量，所以test()不需要外部输入
2. 通过所有模型层运行推理
3. 将每个输出与基准真值进行比较（考虑量化误差的容差）
4. 报告每个输出的成功或失败

**INT16 模型注意事项：** 由于量化舍入误差，INT16 模型允许比较时有 ±1 的差异。

分析模型内存使用情况
-----------------------------

``profile_memory()`` 方法打印跨不同内存类型（内部 RAM、PSRAM、FLASH）的内存使用详细明细。

**API：** ``void dl::Model::profile_memory()``

**用法：**

.. code-block:: cpp

   #include "dl_model_base.hpp"

   // 创建并测试模型后...
   model->profile_memory();

**输出包括：**

+---------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| 名称                | 解释                                                                                                                                                |
+=====================+=====================================================================================================================================================+
| ``fbs_model``       | flatbuffers 模型，包含一个子项，模型参数 ``parameter``。flatbuffers 模型除了模型参数之外，还包括测试输入输出，模型参数/变量的形状，模型结构等信息。 |
|      ``parameter``  |                                                                                                                                                     |
+---------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| ``parameter_copy``  | 复制的模型参数，当 flatbuffers 模型位于 FLASH 的时候，默认情况下会复制到 PSRAM 或者 internal RAM 以提高推理性能。                                   |
+---------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| ``variable``        | 内存管理模块申请的内存，模型输入/输出以及中间的计算结果都会使用这部分空间。                                                                         |
+---------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| ``others``          | 类成员变量所需要的空间, ``heap_caps_aligned_alloc`` / ``heap_caps_aligned_calloc`` 申请过程中对齐的额外部分（很小）。                               |
+---------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+

**显示的内存类型：** 每个类别的内部 RAM、PSRAM 和 FLASH 使用情况。

分析模型推理延迟
-----------------------------

``profile_module()`` 方法打印模型中每个模块（层）的详细延迟信息。

**API：** ``void dl::Model::profile_module(bool sort_module_by_latency = false)``

**参数：**
- ``sort_module_by_latency`` ：如果为 ``true`` ，模块按延迟排序（最高优先）。如果为 ``false`` （默认），模块按拓扑顺序显示。

**用法：**

.. code-block:: cpp

   // 默认：拓扑顺序
   model->profile_module();

   // 按延迟排序（最高优先）
   model->profile_module(true);

**输出包括：**
- 模块名称
- 模块类型（操作类型）
- 推理延迟（微秒，如果启用 ``DL_LOG_LATENCY_UNIT`` 则为周期数）
- 末尾的总推理延迟

**相关 API：**

- ``std::map<std::string, module_info> get_module_info()`` - 以编程方式返回模块信息
- ``void print_module_info(const std::map<std::string, module_info> &info, bool sort_module_by_latency = false)`` - 从映射打印模块信息

组合性能分析：profile() 方法
--------------------------------------------

``profile()`` 方法结合了 ``profile_memory()`` 和 ``profile_module()``，进行综合分析。

**API：** ``void dl::Model::profile(bool sort_module_by_latency = false)``

**用法：**

.. code-block:: cpp

   // 拓扑顺序的综合性能分析
   model->profile();

   // 按延迟排序的综合性能分析
   model->profile(true);

这是获取内存和性能分析的最便捷方式。