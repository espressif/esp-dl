如何加载和测试模型
====================

:link_to_translation:`en:[English]`

在本教程中，我们将介绍如何加载和测试一个 espdl 模型。:project:`参考例程 <examples/tutorial/how_to_load_test_profile_model>`

.. contents::
  :local:
  :depth: 2

准备工作
----------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :doc:`量化导出 espdl 模型 </tutorials/how_to_quantize_model>`

从 ``rodata`` 中加载模型
-------------------------

1. **在** ``CMakeLists.txt`` **中添加模型文件**

   如果要将 ``.espdl`` 模型文件放到 FLASH 芯片的 ``.rodata`` 段，需要在 ``CMakeLists.txt`` 中添加以下代码。前几行应放在 ``idf_component_register()`` 之前，最后一行应放在 ``idf_component_register()`` 之后。

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

   .. code-block:: cpp

      // "_binary_model_espdl_start" is composed of three parts: the prefix "binary", the filename "model_espdl", and the suffix "_start".
      extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

      dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
      
      // Keep parameter in FLASH, saves PSRAM/internal RAM, lower performance.
      // dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
      
.. note::

   1. 使用 `从 rodata 中加载模型`_ 时，由于 ``.rodata`` 段属于 app 分区，每次修改代码，模型文件都会被烧录。如果模型文件较大，可能需要调整 app 分区的大小。使用 `从 partition 中加载模型`_ 或者 `从 sdcard 中加载模型`_ 可以不重复烧录模型，有助于减少烧录的时间。
   2. 使用 `从 rodata 中加载模型`_ 或者 `从 partition 中加载模型`_ 时, 关闭 Model 构造函数中的 param_copy 选项可以防止 FLASH 中模型权重的复制到 PSRAM 或者 internal RAM。这样可以减少 PSRAM 或者 internal RAM 空间的使用。但是由于 PSRAM 或者 internal RAM 的频率高于 FLASH， 模型的推理性能会下降。


从 ``partition`` 中加载模型
----------------------------

1. **在** ``partition.csv`` **中添加模型信息**

   关于 ``partition.csv`` ， 请查阅 `分区表文档 <https://docs.espressif.com/projects/esp-idf/zh_CN/latest/esp32/api-guides/partition-tables.html>`_。

   ::

      # Name,   Type, SubType, Offset,  Size, Flags
      factory,  app,  factory,  0x010000,  4000K,
      model,   data,  spiffs,        ,  4000K,

   模型的 ``Name`` 字段可以是任何有意义的名称，但不能超过 16 个字节，其中包括一个空字节（之后的内容将被截断）。``SubType`` 字段必须是 spiffs。``Offset`` 在别的分区之后可以不填，会自动计算。``Size`` 必须大于模型文件的大小。

2. **在** ``CMakeLists.txt`` **中添加模型烧录信息**

   .. code-block:: cmake

      idf_component_register(...)
      set(image_file your_model_path)
      esptool_py_flash_to_partition(flash "model" "${image_file}")
   
   ``esptool_py_flash_to_partition`` 中的第二个参数必须和 ``partition.csv`` 中的 ``Name`` 字段一致。

3. **在程序中加载模型**

   .. code-block:: cpp

      dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

      // Keep parameter in FLASH, saves PSRAM/internal RAM, lower performance.
      // dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY,
      // nullptr, false);
  
   构造函数的第一个参数必须和 ``partition.csv`` 中的 ``Name`` 字段一致。

.. note::

   使用 ``idf.py app-flash`` 代替 ``idf.py flash`` ，可以只烧录 app 分区，不烧录模型分区，减少烧录时间。

从 ``sdcard`` 中加载模型
--------------------------

1. **检查 sdcard 是否是正确格式**

   首先备份数据，然后尝试在板端挂载。如果 sdcard 格式不正确，它会被自动格式化成正确的格式。

- 如果使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__

  在 menuconfig 中打开 ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` 选项。
  
  .. code-block:: cpp
  
     ESP_ERROR_CHECK(bsp_sdcard_mount());

- 如果不使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__

  将 esp_vfs_fat_sdmmc_mount_config_t 结构体中的 format_if_mount_failed 设置为 true。
  
  .. code-block:: cpp
  
     esp_vfs_fat_sdmmc_mount_config_t mount_config = {
           .format_if_mount_failed = true,
           .max_files = 5,
           .allocation_unit_size = 16 * 1024
     };
     // 挂载sdcard.

2. **将模型复制到 sdcard**
   
   将 .espdl 模型复制到 sdcard。

3. **在程序中加载模型**

- 如果使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

  .. code-block:: cpp
  
     ESP_ERROR_CHECK(bsp_sdcard_mount());
     const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
     Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);
   
- 如果不使用 `BSP(Board Support Package)  <https://github.com/espressif/esp-bsp/tree/master/bsp>`__  

  .. code-block:: cpp
  
     // 挂载sdcard.
     const char *model_path = "/your_sdcard_mount_point/your_model_path/model_name.espdl";
     Model *model = new Model(model_path, fbs::MODEL_LOCATION_IN_SDCARD);

.. note::

   使用 `从 sdcard 中加载模型`_ 时，模型加载过程将花费更长的时间，因为模型数据需要从 sdcard 复制到 PSRAM 或者 internal RAM。如果你的 FLASH 空间紧张，这种方法很有用。

测试模型板端推理是否正确
-------------------------

为了在板端测试推理是否正确，``.espdl`` 模型在导出的时候需要 :ref:`添加测试输入/输出 <add_test_input_output>`。实际部署的时候可以再导出一个没有测试输入输出的版本，以减少模型文件的大小。

.. code-block:: cpp

   ESP_ERROR_CHECK(model->test());

测试模型内存占用
-------------------

.. code-block:: cpp

   model->profile_memory();

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

测试模型推理耗时
-------------------

.. code-block:: cpp

   model->profile_module();

默认按照 ONNX 拓扑排序打印模型各层。如果想按照各层的耗时来排序，可以将 ``profile_module`` 的入参设为 ``True``。

.. code-block:: cpp

   model->profile_module(true);