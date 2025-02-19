ESP-DL 项目组织
===========================

:link_to_translation:`en:[English]`

**ESP-DL** 的模块化设计使其开发、维护和扩展变得高效。项目的组织结构如下：

**dl（深度学习）**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

核心深度学习模块和工具，分为子模块：

* **model**  
  加载、管理和分配深度学习模型的内存。包含 ``dl_model_base`` 和 ``dl_memory_manager``。

* **module**  
  算子接口（卷积、池化、激活）。文件：``dl_module_conv``，``dl_module_pool``，``dl_module_relu``。

* **base**  
  具体的算子实现，包括对芯片（esp32s3/esp32p4）的汇编加速。

* **math**  
  数学操作（矩阵函数）。文件：``dl_math.hpp`` 和 ``dl_math.cpp``。

* **tool**  
  辅助功能（实用工具）。文件：``dl_tool.hpp`` 和 ``dl_tool.cpp``。

* **tensor**  
  张量类。文件：``dl_tensor_base.hpp``。


**vision（计算机视觉）**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算机视觉模块，分为子模块：

* **classification**  
  图像分类（模型推理）。推理：``dl_cls_base``。后处理器：``imagenet_cls_postprocessor``，``dl_cls_postprocessor``。

* **recognition**  
  特征提取（模型推理）。特征数据库管理（注册、删除、查询）。预处理器：``dl_feat_image_preprocessor``。推理：``dl_feat_base``。后处理器：``dl_feat_postprocessor``。数据库：``dl_recognition_database``

* **image**  
  图像处理（调整大小、裁剪、仿射变换）。颜色转换（像素、图像）。图像预处理器（调整大小、裁剪、颜色转换、规范化、量化的管道）。图像解码/编码（JPEG/BMP）。绘制工具（点、空心矩形）。

  图像处理：``dl_image_process``。颜色转换：``dl_image_color``。图像预处理器：``dl_image_preprocessor``。图像解码/编码：``dl_image_jpeg``、``dl_image_bmp``。绘制工具：``dl_image_draw``。

* **detect**  
  目标检测（模型推理）。推理：``dl_detect_base``。后处理器：``dl_detect_msr_postprocessor``、``dl_detect_mnp_postprocessor``、``dl_detect_pico_postprocessor``。


**fbs_loader（FlatBuffers 加载器）**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

处理 FlatBuffers 模型：

* **include**  
  头文件：``fbs_loader.hpp``，``fbs_model.hpp``。

* **src**  
  实现：``fbs_loader.cpp``。


**其他文件**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CMakeLists.txt**  
  项目构建配置。

* **idf_component.yml**  
  组件元数据（名称、版本、依赖项）。

* **README.md**  
  项目文档和使用说明。

* **LICENSE**  
  许可条款。