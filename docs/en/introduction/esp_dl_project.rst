ESP-DL Project Organization
===========================

:link_to_translation:`zh_CN:[中文]`

**ESP-DL**'s modular design enables efficient development, maintenance, and scalability. The project is organized as follows:

**dl (Deep Learning)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core deep learning modules and tools, divided into submodules:

* **model**  
  Loads, manages, and allocates memory for deep learning models. Includes ``dl_model_base`` and ``dl_memory_manager``.

* **module**  
  Interfaces for operations (convolution, pooling, activation). Files: ``dl_module_conv``, ``dl_module_pool``, ``dl_module_relu``.

* **base**  
  Implements operations for chips (esp32s3/esp32p4) with assembly support.

* **math**  
  Mathematical operations (matrix functions). Files: ``dl_math.hpp`` and ``dl_math.cpp``.

* **tool**  
  Auxiliary functions (utility tools). Files: ``dl_tool.hpp`` and ``dl_tool.cpp``.

* **tensor**  
  Tensor classes (variables, constants). Files: ``dl_tensor_base.hpp``.


**vision**
~~~~~~~~~~~~~~~~~~

Computer vision modules divided into submodules:

* **classification**  
  Image classification (preprocessing, inference, post-processing). Post-processors: ``imagenet_cls_postprocessor``, ``dl_cls_postprocessor``.

* **recognition**  
  Face recognition (feature extraction/matching, database management). Implements ``dl_feat_postprocessor`` and ``dl_feat_image_preprocessor``.

* **image**  
  Image processing (decoding, color conversion). Supports JPEG/BMP with decoding/encoding.

* **detect**  
  Object detection post-processing. Post-processors: ``dl_detect_msr_postprocessor``, ``dl_detect_mnp_postprocessor``, ``dl_detect_pico_postprocessor``.


**fbs_loader (FlatBuffers Loader)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles FlatBuffers models:

* **include**  
  Headers: ``fbs_loader.hpp``, ``fbs_model.hpp``.

* **src**  
  Implementations: ``fbs_loader.cpp``.


**Other Files**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CMakeLists.txt**  
  Project build configuration.

* **idf_component.yml**  
  Component metadata (name, version, dependencies).

* **README.md**  
  Project documentation and usage.

* **LICENSE**  
  License terms.