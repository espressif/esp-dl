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
  Interfaces for 60+ neural network operators (convolution, pooling, activation, etc.). Files: ``dl_module_base.hpp``, ``dl_module_conv.hpp``, ``dl_module_pool.hpp``, ``dl_module_relu.hpp``, etc.

* **base**
  Implements operations for chips (esp32, esp32s3, esp32p4) with ISA-specific assembly support. Includes operator implementations in ``dl_base_conv2d.cpp/hpp``, ``dl_base_avg_pool2d.cpp/hpp``, etc., and ISA-specific code in ``isa/`` subdirectories.

* **math**
  Mathematical operations (matrix functions). Files: ``dl_math.hpp`` and ``dl_math_matrix.hpp``.

* **tool**
  Auxiliary functions (utility tools). Files: ``dl_tool.hpp`` and ``dl_tool.cpp``. Includes ISA-specific tools in ``isa/`` subdirectories.

* **tensor**
  Tensor classes and operations. Files: ``dl_tensor_base.hpp``.


**vision**
~~~~~~~~~~~~~~~~~~

Computer vision modules divided into submodules:

* **classification**
  Image classification (model inference). Inference: ``dl_cls_base``. Post-processors: ``imagenet_cls_postprocessor``, ``hand_gesture_cls_postprocessor``, ``dl_cls_postprocessor``.

* **recognition**
  Feature extraction (model inference). Feature database management (Enroll, delete, query). Pre-processor: ``dl_feat_image_preprocessor``. Inference: ``dl_feat_base``. Post-processor: ``dl_feat_postprocessor``. Database: ``dl_recognition_database``

* **image**
  Image processing (resize, crop, warp affine). Color conversion (pixel, img). Image preprocessor (pipeline of resize, crop, color conversion, normalization, quantization). Image decoding/encoding (JPEG/BMP). Draw utility (point, hollow rectangle).

  Image process: ``dl_image_process``. Color conversion: ``dl_image_color``. Image preprocessor: ``dl_image_preprocessor``. Image decoding/encoding: ``dl_image_jpeg``, ``dl_image_bmp``. Draw utility: ``dl_image_draw``.

* **detect**
  Object detection (model inference). Inference: ``dl_detect_base``. Post-processors: ``dl_detect_yolo11_postprocessor``, ``dl_detect_espdet_postprocessor``, ``dl_detect_msr_postprocessor``, ``dl_detect_mnp_postprocessor``, ``dl_detect_pico_postprocessor``. Pose estimation: ``dl_pose_yolo11_postprocessor``.


**audio**
~~~~~~~~~~~~~~~~~~

Audio processing modules divided into submodules:

* **common**
  Common audio utilities. Files: ``dl_audio_common.cpp/hpp``, ``dl_audio_wav.cpp/hpp``.

* **speech_features**
  Speech feature extraction. Files: ``dl_speech_features.cpp/hpp`` (base class), ``dl_fbank.cpp/hpp`` (Filter Bank), ``dl_mfcc.cpp/hpp`` (MFCC), ``dl_spectrogram.cpp/hpp`` (Spectrogram).


**fbs_loader (FlatBuffers Loader)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles FlatBuffers models:

* **include**
  Headers: ``fbs_loader.hpp``, ``fbs_model.hpp``.

* **src**
  Implementations: ``fbs_loader.cpp``.

* **lib/**
  Pre-compiled libraries for different targets: ``esp32/``, ``esp32s3/``, ``esp32p4/``.

* **espidl.fbs**
  FlatBuffers schema file.

* **pack_espdl_models.py**
  Model packing script.


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