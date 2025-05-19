How to deploy YOLO11n
============================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will introduce how to quantize a pre-trained YOLO11n model using ESP-PPQ and deploy the quantized YOLO11n model using ESP-DL.

.. contents::
  :local:
  :depth: 2

Preparation
----------------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_yolo11n:

Model quantization
------------------------

Pre-trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can download pre-trained yolo11n model from `Ultralytics release <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>`__.

Currently, ESP-PPQ supports ONNX, PyTorch, and TensorFlow models. During the quantization process, PyTorch and TensorFlow models are first converted to ONNX models, so the pre-trained yolo11n model needs to be converted to an ONNX model.

Specificially, refer to the script :project_file:`export_onnx.py <models/coco_detect/models/export_onnx.py>` to convert the pre-trained yolo11n model to an ONNX model.

In the srcipt, we have overridden the forward method of the Detect class, which offers following advantages:

- Faster inference. Compared to the original yolo11n model, operations related to decoding bounding boxes in Detect head are moved from the inference pass to the post-processing phase, resulting in a significant reduction in inference latency. On one hand, operations like ``Conv``, ``Transpose``, ``Slice``, ``Split`` and ``Concat`` are time-consuming when applied during inference pass. On the other hand, the inference outputs are first filtered using a score threshold before decoding the boxes in the post-processing pass, which significantly reduces the number of calculations, thereby acclerating the overall inference speed.

- Lower quantization Error. The ``Concat`` and ``Add`` operators adopt joint quantization in ESP-PPQ. To reduce quantization errors, the box and score are output by separate branches, rather than being concatenated, due to the significant difference in their ranges. Similarly, since the ranges of the two inputs of ``Add`` and ``Sub`` differ significantly, the calculations are performed in the post-processing phase to avoid quantization errors.


Calibration Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration dataset needs to match the input format of the model. The calibration dataset should cover all possible input scenarios to better quantize the model. Here, the calibration dataset used in this example is `calib_yolo11n <https://dl.espressif.com/public/calib_yolo11n.zip>`__.

8bit default configuration quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Quantization settings**

.. code-block:: python

   target="esp32p4"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting

**Quantization results**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 36.042%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ████████████████     | 28.761%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████        | 22.876%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 21.570%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | ████████████         | 21.467%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ████████████         | 21.021%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ████████████         | 20.973%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ███████████          | 19.432%
   /model.22/m.0/cv2/conv/Conv:                 | ███████████          | 19.320%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 19.243%
   /model.22/m.0/cv3/conv/Conv:                 | ███████████          | 19.029%
   /model.22/cv2/conv/Conv:                     | ██████████           | 18.488%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ██████████           | 18.222%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ██████████           | 17.400%
   /model.8/m.0/cv2/conv/Conv:                  | █████████            | 16.189%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █████████            | 15.585%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ████████             | 14.687%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ████████             | 14.601%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████             | 14.154%
   /model.22/cv1/conv/Conv:                     | ████████             | 14.102%
   /model.10/m/m.0/attn/MatMul_1:               | ████████             | 13.998%
   /model.10/cv1/conv/Conv:                     | ███████              | 13.560%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 11.771%
   /model.19/m.0/cv2/conv/Conv:                 | ██████               | 11.216%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████               | 11.140%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████               | 11.057%
   /model.13/m.0/cv2/conv/Conv:                 | ██████               | 10.881%
   /model.20/conv/Conv:                         | ██████               | 10.692%
   /model.23/cv2.2/cv2.2.2/Conv:                | █████                | 9.888%
   /model.10/cv2/conv/Conv:                     | █████                | 9.788%
   /model.8/cv2/conv/Conv:                      | █████                | 9.477%
   /model.8/m.0/cv1/conv/Conv:                  | █████                | 9.422%
   /model.19/cv2/conv/Conv:                     | █████                | 9.102%
   /model.8/cv1/conv/Conv:                      | █████                | 9.101%
   /model.8/m.0/cv3/conv/Conv:                  | █████                | 9.068%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 9.014%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████                | 8.996%
   /model.6/m.0/cv2/conv/Conv:                  | █████                | 8.882%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████                | 8.637%
   /model.13/cv2/conv/Conv:                     | █████                | 8.556%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | █████                | 8.461%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 8.362%
   /model.19/cv1/conv/Conv:                     | ████                 | 8.194%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 8.021%
   /model.13/cv1/conv/Conv:                     | ████                 | 7.910%
   /model.10/m/m.0/attn/MatMul:                 | ████                 | 7.861%
   /model.19/m.0/cv1/conv/Conv:                 | ████                 | 7.520%
   /model.22/m.0/cv1/conv/Conv:                 | ████                 | 7.239%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ████                 | 7.054%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 7.042%
   /model.13/m.0/cv1/conv/Conv:                 | ████                 | 6.987%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 6.739%
   /model.23/cv2.1/cv2.1.2/Conv:                | ████                 | 6.734%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ████                 | 6.660%
   /model.17/conv/Conv:                         | ███                  | 6.025%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 5.897%
   /model.6/cv2/conv/Conv:                      | ███                  | 5.815%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 5.814%
   /model.6/cv1/conv/Conv:                      | ███                  | 5.693%
   /model.7/conv/Conv:                          | ███                  | 5.570%
   /model.9/cv2/conv/Conv:                      | ███                  | 5.382%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ███                  | 5.173%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 5.168%
   /model.16/m.0/cv1/conv/Conv:                 | ███                  | 5.087%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ███                  | 5.010%
   /model.16/cv2/conv/Conv:                     | ███                  | 4.991%
   /model.2/cv2/conv/Conv:                      | ██                   | 4.552%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 4.443%
   /model.3/conv/Conv:                          | ██                   | 4.318%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██                   | 4.304%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.968%
   /model.5/conv/Conv:                          | ██                   | 3.948%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 3.863%
   /model.4/cv1/conv/Conv:                      | ██                   | 3.720%
   /model.2/cv1/conv/Conv:                      | ██                   | 3.565%
   /model.4/cv2/conv/Conv:                      | ██                   | 3.538%
   /model.16/cv1/conv/Conv:                     | ██                   | 3.110%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 2.844%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | █                    | 2.762%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 2.532%
   /model.9/cv1/conv/Conv:                      | █                    | 2.015%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.761%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 1.317%
   /model.1/conv/Conv:                          | █                    | 1.315%
   /model.23/cv3.2/cv3.2.2/Conv:                | █                    | 1.114%
   /model.2/m.0/cv1/conv/Conv:                  |                      | 0.731%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.491%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.282%
   /model.0/conv/Conv:                          |                      | 0.159%
   Analysing Layerwise quantization error:: 100%|██| 89/89 [07:46<00:00,  5.24s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.1/conv/Conv:                          | ████████████████████ | 0.384%
   /model.22/cv1/conv/Conv:                     | █████████████        | 0.247%
   /model.4/cv2/conv/Conv:                      | ████████████         | 0.233%
   /model.2/cv2/conv/Conv:                      | ██████████           | 0.201%
   /model.0/conv/Conv:                          | ██████████           | 0.192%
   /model.9/cv2/conv/Conv:                      | ████████             | 0.156%
   /model.10/cv1/conv/Conv:                     | ███████              | 0.132%
   /model.3/conv/Conv:                          | ██████               | 0.108%
   /model.4/cv1/conv/Conv:                      | ████                 | 0.074%
   /model.16/cv1/conv/Conv:                     | ███                  | 0.066%
   /model.2/cv1/conv/Conv:                      | ███                  | 0.060%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███                  | 0.052%
   /model.2/m.0/cv1/conv/Conv:                  | ██                   | 0.044%
   /model.6/cv1/conv/Conv:                      | ██                   | 0.033%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██                   | 0.029%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.028%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █                    | 0.023%
   /model.16/cv2/conv/Conv:                     | █                    | 0.021%
   /model.16/m.0/cv2/conv/Conv:                 | █                    | 0.020%
   /model.19/m.0/cv1/conv/Conv:                 | █                    | 0.020%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.018%
   /model.19/cv2/conv/Conv:                     | █                    | 0.017%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.016%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.016%
   /model.19/cv1/conv/Conv:                     | █                    | 0.015%
   /model.13/cv2/conv/Conv:                     | █                    | 0.015%
   /model.8/cv1/conv/Conv:                      | █                    | 0.013%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | █                    | 0.013%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | █                    | 0.012%
   /model.13/cv1/conv/Conv:                     | █                    | 0.012%
   /model.10/cv2/conv/Conv:                     | █                    | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 | █                    | 0.011%
   /model.6/cv2/conv/Conv:                      | █                    | 0.011%
   /model.13/m.0/cv2/conv/Conv:                 | █                    | 0.010%
   /model.5/conv/Conv:                          |                      | 0.010%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.009%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.009%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.008%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.008%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.008%
   /model.9/cv1/conv/Conv:                      |                      | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.007%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.007%
   /model.17/conv/Conv:                         |                      | 0.007%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.007%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.007%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.006%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.006%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.005%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.005%
   /model.7/conv/Conv:                          |                      | 0.005%
   /model.8/cv2/conv/Conv:                      |                      | 0.004%
   /model.22/cv2/conv/Conv:                     |                      | 0.004%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.004%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.004%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.003%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.003%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.003%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.003%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.002%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.002%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.001%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.20/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%

**Quantization error analysis**

With the same inputs, The mAP50:95 on COCO val2017 after quantization is only 30.7%, which is lower than that of the float model. There is a accuracy loss with:

- **Graphwise Error**

  The output layers of the model are /model.23/cv3.2/cv3.2.2/Conv, /model.23/cv2.2/cv2.2.2/Conv, /model.23/cv3.1/cv3.1.2/Conv, /model.23/cv2.1/cv2.1.2/Conv, /model.23/cv3.0/cv3.0.2/Conv and /model.23/cv2.0/cv2.0.2/Conv. The cumulative error for these layers are 1.114%, 9.888%, 0.491%, 6.734%, 0.282% and 6.739% respectively. Generally, if the cumulative error of the output layer is less than 10%, the loss in accuracy of the quantized model is minimal.

- **Layerwise error**

  Observing the Layerwise error, it is found that the errors for all layers are below 1%, indicating that the quantization errors for all layers are small. 

We noticed that although the layer-wise errors for all layers are small, the cumulative errors in some layers are relatively large. This may be related to the complex CSP structure in the yolo11n model, where the inputs to the ``Concat`` or ``Add`` layers may have different distributions or scales. We can choose to quantize certain layers using int16 and optimize the quantization with horizontal layer split pass. For more details, please refer to the mixed-precision + horizontal layer split pass quantization test.

.. _horizontal_layer_split_label:

Mixed-Precision + Horizontal Layer Split Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spliting convolution layers or GEMM layers can reduce quantization error for better performance.

**Quantization settings**

.. code-block:: python

   from ppq.api import get_target_platform
   target="esp32p4"
   num_of_bits=8
   batch_size=32

   # Quantize the following layers with 16-bits
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.dispatching_table.append("/model.2/cv2/conv/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/model.3/conv/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/model.4/cv2/conv/Conv", get_target_platform(TARGET, 16))

   # Horizontal Layer Split Pass
   quant_setting.weight_split = True
   quant_setting.weight_split_setting.method = 'balance'
   quant_setting.weight_split_setting.value_threshold = 1.5
   quant_setting.weight_split_setting.interested_layers = ['/model.0/conv/Conv', '/model.1/conv/Conv']
    

**Quantization results**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 24.841%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ███████████████      | 19.061%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ██████████████       | 17.927%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | ██████████████       | 17.396%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ██████████████       | 17.061%
   /model.22/m.0/cv3/conv/Conv:                 | ████████████         | 15.563%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ████████████         | 15.427%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ████████████         | 14.890%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████████         | 14.784%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | ███████████          | 14.243%
   /model.22/cv2/conv/Conv:                     | ███████████          | 14.098%
   /model.22/m.0/cv2/conv/Conv:                 | ███████████          | 13.945%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ███████████          | 13.489%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | █████████            | 10.919%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ████████             | 10.073%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████             | 9.819%
   /model.22/cv1/conv/Conv:                     | ███████              | 9.093%
   /model.10/m/m.0/attn/MatMul_1:               | ███████              | 8.414%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ███████              | 8.245%
   /model.23/cv2.2/cv2.2.2/Conv:                | ███████              | 8.208%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 8.031%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ██████               | 7.818%
   /model.13/m.0/cv2/conv/Conv:                 | ██████               | 7.717%
   /model.19/m.0/cv2/conv/Conv:                 | ██████               | 7.404%
   /model.20/conv/Conv:                         | ██████               | 7.161%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████               | 7.080%
   /model.10/m/m.0/attn/pe/conv/Conv:           | █████                | 6.814%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 6.764%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████                | 6.539%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████                | 6.418%
   /model.19/cv2/conv/Conv:                     | █████                | 6.206%
   /model.13/cv2/conv/Conv:                     | █████                | 5.894%
   /model.10/cv1/conv/Conv:                     | █████                | 5.757%
   /model.10/cv2/conv/Conv:                     | █████                | 5.716%
   /model.19/cv1/conv/Conv:                     | ████                 | 5.279%
   /model.22/m.0/cv1/conv/Conv:                 | ████                 | 5.072%
   /model.19/m.0/cv1/conv/Conv:                 | ████                 | 5.036%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ████                 | 4.979%
   /model.8/m.0/cv2/conv/Conv:                  | ████                 | 4.862%
   /model.10/m/m.0/attn/MatMul:                 | ████                 | 4.670%
   /model.13/cv1/conv/Conv:                     | ████                 | 4.594%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 4.441%
   /model.23/cv2.0/cv2.0.2/Conv:                | ███                  | 4.308%
   /model.13/m.0/cv1/conv/Conv:                 | ███                  | 4.278%
   /model.23/cv2.1/cv2.1.2/Conv:                | ███                  | 4.214%
   /model.6/m.0/cv2/conv/Conv:                  | ███                  | 4.031%
   /model.17/conv/Conv:                         | ███                  | 3.760%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 3.521%
   /model.8/m.0/cv1/conv/Conv:                  | ███                  | 3.227%
   /model.16/m.0/cv1/conv/Conv:                 | ██                   | 3.185%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██                   | 3.178%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██                   | 3.150%
   /model.8/cv2/conv/Conv:                      | ██                   | 3.067%
   /model.8/m.0/cv3/conv/Conv:                  | ██                   | 3.067%
   /model.16/cv2/conv/Conv:                     | ██                   | 3.054%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.053%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.049%
   /model.6/m.0/cv3/conv/Conv:                  | ██                   | 3.049%
   /model.8/cv1/conv/Conv:                      | ██                   | 2.984%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 2.934%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██                   | 2.794%
   /model.6/cv1/conv/Conv:                      | ██                   | 2.783%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.753%
   /model.2/cv1/conv/Conv:                      | ██                   | 2.697%
   /model.6/cv2/conv/Conv:                      | ██                   | 2.616%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.596%
   /model.9/cv2/conv/Conv:                      | ██                   | 2.500%
   /model.3/conv/Conv:                          | ██                   | 2.499%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 2.469%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ██                   | 2.235%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.233%
   /model.4/cv2/conv/Conv:                      | ██                   | 2.150%
   /model.7/conv/Conv:                          | ██                   | 2.075%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 2.069%
   /model.5/conv/Conv:                          | ██                   | 1.998%
   /model.16/cv1/conv/Conv:                     | █                    | 1.899%
   /model.4/cv1/conv/Conv:                      | █                    | 1.808%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 1.741%
   /model.6/m.0/cv1/conv/Conv:                  | █                    | 1.734%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | █                    | 1.523%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.248%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 0.875%
   /model.23/cv3.2/cv3.2.2/Conv:                | █                    | 0.784%
   /model.1/conv/Conv:                          | █                    | 0.781%
   PPQ_Operation_2:                             |                      | 0.698%
   /model.9/cv1/conv/Conv:                      |                      | 0.680%
   /model.2/m.0/cv1/conv/Conv:                  |                      | 0.508%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.360%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.189%
   PPQ_Operation_0:                             |                      | 0.110%
   /model.0/conv/Conv:                          |                      | 0.099%
   Analysing Layerwise quantization error:: 100%|██| 91/91 [12:32<00:00,  8.27s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/cv1/conv/Conv:                     | ████████████████████ | 0.244%
   /model.9/cv2/conv/Conv:                      | █████████████        | 0.156%
   /model.10/cv1/conv/Conv:                     | ███████████          | 0.132%
   /model.1/conv/Conv:                          | ██████               | 0.077%
   /model.4/cv1/conv/Conv:                      | ██████               | 0.074%
   /model.16/cv1/conv/Conv:                     | █████                | 0.066%
   /model.0/conv/Conv:                          | █████                | 0.061%
   /model.2/cv1/conv/Conv:                      | █████                | 0.060%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████                 | 0.052%
   PPQ_Operation_0:                             | ████                 | 0.047%
   /model.2/m.0/cv1/conv/Conv:                  | ████                 | 0.045%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ██                   | 0.029%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 0.029%
   /model.6/cv1/conv/Conv:                      | ██                   | 0.025%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | ██                   | 0.023%
   /model.16/cv2/conv/Conv:                     | ██                   | 0.021%
   /model.16/m.0/cv2/conv/Conv:                 | ██                   | 0.020%
   /model.19/m.0/cv1/conv/Conv:                 | ██                   | 0.020%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.018%
   /model.19/cv2/conv/Conv:                     | █                    | 0.017%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.016%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | █                    | 0.016%
   /model.19/cv1/conv/Conv:                     | █                    | 0.015%
   /model.13/cv2/conv/Conv:                     | █                    | 0.015%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | █                    | 0.013%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | █                    | 0.012%
   /model.13/cv1/conv/Conv:                     | █                    | 0.012%
   /model.6/cv2/conv/Conv:                      | █                    | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 | █                    | 0.011%
   /model.8/cv1/conv/Conv:                      | █                    | 0.010%
   /model.13/m.0/cv2/conv/Conv:                 | █                    | 0.010%
   /model.5/conv/Conv:                          | █                    | 0.010%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | █                    | 0.009%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █                    | 0.008%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | █                    | 0.008%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | █                    | 0.008%
   /model.19/m.0/cv2/conv/Conv:                 | █                    | 0.008%
   /model.8/cv2/conv/Conv:                      | █                    | 0.008%
   /model.9/cv1/conv/Conv:                      | █                    | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █                    | 0.007%
   /model.16/m.0/cv1/conv/Conv:                 | █                    | 0.007%
   /model.17/conv/Conv:                         | █                    | 0.007%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | █                    | 0.007%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | █                    | 0.007%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.006%
   /model.10/cv2/conv/Conv:                     |                      | 0.006%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.006%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.005%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.005%
   /model.22/cv2/conv/Conv:                     |                      | 0.005%
   /model.7/conv/Conv:                          |                      | 0.004%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.004%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.004%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.003%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.003%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.003%
   PPQ_Operation_2:                             |                      | 0.003%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.003%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.002%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.002%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.002%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.001%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.001%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.001%
   /model.2/cv2/conv/Conv:                      |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.20/conv/Conv:                         |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%
   /model.3/conv/Conv:                          |                      | 0.000%
   /model.4/cv2/conv/Conv:                      |                      | 0.000%

**Quantization error analysis**

After using 16-bits quantization on layers with higher layer-wise error and employing horizontal layer split pass, the quantized model's mAP50:95 on COCO val2017 improves to 33.3% with the same inputs. Additionally, a noticeable decrease in cumulative error of output layers can be observed. 

The graphwise error for the output layers of the model, /model.23/cv3.2/cv3.2.2/Conv, /model.23/cv2.2/cv2.2.2/Conv, /model.23/cv3.1/cv3.1.2/Conv, /model.23/cv2.1/cv2.1.2/Conv, /model.23/cv3.0/cv3.0.2/Conv and /model.23/cv2.0/cv2.0.2/Conv, are 0.784%, 8.208%, 0.360%, 4.214%, 0.189% and 4.308% respectively.

.. _quantization_aware_label:

Quantization-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To further improve the accuracy of the quantized model, we adopt the quantization-aware training(QAT) strategy. Here, QAT is performed based on 8-bit quantization.

**Quantization settings**

- :project_file:`yolo11n_qat.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n/yolo11n_qat.py>`
- :project_file:`trainer.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n/trainer.py>`

**Quantization results**

.. code-block::

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████████ | 23.754%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ██████████████       | 16.118%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | █████████            | 10.878%
   /model.8/m.0/cv2/conv/Conv:                  | █████████            | 10.527%
   /model.22/m.0/cv3/conv/Conv:                 | █████████            | 10.298%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████            | 10.188%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ████████             | 10.093%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████             | 9.891%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | ████████             | 9.839%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ████████             | 9.827%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████             | 9.658%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ████████             | 9.168%
   /model.22/m.0/cv2/conv/Conv:                 | ███████              | 8.604%
   /model.10/m/m.0/attn/MatMul_1:               | ███████              | 8.596%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ███████              | 8.541%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ███████              | 8.528%
   /model.22/cv2/conv/Conv:                     | ███████              | 8.442%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ███████              | 8.306%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ███████              | 8.015%
   /model.10/cv1/conv/Conv:                     | ███████              | 7.998%
   /model.22/cv1/conv/Conv:                     | ██████               | 7.307%
   /model.8/cv1/conv/Conv:                      | ██████               | 7.265%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ██████               | 6.989%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████               | 6.716%
   /model.6/m.0/cv2/conv/Conv:                  | █████                | 6.595%
   /model.2/cv2/conv/Conv:                      | █████                | 6.131%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | █████                | 6.078%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 6.055%
   /model.19/m.0/cv2/conv/Conv:                 | █████                | 5.999%
   /model.8/m.0/cv1/conv/Conv:                  | █████                | 5.919%
   /model.13/m.0/cv2/conv/Conv:                 | █████                | 5.863%
   /model.20/conv/Conv:                         | █████                | 5.638%
   /model.8/cv2/conv/Conv:                      | █████                | 5.616%
   /model.10/cv2/conv/Conv:                     | █████                | 5.464%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | █████                | 5.443%
   /model.2/m.0/cv2/conv/Conv:                  | ████                 | 5.426%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ████                 | 5.390%
   /model.13/cv2/conv/Conv:                     | ████                 | 5.256%
   /model.19/cv2/conv/Conv:                     | ████                 | 5.231%
   /model.13/cv1/conv/Conv:                     | ████                 | 5.131%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ████                 | 5.122%
   /model.6/cv1/conv/Conv:                      | ████                 | 5.049%
   /model.6/cv2/conv/Conv:                      | ████                 | 4.788%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 4.706%
   /model.19/cv1/conv/Conv:                     | ████                 | 4.586%
   /model.7/conv/Conv:                          | ████                 | 4.586%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 4.541%
   /model.8/m.0/cv3/conv/Conv:                  | ████                 | 4.529%
   /model.3/conv/Conv:                          | ████                 | 4.361%
   /model.13/m.0/cv1/conv/Conv:                 | ████                 | 4.359%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | ████                 | 4.328%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 4.156%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | ███                  | 4.083%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███                  | 3.998%
   /model.19/m.0/cv1/conv/Conv:                 | ███                  | 3.974%
   /model.23/cv2.2/cv2.2.2/Conv:                | ███                  | 3.817%
   /model.16/m.0/cv1/conv/Conv:                 | ███                  | 3.797%
   /model.16/m.0/cv2/conv/Conv:                 | ███                  | 3.654%
   /model.4/cv1/conv/Conv:                      | ███                  | 3.544%
   /model.4/cv2/conv/Conv:                      | ███                  | 3.488%
   /model.22/m.0/cv1/conv/Conv:                 | ███                  | 3.423%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | ███                  | 3.382%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ███                  | 3.299%
   /model.17/conv/Conv:                         | ███                  | 3.296%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 3.267%
   /model.5/conv/Conv:                          | ███                  | 3.147%
   /model.23/cv2.1/cv2.1.2/Conv:                | ███                  | 3.102%
   /model.16/cv2/conv/Conv:                     | ███                  | 3.091%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ███                  | 3.080%
   /model.23/cv2.0/cv2.0.2/Conv:                | ██                   | 3.056%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ██                   | 2.989%
   /model.2/cv1/conv/Conv:                      | ██                   | 2.874%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██                   | 2.843%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 2.819%
   /model.9/cv2/conv/Conv:                      | ██                   | 2.662%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 2.633%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██                   | 2.581%
   /model.4/m.0/cv1/conv/Conv:                  | ██                   | 2.545%
   /model.16/cv1/conv/Conv:                     | ██                   | 2.171%
   /model.4/m.0/cv2/conv/Conv:                  | ██                   | 1.942%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 1.925%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.721%
   /model.9/cv1/conv/Conv:                      | █                    | 1.140%
   /model.1/conv/Conv:                          | █                    | 1.117%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █                    | 0.831%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.443%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.247%
   /model.0/conv/Conv:                          |                      | 0.150%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|██████████| 89/89 [04:44<00:00,  3.20s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.2/cv2/conv/Conv:                      | ████████████████████ | 1.462%
   /model.3/conv/Conv:                          | ██████████           | 0.764%
   /model.4/cv2/conv/Conv:                      | ██████████           | 0.763%
   /model.10/cv2/conv/Conv:                     | ███████              | 0.535%
   /model.9/cv2/conv/Conv:                      | ██████               | 0.439%
   /model.2/cv1/conv/Conv:                      | █████                | 0.395%
   /model.4/cv1/conv/Conv:                      | █████                | 0.361%
   /model.1/conv/Conv:                          | █████                | 0.347%
   /model.2/m.0/cv1/conv/Conv:                  | ███                  | 0.192%
   /model.4/m.0/cv2/conv/Conv:                  | ███                  | 0.184%
   /model.22/cv1/conv/Conv:                     | ██                   | 0.179%
   /model.5/conv/Conv:                          | ██                   | 0.161%
   /model.16/cv1/conv/Conv:                     | ██                   | 0.154%
   /model.10/cv1/conv/Conv:                     | ██                   | 0.145%
   /model.16/m.0/cv2/conv/Conv:                 | ██                   | 0.142%
   /model.16/m.0/cv1/conv/Conv:                 | ██                   | 0.113%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.107%
   /model.0/conv/Conv:                          | █                    | 0.100%
   /model.10/m/m.0/attn/pe/conv/Conv:           | █                    | 0.095%
   /model.6/cv1/conv/Conv:                      | █                    | 0.082%
   /model.23/cv2.2/cv2.2.2/Conv:                | █                    | 0.082%
   /model.16/cv2/conv/Conv:                     | █                    | 0.076%
   /model.6/cv2/conv/Conv:                      | █                    | 0.066%
   /model.22/m.0/cv1/conv/Conv:                 | █                    | 0.060%
   /model.13/cv2/conv/Conv:                     | █                    | 0.056%
   /model.19/cv2/conv/Conv:                     | █                    | 0.041%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.034%
   /model.7/conv/Conv:                          |                      | 0.033%
   /model.13/cv1/conv/Conv:                     |                      | 0.033%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.032%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.032%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.029%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.029%
   /model.2/m.0/cv2/conv/Conv:                  |                      | 0.026%
   /model.19/cv1/conv/Conv:                     |                      | 0.025%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.024%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.024%
   /model.17/conv/Conv:                         |                      | 0.023%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.021%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.019%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.019%
   /model.9/cv1/conv/Conv:                      |                      | 0.017%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.015%
   /model.8/cv1/conv/Conv:                      |                      | 0.014%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.014%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.014%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.012%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.011%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.011%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.010%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.009%
   /model.20/conv/Conv:                         |                      | 0.009%
   /model.8/cv2/conv/Conv:                      |                      | 0.009%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.008%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.008%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.008%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.008%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.007%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.007%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.007%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.007%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.007%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.006%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.005%
   /model.22/cv2/conv/Conv:                     |                      | 0.005%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.004%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.004%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.003%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.003%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.003%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.003%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.003%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.003%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.002%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.002%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.002%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.002%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.002%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.002%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.000%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%


**Quantization error analysis**

After applying QAT to 8-bit quantization, the quantized model's mAP50:95 on COCO val2017 improves to 35.0% with the same inputs, while cumulative errors of out layers are significantly reduced. Compared to the other two quantization methods, the 8-bit QAT quantized model achieves the highest quantization accuracy with the lowest inference latency.

The graphwise error for the output layers of the model, /model.23/cv3.2/cv3.2.2/Conv, /model.23/cv2.2/cv2.2.2/Conv, /model.23/cv3.1/cv3.1.2/Conv, /model.23/cv2.1/cv2.1.2/Conv, /model.23/cv3.0/cv3.0.2/Conv and /model.23/cv2.0/cv2.0.2/Conv, are 0.443%, 3.817%, 0.247%, 3.102%, 0.119% and 3.056% respectively.

.. note::

   If the model inference speed is a higher priority and a certain degree of accuracy loss is acceptable, you may consider quantizing the model with an input size of 320x320 for the YOLO11N model. The model inference speed of different input resolutions can be found in :project_file:`README.md <models/coco_detect/README.md>` .


Model deployment
-----------------------

:project:`example <examples/yolo11_detect>`

Object detection base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_detect_base.hpp <esp-dl/vision/detect/dl_detect_base.hpp>`
- :project_file:`dl_detect_base.cpp <esp-dl/vision/detect/dl_detect_base.cpp>`

Pre-process
^^^^^^^^^^^^^^^^^^^^

``ImagePreprocessor`` class contains the common pre-precoess pipeline, ``color conversion``, ``crop``, ``resize``, ``normalization``, ``quantize``。

- :project_file:`dl_image_preprocessor.hpp <esp-dl/vision/image/dl_image_preprocessor.hpp>`
- :project_file:`dl_image_preprocessor.cpp <esp-dl/vision/image/dl_image_preprocessor.cpp>`

Post-process
^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_detect_postprocessor.hpp <esp-dl/vision/detect/dl_detect_postprocessor.hpp>`
- :project_file:`dl_detect_postprocessor.cpp <esp-dl/vision/detect/dl_detect_postprocessor.cpp>`
- :project_file:`dl_detect_yolo11_postprocessor.hpp <esp-dl/vision/detect/dl_detect_yolo11_postprocessor.hpp>`
- :project_file:`dl_detect_yolo11_postprocessor.cpp <esp-dl/vision/detect/dl_detect_yolo11_postprocessor.cpp>`

