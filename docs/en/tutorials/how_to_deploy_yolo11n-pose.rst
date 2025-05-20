How to deploy YOLO11n-pose
============================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will introduce how to quantize a pre-trained YOLO11n-pose model using ESP-PPQ and deploy the quantized YOLO11n-pose model using ESP-DL.

.. contents::
  :local:
  :depth: 2

Preparation
----------------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_yolo11n-pose:

Model quantization
------------------------

Pre-trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can download pre-trained yolo11n-pose model from `Ultralytics release <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt>`__.

Currently, ESP-PPQ supports ONNX, PyTorch, and TensorFlow models. During the quantization process, PyTorch and TensorFlow models are first converted to ONNX models, so the pre-trained yolo11n-pose model needs to be converted to an ONNX model.

Specificially, refer to the script :project_file:`export_onnx.py <models/coco_pose/models/export_onnx.py>` to convert the pre-trained yolo11n-pose model to an ONNX model.

In the srcipt, we have overridden the forward method of the Pose class, which offers following advantages:

- Faster inference. Compared to the original yolo11n-pose model, operations related to decoding bounding boxes and keypoints in Pose head are moved from the inference pass to the post-processing phase, resulting in a significant reduction in inference latency. On one hand, operations like ``Conv``, ``Transpose``, ``Slice``, ``Split`` and ``Concat`` are time-consuming when applied during inference pass. On the other hand, the inference outputs are first filtered using a score threshold before decoding the boxes in the post-processing pass, which significantly reduces the number of calculations, thereby acclerating the overall inference speed.

- Lower quantization Error. The ``Concat`` and ``Add`` operators adopt joint quantization in ESP-PPQ. To reduce quantization errors, the box and score are output by separate branches, rather than being concatenated, due to the significant difference in their ranges. Similarly, since the ranges of the two inputs of ``Add`` and ``Sub`` differ significantly, the calculations are performed in the post-processing phase to avoid quantization errors.


Calibration Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration dataset needs to match the input format of the model. The calibration dataset should cover all possible input scenarios to better quantize the model. Here, the calibration dataset used in this example is `calib_yolo11n-pose <https://dl.espressif.com/public/calib_yolo11n-pose.zip>`__.

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

   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 29.088%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ███████████████████  | 27.302%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ██████████████████   | 26.342%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████████████████   | 25.969%
   /model.20/conv/Conv:                         | █████████████████    | 24.730%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████████    | 24.264%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | ███████████████      | 22.389%
   /model.19/m.0/cv2/conv/Conv:                 | ███████████████      | 22.368%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ███████████████      | 22.253%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ███████████████      | 21.697%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ███████████████      | 21.245%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ██████████████       | 21.086%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████████████       | 20.461%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████████       | 20.134%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 20.002%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ██████████████       | 19.908%
   /model.22/m.0/cv3/conv/Conv:                 | ██████████████       | 19.885%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | █████████████        | 19.304%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████████        | 19.237%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | █████████████        | 19.148%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | █████████████        | 18.741%
   /model.22/cv1/conv/Conv:                     | █████████████        | 18.517%
   /model.19/cv2/conv/Conv:                     | █████████████        | 18.392%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████████         | 17.707%
   /model.17/conv/Conv:                         | ████████████         | 17.268%
   /model.19/cv1/conv/Conv:                     | ████████████         | 17.171%
   /model.22/cv2/conv/Conv:                     | ████████████         | 16.800%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | ███████████          | 16.429%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ███████████          | 16.023%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ███████████          | 15.607%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ███████████          | 15.490%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 15.414%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ██████████           | 15.284%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ██████████           | 15.190%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████████           | 15.014%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ██████████           | 14.867%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 14.687%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 13.101%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 13.023%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████████            | 12.945%
   /model.10/m/m.0/attn/pe/conv/Conv:           | █████████            | 12.775%
   /model.23/cv4.1/cv4.1.2/Conv:                | ████████             | 12.265%
   /model.13/cv2/conv/Conv:                     | ████████             | 11.931%
   /model.23/cv4.2/cv4.2.2/Conv:                | ████████             | 11.831%
   /model.13/m.0/cv1/conv/Conv:                 | ████████             | 11.601%
   /model.16/cv2/conv/Conv:                     | ███████              | 10.853%
   /model.13/cv1/conv/Conv:                     | ███████              | 10.803%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███████              | 10.393%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 10.237%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 10.217%
   /model.8/m.0/cv2/conv/Conv:                  | ███████              | 9.646%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██████               | 8.986%
   /model.10/cv1/conv/Conv:                     | ██████               | 8.670%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ██████               | 8.661%
   /model.16/m.0/cv1/conv/Conv:                 | ██████               | 8.658%
   /model.10/m/m.0/attn/MatMul_1:               | ██████               | 8.325%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 8.269%
   /model.16/cv1/conv/Conv:                     | █████                | 7.914%
   /model.10/cv2/conv/Conv:                     | █████                | 7.722%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █████                | 7.590%
   /model.8/cv2/conv/Conv:                      | █████                | 7.585%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | █████                | 7.046%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 7.027%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 6.661%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 6.307%
   /model.8/cv1/conv/Conv:                      | ████                 | 6.270%
   /model.23/cv2.1/cv2.1.2/Conv:                | ████                 | 6.265%
   /model.23/cv3.2/cv3.2.2/Conv:                | ████                 | 5.813%
   /model.6/m.0/cv2/conv/Conv:                  | ████                 | 5.743%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 5.674%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 5.361%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 5.302%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 4.913%
   /model.8/m.0/cv1/conv/Conv:                  | ███                  | 4.662%
   /model.6/cv1/conv/Conv:                      | ███                  | 4.585%
   /model.9/cv1/conv/Conv:                      | ███                  | 3.981%
   /model.7/conv/Conv:                          | ██                   | 3.646%
   /model.9/cv2/conv/Conv:                      | ██                   | 3.638%
   /model.6/cv2/conv/Conv:                      | ██                   | 3.517%
   /model.3/conv/Conv:                          | ██                   | 3.372%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.293%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.111%
   /model.4/cv1/conv/Conv:                      | ██                   | 2.925%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 2.880%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.706%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.577%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 2.504%
   /model.5/conv/Conv:                          | ██                   | 2.437%
   /model.2/cv1/conv/Conv:                      | ██                   | 2.321%
   /model.4/cv2/conv/Conv:                      | █                    | 2.144%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 2.106%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 1.901%
   /model.1/conv/Conv:                          | █                    | 1.735%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.664%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 1.408%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.040%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.365%
   /model.0/conv/Conv:                          |                      | 0.087%
   Analysing Layerwise quantization error:: 100%|████████████████████████████████████████████████████| 98/98 [00:48<00:00,  2.03it/s]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.0/conv/Conv:                          | ████████████████████ | 1.017%
   /model.9/cv2/conv/Conv:                      | ██████████           | 0.493%
   /model.8/cv1/conv/Conv:                      | ████████             | 0.410%
   /model.2/cv2/conv/Conv:                      | ██████               | 0.287%
   /model.1/conv/Conv:                          | ████                 | 0.228%
   /model.2/cv1/conv/Conv:                      | ███                  | 0.163%
   /model.16/cv2/conv/Conv:                     | ███                  | 0.130%
   /model.4/cv2/conv/Conv:                      | ██                   | 0.096%
   /model.3/conv/Conv:                          | █                    | 0.070%
   /model.4/cv1/conv/Conv:                      | █                    | 0.068%
   /model.10/cv1/conv/Conv:                     | █                    | 0.049%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.047%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.043%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.041%
   /model.13/cv2/conv/Conv:                     | █                    | 0.037%
   /model.16/cv1/conv/Conv:                     | █                    | 0.030%
   /model.22/cv2/conv/Conv:                     | █                    | 0.027%
   /model.8/cv2/conv/Conv:                      | █                    | 0.027%
   /model.13/cv1/conv/Conv:                     |                      | 0.025%
   /model.5/conv/Conv:                          |                      | 0.025%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.025%
   /model.6/cv2/conv/Conv:                      |                      | 0.024%
   /model.4/m.0/cv1/conv/Conv:                  |                      | 0.022%
   /model.6/cv1/conv/Conv:                      |                      | 0.021%
   /model.19/cv1/conv/Conv:                     |                      | 0.020%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.018%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           |                      | 0.017%
   /model.9/cv1/conv/Conv:                      |                      | 0.015%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.014%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.014%
   /model.19/cv2/conv/Conv:                     |                      | 0.014%
   /model.16/m.0/cv2/conv/Conv:                 |                      | 0.014%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           |                      | 0.014%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.013%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.013%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.013%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.013%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.013%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.012%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.012%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.012%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           |                      | 0.011%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.011%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.011%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.011%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.011%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.010%
   /model.7/conv/Conv:                          |                      | 0.010%
   /model.17/conv/Conv:                         |                      | 0.009%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.009%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.009%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.009%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.008%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.008%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.008%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.008%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.008%
   /model.10/cv2/conv/Conv:                     |                      | 0.007%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.007%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.007%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.007%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.006%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.006%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.006%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.005%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.22/cv1/conv/Conv:                     |                      | 0.004%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.004%
   /model.23/cv4.2/cv4.2.2/Conv:                |                      | 0.004%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.004%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.004%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.003%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.003%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.003%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.003%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.002%
   /model.20/conv/Conv:                         |                      | 0.002%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.000%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.000%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000% 

**Quantization error analysis**

With the same inputs, The Pose mAP50:95 on COCO after quantization is only 42.9%, which is lower than that of the float model（50.0%）. 


.. _quantization_aware_pose_label:

Quantization-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To further improve the accuracy of the quantized model, we adopt the quantization-aware training(QAT) strategy. Here, QAT is performed based on 8-bit quantization.

**Quantization settings**

- :project_file:`yolo11n-pose_qat.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/yolo11n-pose_qat.py>`
- :project_file:`trainer.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/trainer.py>`

**Quantization results**

.. code-block::
   
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 29.427%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ███████████████████  | 28.661%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ███████████████████  | 27.500%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ██████████████████   | 27.128%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████████████████   | 26.522%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █████████████████    | 25.263%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████████    | 25.103%
   /model.20/conv/Conv:                         | █████████████████    | 24.669%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | █████████████████    | 24.407%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | █████████████████    | 24.301%
   /model.19/m.0/cv2/conv/Conv:                 | ███████████████      | 22.689%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ███████████████      | 22.297%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ███████████████      | 22.235%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ███████████████      | 21.825%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ███████████████      | 21.686%
   /model.22/m.0/cv3/conv/Conv:                 | ███████████████      | 21.669%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ███████████████      | 21.551%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 21.208%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██████████████       | 21.207%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████████       | 20.239%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | ██████████████       | 19.969%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████████        | 19.811%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | █████████████        | 19.416%
   /model.22/cv1/conv/Conv:                     | █████████████        | 18.922%
   /model.19/cv2/conv/Conv:                     | █████████████        | 18.922%
   /model.10/m/m.0/attn/proj/conv/Conv:         | █████████████        | 18.593%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | █████████████        | 18.513%
   /model.19/cv1/conv/Conv:                     | ████████████         | 18.177%
   /model.22/cv2/conv/Conv:                     | ████████████         | 18.077%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ████████████         | 17.960%
   /model.17/conv/Conv:                         | ████████████         | 17.859%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 17.706%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████████         | 17.129%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ████████████         | 17.100%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ███████████          | 16.427%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ███████████          | 16.426%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ███████████          | 16.309%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 14.961%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | ██████████           | 14.286%
   /model.23/cv4.2/cv4.2.2/Conv:                | █████████            | 13.920%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 13.769%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 13.733%
   /model.23/cv4.1/cv4.1.2/Conv:                | █████████            | 13.589%
   /model.13/cv1/conv/Conv:                     | █████████            | 12.879%
   /model.10/m/m.0/attn/MatMul_1:               | █████████            | 12.677%
   /model.13/cv2/conv/Conv:                     | ████████             | 12.362%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ████████             | 12.167%
   /model.10/cv1/conv/Conv:                     | ████████             | 11.927%
   /model.13/m.0/cv1/conv/Conv:                 | ████████             | 11.909%
   /model.16/cv2/conv/Conv:                     | ████████             | 11.742%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████████             | 11.469%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ████████             | 11.240%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 10.769%
   /model.16/m.0/cv1/conv/Conv:                 | ███████              | 10.585%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 10.513%
   /model.8/m.0/cv2/conv/Conv:                  | ███████              | 10.475%
   /model.10/cv2/conv/Conv:                     | ██████               | 8.883%
   /model.23/cv2.1/cv2.1.2/Conv:                | ██████               | 8.799%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | ██████               | 8.574%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 8.389%
   /model.16/cv1/conv/Conv:                     | ██████               | 8.319%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | ██████               | 8.244%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | █████                | 7.842%
   /model.8/cv2/conv/Conv:                      | █████                | 7.821%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 7.740%
   /model.8/cv1/conv/Conv:                      | █████                | 7.427%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | █████                | 7.362%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | █████                | 6.731%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 6.659%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 6.448%
   /model.6/m.0/cv2/conv/Conv:                  | ████                 | 6.250%
   /model.6/m.0/cv3/conv/Conv:                  | ████                 | 6.113%
   /model.8/m.0/cv1/conv/Conv:                  | ████                 | 5.575%
   /model.23/cv3.2/cv3.2.2/Conv:                | ████                 | 5.471%
   /model.6/cv1/conv/Conv:                      | ████                 | 5.357%
   /model.9/cv2/conv/Conv:                      | ███                  | 5.128%
   /model.9/cv1/conv/Conv:                      | ███                  | 4.499%
   /model.3/conv/Conv:                          | ███                  | 4.151%
   /model.7/conv/Conv:                          | ███                  | 4.149%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ███                  | 4.037%
   /model.6/cv2/conv/Conv:                      | ███                  | 3.855%
   /model.4/cv1/conv/Conv:                      | ██                   | 3.610%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.538%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.504%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.333%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 3.285%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 3.008%
   /model.4/cv2/conv/Conv:                      | ██                   | 2.816%
   /model.5/conv/Conv:                          | ██                   | 2.787%
   /model.4/m.0/cv1/conv/Conv:                  | ██                   | 2.775%
   /model.2/cv1/conv/Conv:                      | ██                   | 2.455%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 2.363%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 2.013%
   /model.1/conv/Conv:                          | █                    | 1.755%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.642%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.296%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.762%
   /model.0/conv/Conv:                          |                      | 0.066%
   Analysing Layerwise quantization error:: 100%|███████████████████████████████████████████████████████████| 98/98 [00:49<00:00,  1.99it/s]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.9/cv2/conv/Conv:                      | ████████████████████ | 3.566%
   /model.2/cv2/conv/Conv:                      | ███████████          | 1.952%
   /model.3/conv/Conv:                          | ██████               | 1.071%
   /model.2/cv1/conv/Conv:                      | █████                | 0.891%
   /model.1/conv/Conv:                          | ███                  | 0.523%
   /model.4/cv2/conv/Conv:                      | ███                  | 0.462%
   /model.8/cv1/conv/Conv:                      | ██                   | 0.417%
   /model.2/m.0/cv2/conv/Conv:                  | ██                   | 0.344%
   /model.5/conv/Conv:                          | ██                   | 0.326%
   /model.6/m.0/cv3/conv/Conv:                  | ██                   | 0.298%
   /model.0/conv/Conv:                          | ██                   | 0.290%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.206%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.201%
   /model.13/cv2/conv/Conv:                     | █                    | 0.175%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | █                    | 0.159%
   /model.16/cv2/conv/Conv:                     | █                    | 0.137%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | █                    | 0.137%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | █                    | 0.107%
   /model.23/cv2.2/cv2.2.2/Conv:                | █                    | 0.102%
   /model.4/cv1/conv/Conv:                      | █                    | 0.101%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.091%
   /model.23/cv4.2/cv4.2.2/Conv:                | █                    | 0.091%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.087%
   /model.19/cv2/conv/Conv:                     |                      | 0.081%
   /model.7/conv/Conv:                          |                      | 0.074%
   /model.6/cv1/conv/Conv:                      |                      | 0.071%
   /model.17/conv/Conv:                         |                      | 0.070%
   /model.6/cv2/conv/Conv:                      |                      | 0.069%
   /model.16/cv1/conv/Conv:                     |                      | 0.068%
   /model.22/cv2/conv/Conv:                     |                      | 0.054%
   /model.10/cv1/conv/Conv:                     |                      | 0.050%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.049%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.047%
   /model.19/cv1/conv/Conv:                     |                      | 0.046%
   /model.10/cv2/conv/Conv:                     |                      | 0.046%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.041%
   /model.8/cv2/conv/Conv:                      |                      | 0.041%
   /model.13/cv1/conv/Conv:                     |                      | 0.037%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.036%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.035%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.033%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.033%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.033%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.032%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.031%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.028%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.028%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           |                      | 0.027%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.027%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.027%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.025%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.023%
   /model.9/cv1/conv/Conv:                      |                      | 0.022%
   /model.16/m.0/cv2/conv/Conv:                 |                      | 0.020%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.020%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.020%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.019%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.019%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.018%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.018%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.015%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.015%
   /model.20/conv/Conv:                         |                      | 0.013%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.013%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.013%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.011%
   /model.22/cv1/conv/Conv:                     |                      | 0.011%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.011%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.010%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.009%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.009%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.008%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.008%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.008%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.007%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.007%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.007%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.006%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.005%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.005%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.004%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.004%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.003%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.003%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.003%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.002%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.001%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%


**Quantization error analysis**

After applying QAT to 8-bit quantization, the quantized model's Pose mAP50:95 on COCO improves to 45.4% with the same inputs, while cumulative errors of out layers are significantly reduced. Compared to the other two quantization methods, the 8-bit QAT quantized model achieves the highest quantization accuracy with the lowest inference latency.


Model deployment
-----------------------

:project:`example <examples/yolo11_pose>`

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
- :project_file:`dl_pose_yolo11_postprocessor.hpp <esp-dl/vision/detect/dl_pose_yolo11_postprocessor.hpp>`
- :project_file:`dl_pose_yolo11_postprocessor.cpp <esp-dl/vision/detect/dl_pose_yolo11_postprocessor.cpp>`

