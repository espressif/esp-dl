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
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 29.305%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ██████████████████   | 26.959%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ██████████████████   | 26.555%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | █████████████████    | 25.611%
   /model.20/conv/Conv:                         | █████████████████    | 24.738%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | ████████████████     | 24.122%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | ███████████████      | 22.512%
   /model.19/m.0/cv2/conv/Conv:                 | ███████████████      | 22.397%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | ███████████████      | 22.174%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ███████████████      | 21.621%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ███████████████      | 21.489%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ███████████████      | 21.445%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ██████████████       | 20.528%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ██████████████       | 20.083%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 20.066%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████████       | 20.042%
   /model.22/m.0/cv3/conv/Conv:                 | █████████████        | 19.737%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | █████████████        | 19.585%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | █████████████        | 19.392%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | █████████████        | 18.773%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████████        | 18.688%
   /model.22/cv1/conv/Conv:                     | █████████████        | 18.579%
   /model.19/cv2/conv/Conv:                     | █████████████        | 18.494%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | ████████████         | 17.576%
   /model.17/conv/Conv:                         | ████████████         | 17.224%
   /model.19/cv1/conv/Conv:                     | ████████████         | 17.140%
   /model.22/cv2/conv/Conv:                     | ███████████          | 16.785%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | ███████████          | 16.375%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ███████████          | 16.167%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ███████████          | 15.655%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 15.504%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ███████████          | 15.431%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ██████████           | 15.251%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ██████████           | 15.171%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ██████████           | 15.006%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 14.692%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ██████████           | 14.548%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 13.065%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 12.980%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████████            | 12.921%
   /model.10/m/m.0/attn/pe/conv/Conv:           | █████████            | 12.745%
   /model.23/cv4.1/cv4.1.2/Conv:                | ████████             | 12.498%
   /model.13/cv2/conv/Conv:                     | ████████             | 11.932%
   /model.23/cv4.2/cv4.2.2/Conv:                | ████████             | 11.797%
   /model.13/m.0/cv1/conv/Conv:                 | ████████             | 11.777%
   /model.16/cv2/conv/Conv:                     | ███████              | 10.892%
   /model.13/cv1/conv/Conv:                     | ███████              | 10.760%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ███████              | 10.352%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 10.325%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 10.257%
   /model.8/m.0/cv2/conv/Conv:                  | ███████              | 9.687%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ██████               | 8.997%
   /model.10/cv1/conv/Conv:                     | ██████               | 8.787%
   /model.16/m.0/cv1/conv/Conv:                 | ██████               | 8.629%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ██████               | 8.600%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 8.328%
   /model.10/m/m.0/attn/MatMul_1:               | ██████               | 8.293%
   /model.16/cv1/conv/Conv:                     | █████                | 7.947%
   /model.10/cv2/conv/Conv:                     | █████                | 7.824%
   /model.8/cv2/conv/Conv:                      | █████                | 7.696%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █████                | 7.615%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | █████                | 7.145%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 7.033%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 6.707%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | ████                 | 6.376%
   /model.23/cv2.1/cv2.1.2/Conv:                | ████                 | 6.321%
   /model.8/cv1/conv/Conv:                      | ████                 | 6.296%
   /model.6/m.0/cv2/conv/Conv:                  | ████                 | 5.605%
   /model.23/cv3.2/cv3.2.2/Conv:                | ████                 | 5.599%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 5.559%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 5.262%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 5.207%
   /model.6/m.0/cv3/conv/Conv:                  | ███                  | 4.840%
   /model.8/m.0/cv1/conv/Conv:                  | ███                  | 4.667%
   /model.6/cv1/conv/Conv:                      | ███                  | 4.523%
   /model.9/cv1/conv/Conv:                      | ███                  | 4.038%
   /model.9/cv2/conv/Conv:                      | ██                   | 3.733%
   /model.7/conv/Conv:                          | ██                   | 3.605%
   /model.6/cv2/conv/Conv:                      | ██                   | 3.478%
   /model.3/conv/Conv:                          | ██                   | 3.352%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.230%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.136%
   /model.4/cv1/conv/Conv:                      | ██                   | 2.913%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 2.830%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.692%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 2.557%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 2.475%
   /model.5/conv/Conv:                          | ██                   | 2.413%
   /model.2/cv1/conv/Conv:                      | █                    | 2.267%
   /model.4/cv2/conv/Conv:                      | █                    | 2.135%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 2.104%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 1.910%
   /model.1/conv/Conv:                          | █                    | 1.708%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.658%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 1.455%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.056%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.364%
   /model.0/conv/Conv:                          |                      | 0.087%
   Analysing Layerwise quantization error:: 100%|██████████| 98/98 [16:09<00:00,  9.89s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.0/conv/Conv:                          | ████████████████████ | 1.017%
   /model.9/cv2/conv/Conv:                      | ██████████           | 0.493%
   /model.8/cv1/conv/Conv:                      | ████████             | 0.410%
   /model.2/cv2/conv/Conv:                      | ██████               | 0.287%
   /model.1/conv/Conv:                          | ████                 | 0.228%
   /model.2/cv1/conv/Conv:                      | ███                  | 0.163%
   /model.16/cv2/conv/Conv:                     | ███                  | 0.130%
   /model.4/cv2/conv/Conv:                      | ██                   | 0.096%
   /model.3/conv/Conv:                          | █                    | 0.070%
   /model.4/cv1/conv/Conv:                      | █                    | 0.068%
   /model.10/cv1/conv/Conv:                     | █                    | 0.049%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.047%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.043%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 0.041%
   /model.13/cv2/conv/Conv:                     | █                    | 0.037%
   /model.16/cv1/conv/Conv:                     | █                    | 0.030%
   /model.22/cv2/conv/Conv:                     | █                    | 0.027%
   /model.8/cv2/conv/Conv:                      | █                    | 0.027%
   /model.13/cv1/conv/Conv:                     |                      | 0.025%
   /model.5/conv/Conv:                          |                      | 0.025%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.025%
   /model.6/cv2/conv/Conv:                      |                      | 0.024%
   /model.4/m.0/cv1/conv/Conv:                  |                      | 0.022%
   /model.6/cv1/conv/Conv:                      |                      | 0.021%
   /model.19/cv1/conv/Conv:                     |                      | 0.020%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.018%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           |                      | 0.017%
   /model.9/cv1/conv/Conv:                      |                      | 0.015%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.014%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.014%
   /model.19/cv2/conv/Conv:                     |                      | 0.014%
   /model.16/m.0/cv2/conv/Conv:                 |                      | 0.014%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           |                      | 0.014%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.013%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.013%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.013%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.013%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.013%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.012%
   /model.6/m.0/cv3/conv/Conv:                  |                      | 0.012%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.012%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           |                      | 0.011%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.011%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.011%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.011%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.011%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.011%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.010%
   /model.7/conv/Conv:                          |                      | 0.010%
   /model.17/conv/Conv:                         |                      | 0.009%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.009%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.009%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.009%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.008%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.008%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.008%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.008%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.008%
   /model.10/cv2/conv/Conv:                     |                      | 0.007%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.007%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.007%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.007%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.006%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.006%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.006%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.005%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.005%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.005%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.23/cv2.2/cv2.2.2/Conv:                |                      | 0.005%
   /model.22/cv1/conv/Conv:                     |                      | 0.004%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.004%
   /model.23/cv4.2/cv4.2.2/Conv:                |                      | 0.004%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.004%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.004%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.003%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.003%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.003%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.003%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.002%
   /model.20/conv/Conv:                         |                      | 0.002%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.000%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.000%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.000%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%

**Quantization error analysis**

With the same inputs, The Pose mAP50:95 on COCO after quantization is only 43.1%, which is lower than that of the float model（50.0%）. 


.. _quantization_aware_pose_label:

Quantization-Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To further improve the accuracy of the quantized model, we adopt the quantization-aware training(QAT) strategy. Here, QAT is performed based on 8-bit quantization.

**Quantization settings**

- :project_file:`yolo11n-pose_qat.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/yolo11n_pose_qat.py>`
- :project_file:`trainer.py <examples/tutorial/how_to_quantize_model/quantize_yolo11n-pose/trainer.py>`

**Quantization results**

.. code-block::
   
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.22/m.0/cv2/conv/Conv:                 | ████████████████████ | 27.739%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: | ███████████████████  | 26.872%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | ███████████████████  | 26.229%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           | ██████████████████   | 25.300%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: | ██████████████████   | 24.625%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           | █████████████████    | 23.751%
   /model.20/conv/Conv:                         | █████████████████    | 23.320%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: | █████████████████    | 22.901%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           | ████████████████     | 22.516%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         | ████████████████     | 22.035%
   /model.19/m.0/cv2/conv/Conv:                 | ████████████████     | 21.569%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           | ███████████████      | 21.199%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: | ███████████████      | 20.785%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: | ███████████████      | 20.597%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: | ███████████████      | 20.329%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           | ███████████████      | 20.179%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: | ██████████████       | 19.983%
   /model.22/m.0/cv3/conv/Conv:                 | ██████████████       | 19.919%
   /model.13/m.0/cv2/conv/Conv:                 | ██████████████       | 19.424%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: | ██████████████       | 18.893%
   /model.19/cv2/conv/Conv:                     | █████████████        | 18.055%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: | █████████████        | 17.915%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           | █████████████        | 17.796%
   /model.22/cv1/conv/Conv:                     | █████████████        | 17.777%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           | █████████████        | 17.573%
   /model.19/cv1/conv/Conv:                     | ████████████         | 17.116%
   /model.17/conv/Conv:                         | ████████████         | 16.869%
   /model.22/cv2/conv/Conv:                     | ████████████         | 16.750%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           | ████████████         | 16.540%
   /model.10/m/m.0/attn/proj/conv/Conv:         | ████████████         | 16.491%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           | ████████████         | 16.421%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           | ████████████         | 16.205%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | ████████████         | 16.116%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: | ███████████          | 15.400%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           | ███████████          | 15.251%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: | ███████████          | 14.851%
   /model.10/m/m.0/attn/pe/conv/Conv:           | ███████████          | 14.659%
   /model.19/m.0/cv1/conv/Conv:                 | ██████████           | 14.289%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           | █████████            | 13.038%
   /model.16/m.0/cv2/conv/Conv:                 | █████████            | 12.941%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           | █████████            | 12.791%
   /model.23/cv4.2/cv4.2.2/Conv:                | █████████            | 12.508%
   /model.23/cv4.1/cv4.1.2/Conv:                | █████████            | 12.226%
   /model.13/cv1/conv/Conv:                     | ████████             | 11.821%
   /model.13/cv2/conv/Conv:                     | ████████             | 11.612%
   /model.13/m.0/cv1/conv/Conv:                 | ████████             | 11.515%
   /model.10/m/m.0/attn/MatMul_1:               | ████████             | 11.303%
   /model.16/cv2/conv/Conv:                     | ████████             | 11.028%
   /model.10/m/m.0/attn/qkv/conv/Conv:          | ████████             | 10.951%
   /model.10/cv1/conv/Conv:                     | ████████             | 10.755%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           | ████████             | 10.684%
   /model.22/m.0/cv1/conv/Conv:                 | ███████              | 10.164%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         | ███████              | 9.968%
   /model.16/m.0/cv1/conv/Conv:                 | ███████              | 9.656%
   /model.23/cv4.0/cv4.0.2/Conv:                | ███████              | 9.566%
   /model.8/m.0/cv2/conv/Conv:                  | ███████              | 9.521%
   /model.10/cv2/conv/Conv:                     | ██████               | 8.068%
   /model.16/cv1/conv/Conv:                     | ██████               | 7.989%
   /model.23/cv2.1/cv2.1.2/Conv:                | ██████               | 7.969%
   /model.8/m.0/cv3/conv/Conv:                  | ██████               | 7.725%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: | █████                | 7.570%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            | █████                | 7.339%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            | █████                | 7.283%
   /model.8/cv2/conv/Conv:                      | █████                | 7.092%
   /model.10/m/m.0/attn/MatMul:                 | █████                | 6.654%
   /model.8/cv1/conv/Conv:                      | █████                | 6.492%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            | █████                | 6.451%
   /model.23/cv2.0/cv2.0.2/Conv:                | ████                 | 5.990%
   /model.23/cv2.2/cv2.2.2/Conv:                | ████                 | 5.902%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            | ████                 | 5.898%
   /model.6/m.0/cv2/conv/Conv:                  | ████                 | 5.881%
   /model.6/m.0/cv3/conv/Conv:                  | ████                 | 5.402%
   /model.8/m.0/cv1/conv/Conv:                  | ████                 | 5.210%
   /model.23/cv3.2/cv3.2.2/Conv:                | ████                 | 5.126%
   /model.6/cv1/conv/Conv:                      | ████                 | 4.983%
   /model.9/cv2/conv/Conv:                      | ███                  | 4.616%
   /model.9/cv1/conv/Conv:                      | ███                  | 3.934%
   /model.7/conv/Conv:                          | ███                  | 3.906%
   /model.3/conv/Conv:                          | ███                  | 3.654%
   /model.6/cv2/conv/Conv:                      | ██                   | 3.429%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.319%
   /model.2/cv2/conv/Conv:                      | ██                   | 3.220%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            | ██                   | 3.191%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            | ██                   | 3.157%
   /model.4/cv1/conv/Conv:                      | ██                   | 2.893%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            | ██                   | 2.792%
   /model.6/m.0/cv1/conv/Conv:                  | ██                   | 2.761%
   /model.5/conv/Conv:                          | ██                   | 2.629%
   /model.4/cv2/conv/Conv:                      | ██                   | 2.298%
   /model.2/cv1/conv/Conv:                      | █                    | 2.107%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 2.095%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 2.069%
   /model.23/cv3.1/cv3.1.2/Conv:                | █                    | 1.744%
   /model.1/conv/Conv:                          | █                    | 1.631%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 1.583%
   /model.4/m.0/cv2/conv/Conv:                  | █                    | 1.126%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.535%
   /model.0/conv/Conv:                          |                      | 0.067%
   Analysing Layerwise quantization error:: 100%|██████████| 98/98 [10:49<00:00,  6.63s/it]
   Layer                                        | NOISE:SIGNAL POWER RATIO 
   /model.9/cv2/conv/Conv:                      | ████████████████████ | 2.976%
   /model.2/cv2/conv/Conv:                      | ███████████          | 1.610%
   /model.3/conv/Conv:                          | ██████               | 0.854%
   /model.2/cv1/conv/Conv:                      | ████                 | 0.543%
   /model.1/conv/Conv:                          | ███                  | 0.487%
   /model.8/cv1/conv/Conv:                      | ███                  | 0.414%
   /model.4/cv2/conv/Conv:                      | ███                  | 0.397%
   /model.0/conv/Conv:                          | ██                   | 0.364%
   /model.6/m.0/cv3/conv/Conv:                  | ██                   | 0.230%
   /model.5/conv/Conv:                          | █                    | 0.181%
   /model.2/m.0/cv2/conv/Conv:                  | █                    | 0.144%
   /model.13/cv2/conv/Conv:                     | █                    | 0.140%
   /model.2/m.0/cv1/conv/Conv:                  | █                    | 0.138%
   /model.4/cv1/conv/Conv:                      | █                    | 0.129%
   /model.16/cv2/conv/Conv:                     | █                    | 0.122%
   /model.23/cv4.2/cv4.2.0/conv/Conv:           | █                    | 0.120%
   /model.4/m.0/cv1/conv/Conv:                  | █                    | 0.107%
   /model.23/cv4.1/cv4.1.0/conv/Conv:           | █                    | 0.096%
   /model.19/cv2/conv/Conv:                     | █                    | 0.078%
   /model.23/cv2.2/cv2.2.2/Conv:                | █                    | 0.076%
   /model.4/m.0/cv2/conv/Conv:                  |                      | 0.071%
   /model.8/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.071%
   /model.6/cv2/conv/Conv:                      |                      | 0.067%
   /model.6/cv1/conv/Conv:                      |                      | 0.066%
   /model.17/conv/Conv:                         |                      | 0.060%
   /model.23/cv4.2/cv4.2.1/conv/Conv:           |                      | 0.057%
   /model.22/m.0/m/m.1/cv1/conv/Conv:           |                      | 0.056%
   /model.16/cv1/conv/Conv:                     |                      | 0.051%
   /model.10/cv1/conv/Conv:                     |                      | 0.050%
   /model.23/cv4.2/cv4.2.2/Conv:                |                      | 0.046%
   /model.22/cv2/conv/Conv:                     |                      | 0.044%
   /model.7/conv/Conv:                          |                      | 0.043%
   /model.10/m/m.0/attn/pe/conv/Conv:           |                      | 0.043%
   /model.10/cv2/conv/Conv:                     |                      | 0.037%
   /model.19/cv1/conv/Conv:                     |                      | 0.037%
   /model.8/cv2/conv/Conv:                      |                      | 0.036%
   /model.13/cv1/conv/Conv:                     |                      | 0.036%
   /model.6/m.0/m/m.1/cv1/conv/Conv:            |                      | 0.033%
   /model.22/m.0/cv3/conv/Conv:                 |                      | 0.031%
   /model.19/m.0/cv1/conv/Conv:                 |                      | 0.027%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv: |                      | 0.026%
   /model.8/m.0/cv1/conv/Conv:                  |                      | 0.025%
   /model.19/m.0/cv2/conv/Conv:                 |                      | 0.025%
   /model.8/m.0/cv3/conv/Conv:                  |                      | 0.024%
   /model.10/m/m.0/attn/qkv/conv/Conv:          |                      | 0.023%
   /model.8/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.023%
   /model.22/m.0/cv1/conv/Conv:                 |                      | 0.021%
   /model.6/m.0/m/m.0/cv1/conv/Conv:            |                      | 0.021%
   /model.23/cv2.0/cv2.0.0/conv/Conv:           |                      | 0.020%
   /model.6/m.0/cv1/conv/Conv:                  |                      | 0.020%
   /model.23/cv4.0/cv4.0.0/conv/Conv:           |                      | 0.019%
   /model.9/cv1/conv/Conv:                      |                      | 0.018%
   /model.23/cv4.1/cv4.1.2/Conv:                |                      | 0.018%
   /model.23/cv2.1/cv2.1.1/conv/Conv:           |                      | 0.018%
   /model.13/m.0/cv1/conv/Conv:                 |                      | 0.016%
   /model.23/cv2.1/cv2.1.0/conv/Conv:           |                      | 0.016%
   /model.23/cv4.1/cv4.1.1/conv/Conv:           |                      | 0.016%
   /model.16/m.0/cv2/conv/Conv:                 |                      | 0.015%
   /model.10/m/m.0/attn/proj/conv/Conv:         |                      | 0.013%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv: |                      | 0.013%
   /model.8/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.013%
   /model.16/m.0/cv1/conv/Conv:                 |                      | 0.012%
   /model.23/cv2.2/cv2.2.0/conv/Conv:           |                      | 0.011%
   /model.20/conv/Conv:                         |                      | 0.011%
   /model.22/m.0/m/m.0/cv1/conv/Conv:           |                      | 0.011%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv: |                      | 0.011%
   /model.8/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.010%
   /model.23/cv2.0/cv2.0.2/Conv:                |                      | 0.009%
   /model.10/m/m.0/attn/MatMul:                 |                      | 0.009%
   /model.22/cv1/conv/Conv:                     |                      | 0.009%
   /model.13/m.0/cv2/conv/Conv:                 |                      | 0.008%
   /model.23/cv2.2/cv2.2.1/conv/Conv:           |                      | 0.008%
   /model.23/cv2.1/cv2.1.2/Conv:                |                      | 0.007%
   /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv: |                      | 0.007%
   /model.22/m.0/m/m.1/cv2/conv/Conv:           |                      | 0.007%
   /model.6/m.0/m/m.0/cv2/conv/Conv:            |                      | 0.006%
   /model.22/m.0/m/m.0/cv2/conv/Conv:           |                      | 0.006%
   /model.23/cv4.0/cv4.0.1/conv/Conv:           |                      | 0.005%
   /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv: |                      | 0.005%
   /model.23/cv4.0/cv4.0.2/Conv:                |                      | 0.004%
   /model.6/m.0/m/m.1/cv2/conv/Conv:            |                      | 0.004%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv: |                      | 0.004%
   /model.10/m/m.0/ffn/ffn.1/conv/Conv:         |                      | 0.003%
   /model.23/cv3.2/cv3.2.2/Conv:                |                      | 0.003%
   /model.10/m/m.0/attn/MatMul_1:               |                      | 0.002%
   /model.10/m/m.0/ffn/ffn.0/conv/Conv:         |                      | 0.002%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv: |                      | 0.002%
   /model.23/cv2.0/cv2.0.1/conv/Conv:           |                      | 0.002%
   /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.2/Conv:                |                      | 0.001%
   /model.23/cv3.1/cv3.1.2/Conv:                |                      | 0.001%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv: |                      | 0.001%
   /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv: |                      | 0.000%
   /model.6/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv: |                      | 0.000%
   /model.8/m.0/cv2/conv/Conv:                  |                      | 0.000%
   /model.22/m.0/cv2/conv/Conv:                 |                      | 0.000%


**Quantization error analysis**

After applying QAT to 8-bit quantization, the quantized model's Pose mAP50:95 on COCO improves to 44.9% with the same inputs, while cumulative errors of out layers are significantly reduced. Compared to the other two quantization methods, the 8-bit QAT quantized model achieves the highest quantization accuracy with the lowest inference latency.


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

