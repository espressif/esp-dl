How to deploy MobileNetV2
================================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will introduce how to quantize a pre-trained MobileNetV2 model using ESP-PPQ and deploy the quantized MobileNetV2 model using ESP-DL.

.. contents::
   :local:
   :depth: 2

Preparation
----------------

1. :ref:`Install ESP_IDF <requirements_esp_idf>`
2. :ref:`Install ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_mobilenetv2:

Model quantization
------------------------

:project:`Quantization script <examples/tutorial/how_to_quantize_model/quantize_mobilenetv2>`

Pre-trained model
^^^^^^^^^^^^^^^^^^^^^

Load the pre-trained model of MobileNet_v2 from torchvision. You can also download it from `ONNX models <https://github.com/onnx/models>`__ or `TensorFlow models <https://github.com/tensorflow/models>`__:

.. code-block:: python

   import torchvision
   from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

   model = torchvision.models.mobilenet.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

Calibration dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration dataset needs to be consistent with your model input format. The calibration dataset needs to cover all possible situations of your model input as much as possible to better quantize the model. Here we take the ImageNet dataset as an example to demonstrate how to prepare the calibration dataset.

Use torchvision to load the ImageNet dataset:

.. code-block:: python
   
   import torchvision.datasets as datasets
   from torch.utils.data.dataset import Subset
   dataset = datasets.ImageFolder(
      CALIB_DIR,
      transforms.Compose(
            [
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize(
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
               ),
            ]
      ),
   )
   dataset = Subset(dataset, indices=[_ for _ in range(0, 1024)])
   dataloader = DataLoader(
      dataset=dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=4,
      pin_memory=False,
      collate_fn=collate_fn1,
   )

8bit default configuration quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Quantization settings**

.. code-block:: python

   target="esp32p4"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting

**Quantization results**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 48.831%
   /features/features.15/conv/conv.2/Conv:          | ███████████████████  | 45.268%
   /features/features.17/conv/conv.2/Conv:          | ██████████████████   | 43.112%
   /features/features.18/features.18.0/Conv:        | █████████████████    | 41.586%
   /features/features.14/conv/conv.2/Conv:          | █████████████████    | 41.135%
   /features/features.13/conv/conv.2/Conv:          | ██████████████       | 35.090%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █████████████        | 32.895%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 29.226%
   /features/features.12/conv/conv.2/Conv:          | ████████████         | 28.895%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 27.808%
   /features/features.7/conv/conv.2/Conv:           | ███████████          | 27.675%
   /features/features.10/conv/conv.2/Conv:          | ███████████          | 26.292%
   /features/features.11/conv/conv.2/Conv:          | ███████████          | 26.085%
   /features/features.6/conv/conv.2/Conv:           | ███████████          | 25.892%
   /classifier/classifier.1/Gemm:                   | ██████████           | 25.591%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 25.323%
   /features/features.4/conv/conv.2/Conv:           | ██████████           | 24.787%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | ██████████           | 24.354%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 20.207%
   /features/features.9/conv/conv.2/Conv:           | ████████             | 19.808%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ████████             | 18.465%
   /features/features.5/conv/conv.2/Conv:           | ███████              | 17.868%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███████              | 16.589%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ███████              | 16.143%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ██████               | 15.382%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 15.105%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ██████               | 15.029%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ██████               | 14.875%
   /features/features.2/conv/conv.2/Conv:           | ██████               | 14.869%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ██████               | 14.552%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ██████               | 14.050%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.929%
   /features/features.8/conv/conv.2/Conv:           | ██████               | 13.833%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ██████               | 13.684%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | █████                | 12.942%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | █████                | 12.765%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | █████                | 12.251%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █████                | 11.186%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 11.070%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.371%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ████                 | 10.356%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.149%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.472%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.232%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 9.187%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 8.770%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | ███                  | 8.408%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ███                  | 8.151%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ███                  | 7.156%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | ███                  | 6.328%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 5.392%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.875%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [08:44<00:00,  9.91s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 14.303%
   /features/features.0/features.0.0/Conv:          | █                    | 0.844%
   /features/features.1/conv/conv.1/Conv:           | █                    | 0.667%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.419%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.272%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.238%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.146%
   /features/features.14/conv/conv.2/Conv:          |                      | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.105%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.083%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.076%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.076%
   /features/features.3/conv/conv.2/Conv:           |                      | 0.075%
   /features/features.16/conv/conv.2/Conv:          |                      | 0.074%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.072%
   /features/features.15/conv/conv.2/Conv:          |                      | 0.066%
   /features/features.4/conv/conv.2/Conv:           |                      | 0.065%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.063%
   /classifier/classifier.1/Gemm:                   |                      | 0.063%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.054%
   /features/features.13/conv/conv.2/Conv:          |                      | 0.050%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.040%
   /features/features.2/conv/conv.2/Conv:           |                      | 0.038%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.034%
   /features/features.17/conv/conv.2/Conv:          |                      | 0.030%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.025%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.024%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.022%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.021%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.021%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.019%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.018%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.017%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.014%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

   * Prec@1 60.500 Prec@5 83.275*

**Quantization error analysis**

The top1 accuracy after quantization is only 60.5%, which is far from the accuracy of the float model (71.878%). The quantization model has a large loss in accuracy, including:

- **Graphwise Error**

  The last layer of the model is /classifier/classifier.1/Gemm, and the cumulative error of this layer is 25.591%. In experience, the cumulative error of the last layer is less than 10%, and the accuracy loss of the quantization model is small.

- **Layerwise error**

  Observing the Layerwise error, it is found that the errors of most layers are less than 1%, indicating that the quantization errors of most layers are small, and only a few layers have large errors. We can choose to quantize the layers with large errors using int16. For details, please see mixed precision quantization.

.. _mixed_precision_quantization_label:

Mixed precision quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
esp-dl supports mixed precision quantization, which can be used to quantize some layers using int16 and some layers using int8. The quantization error of the model can be reduced by using mixed precision quantization.

**Quantization settings**

.. code-block:: python

   from esp_ppq.api import get_target_platform
   target="esp32p4"
   num_of_bits=8
   batch_size=32

   # The following layers are quantized using int16
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.0/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.2/Clip", get_target_platform(TARGET, 16))

**Quantization results**

.. code-block::

   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 31.585%
   /features/features.15/conv/conv.2/Conv:          | ███████████████████  | 29.253%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 25.077%
   /features/features.14/conv/conv.2/Conv:          | ████████████████     | 24.819%
   /features/features.17/conv/conv.2/Conv:          | ████████████         | 19.546%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 19.283%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 18.764%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 18.596%
   /features/features.18/features.18.0/Conv:        | ████████████         | 18.541%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 15.633%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 14.784%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 14.773%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | █████████            | 13.700%
   /features/features.6/conv/conv.2/Conv:           | ████████             | 12.824%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 11.727%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ███████              | 10.612%
   /features/features.11/conv/conv.2/Conv:          | ██████               | 10.262%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.967%
   /classifier/classifier.1/Gemm:                   | ██████               | 9.117%
   /features/features.5/conv/conv.2/Conv:           | ██████               | 8.915%
   /features/features.7/conv/conv.2/Conv:           | █████                | 8.690%
   /features/features.3/conv/conv.2/Conv:           | █████                | 8.586%
   /features/features.4/conv/conv.2/Conv:           | █████                | 7.525%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 7.432%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 7.317%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.848%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.711%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.100%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.043%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 5.962%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.873%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.833%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 5.832%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.736%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.639%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.017%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 4.963%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 4.870%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.655%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 4.650%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.648%
   /features/features.1/conv/conv.1/Conv:           | ███                  | 4.318%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.849%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.712%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.394%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.391%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.713%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.637%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.602%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 2.397%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.759%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.433%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53/53 [08:27<00:00,  9.58s/it]
   *
   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.1/conv/conv.1/Conv:           | ████████████████████ | 1.096%
   /features/features.0/features.0.0/Conv:          | ███████████████      | 0.844%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██████████           | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████████             | 0.425%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████                | 0.272%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 0.238%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ███                  | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███                  | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███                  | 0.146%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ██                   | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ██                   | 0.105%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | █                    | 0.083%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █                    | 0.076%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.075%
   /features/features.16/conv/conv.2/Conv:          | █                    | 0.074%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | █                    | 0.072%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.071%
   /features/features.15/conv/conv.2/Conv:          | █                    | 0.066%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.065%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.063%
   /classifier/classifier.1/Gemm:                   | █                    | 0.063%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.059%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.054%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | █                    | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █                    | 0.040%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.038%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | █                    | 0.034%
   /features/features.17/conv/conv.2/Conv:          | █                    | 0.030%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.025%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.024%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.022%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.021%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.021%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.019%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.018%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.017%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.017%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.014%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

   * Prec@1 69.550 Prec@5 88.450*

**Quantization Error Analysis**

After replacing the layer with the largest error with 16-bit quantization, it can be observed that the model accuracy is significantly improved. The top1 accuracy after quantization is 69.550%, which is close to the accuracy of the float model (71.878%). The cumulative error of the last layer of the model ``/classifier/classifier.1/Gemm`` is 9.117%.

.. _layerwise_equalization_quantization_label:

Layerwise equalization quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method is proposed in the paper `Data-Free Quantization Through Weight Equalization and Bias Correction <https://arxiv.org/abs/1906.04721>`_. When using this method, the original ReLU6 in the MobilenetV2 model needs to be replaced with ReLU.

**Quantization Settings**

.. code-block:: python

   import torch.nn as nn
   def convert_relu6_to_relu(model):
      for child_name, child in model.named_children():
         if isinstance(child, nn.ReLU6):
               setattr(model, child_name, nn.ReLU())
         else:
               convert_relu6_to_relu(child)
      return model

   # Replace ReLU6 with ReLU
   model = convert_relu6_to_relu(model)
   # Use layerwise equalization
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.equalization = True
   quant_setting.equalization_setting.iterations = 4
   quant_setting.equalization_setting.value_threshold = .4
   quant_setting.equalization_setting.opt_level = 2
   quant_setting.equalization_setting.interested_layers = None

.. code-block::

   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 34.497%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 30.813%
   /features/features.14/conv/conv.2/Conv:          | ███████████████      | 25.876%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ██████████████       | 24.498%
   /features/features.17/conv/conv.2/Conv:          | ████████████         | 20.290%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 20.177%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 19.993%
   /features/features.18/features.18.0/Conv:        | ███████████          | 19.536%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ██████████           | 17.879%
   /features/features.12/conv/conv.2/Conv:          | ██████████           | 17.150%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | █████████            | 15.970%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 15.254%
   /features/features.1/conv/conv.1/Conv:           | █████████            | 15.122%
   /features/features.10/conv/conv.2/Conv:          | █████████            | 14.917%
   /features/features.6/conv/conv.2/Conv:           | ████████             | 13.446%
   /features/features.11/conv/conv.2/Conv:          | ███████              | 12.533%
   /features/features.9/conv/conv.2/Conv:           | ███████              | 11.479%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ███████              | 11.470%
   /features/features.5/conv/conv.2/Conv:           | ██████               | 10.669%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 10.526%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ██████               | 9.529%
   /features/features.7/conv/conv.2/Conv:           | █████                | 9.500%
   /classifier/classifier.1/Gemm:                   | █████                | 8.965%
   /features/features.4/conv/conv.2/Conv:           | █████                | 8.674%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 8.349%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 8.068%
   /features/features.8/conv/conv.2/Conv:           | █████                | 7.961%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 7.451%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.714%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.399%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.369%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 6.222%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 5.867%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.719%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ███                  | 5.546%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.414%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 5.093%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 4.951%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ███                  | 4.941%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.825%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ██                   | 4.330%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 4.299%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ██                   | 4.283%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.477%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.287%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.787%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.774%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.705%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.636%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 1.846%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.170%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.389%
   /features/features.0/features.0.0/Conv:          |                      | 0.025%
   Analysing Layerwise quantization error:: 100%|██████████| 53/53 [07:46<00:00,  8.80s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 0.989%
   /features/features.0/features.0.0/Conv:          | █████████████████    | 0.845%
   /features/features.16/conv/conv.2/Conv:          | █████                | 0.238%
   /features/features.17/conv/conv.2/Conv:          | ████                 | 0.202%
   /features/features.14/conv/conv.2/Conv:          | ████                 | 0.198%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 0.192%
   /features/features.15/conv/conv.2/Conv:          | ███                  | 0.145%
   /features/features.4/conv/conv.2/Conv:           | ██                   | 0.120%
   /features/features.2/conv/conv.2/Conv:           | ██                   | 0.111%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 0.079%
   /classifier/classifier.1/Gemm:                   | █                    | 0.062%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.050%
   /features/features.12/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █                    | 0.047%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.046%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.045%
   /features/features.5/conv/conv.2/Conv:           | █                    | 0.030%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.028%
   /features/features.6/conv/conv.2/Conv:           | █                    | 0.027%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | █                    | 0.026%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.025%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.023%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.021%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.020%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.020%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.017%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.016%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.012%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.012%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.012%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.012%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.010%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.008%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.008%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.008%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.008%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.006%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.005%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.004%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.004%
   /features/features.18/features.18.0/Conv:        |                      | 0.003%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.003%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.003%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%

   * Prec@1 69.800 Prec@5 88.550

**Quantization Error Analysis**

Note that applying layerwise equalization to 8-bit quantization helps reduce quantization loss. The cumulative error of the last layer of the model ``/classifier/classifier.1/Gemm`` is 8.965%. The top1 accuracy after quantization is 69.800%, which is closer to the accuracy of the float model (71.878%) and higher than the quantization accuracy of mixed precision quantization.

.. note::

   To further reduce the quantization error, you can try using QAT (Auantization Aware Training). For specific methods, please refer to `PPQ QAT example <https://github.com/OpenPPL/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py>`__.


Model deployment
-----------------------

:project:`examples <examples/mobilenetv2_cls>`

Image classification base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_cls_base.hpp <esp-dl/vision/classification/dl_cls_base.hpp>`
- :project_file:`dl_cls_base.cpp <esp-dl/vision/classification/dl_cls_base.cpp>`

Pre-process
^^^^^^^^^^^^^^^^^^^^

``ImagePreprocessor`` class contains the common pre-precoess pipeline, ``color conversion``, ``crop``, ``resize``, ``normalization``, ``quantize``。

- :project_file:`dl_image_preprocessor.hpp <esp-dl/vision/image/dl_image_preprocessor.hpp>`
- :project_file:`dl_image_preprocessor.cpp <esp-dl/vision/image/dl_image_preprocessor.cpp>`

Post-process
^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_cls_postprocessor.hpp <esp-dl/vision/classification/dl_cls_postprocessor.hpp>`
- :project_file:`dl_cls_postprocessor.cpp <esp-dl/vision/classification/dl_cls_postprocessor.cpp>`
- :project_file:`imagenet_cls_postprocessor.hpp <esp-dl/vision/classification/imagenet_cls_postprocessor.hpp>`
- :project_file:`imagenet_cls_postprocessor.cpp <esp-dl/vision/classification/imagenet_cls_postprocessor.cpp>`
