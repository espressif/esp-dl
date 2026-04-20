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

.. note::

   - **ESP32P4**: **Per-channel** quantization is used for Conv and Gemm weights; other operators stay **per-tensor**.
   - **ESP32S3 and other chips**: due to ISA limits, all operators use **per-tensor** quantization.

   **Per-channel** quantization usually preserves more detail than **per-tensor**, so the same model often quantizes better on ESP32P4 than on ESP32S3.

   The walk-through below mainly targets **ESP32S3**. The **8-bit default configuration** section also lists **ESP32P4** results for comparison.

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

8-bit default configuration quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ESP32P4 quantization
""""""""""""""""""""

**Quantization settings**

.. code-block:: python

   import torch.nn as nn

   def convert_relu6_to_relu(model):
      for child_name, child in model.named_children():
         if isinstance(child, nn.ReLU6):
               setattr(model, child_name, nn.ReLU())
         else:
               convert_relu6_to_relu(child)
      return model

   target="esp32p4"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting
   # Replace ReLU6 with ReLU. When using per-channel quantization,
   # ReLU tends to yield higher accuracy after quantization.
   model = convert_relu6_to_relu(model)

**Quantization results**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 28.455%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 24.981%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ███████████████      | 21.120%
   /features/features.14/conv/conv.2/Conv:          | ██████████████       | 20.162%
   /features/features.17/conv/conv.2/Conv:          | █████████████        | 18.303%
   /features/features.18/features.18.0/Conv:        | ████████████         | 17.422%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 16.590%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 15.886%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███████████          | 15.033%
   /features/features.12/conv/conv.2/Conv:          | ██████████           | 13.968%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | █████████            | 12.973%
   /features/features.6/conv/conv.2/Conv:           | █████████            | 12.624%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 12.366%
   /features/features.10/conv/conv.2/Conv:          | █████████            | 12.207%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ███████              | 10.319%
   /features/features.11/conv/conv.2/Conv:          | ███████              | 10.139%
   /features/features.5/conv/conv.2/Conv:           | ███████              | 9.430%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.324%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 9.298%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ██████               | 8.819%
   /features/features.7/conv/conv.2/Conv:           | ██████               | 8.649%
   /features/features.1/conv/conv.1/Conv:           | ██████               | 8.423%
   /features/features.4/conv/conv.2/Conv:           | ██████               | 7.972%
   /classifier/classifier.1/Gemm:                   | █████                | 7.204%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 6.975%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 6.867%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | █████                | 6.626%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.296%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 6.113%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.068%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 6.044%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.839%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.814%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 5.790%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.580%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.575%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.476%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ████                 | 5.182%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.157%
   /features/features.2/conv/conv.2/Conv:           | ████                 | 5.115%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 5.003%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.470%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 4.412%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.133%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.012%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.524%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.370%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.004%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.680%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.617%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.962%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.477%
   /features/features.0/features.0.0/Conv:          |                      | 0.114%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:33<00:00, 10.81s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.0/features.0.0/Conv:          | ████████████████████ | 0.879%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████         | 0.526%
   /features/features.16/conv/conv.2/Conv:          | █████                | 0.215%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 0.165%
   /features/features.15/conv/conv.2/Conv:          | ███                  | 0.127%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.110%
   /classifier/classifier.1/Gemm:                   | █                    | 0.061%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.055%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.048%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.043%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.039%
   /features/features.17/conv/conv.2/Conv:          | █                    | 0.037%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.035%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.034%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.033%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.027%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.020%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.017%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.015%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.013%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.011%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.010%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.007%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.007%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.006%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.006%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.005%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.004%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.004%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.004%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.003%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.003%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.002%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.001%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.001%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.000%

   * Prec@1 71.150 Prec@5 89.350

ESP32S3 quantization
""""""""""""""""""""

**Quantization settings**

.. code-block:: python

   target="esp32s3"
   num_of_bits=8
   batch_size=32
   quant_setting = QuantizationSettingFactory.espdl_setting() # default setting

**Quantization results**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 49.069%
   /features/features.17/conv/conv.2/Conv:          | ███████████████████  | 45.605%
   /features/features.15/conv/conv.2/Conv:          | ███████████████████  | 45.446%
   /features/features.18/features.18.0/Conv:        | ██████████████████   | 43.709%
   /features/features.14/conv/conv.2/Conv:          | █████████████████    | 41.199%
   /features/features.13/conv/conv.2/Conv:          | ██████████████       | 35.330%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █████████████        | 33.151%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 29.433%
   /features/features.12/conv/conv.2/Conv:          | ████████████         | 29.049%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ███████████          | 27.952%
   /features/features.7/conv/conv.2/Conv:           | ███████████          | 27.420%
   /classifier/classifier.1/Gemm:                   | ███████████          | 27.097%
   /features/features.11/conv/conv.2/Conv:          | ███████████          | 26.299%
   /features/features.10/conv/conv.2/Conv:          | ███████████          | 26.026%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 25.428%
   /features/features.6/conv/conv.2/Conv:           | ██████████           | 25.363%
   /features/features.4/conv/conv.2/Conv:           | ██████████           | 24.751%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | ██████████           | 24.485%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 20.296%
   /features/features.9/conv/conv.2/Conv:           | ████████             | 19.854%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ████████             | 18.552%
   /features/features.5/conv/conv.2/Conv:           | ███████              | 17.704%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███████              | 16.861%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ███████              | 16.398%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ██████               | 15.579%
   /features/features.2/conv/conv.2/Conv:           | ██████               | 15.369%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ██████               | 15.193%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 14.844%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ██████               | 14.811%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ██████               | 14.660%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.995%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ██████               | 13.825%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ██████               | 13.750%
   /features/features.8/conv/conv.2/Conv:           | █████                | 13.557%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | █████                | 12.814%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | █████                | 12.610%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | █████                | 12.173%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | █████                | 11.607%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █████                | 11.136%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.368%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ████                 | 10.333%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ████                 | 10.038%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ████                 | 9.632%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████                 | 9.282%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 8.970%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ████                 | 8.947%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | ███                  | 8.389%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ███                  | 8.067%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ███                  | 7.462%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | ███                  | 6.513%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 5.537%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.875%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [34:02<00:00, 38.53s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 14.306%
   /features/features.0/features.0.0/Conv:          | █                    | 0.843%
   /features/features.1/conv/conv.1/Conv:           | █                    | 0.667%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.425%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.271%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.237%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: |                      | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.146%
   /features/features.14/conv/conv.2/Conv:          |                      | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.104%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.083%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.076%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.076%
   /features/features.3/conv/conv.2/Conv:           |                      | 0.075%
   /features/features.16/conv/conv.2/Conv:          |                      | 0.073%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.072%
   /features/features.15/conv/conv.2/Conv:          |                      | 0.066%
   /features/features.4/conv/conv.2/Conv:           |                      | 0.064%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.063%
   /classifier/classifier.1/Gemm:                   |                      | 0.063%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.054%
   /features/features.13/conv/conv.2/Conv:          |                      | 0.050%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.040%
   /features/features.2/conv/conv.2/Conv:           |                      | 0.039%
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

   * Prec@1 60.325 Prec@5 83.100

**Quantization error analysis**

Experiments show that with **per-channel** quantization on ESP32P4, Prec@1 of the quantized model is **71.150%**, close to the float model (**71.878%**); with **per-tensor** quantization on ESP32S3, Prec@1 is only about **60.5%**, a large gap versus the same float baseline, so other quantization techniques are worth exploring. The following briefly analyzes quantization error on **ESP32S3**:

- **Cumulative error (Graphwise error)**

  The last layer is ``/classifier/classifier.1/Gemm``; its cumulative error is **27.097%**. In practice, if the last layer stays below **10%**, accuracy loss from quantization is usually small.

- **Layerwise error**

  Most layers are below **1%**, so quantization error is small for most of the network; only a few layers dominate. Those layers can be quantized with **int16**—see mixed-precision quantization below.


.. _mixed_precision_quantization_label:

Mixed precision quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Quantization settings**

.. code-block:: python

   from esp_ppq.api import get_target_platform
   target="esp32s3"
   num_of_bits=8
   batch_size=32

   # Quantize the following layers with int16
   quant_setting = QuantizationSettingFactory.espdl_setting()
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.0/Conv", get_target_platform(TARGET, 16))
   quant_setting.dispatching_table.append("/features/features.1/conv/conv.0/conv.0.2/Clip", get_target_platform(TARGET, 16))

**Quantization results**

.. code-block::

   Analysing Graphwise Quantization Error::
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 31.995%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 29.554%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 25.400%
   /features/features.14/conv/conv.2/Conv:          | ████████████████     | 25.125%
   /features/features.17/conv/conv.2/Conv:          | █████████████        | 21.358%
   /features/features.18/features.18.0/Conv:        | █████████████        | 20.177%
   /features/features.13/conv/conv.2/Conv:          | ████████████         | 19.204%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 18.988%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ████████████         | 18.846%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 15.831%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 14.938%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 14.918%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | █████████            | 13.865%
   /features/features.6/conv/conv.2/Conv:           | ████████             | 12.909%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 11.843%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ███████              | 10.648%
   /features/features.11/conv/conv.2/Conv:          | ██████               | 10.391%
   /classifier/classifier.1/Gemm:                   | ██████               | 10.289%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.940%
   /features/features.5/conv/conv.2/Conv:           | ██████               | 8.902%
   /features/features.7/conv/conv.2/Conv:           | █████                | 8.745%
   /features/features.3/conv/conv.2/Conv:           | █████                | 8.614%
   /features/features.4/conv/conv.2/Conv:           | █████                | 7.645%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | █████                | 7.501%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | █████                | 7.428%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.915%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.488%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ████                 | 6.148%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ████                 | 6.042%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.955%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.904%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ████                 | 5.894%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ████                 | 5.847%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ████                 | 5.793%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ████                 | 5.737%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 5.293%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.022%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 5.004%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 4.670%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.662%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ███                  | 4.643%
   /features/features.1/conv/conv.1/Conv:           | ███                  | 4.320%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.878%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.740%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 3.420%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.306%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.734%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.639%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.603%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 2.404%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 1.765%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.433%
   /features/features.0/features.0.0/Conv:          |                      | 0.119%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:29<00:00, 10.74s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.1/Conv:           | ████████████████████ | 1.099%
   /features/features.0/features.0.0/Conv:          | ███████████████      | 0.843%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | ██████████           | 0.574%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ████████             | 0.425%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████                | 0.271%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ████                 | 0.237%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ████                 | 0.214%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ███                  | 0.180%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 0.151%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ███                  | 0.148%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███                  | 0.146%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.136%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ██                   | 0.105%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ██                   | 0.104%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | █                    | 0.083%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | █                    | 0.076%
   /features/features.3/conv/conv.2/Conv:           | █                    | 0.075%
   /features/features.16/conv/conv.2/Conv:          | █                    | 0.073%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | █                    | 0.072%
   /features/features.7/conv/conv.2/Conv:           | █                    | 0.071%
   /features/features.15/conv/conv.2/Conv:          | █                    | 0.066%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.064%
   /features/features.11/conv/conv.2/Conv:          | █                    | 0.063%
   /classifier/classifier.1/Gemm:                   | █                    | 0.063%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 0.054%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | █                    | 0.042%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | █                    | 0.040%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.039%
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
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.017%
   /features/features.12/conv/conv.2/Conv:          |                      | 0.017%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.014%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.014%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.013%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.012%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.009%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.008%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.006%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.005%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.18/features.18.0/Conv:        |                      | 0.002%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.002%

   * Prec@1 69.225 Prec@5 88.700

**Quantization error analysis**

After moving the worst layers to 16-bit, accuracy improves clearly: top-1 is **69.225%**, close to the float model (**71.878%**). The cumulative error of the last layer ``/classifier/classifier.1/Gemm`` is **10.289%**.

.. _layerwise_equalization_quantization_label:

Layerwise equalization quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method is described in `Data-Free Quantization Through Weight Equalization and Bias Correction <https://arxiv.org/abs/1906.04721>`_. Replace the original ReLU6 in MobileNetV2 with ReLU when using it.

**Quantization settings**

.. code-block:: python

   import torch.nn as nn

   target="esp32s3"
   num_of_bits=8
   batch_size=32

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
   quant_setting.equalization_setting.iterations = 6
   quant_setting.equalization_setting.value_threshold = 0.5
   quant_setting.equalization_setting.opt_level = 2
   quant_setting.equalization_setting.interested_layers = None

**Quantization results**

.. code-block::

   Analysing Graphwise Quantization Error
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.16/conv/conv.2/Conv:          | ████████████████████ | 34.197%
   /features/features.15/conv/conv.2/Conv:          | ██████████████████   | 30.936%
   /features/features.14/conv/conv.2/Conv:          | ████████████████     | 27.411%
   /features/features.17/conv/conv.0/conv.0.0/Conv: | ████████████████     | 27.235%
   /features/features.16/conv/conv.0/conv.0.0/Conv: | ████████████         | 21.350%
   /features/features.17/conv/conv.2/Conv:          | ████████████         | 20.880%
   /features/features.18/features.18.0/Conv:        | ████████████         | 19.900%
   /features/features.16/conv/conv.1/conv.1.0/Conv: | ███████████          | 19.559%
   /features/features.13/conv/conv.2/Conv:          | ███████████          | 19.325%
   /features/features.15/conv/conv.0/conv.0.0/Conv: | ██████████           | 17.006%
   /features/features.15/conv/conv.1/conv.1.0/Conv: | █████████            | 15.475%
   /features/features.12/conv/conv.2/Conv:          | █████████            | 14.920%
   /features/features.14/conv/conv.1/conv.1.0/Conv: | ████████             | 13.397%
   /features/features.10/conv/conv.2/Conv:          | ███████              | 12.634%
   /features/features.6/conv/conv.2/Conv:           | ███████              | 11.711%
   /classifier/classifier.1/Gemm:                   | ██████               | 10.808%
   /features/features.11/conv/conv.2/Conv:          | ██████               | 10.234%
   /features/features.14/conv/conv.0/conv.0.0/Conv: | ██████               | 10.109%
   /features/features.9/conv/conv.2/Conv:           | ██████               | 9.563%
   /features/features.3/conv/conv.2/Conv:           | ██████               | 9.448%
   /features/features.5/conv/conv.2/Conv:           | █████                | 8.716%
   /features/features.7/conv/conv.2/Conv:           | █████                | 8.040%
   /features/features.4/conv/conv.2/Conv:           | ████                 | 7.647%
   /features/features.12/conv/conv.1/conv.1.0/Conv: | ████                 | 7.205%
   /features/features.13/conv/conv.1/conv.1.0/Conv: | ████                 | 7.021%
   /features/features.13/conv/conv.0/conv.0.0/Conv: | ████                 | 6.244%
   /features/features.8/conv/conv.2/Conv:           | ████                 | 6.205%
   /features/features.1/conv/conv.1/Conv:           | ████                 | 6.036%
   /features/features.10/conv/conv.1/conv.1.0/Conv: | ███                  | 5.824%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.452%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | ███                  | 5.408%
   /features/features.17/conv/conv.1/conv.1.0/Conv: | ███                  | 5.324%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  | ███                  | 5.187%
   /features/features.12/conv/conv.0/conv.0.0/Conv: | ███                  | 4.646%
   /features/features.2/conv/conv.2/Conv:           | ███                  | 4.576%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.493%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  | ███                  | 4.364%
   /features/features.10/conv/conv.0/conv.0.0/Conv: | ███                  | 4.305%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | ██                   | 4.160%
   /features/features.11/conv/conv.0/conv.0.0/Conv: | ██                   | 4.031%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  | ██                   | 3.667%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  | ██                   | 2.905%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  | ██                   | 2.730%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  | █                    | 2.202%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  | █                    | 2.170%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  | █                    | 2.150%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  | █                    | 2.117%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  | █                    | 2.101%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  | █                    | 1.506%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  | █                    | 1.238%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  | █                    | 0.965%
   /features/features.1/conv/conv.0/conv.0.0/Conv:  |                      | 0.357%
   /features/features.0/features.0.0/Conv:          |                      | 0.025%
   Analysing Layerwise quantization error:: 100%|██████████████████████████| 53/53 [09:22<00:00, 10.61s/it]
   Layer                                            | NOISE:SIGNAL POWER RATIO 
   /features/features.1/conv/conv.0/conv.0.0/Conv:  | ████████████████████ | 1.770%
   /features/features.0/features.0.0/Conv:          | ██████████           | 0.844%
   /features/features.16/conv/conv.2/Conv:          | ██                   | 0.180%
   /features/features.14/conv/conv.2/Conv:          | ██                   | 0.176%
   /features/features.1/conv/conv.1/Conv:           | ██                   | 0.153%
   /features/features.15/conv/conv.2/Conv:          | █                    | 0.109%
   /features/features.2/conv/conv.2/Conv:           | █                    | 0.101%
   /features/features.4/conv/conv.2/Conv:           | █                    | 0.092%
   /features/features.3/conv/conv.1/conv.1.0/Conv:  | █                    | 0.081%
   /classifier/classifier.1/Gemm:                   | █                    | 0.062%
   /features/features.11/conv/conv.1/conv.1.0/Conv: | █                    | 0.061%
   /features/features.12/conv/conv.2/Conv:          | █                    | 0.050%
   /features/features.13/conv/conv.2/Conv:          | █                    | 0.048%
   /features/features.17/conv/conv.2/Conv:          |                      | 0.043%
   /features/features.7/conv/conv.2/Conv:           |                      | 0.041%
   /features/features.8/conv/conv.1/conv.1.0/Conv:  |                      | 0.038%
   /features/features.5/conv/conv.1/conv.1.0/Conv:  |                      | 0.035%
   /features/features.11/conv/conv.2/Conv:          |                      | 0.033%
   /features/features.6/conv/conv.2/Conv:           |                      | 0.032%
   /features/features.2/conv/conv.1/conv.1.0/Conv:  |                      | 0.030%
   /features/features.3/conv/conv.2/Conv:           |                      | 0.029%
   /features/features.5/conv/conv.2/Conv:           |                      | 0.028%
   /features/features.6/conv/conv.1/conv.1.0/Conv:  |                      | 0.026%
   /features/features.4/conv/conv.0/conv.0.0/Conv:  |                      | 0.026%
   /features/features.15/conv/conv.1/conv.1.0/Conv: |                      | 0.025%
   /features/features.10/conv/conv.2/Conv:          |                      | 0.022%
   /features/features.14/conv/conv.0/conv.0.0/Conv: |                      | 0.020%
   /features/features.16/conv/conv.1/conv.1.0/Conv: |                      | 0.019%
   /features/features.4/conv/conv.1/conv.1.0/Conv:  |                      | 0.018%
   /features/features.13/conv/conv.1/conv.1.0/Conv: |                      | 0.014%
   /features/features.13/conv/conv.0/conv.0.0/Conv: |                      | 0.014%
   /features/features.16/conv/conv.0/conv.0.0/Conv: |                      | 0.012%
   /features/features.17/conv/conv.0/conv.0.0/Conv: |                      | 0.012%
   /features/features.12/conv/conv.0/conv.0.0/Conv: |                      | 0.011%
   /features/features.12/conv/conv.1/conv.1.0/Conv: |                      | 0.011%
   /features/features.9/conv/conv.2/Conv:           |                      | 0.009%
   /features/features.8/conv/conv.2/Conv:           |                      | 0.009%
   /features/features.2/conv/conv.0/conv.0.0/Conv:  |                      | 0.008%
   /features/features.7/conv/conv.0/conv.0.0/Conv:  |                      | 0.007%
   /features/features.10/conv/conv.1/conv.1.0/Conv: |                      | 0.006%
   /features/features.10/conv/conv.0/conv.0.0/Conv: |                      | 0.006%
   /features/features.15/conv/conv.0/conv.0.0/Conv: |                      | 0.006%
   /features/features.3/conv/conv.0/conv.0.0/Conv:  |                      | 0.004%
   /features/features.6/conv/conv.0/conv.0.0/Conv:  |                      | 0.004%
   /features/features.11/conv/conv.0/conv.0.0/Conv: |                      | 0.003%
   /features/features.5/conv/conv.0/conv.0.0/Conv:  |                      | 0.003%
   /features/features.9/conv/conv.1/conv.1.0/Conv:  |                      | 0.003%
   /features/features.14/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.7/conv/conv.1/conv.1.0/Conv:  |                      | 0.002%
   /features/features.17/conv/conv.1/conv.1.0/Conv: |                      | 0.002%
   /features/features.8/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.9/conv/conv.0/conv.0.0/Conv:  |                      | 0.001%
   /features/features.18/features.18.0/Conv:        |                      | 0.001%

   * Prec@1 69.800 Prec@5 88.400

**Quantization error analysis**

Layerwise equalization on 8-bit quantization further reduces loss: top-1 is **69.80%**, closer to the float model (**71.878%**) and **better than** the mixed-precision result above.

.. note::

   To further reduce the quantization error, you can try using QAT (Quantization Aware Training). For specific methods, please refer to `PPQ QAT example <https://github.com/OpenPPL/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py>`__.


Model deployment
-----------------------

:project:`Reference example <examples/mobilenetv2_cls>`

Image classification base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_cls_base.hpp <esp-dl/vision/classification/dl_cls_base.hpp>`
- :project_file:`dl_cls_base.cpp <esp-dl/vision/classification/dl_cls_base.cpp>`

Pre-process
^^^^^^^^^^^^^^^^^^^^

``ImagePreprocessor`` class contains the common pre-process pipeline, ``color conversion``, ``crop``, ``resize``, ``normalization``, ``quantize``.

- :project_file:`dl_image_preprocessor.hpp <esp-dl/vision/image/dl_image_preprocessor.hpp>`
- :project_file:`dl_image_preprocessor.cpp <esp-dl/vision/image/dl_image_preprocessor.cpp>`

Post-process
^^^^^^^^^^^^^^^^^^^^

- :project_file:`dl_cls_postprocessor.hpp <esp-dl/vision/classification/dl_cls_postprocessor.hpp>`
- :project_file:`dl_cls_postprocessor.cpp <esp-dl/vision/classification/dl_cls_postprocessor.cpp>`
- :project_file:`imagenet_cls_postprocessor.hpp <esp-dl/vision/classification/imagenet_cls_postprocessor.hpp>`
- :project_file:`imagenet_cls_postprocessor.cpp <esp-dl/vision/classification/imagenet_cls_postprocessor.cpp>`
