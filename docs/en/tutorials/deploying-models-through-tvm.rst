========================================================
Auto-Generating Model Deployment Project using TVM
========================================================

:link_to_translation:`zh_CN:[中文]`

This case introduces the complete process of deploying a model with TVM.

Preparation
-----------

ESP-DL is a deep learning inference framework tailored for the ESP series of chips. This library cannot accomplish model training, and users can utilize training platforms such as `TensorFlow <https://www.tensorflow.org/>`__，`PyTorch <https://pytorch.org/>`__ to train their models, and then deploy the models through ESP-DL.

To help you understand the concepts in this guide, it is recommended to download and familiarize yourself with the following tools:

-  ESP-DL library: A library that includes quantization specifications, data layout formats, and supported acceleration layers.
-  ONNX: An open format for representing deep learning models.
-  TVM: An end-to-end deep learning compilation framework suitable for CPUs, GPUs, and various machine learning acceleration chips.

Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Environment requirements:

- Python == 3.7 or 3.8
- `ONNX <https://github.com/onnx/onnx>`__ == 1.12.0
- `ONNX Runtime <https://github.com/microsoft/onnxruntime>`__ == 1.14.0
- `ONNX Optimizer <https://github.com/onnx/optimizer>`__ == 0.2.6
- `ONNX Simplifier <https://github.com/daquexian/onnx-simplifier>`__ == 0.4.17
- numpy
- decorator
- attrs
- typing-extensions
- psutil
- scipy

You can use the :project_file:`tools/tvm/requirements.txt` file to install the related Python packages:

.. code:: none

    pip install -r requirements.txt

Set TVM Package Path
~~~~~~~~~~~~~~~~~~~~~

You can use the :project_file:`tools/tvm/download.sh` file to download the compiled TVM packages:

.. code:: none

    . ./download.sh

The TVM package will be downloaded to ``esp-dl/tvm/python/tvm``. After finish downloading, you can set the PYTHONPATH environment variable to specify the location of the TVM library. To achieve this, run the following command in the terminal, or add the following line to the ``~/.bashrc`` file.

.. code:: python

    export PYTHONPATH='$PYTHONPATH:/path-to-esp-dl/esp-dl/tvm/python'

Step 1: Quantize the Model
--------------------------

In order to run the deployed model quickly on the chip, the trained floating-point model needs to be converted to a fixed-point model.

Common quantization methods are divided into two types:

1. Post-training quantization: Converts the existing model to a fixed-point representation. This method is relatively simple and does not require retraining of the network, but in some cases there may be some loss of accuracy.
2. Quantization-aware training: Considers the truncation error and saturation effect brought by quantization during network training. This method is more complex to use, but the effect will be better.

ESP-DL currently only supports the first method. If you cannot accept the loss of accuracy after quantization, please consider using the second method.

Step 1.1: Convert the Model to ONNX Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantization script is based on the open-source AI model format `ONNX <https://github.com/onnx/onnx>`__. Models trained on other platforms need to be converted to the ONNX format to use this toolkit.

Taking the TensorFlow platform as an example. To convert the trained TensorFlow model to the ONNX model format, you can use `tf2onnx <https://github.com/onnx/tensorflow-onnx>`__ in a script. Example code is as follows:

.. code:: python

    model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=13, output_path="mnist_model.onnx")

For more examples about converting model formats, please refer to :project:`xxx_to_onnx <tools/quantization_tool/examples>`.

Step 1.2: Preprocess the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During preprocessing, a series of operations will be performed on the float32 model to prepare for quantization.

.. code:: python

    python -m onnxruntime.quantization.preprocess --input model.onnx --output model_opt.onnx

Parameter descriptions:

-  input: Specifies the path of the float32 model file to be processed.
-  output: Specifies the path of the processed float32 model file.

Preprocessing includes the following optional steps:

-  Symbolic Shape Inference: Infers the shape of the input and output tensors. Symbolic shape inference can help determine the shape of the tensor before inference, to better perform subsequent optimization and processing.
-  ONNX Runtime Model Optimization: Optimizes the model with ONNX Runtime, a high-performance inference engine that can optimize models for specific hardware and platforms to improve inference speed and efficiency. Models can be optimized by techniques such as graph optimization, kernel fusion, quantization for better execution.
-  ONNX Shape Inference: Infers the shape of the tensor based on the ONNX format model to better understand and optimize the model. ONNX shape inference can allocate the correct shape for the tensors in the model and help with subsequent optimization and inference.

Step 1.3：Quantize the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantization tool takes the preprocessed float32 model as input and generates an int8 quantized model.

.. code:: python

    python esp_quantize_onnx.py --input_model model_opt.onnx --output_model model_quant.onnx --calibrate_dataset calib_img.npy

Parameter descriptions:

-  input_model: Specifies the path and filename of the input model, which should be a preprocessed float32 model saved in ONNX format (.onnx).
-  output_model: Specifies the path and filename of the output model after quantization, saved in ONNX format (.onnx).
-  calibrate_dataset: Specifies the path and filename of the dataset used for calibration. The dataset should be a NumPy array file (.npy) containing calibration data, used to generate the calibration statistics for the quantizer.

:project_file:`tools/tvm/esp_quantize_onnx.py` creates a data reader for the input data of the model, uses this input data to run the model, calibrates the quantization parameters of each tensor, and generates a quantized model. The specific process is as follows:

-  Create an input data reader: First, an input data reader will be created to read the calibration data from the data source. The dataset used for calibration should be saved as a NumPy array file. It contains a collection of input images. For example, the input size of model.onnx is [32, 32, 3], and calibe_images.npy stores the data of 500 calibration images with a shape of [500, 32, 32, 3].
-  Run the model for calibration: Next, the code will run the model using the data provided by the input data reader. By passing the input data to the model, the model will perform the inference operation and generate output results. During this process, the code will calibrate the quantization parameters of each tensor according to the actual output results and the expected results. This calibration process aims to determine the quantization range, scaling factor and other parameters of each tensor, so as to accurately represent the data in the subsequent quantization conversion.
-  Generate Quantized Model: After the quantization parameters have been calibrated, the code will use these parameters to perform quantization conversion on the model. This conversion process will replace the floating-point weights and biases in the model with quantized representations, using lower bit precision to represent numerical values. The generated quantized model will retain the quantization parameters, so the data can be correctly restored during the subsequent deployment process. Please do not run the inference process on this quantized model, as it may produce results inconsistent with those obtained when running on the board. For specific debugging procedures, please refer to the following sections.

Step 2: Deploy the Model
------------------------

Deploy the quantized ONNX model on the ESP series chips. Only some operators running on ESP32-S3 are supported by ISA related acceleration.

For operators supported by acceleration, please see :project:`include/layer`. For more information about ISA, please refer to `ESP32-S3 Technical Reference Manual <https://www.espressif.com.cn/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf>`__.

Step 2.1: Prepare the Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare an input image, whose size should be consistent with the input size of the obtained ONNX model. You can view the model input size through the Netron tool.

Step 2.2: Generate the Project for Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use TVM to automatically generate a project for inferring model with the given input.

.. code:: python

    python export_onnx_model.py --target_chip esp32s3 --model_path model_quant.onnx --img_path input_sample.npy --template_path "esp_dl/tools/tvm/template_project_for_model" --out_path "esp_dl/example"

Parameter descriptions:

-  target_chip: The name of the target chip, which is esp32s3 in the command above. It specifies that the generated example project will be optimized for the ESP32-S3 chip.
-  model_path: The path of the quantized ONNX model. Please provide the full path and filename of the model.
-  img_path: The path of the input image. Please provide the full path and filename of the input image.
-  template_path: The template path for the example project. The template program by default is :project:`tools/tvm/template_project_for_model`.
-  out_path: The output path of the generated example project. Please provide a path to a target directory.

The script :project_file:`tools/tvm/export_onnx_model.py` loads the quantized ONNX model into TVM, and converts and optimizes the model's layout. After  preprocessing, it finally compiles the model into code suitable for the ESP backend. The specific process is as follows:

-  Convert the ONNX model to TVM's intermediate representation (Relay IR) via the ``tvm.relay.frontend.from_onnx`` function.
-  Convert the default NCHW layout of ONNX to the NHWC layout expected by ESP-DL. Define the ``desired_layouts`` dictionary, specifying the operations to convert layout and the expected layout. In this case, the layout of "qnn.conv2d" and "nn.avg_pool2d" in the model will be converted. The conversion is done via TVM's transform mechanism.
-  Execute preprocessing steps for deploying to ESP chips, including operator rewriting, fusion, and annotation.
-  Generate the model's C code via TVM's BYOC (Bring Your Own Codegen) mechanism, including supported accelerated operators. BYOC is a mechanism of TVM that allows users to customize the behavior of code generation. By using BYOC, specific parts of the model are compiled into ESP-DL's accelerated operators for acceleration on the target hardware. Using TVM's ``tvm.build`` function, Relay IR is compiled into executable code on the target hardware.
-  Integrate the generated model code into the provided template project files.

Step 3: Run the Model
---------------------

Step 3.1: Run Inference
~~~~~~~~~~~~~~~~~~~~~~~~

The structure of the project files ``new_project`` generated in the previous step is as follows:

::

    ├── CMakeLists.txt
    ├── components
    │   ├── esp-dl
    │   └── tvm_model
    │       ├── CMakeLists.txt
    │       ├── crt_config
    │       └── model
    ├── main
    │   ├── app_main.c
    │   ├── input_data.h
    │   ├── output_data.h
    │   └── CMakeLists.txt
    ├── partitions.csv
    ├── sdkconfig.defaults
    ├── sdkconfig.defaults.esp32
    ├── sdkconfig.defaults.esp32s2
    ├── sdkconfig.defaults.esp32s3

Once the ESP-IDF terminal environment is properly configured (please note the version of ESP-IDF), you can run the project:

::

    idf.py set-target esp32s3
    idf.py flash monitor

Step 3.2: Debug
~~~~~~~~~~~~~~~

The inference process of the model is defined in the function ``tvmgen_default___tvm_main__`` located in components/tvm_model/model/codegen/host/src/default_lib1.c. To verify whether the output of the model running on the board matches the expected output, you can follow the steps below.

The first layer of the model is a conv2d operator. From the function body, it can be seen that ``tvmgen_default_esp_main_0`` calls the conv2d acceleration operator provided by ESP-DL to perform the convolution operation of the first layer. You can add the following code snippet to obtain the results of this layer. In this example code, only the first 16 numbers are outputted.

::

    int8_t *out = (int8_t *)sid_4_let;
    for(int i=0; i<16; i++)
        printf("%d,",out[i]);
    printf("\n");

``export_onnx_model.py`` provides the ``debug_onnx_model`` function for debugging the results of the model running on the board, so as to verify if they match the expected output. Make sure to call the ``debug_onnx_model`` function after the model has been deployed and executed on the board to examine the results and evaluate if they align with the expected outcomes.

::

    debug_onnx_model(args.target_chip, args.model_path, args.img_path)

The ``evaluate_onnx_for_esp`` function inside ``debug_onnx_model`` is used to align Relay with the computation method on the board, specifically for debugging purposes. It is important to note that this function is intended for use only during the debugging phase.

::

    mod = evaluate_onnx_for_esp(mod, params)
    
    m = GraphModuleDebug(
            lib["debug_create"]("default", dev),
            [dev],
            lib.graph_json,
            dump_root = os.path.dirname(os.path.abspath(model_path))+"/tvmdbg",
        )

The GraphModuleDebug in TVM can be used to output all the information about the computational graph to the ``tvmdbg`` directory. The resulting ``tvmdbg_graph_dump.json`` file contains information about each operation node in the graph. For more details, you can refer to the TVM Debugger documentation at `TVM Debugger <https://tvm.apache.org/docs/arch/debugger.html>`__.
From the file, we can see that the name of the first convolutional output layer is ``tvmgen_default_fused_nn_relu``, the output shape of this layer is [1, 32, 32, 16], and the data type of the output is int8.

::

    tvm_out = tvm.nd.empty((1,32,32,16),dtype="int8")
    m.debug_get_output("tvmgen_default_fused_nn_relu", tvm_out)
    print(tvm_out.numpy().flatten()[0:16])

Create a variable based on the provided information to store the output of this layer. You can then compare this output to the results obtained from running the model on the board to verify if they are consistent.
