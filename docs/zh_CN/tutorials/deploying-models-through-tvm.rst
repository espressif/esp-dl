===========================
使用TVM自动生成模型部署项目
===========================

:link_to_translation:`en:[English]`

本案例介绍了使用 TVM 部署模型的完整流程。

准备
----

ESP-DL 是适配 ESP 系列芯片的深度学习推理框架。本库无法完成模型的训练，用户可使用 `TensorFlow <https://www.tensorflow.org/>`__，`PyTorch <https://pytorch.org/>`__ 等训练平台来训练模型，然后再通过 ESP-DL 部署模型。

为了帮助您理解本指南中的概念，建议您下载并熟悉以下工具：

- ESP-DL 库：详细了解 ESP-DL，包括量化规范、数据排布格式、支持的加速层。
- ONNX：一种用于表示深度学习模型的开放格式。
- TVM：一个端到端的深度学习编译框架，适用于CPU、GPU 和各种机器学习加速芯片。

安装 Python 依赖包
~~~~~~~~~~~~~~~~~~~~

环境要求：

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

您可以使用 :project_file:`tools/tvm/requirements.txt` 来安装相关 Python 依赖包：

.. code:: none

    pip install -r requirements.txt

配置 TVM 包
~~~~~~~~~~~~~~~~~~

您可以使用 :project_file:`tools/tvm/download.sh` 来下载我们已经编译好的 TVM 包：

.. code:: none

    . ./download.sh

TVM 包将被下载到 ``esp-dl/tvm/python/tvm`` 中. 下载完包后，需要设置环境变量 PYTHONPATH，指定 TVM 库的位置。可以在终端运行以下命令，也可以在 ``~/.bashrc`` 文件中添加以下行。

.. code:: python

    export PYTHONPATH='$PYTHONPATH:/path-to-esp-dl/esp-dl/tvm/python'

步骤 1：模型量化
----------------

为了部署的模型在芯片上能快速运行，需要将训练好的浮点模型转换定点模型。

常见的量化手段分为两种：

1. 训练后量化（post-training quantization）：将已有的模型转化为定点数表示。这种方法相对简单，不需要重新训练网络，但在有些情况下会有一定的精度损失。
2. 量化训练（quantization-aware training）：在网络训练过程中考虑量化带来的截断误差和饱和效应。这种方式使用上更复杂，但效果会更好。

ESP-DL 中目前只支持第一种方法。若无法接受量化后的精度损失，请考虑使用第二种方式。

步骤 1.1：转换为 ONNX 格式模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

量化脚本基于开源的 AI 模型格式 `ONNX <https://github.com/onnx/onnx>`__ 运行。其他平台训练得到的模型需要先转换为 ONNX 格式才能使用该工具包。

以 TensorFlow 平台为例，您可在脚本中使用 `tf2onnx <https://github.com/onnx/tensorflow-onnx>`__ 将训练好的 TensorFlow 模型转换成 ONNX 模型格式，实例代码如下：

.. code:: python

    model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=13, output_path="mnist_model.onnx")

更多平台转换实例可参考 :project:`xxx_to_onnx <tools/quantization_tool/examples>`。

步骤 1.2：预处理
~~~~~~~~~~~~~~~~

在预处理过程中，将会对 float32 模型进行一系列操作，以便为量化做好准备。

.. code:: python

    python -m onnxruntime.quantization.preprocess --input model.onnx --output model_opt.onnx

参数说明：

-  input：指定输入的待处理 float32 模型文件路径。
-  output：指定输出的处理后 float32 模型文件路径。

预处理包括以下可选步骤：

-  符号形状推断（Symbolic Shape Inference）：这个步骤会对输入和输出的张量形状进行推断。符号形状推断可以帮助模型在推理之前确定张量的形状，以便更好地进行后续优化和处理。
-  ONNX Runtime模型优化（ONNX Runtime Model Optimization）：这个步骤使用 ONNX Runtime 来进行模型优化。ONNX Runtime 是一个高性能推理引擎，可以针对特定硬件和平台进行模型优化，以提高推理速度和效率。模型优化包括诸如图优化、内核融合、量化等技术，以优化模型的执行。
-  ONNX 形状推断（ONNX Shape Inference）：这个步骤根据ONNX 格式模型推断张量形状，从而更好地理解和优化模型。ONNX 形状推断可以为模型中的张量分配正确的形状，帮助后续的优化和推理。

步骤 1.3：量化
~~~~~~~~~~~~~~

量化工具接受预处理后的 float32 模型作为输入，并生成一个 int8 量化模型。

.. code:: python

    python esp_quantize_onnx.py --input_model model_opt.onnx --output_model model_quant.onnx --calibrate_dataset calib_img.npy

参数说明：

-  input_model：指定输入模型的路径和文件名，应为预处理过的 float32 模型，以 ONNX 格式（.onnx）保存。
-  output_model：指定输出模型的路径和文件名，将是量化处理后的模型，以ONNX格式（.onnx）保存。
-  calibrate_dataset：指定用于校准的数据集路径和文件名，应为包含校准数据的 NumPy 数组文件（.npy），用于生成量化器的校准统计信息。

:project_file:`tools/tvm/esp_quantize_onnx.py` 中创建了一个用于模型的输入数据读取器，使用这些输入数据来运行模型，以校准每个张量的量化参数，并生成量化模型。具体流程如下：

-  创建输入数据读取器：首先，创建一个输入数据读取器，用于从数据源中读取输入的校准数据。用于校准的数据集应保存为 NumPy 数组文件，其中包含输入图片的集合。例如 model.onnx 的输入大小为 [32, 32, 3]，calibe_images.npy 存储的则是 500 张校准图片的数据，形状为 [500, 32, 32, 3]。
-  运行模型进行校准：接下来，代码会使用输入数据读取器提供的数据来运行模型。通过将输入数据传递给模型，模型会进行推断（inference），生成输出结果。在这个过程中，代码会根据实际输出结果和预期结果，校准每个张量的量化参数。这个校准过程旨在确定每个张量的量化范围、缩放因子等参数，以便在后续的量化转换中准确地表示数据。
-  生成量化模型：校准完量化参数后，代码将使用这些参数对模型进行量化转换。这个转换过程会将模型中的浮点数权重和偏差替换为量化表示，使用较低的位精度来表示数值。生成的量化模型会保留量化参数，以便在后续的部署过程中正确还原数据。请注意，不要在这个量化模型上运行推理过程，可能会与板上运行的结果不一致，具体的调试流程请参考后续章节。

步骤 2：部署模型
----------------

将量化后的 ONNX 模型部署到 ESP 系列芯片上。只有在 ESP32-S3 上运行的部分算子支持 ISA 加速。

支持加速的算子请查看 :project:`include/layer`。更多 ISA 相关介绍请查看 `《ESP32-S3 技术参考手册》 <https://www.espressif.com.cn/sites/default/files/documentation/esp32-s3_technical_reference_manual_cn.pdf>`__。

步骤 2.1：准备输入
~~~~~~~~~~~~~~~~~~

准备一张输入图像，输入的图像大小应该与得到的 ONNX 模型输入大小一致。模型输入大小可通过 Netron 工具查看。

步骤 2.2：部署项目生成
~~~~~~~~~~~~~~~~~~~~~~

使用 TVM 自动生成一个项目，用来运行给定输入的模型推理。

.. code:: python

    python export_onnx_model.py --target_chip esp32s3 --model_path model_quant.onnx --img_path input_sample.npy --template_path "esp_dl/tools/tvm/template_project_for_model" --out_path "esp_dl/example"



参数说明：

-  target_chip: 目标芯片的名称。上述命令中目标芯片是esp32s3，表示生成的示例项目将针对 ESP32-S3 芯片进行优化。
-  model_path: 经过量化的 ONNX 模型的路径。请提供模型的完整路径和文件名。
-  img_path: 输入图像的路径。请提供输入图像的完整路径和文件名。
-  template_path: 用于示例项目的模板路径。默认提供的模板程序为 :project:`tools/tvm/template_project_for_model`。
-  out_path: 生成的示例项目的输出路径。请提供目标目录的路径。

:project_file:`tools/tvm/export_onnx_model.py` 将量化的 ONNX 模型加载到 TVM 中，并对模型进行布局转换和优化，经过一定的预处理后最终编译成适配 ESP 后端的代码。具体流程如下：

-  通过 ``tvm.relay.frontend.from_onnx`` 函数将 ONNX 模型转换为 TVM 的中间表示（Relay IR）。
-  将 ONNX 默认的 NCHW 布局转换为 ESP-DL 期望的布局 NHWC。定义 ``desired_layouts`` 字典，指定要进行布局转换的操作和期望的布局。这里将对模型中的 "qnn.conv2d" 和 "nn.avg_pool2d" 的布局进行转换。转换通过 TVM 的 transform 机制来完成。
-  执行针对部署到 ESP 芯片的预处理，包括算子的重写、融合、标注。
-  通过 TVM 的 BYOC（Bring Your Own Codegen） 机制编译生成模型的 C 代码，包括支持的加速算子。BYOC 是 TVM 的机制，允许用户自定义代码生成。BYOC 可以将模型的特定部分编译为 ESP-DL 的加速算子，以便在目标硬件上进行加速。使用 TVM 的 ``tvm.build`` 函数，将 Relay IR 编译为目标硬件上的可执行代码。
-  将生成的模型部分的代码集成到提供的模板工程文件中。

步骤 3：运行模型
----------------

步骤 3.1：运行推理
~~~~~~~~~~~~~~~~~~

上一步生成的工程文件 ``new_project`` 结构如下：

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

配置好终端 ESP-IDF（请注意 ESP-IDF 的版本）环境后，即可运行项目：

::

    idf.py set-target esp32s3
    idf.py flash monitor

步骤 3.2：调试
~~~~~~~~~~~~~~

模型的推理过程在 components/tvm_model/model/codegen/host/src/default_lib1.c 里的 ``tvmgen_default___tvm_main__`` 函数中定义。如果想查看板子上运行的模型的输出是否与预期相符，可以参考以下步骤。

模型的第一层为 conv2d 算子，从函数体中可以看到 ``tvmgen_default_esp_main_0`` 调用了 ESP-DL 提供的 conv2d 加速算子来实现第一层的卷积操作。添加下列示例代码可以获得该层的结果，示例代码只输出了前 16 个数。

::

    int8_t *out = (int8_t *)sid_4_let;
    for(int i=0; i<16; i++)
        printf("%d,",out[i]);
    printf("\n");

``export_onnx_model.py`` 中的 ``debug_onnx_model`` 函数用于调试模型板上运行的结果，验证是否符合预期。请确保模型完成部署、并在板上运行后，再调用 ``debug_onnx_model`` 函数。


::

    debug_onnx_model(args.target_chip, args.model_path, args.img_path)

``debug_onnx_model`` 函数里使用``evaluate_onnx_for_esp`` 函数处理 Relay 使其与板上计算方法一致，请注意这个函数仅适用于调试阶段。

::

    mod = evaluate_onnx_for_esp(mod, params)

    m = GraphModuleDebug(
            lib["debug_create"]("default", dev),
            [dev],
            lib.graph_json,
            dump_root = os.path.dirname(os.path.abspath(model_path))+"/tvmdbg",
        )

通过 TVM 的 GraphModuleDebug 将计算图的全部信息输出到 ``tvmdbg`` 目录下，输出的 ``tvmdbg_graph_dump.json`` 文件中包含了图中各个运算结点的信息。更多说明可查看 `TVM Debugger 文档 <https://tvm.apache.org/docs/arch/debugger.html>`__。输出文件中第一个卷积输出层的名称为 ``tvmgen_default_fused_nn_relu``，输出的大小为[1, 32, 32, 16]，输出类型为 int8。

::

    tvm_out = tvm.nd.empty((1,32,32,16),dtype="int8")
    m.debug_get_output("tvmgen_default_fused_nn_relu", tvm_out)
    print(tvm_out.numpy().flatten()[0:16])

根据上述信息创建一个变量存储这一层的输出，可以比较这一输出是否与板子上运行得到的结果一致。
