# 使用 ESP-DL 部署模型的教程

在本教程中，我们将介绍如何量化模型，如何将 ESP-DL 前向推理结果与电脑端 esp-ppq 前向推理结果做比较。
其重在介绍如何量化模型，如何部署模型，不涉及模型输入数据的获取和处理，模型输入为随机数。

## 准备工作

在开始之前，请确保您已经安装了 ESP-IDF 开发环境，并且已经配置好了您的开发板。

除此之外，您还需要安装量化工具 [esp-ppq](https://github.com/espressif/esp-ppq)，该工具基于优秀的开源量化工具 [ppq](https://github.com/OpenPPL/ppq)，并添加适合 ESPRESSIF 芯片平台的客制化配置形成。

### esp-ppq 安装步骤

   ```bash
   git clone https://github.com/espressif/esp-ppq.git
   cd esp-ppq
   pip install -r requirements.txt
   python setup.py install
   ```

esp-ppq 的更多资料可参考其源码以及 ppq 的官方教学文档。

## 模型量化

ESP-DL 的量化工具是 [esp-ppq](https://github.com/espressif/esp-ppq)，其主体代码为开源量化工具 [ppq](https://github.com/OpenPPL/ppq)，esp-ppq 在其中添加了 ESPDL_INT8 目标平台，以及与其对应的 EspdlQuantizer.py, espdl_exporter.py，目的是使量化方式适配 ESPRESSIF 芯片平台。ppq 的主体代码改动较少，ppq 的相关学习资料同样适用于 esp-ppq。
**由于 ppq 加载解析的是 ONNX 文件，所以需要先将模型文件转换为 ONNX 文件格式**

esp-ppq 的使用方式，可参考 qmodel_constructor.py，运行方式如下：

   ```bash
   cd tools/qmodel_constructor
   python qmodel_constructor.py -c config/model_cfg.toml
   ```

主要文件介绍如下：
- `config/model_cfg.toml` 是模型量化时的配置，采用 TOML 文件格式编写，其中包含量化位宽，模型输入形状，onnx模型文件路径等配置项。
- `qmodel_constructor.py` 是量化脚本，调用 esp-ppq 接口：
   - `quantize_model_wrapper(...)` 是参照 ppq 的 [QAT 接口调用方式](https://github.com/espressif/esp-ppq/blob/master/ppq/samples/QAT/imagenet.py) 封装的函数，仅保留了 PTQ 量化部分，如还需进行 QAT 量化，可参考 ppq 的 QAT 示例代码。
   - 量化过程使用的 calibration 数据集，该示例是通过 `load_calibration_dataset` 接口随机生成。在使用时，需要替换为真实数据集。
   - 示例代码中调用了 graphwise_error_analyse, layerwise_error_analyse 等接口分析量化误差。如您也需调用这些接口，请记得将 calibration DataLoader 的 shuffle 置为 False，否则误差结果较大。
   - generate_test_value 接口用于生成模型的输入值，以及 esp-ppq 前向的输出值，它们将会被打包进生成的 ".espdl" 文件中，用于和 ESPRESSIF 芯片的前向结果做精度对齐测试，这将会导致生成的".espdl"文件体积增大。如无需精度测试，对 `PFL.Exporter(platform=self.platform).export(...)` 的入参 `valuesForTest` 置为 None 即可。
   - `PFL.Exporter(platform=self.platform).export(...)` 将会导出 ESPRESSIF 所需的模型文件：
      - `models/mobilenet_v2.espdl` 是 ESP-DL 的模型文件，由 esp-ppq 对 `models/mobilenet_v2.onnx` 量化后导出。
      - `models/mobilenet_v2.onnx` 是原始模型文件，由 pytorch/tensorflow 等框架导出。
      - `models/mobilenet_v2.info` 是 ".espdl" 模型文件的信息展示，包括：模型拓扑结构，量化 exponent 值，参数值，若模型包含测试输入值/输出值，则也会被包含其中。
      - `models/mobilenet_v2.json` 是模型的量化策略信息，包含量化位宽，min/max，scale 等信息，与 "*.info" 文件中的信息有部分重叠。


## 模型部署及推理精度测试

示例工程见 `examples/mobilenet_v2`，其目录结构如下：

   ```bash
   $ tree examples/mobilenet_v2
   examples/mobilenet_v2
   ├── CMakeLists.txt
   ├── main
   │   ├── app_main.cpp
   │   ├── CMakeLists.txt
   │   └── Kconfig.projbuild
   ├── models
   │   ├── mobilenet_v2.espdl
   │   ├── mobilenet_v2.info
   │   ├── mobilenet_v2.json
   │   └── mobilenet_v2.onnx
   ├── pack_model.py
   ├── partitions_esp32p4.csv
   ├── sdkconfig.defaults
   └── sdkconfig.defaults.esp32p4

   2 directories, 12 files
   ```

主要文件介绍如下：
- `main/app_main.cpp` 展示了如何调用 ESP-DL 接口加载、运行模型。
- `models` 目录存放模型相关文件，其中只有 `mobilenet_v2.espdl` 文件是必须的，将会被烧录到 flash 分区中。
- `pack_model.py` 模型打包脚本，会被 `main/CMakeLists.txt` 调用执行。
- `partitions_esp32p4.csv` 是分区表，在该工程中，模型文件 `models/mobilenet_v2.espdl` 将会被烧录到其中的 `model` 分区。
- `sdkconfig.defaults.esp32p4` 是项目配置，其中 `CONFIG_MODEL_FILE_PATH` 配置了模型文件路径，是基于该项目的相对路径。


### 模型加载运行

ESP-DL 支持自动构图及内存规划，目前支持的算子见 `esp-dl/dl/module/include`。
对于模型的加载运行，只需要参照下面，简单调用几个接口即可。该示例采用构造函数，以系统分区的形式加载模型。
更多加载方式请参考 [how_to_load_model](how_to_load_model_cn.md)

   ```cpp
   Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
   ......
   model->run(graph_test_inputs);
   ```

模型输入 `graph_test_inputs`，在该示例中，通过 `get_graph_test_inputs` 函数获得。
如下所示，该函数实现主要是构建 `TensorBase` 对象，传参 `input_data` 为模型输入数据 buffer 的首地址，buffer 中的数据需要是已经量化后的数据。
由于该示例展示的是如何测试 ESP-DL 推理精度，所以这里 `input_data` 获取的是已经被 esp-ppq 打包进 `mobilenet_v2.espdl` 文件中的测试输入值。
**input_data 需要是首地址16字节对齐的内存块，可通过 IDF 接口 `heap_caps_aligned_alloc` 分配**

   ```cpp
   const void *input_data = parser_instance->get_test_input_tensor_raw_data(input_name);
   if (input_data) {
         TensorBase *test_input =
            new TensorBase(input->shape, input_data, input->exponent, input->dtype, false, MALLOC_CAP_SPIRAM);
         test_inputs.emplace(input_name, test_input);
   }
   ```

> 对于输入数据的量化处理，ESP-DL P4 采用的 round 策略为 "Rounding half to even"，可参考 [bool TensorBase::assign(TensorBase *tensor)](../esp-dl/dl/typedef/src/dl_tensor_base.cpp) 中相关实现。量化所需的 exponent 等信息，可在 "*.info" 相关模型文件中查找。


### 推理结果获取及测试

在 `model->run(graph_test_inputs)` 运行完之后，我们就可以通过 `model->get_outputs()` 获取 ESP-DL 的推理结果了，返回的是 std::map 对象。之后，就可以参考 `compare_test_outputs` 函数实现，与模型文件中的 esp-ppq 推理结果做比较。
如果需要在 ESP-DL 中获取模型推理的中间结果，则需额外构建中间层对应的 `TensorBase` 对象，与其名字组成 `std::map` 对象传给 `user_outputs` 入参。`TensorBase` 对象的构造参照前面 `inputs TensorBase` 对象的构造。
   ```cpp
   void Model::run(std::map<std::string, TensorBase *> &user_inputs,
                  runtime_mode_t mode,
                  std::map<std::string, TensorBase *> user_outputs);
   ```

