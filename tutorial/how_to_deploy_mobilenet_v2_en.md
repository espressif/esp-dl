# Tutorial on Deploying Models Using ESP-DL

In this tutorial, we will introduce how to compare the forward inference results of ESP-DL with those of esp-ppq on a computer. The focus is on how to align ESP-DL and ESP-PPQ, without involving the acquisition and processing of model input data, which is random numbers in this case.

## Prerequisites

Before you begin, ensure that you have installed the ESP-IDF development environment and configured your development board.

Additionally, you need to install the quantization tool [esp-ppq](https://github.com/espressif/esp-ppq). This tool is based on the excellent open-source quantization tool [ppq](https://github.com/OpenPPL/ppq) and includes custom configurations suitable for ESPRESSIF chip platforms.

```bash
pip uninstall ppq
pip install git+https://github.com/espressif/esp-ppq.git
```

## Model Quantization

For MobileNet_v2 model quantization, please refer to [how_to_quantize_model_cn.md](./how_to_quantize_model_cn.md).

## Model Deployment and Inference Accuracy Testing

The example project can be found in `examples/mobilenet_v2`, with the following directory structure:

```bash
$ tree examples/mobilenet_v2
examples/mobilenet_v2
├── CMakeLists.txt
├── main
│   ├── app_main.cpp
│   ├── CMakeLists.txt
│   └── Kconfig.projbuild
├── models
│   ├── mobilenet_v2.espdl
│   ├── mobilenet_v2.info
│   ├── mobilenet_v2.json
│   └── mobilenet_v2.onnx
├── pack_model.py
├── partitions_esp32p4.csv
├── sdkconfig.defaults
└── sdkconfig.defaults.esp32p4

2 directories, 12 files
```

The main files are described as follows:
- `main/app_main.cpp` demonstrates how to load and run the model using ESP-DL interfaces.
- The `models` directory stores model-related files, with only the `mobilenet_v2.espdl` file being essential and will be flashed to the flash partition.
- `pack_model.py` is the model packaging script, which is invoked by `main/CMakeLists.txt`.
- `partitions_esp32p4.csv` is the partition table. In this project, the model file `models/mobilenet_v2.espdl` will be flashed to the `model` partition.
- `sdkconfig.defaults.esp32p4` is the project configuration, where `CONFIG_MODEL_FILE_PATH` configures the model file path, which is relative to the project.

### Model Loading and Running

ESP-DL supports automatic graph construction and memory planning. The currently supported operators can be found in `esp-dl/dl/module/include`.
For loading and running the model, you only need to call a few interfaces as shown below. This example uses the constructor to load the model in the form of a system partition.
For more loading methods, please refer to [how_to_load_model](how_to_load_model_cn.md).

```cpp
Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
......
model->run(graph_test_inputs);
```

The model input `graph_test_inputs` is obtained in this example through the `get_graph_test_inputs` function.   

As shown below, this function mainly constructs `TensorBase` objects. The parameter `input_data` is the starting address of the model input data buffer, and the data in the buffer needs to be already quantized.
Since this example demonstrates how to test the inference accuracy of ESP-DL, the `input_data` here is obtained from the test input values already packaged into the `mobilenet_v2.espdl` file by esp-ppq.
**The `input_data` needs to be a memory block aligned to a 16-byte boundary, which can be allocated using the IDF interface `heap_caps_aligned_alloc`.**

```cpp
const void *input_data = parser_instance->get_test_input_tensor_raw_data(input_name);
if (input_data) {
    TensorBase *test_input =
        new TensorBase(input->shape, input_data, input->exponent, input->dtype, false, MALLOC_CAP_SPIRAM);
    test_inputs.emplace(input_name, test_input);
}
```

> For the quantization processing of input data, ESP-DL P4 uses the "Rounding half to even" strategy. You can refer to the relevant implementation in [bool TensorBase::assign(TensorBase *tensor)](../esp-dl/dl/tensor/src/dl_tensor_base.cpp). The required exponent and other information for quantization can be found in the "*.info" related model files.

### Inference Result Testing

After running `model->run(graph_test_inputs)`, we can obtain the inference results of ESP-DL through `model->get_outputs()`, which returns an `std::map` object. Afterwards, you can refer to the `compare_test_outputs` function implementation to compare with the esp-ppq inference results in the model file.
If you need to obtain intermediate results of model inference in ESP-DL, you need to additionally construct `TensorBase` objects corresponding to the intermediate layers and form an `std::map` object with their names passed to the `user_outputs` parameter. The construction of `TensorBase` objects should refer to the construction of `inputs TensorBase` objects as mentioned earlier.

```cpp
void Model::run(std::map<std::string, TensorBase *> &user_inputs,
                runtime_mode_t mode,
                std::map<std::string, TensorBase *> user_outputs);
```