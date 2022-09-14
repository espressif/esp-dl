# 模型部署的步骤介绍 [[English]](./README.md)

本案例介绍了如何使用我们提供的[量化工具包](../../tools/quantization_tool/README_cn.md) 来完成模型的部署。

注意：如果模型已通过其他平台量化，但使用的量化方法同 ESP-DL 的[量化规范](../../docs/zh_CN/quantization_specification.md)不同（如 TFLite int8 模型)，则无法使用 ESP-DL 进行部署；若量化方法一致，则可参考 [convert_tool_example](../convert_tool_example/README_cn.md) 案例来完成部署。

建议先学习训练后量化 (post-training quantization) 的相关知识。

# 准备

## 步骤 1：模型转换

为了部署模型，必须将训练好的浮点模型转换为 ESP-DL 适配的整型模型格式。由于本库使用的量化方式和参数排列方式与一些平台不同，请使用我们提供的工具[量化工具包](../../tools/quantization_tool/README_cn.md)来完成转换。

### 步骤 1.1：转换为 ONNX 格式模型

量化工具包基于开源的 AI 模型格式 [ONNX](https://github.com/onnx/onnx) 运行。其他平台训练得到的模型需要先转换为 ONNX 格式才能使用该工具包。

以 TensorFlow 平台为例，您可在脚本中使用 [tf2onnx](https://github.com/onnx/tensorflow-onnx) 将训练好的 TensorFlow 模型转换成 ONNX 模型格式，实例代码如下：

``` python
model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=13, output_path="mnist_model.onnx")
```

更多平台转换实例可参考 [xxx_to_onnx](../../tools/quantization_tool/examples)。

### 步骤 1.2：转换为 ESP-DL 适配模型

准备好 ONNX 模型后，即可使用量化工具包来完成量化。

本小节以 [mnist_model_example.onnx](../../tools/quantization_tool/examples/mnist_model_example.onnx) 和 [example.py](../../tools/quantization_tool/examples/example.py) 为例。

#### 步骤 1.2.1：环境准备

环境要求：
- Python == 3.7
- [Numba](https://github.com/numba/numba) == 0.53.1
- [ONNX](https://github.com/onnx/onnx) == 1.9.0
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) == 1.7.0
- [ONNX Optimizer](https://github.com/onnx/optimizer) == 0.2.6

您可以使用 [requirement.txt](../../tools/quantization_tool/requirements.txt) 来安装相关 Python 依赖包：

```cpp
pip install -r requirements.txt
```

#### 步骤 1.2.2：模型优化

量化工具包中的优化器可优化 ONNX 模型图结构：

```python
# Optimize the onnx model
model_path = 'mnist_model_example.onnx'
optimized_model_path = optimize_fp_model(model_path)
```

#### 步骤 1.2.3：模型量化和转换

创建 Python 脚本 *example.py* 来完成转换。

量化工具包中的校准器可将浮点模型量化成可适配 ESP-DL 的整型模型。为了实现训练后量化，请参考以下实例准备校准集，该校准集可以是训练集或验证集的子集：

```python
# Prepare the calibration dataset
# 'mnist_test_data.pickle': this pickle file stores test images from keras.datasets.mnist
with open('mnist_test_data.pickle', 'rb') as f: 
    (test_images, test_labels) = pickle.load(f)

# Normalize the calibration dataset in the same way as for training
test_images = test_images / 255.0

# Prepare the calibration dataset
calib_dataset = test_images[0:5000:50]
```

```python
# Calibration
model_proto = onnx.load(optimized_model_path)
print('Generating the quantization table:')

# Initialize an calibrator to quantize the optimized MNIST model to an int16 model using per-tensor minmax quantization method
calib = Calibrator('int16', 'per-tensor', 'minmax')
calib.set_providers(['CPUExecutionProvider'])

# Obtain the quantization parameter 
calib.generate_quantization_table(model_proto, calib_dataset, 'mnist_calib.pickle')

# Generate the coefficient files for esp32s3
calib.export_coefficient_to_cpp(model_proto, pickle_file_path, 'esp32s3', '.', 'mnist_coefficient', True)
```

使用以下命令运行准备好的转换脚本：

```python
python example.py
```

然后会看到如下的打印日志，其中包含了模型输入和每层输出的量化指数位，会用于接下来定义模型的步骤中：

```python
Generating the quantization table:
Converting coefficient to int16 per-tensor quantization for esp32s3
Exporting finish, the output files are: ./mnist_coefficient.cpp, ./mnist_coefficient.hpp

Quantized model info:
model input name: input, exponent: -15
Reshape layer name: sequential/flatten/Reshape, output_exponent: -15
Gemm layer name: fused_gemm_0, output_exponent: -11
Gemm layer name: fused_gemm_1, output_exponent: -11
Gemm layer name: fused_gemm_2, output_exponent: -9
```

关于工具包中更多 API 的介绍可阅读[量化工具包 API](../../tools/quantization_tool/quantization_tool_api_cn.md)。

# 部署模型

## 步骤 2：构建模型

### 步骤 2.1：从 [`"dl_layer_model.hpp"`](../../include/layer/dl_layer_model.hpp) 中的模型类派生一个新类

量化时配置的为 int16 量化，故模型以及之后的层均继承 ```<int16_t>```类型。

```c++
class MNIST : public Model<int16_t>
{
};
```

### 步骤 2.2：将层声明为成员变量

```c++
class MNIST : public Model<int16_t>
{
private:
    // Declare layers as member variables
    Reshape<int16_t> l1;
    Conv2D<int16_t> l2;
    Conv2D<int16_t> l3;

public:
    Conv2D<int16_t> l4; // Make the l4 public, as the l4.get_output() will be fetched outside the class.
};
```


### 步骤 2.3：用构造函数初始化层

根据[模型量化](#模型量化)得到的文件和打印日志来初始化层。量化后的模型参数存储在 [mnist_coefficient.cpp](./model/mnist_coefficient.cpp) 中，获取参数的函数头文件为 [mnist_coefficient.hpp](./model/mnist_coefficient.hpp)。 

例如定义[卷积层](https://github.com/espressif/esp-dl/blob/master/include/layer/dl_layer_conv2d.hpp#L23) “l2”，根据打印得知输出的指数位为 “-11”，该层的名称为 “fused_gemm_0”。您可调用 ```get_fused_gemm_0_filter()```获取改卷积层权重，调用 ```get_fused_gemm_0_bias()```获取该卷积层偏差，调用 ```get_fused_gemm_0_activation()```获取该卷积层激活参数。同理，配置其他参数，可构造整个 MNIST 模型结构如下：

```c++
class MNIST : public Model<int16_t>
{
    // ellipsis member variables
    
    MNIST() : l1(Reshape<int16_t>({1,1,784})),
              l2(Conv2D<int16_t>(-11, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_SAME_END, {}, 1, 1, "l1")),
              l3(Conv2D<int16_t>(-11, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), get_fused_gemm_1_activation(), PADDING_SAME_END, {}, 1, 1, "l2")),
              l4(Conv2D<int16_t>(-9, get_fused_gemm_2_filter(), get_fused_gemm_2_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l3")){}

};
```

有关如何初始化不同运算层，请查看 [esp-dl/include/layer/](../../include/layer/) 文件夹中相应的 .hpp 文件。


### 步骤 2.4：实现 `void build(Tensor<input_t> &input)`

为了便于区分`模型` `build()` 和`层` `build()`，现定义：

* `模型` `build()` 为 `Model.build()`；
* `层` `build()` 为 `Layer.build()`。

`Model.build()` 会调用所有 `Layer.build()`。`Model.build()` 仅在输入形状变化时有效。若输入形状没有变化，则 `Model.build()` 不会被调用，从而节省计算时间。

有关 `Model.build()` 何时被调用，请查看[步骤 3：运行模型](#步骤-3运行模型)。

有关如何调用每一层的 `Layer.build()`，请查看 [esp-dl/include/layer/](../../include/layer/) 文件夹中相应的 .hpp 文件。

```c++
class MNIST : public Model<int16_t>
{
    // ellipsis member variables
    // ellipsis constructor function
    
    void build(Tensor<int16_t> &input)
    {
        this->l1.build(input);
        this->l2.build(this->l1.get_output());
        this->l3.build(this->l2.get_output());
        this->l4.build(this->l3.get_output());
    }
};
```

### 步骤 2.5：实现 `void call(Tensor<input_t> &input)`

`Model.call()` 会调用所有 `Layer.call()`。有关如何调用每一层的 `Layer.call()`，请查看 [esp-dl/include/layer/](../../include/layer/) 文件夹中相应的 .hpp 文件。

```c++
class MNIST : public Model<int16_t>
{
    // ellipsis member variables
    // ellipsis constructor function
    // ellipsis build(...)

    void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();
    }
};
```

## 步骤 3：运行模型

- 创建模型对象
- 定义输入
    - 输入的图像大小：与模型输入大小一致（若原始图像是通过摄像头获取的，可能需要调整大小）
    - 量化输入：用训练时相同的方式对输入进行归一化，并使用步骤[步骤 1.2.2：模型量化和转换](#步骤-122模型量化和转换)输出日志中的 *input_exponent* 对归一化后的浮点值进行定点化，设置输入的指数位

        ```c++
        int input_height = 28;
        int input_width = 28;
        int input_channel = 1;
        int input_exponent = -15;
        int16_t *model_input = (int16_t *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof(int16_t *));
        for(int i=0 ;i<input_height*input_width*input_channel; i++){
            float normalized_input = example_element[i] / 255.0; //normalization
            model_input[i] = (int16_t)DL_CLIP(normalized_input * (1 << -input_exponent), -32768, 32767);
        }
        ```

    - 定义输入张量

        ```c++
        Tensor<int16_t> input;
        input.set_element((int16_t *)model_input).set_exponent(input_exponent).set_shape({28, 28, 1}).set_auto_free(false);
        ```
- 运行 `Model.forward()` 进行神经网络推理。`Model.forward()` 的过程如下：

  ```c++
  forward()
  {
    if (input_shape is changed)
    {
        Model.build();
    }
    Model.call();
  }
  ```

  

**示例**：[`./main/main.cpp`](./main/app_main.cpp) 文件中的 MNIST 对象和 `forward()` 函数。

```c++
// model forward
MNIST model;
model.forward(input);
```
