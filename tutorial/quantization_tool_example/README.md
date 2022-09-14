# Deploy a Model Step by Step [[中文]](./README_cn.md)

This tutorial shows how to deploy a model with the [Quantization Toolkit](../../tools/quantization_tool/README.md).

Note: For a model quantized on other platforms:
- If the quantization scheme (e.g. TFLite int8 model) is different from that of ESP-DL (see [Quantization Specification](../../docs/en/quantization_specification.md)), then the model cannot be deployed with ESP-DL.
- If the quantization scheme is identical, the model can be deployed with reference to [convert_tool_example](../convert_tool_example/README.md).

It is recommended to learn post-training quantization first.

# Preparation

## Step 1: Convert Your Model

In order to be deployed, the trained floating-point model must be converted to an integer model, the format compatible with ESP-DL. Given that ESP-DL uses a different quantization scheme and element arrangements compared with other platforms, please convert your model with our [Quantization Toolkit](../../tools/quantization_tool/README.md).

### Step 1.1: Convert to ONNX Format

The quantization toolkit runs based on [Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx), an open source format for AI models. Models trained on other platforms must be converted to ONXX format before using this toolkit.

Take TensorFlow for example. You can convert the trained TensorFlow model to an ONNX model by using [tf2onnx](https://github.com/onnx/tensorflow-onnx) in the script:

``` python
model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=13, output_path="mnist_model.onnx")
```

For more conversion examples, please refer to [xxx_to_onnx](../../tools/quantization_tool/examples).

### Step 1.2: Convert to ESP-DL Format

Once the ONNX model is ready, you can quantize the model with the quantization toolkit.

This section takes the example of [mnist_model_example.onnx](../../tools/quantization_tool/examples/mnist_model_example.onnx) and [example.py](../../tools/quantization_tool/examples/example.py).

#### Step 1.2.1: Set up the Environment

Environment requirements:
- Python == 3.7
- [Numba](https://github.com/numba/numba) == 0.53.1
- [ONNX](https://github.com/onnx/onnx) == 1.9.0
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) == 1.7.0
- [ONNX Optimizer](https://github.com/onnx/optimizer) == 0.2.6

You can install Python dependencies with [requirement.txt](../../tools/quantization_tool/requirements.txt):

```cpp
pip install -r requirements.txt
```

#### Step 1.2.2: Optimize Your Model

The optimizer in the quantization toolkit can optimize ONNX graph structures:

```python
# Optimize the onnx model
model_path = 'mnist_model_example.onnx'
optimized_model_path = optimize_fp_model(model_path)
```

#### Step 1.2.3: Convert and Quantize Your Model

Create a Python script *example.py* for conversion.

The calibrator in the quantization toolkit can quantize a floating-point model to an integer model which is compatible with ESP-DL. For post-training quantization, please prepare the calibration dataset (can be the subset of training dataset or validation dataset) with reference to the following example:

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

Run the conversion script with the following command:

```python
python example.py
```

And you will see the following log which includes the quantized coefficients for the model's input and output. These coefficients will be used in later steps when defining the model.

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

For more information about quantization toolkit API, please refer to [Quantization Toolkit API](../../tools/quantization_tool/quantization_tool_api.md).

# Deploy Your Model

## Step 2: Build Your Model

### Step 2.1: Derive a Class from the Model Class in [`"dl_layer_model.hpp"`](../../include/layer/dl_layer_model.hpp)

The quantization configuration is int16, so the model and subsequent layers inherit from ```<int16_t>```.

```c++
class MNIST : public Model<int16_t>
{
};
```

### Step 2.2: Declare Layers as Member Variables

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


### Step 2.3: Initialize Layers in Constructor Function

Initialize layers according to the files and log generated during [Model Quantization](#model-quantization). Parameters for the quantized model are sotred in [mnist_coefficient.cpp](./model/mnist_coefficient.cpp), and functions to get these parameters are stored in the header file [mnist_coefficient.hpp](./model/mnist_coefficient.hpp). 

For example, assume you want to define [convolutional layer](https://github.com/espressif/esp-dl/blob/master/include/layer/dl_layer_conv2d.hpp#L23) "l2". According to the log, the output coefficient is "-11", and this layer is named as "fused_gemm_0". You can call ```get_fused_gemm_0_filter()``` to get the layer's weight, call ```get_fused_gemm_0_bias()``` to get the layer's bias, and call ```get_fused_gemm_0_activation()``` to get the layer's activation. By configuring other parameters likewise, you can build a MNIST model as follows:

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

For how to initialize each Layer, please check the corresponding .hpp file in [esp-dl/include/layer/](../../include/layer/).


### Step 2.4: Implement `void build(Tensor<input_t> &input)`

To distinguish `build()` of `Model` and `build()` of `Layer`, we define:

* `Model.build()` as `build()` of `Model`;
* `Layer.build()` as `build()` of `Layer`.

In `Model.build()`, all `Layer.build()` are called. `Model.build()` is effective when input shape changes. If input shape does not change, `Model.build()` will not be called, thus saving computing time.

For when `Model.build()` is called, please check [Step 3: Run Your Model](#step-3-run-your-model).

For how to call `Layer.build()` of each layer, please refer to the corresponding .hpp file in [esp-dl/include/layer/](../../include/layer/).

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

### Step 2.5: Implement `void call(Tensor<input_t> &input)`

In `Model.call()`, all `Layer.call()` are called. For how to call `Layer.call()` of each layer, please refer to the corresponding .hpp file in [esp-dl/include/layer/](../../include/layer/).

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

## Step 3: Run Your Model

- Create an object of Model
- Define the input
    - Define the input image: The same size as the model's input (if the original image is obtained from a camera, the size might need to be adjusted)
    - Quantize the input: Normalize the input with the same method used in the training, convert the floating-point values after normalization to fixed-point values using *input_exponent* generated at [Step 1.2.2: Convert and Quantize Your Model](#step-122-convert-and-quantize-your-model), and configure the input coefficients

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

    - Define input tensor

        ```c++
        Tensor<int16_t> input;
        input.set_element((int16_t *)model_input).set_exponent(input_exponent).set_shape({28, 28, 1}).set_auto_free(false);
        ```
- Run `Model.forward()` for neural network inference. The progress of `Model.forward()` is:

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

  

**Example**: The object of MNIST and the `forward()` function in [`./main/main.cpp`](./main/app_main.cpp).

```c++
// model forward
MNIST model;
model.forward(input);
```
