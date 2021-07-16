# How to Implement a Custom Model Step by Step

This tutorial shows how to implement a custom model with ESP-DL step by step. For a more vivid explanation, the example is an runnable project about [MNIST](https://tensorflow.google.cn/datasets/catalog/mnist?hl=en) classification mission, hereinafter referred to as MNIST.

About how to implement a custom layer, please check [Implement Custom Layer](../docs/en/implement_custom_layer.md) to know.

Here is the file structure,

```bash
tutorial/
├── CMakeLists.txt
├── main
│   ├── app_main.cpp
│   └── CMakeLists.txt
├── model
│   ├── mnist_coefficient.cpp	(generated after step3)
│   ├── mnist_coefficient.hpp	(generated after step3)
│   ├── mnist_model.hpp
│   └── npy
│       ├── config.json
│       ├── l1_bias.npy
│       ├── l1_filter.npy
│       ├── l2_compress_bias.npy
│       ├── l2_compress_filter.npy
│       ├── l2_depth_filter.npy
│       ├── l3_a_compress_bias.npy
│       ├── l3_a_compress_filter.npy
│       ├── l3_a_depth_filter.npy
│       ├── l3_b_compress_bias.npy
│       ├── l3_b_compress_filter.npy
│       ├── l3_b_depth_filter.npy
│       ├── l3_c_compress_bias.npy
│       ├── l3_c_compress_filter.npy
│       ├── l3_c_depth_filter.npy
│       ├── l3_d_compress_bias.npy
│       ├── l3_d_compress_filter.npy
│       ├── l3_d_depth_filter.npy
│       ├── l3_e_compress_bias.npy
│       ├── l3_e_compress_filter.npy
│       ├── l3_e_depth_filter.npy
│       ├── l4_compress_bias.npy
│       ├── l4_compress_filter.npy
│       ├── l4_depth_activation.npy
│       ├── l4_depth_filter.npy
│       ├── l5_compress_bias.npy
│       ├── l5_compress_filter.npy
│       ├── l5_depth_activation.npy
│       └── l5_depth_filter.npy
└── README.md
```



## Step-1: Save Model Coefficient

Save the float-point coefficients of model to npy files by [numpy.save(file=f'{filename}', arr=coefficient)](https://numpy.org/doc/stable/reference/generated/numpy.save.html?highlight=save#numpy.save). For each neural-network operation, we probably need:

- **filter**: saved with filename `'{layer_name}_filter.npy'`
- **bias**: saved with filename `'{layer_name}_bias.npy'`
- **activation**: activations with coefficient such as *LeakyReLU* and *PReLU* saved with filename `'{layer_name}_activation.npy'`

**Example:** [`./model/npy/`](./model/npy/) contains the coefficient.npy files of the MNIST.



## Step-2: Write Model Configuration

Follow the [**Specification of config.json**](../docs/en/specification_of_config_json.md) and write the config.json file about the model.

**Example:** [`./model/npy/config.json`](./model/npy/config.json) is the configuration of the MNIST.



## Step-3 Convert Model Coefficient

Make sure that coefficient.npy files and config.json file are ready and in the same folder. Follow the [**Usage of convert.py**](../docs/en/usage_of_convert_py.md) to convert coefficients into C/C++ code. Then, the coefficients of each layer could be fetched by calling `get_{layer_name}_***()`, for example, getting filter of "l1" by calling `get_l1_filter()`.

**Example:**

Run 

```bash
python ../convert.py -i ./model/npy/ -n mnist_coefficient -o ./model/
```

Two files `mnist_coefficient.cpp` and `mnist_coefficient.hpp` are generated in [`./model/`](./model/).



## Step-4 Build a Model

### Step-4.1 Derive from the Model class in [`"dl_layer_model.hpp"`](../include/layer/dl_layer_model.hpp)

```c++
class MNIST : public Model<int16_t>
{
};
```



### Step-4.2 Declare layers as member variables

```c++
class MNIST : public Model<int16_t>
{
private:
    Conv2D<int16_t> l1;                  // a layer named l1
    DepthwiseConv2D<int16_t> l2_depth;   // a layer named l2_depth
    Conv2D<int16_t> l2_compress;         // a layer named l2_compress
    DepthwiseConv2D<int16_t> l3_a_depth; // a layer named l3_a_depth
    Conv2D<int16_t> l3_a_compress;       // a layer named l3_a_compress
    DepthwiseConv2D<int16_t> l3_b_depth; // a layer named l3_b_depth
    Conv2D<int16_t> l3_b_compress;       // a layer named l3_b_compress
    DepthwiseConv2D<int16_t> l3_c_depth; // a layer named l3_c_depth
    Conv2D<int16_t> l3_c_compress;       // a layer named l3_c_compress
    DepthwiseConv2D<int16_t> l3_d_depth; // a layer named l3_d_depth
    Conv2D<int16_t> l3_d_compress;       // a layer named l3_d_compress
    DepthwiseConv2D<int16_t> l3_e_depth; // a layer named l3_e_depth
    Conv2D<int16_t> l3_e_compress;       // a layer named l3_e_compress
    Concat2D<int16_t> l3_concat;         // a layer named l3_concat
    DepthwiseConv2D<int16_t> l4_depth;   // a layer named l4_depth
    Conv2D<int16_t> l4_compress;         // a layer named l4_compress
    DepthwiseConv2D<int16_t> l5_depth;   // a layer named l5_depth

public:
    Conv2D<int16_t> l5_compress; // a layer named l5_compress. Make the l5_compress public, as the l5_compress.output will be fetched outside the class.
};
```



### Step-4.3 Initialize layers in constructor function

Initialize layers with the coefficients from the `"mnist_coefficient.hpp"` generated in [Step-3](#Step-3-Convert-Model-Coefficient).

The details about how to initialize each kind of Layer, please check the [declaration of them](../include/layer/).

```c++
class MNIST : public Model<int16_t>
{
    // ellipsis menber variables
    
    MNIST() : l1(Conv2D<int16_t>(-2, get_l1_filter(), get_l1_bias(), get_l1_relu(), PADDING_VALID, 2, 2, "l1")),
              l2_depth(DepthwiseConv2D<int16_t>(-1, get_l2_depth_filter(), NULL, get_l2_depth_relu(), PADDING_SAME, 2, 2, "l2_depth")),
              l2_compress(Conv2D<int16_t>(-3, get_l2_compress_filter(), get_l2_compress_bias(), NULL, PADDING_SAME, 1, 1, "l2_compress")),
              l3_a_depth(DepthwiseConv2D<int16_t>(-1, get_l3_a_depth_filter(), NULL, get_l3_a_depth_relu(), PADDING_VALID, 1, 1, "l3_a_depth")),
              l3_a_compress(Conv2D<int16_t>(-12, get_l3_a_compress_filter(), get_l3_a_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_a_compress")),
              l3_b_depth(DepthwiseConv2D<int16_t>(-2, get_l3_b_depth_filter(), NULL, get_l3_b_depth_relu(), PADDING_VALID, 1, 1, "l3_b_depth")),
              l3_b_compress(Conv2D<int16_t>(-12, get_l3_b_compress_filter(), get_l3_b_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_b_compress")),
              l3_c_depth(DepthwiseConv2D<int16_t>(-12, get_l3_c_depth_filter(), NULL, get_l3_c_depth_relu(), PADDING_SAME, 1, 1, "l3_c_depth")),
              l3_c_compress(Conv2D<int16_t>(-12, get_l3_c_compress_filter(), get_l3_c_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_c_compress")),
              l3_d_depth(DepthwiseConv2D<int16_t>(-12, get_l3_d_depth_filter(), NULL, get_l3_d_depth_relu(), PADDING_SAME, 1, 1, "l3_d_depth")),
              l3_d_compress(Conv2D<int16_t>(-11, get_l3_d_compress_filter(), get_l3_d_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_d_compress")),
              l3_e_depth(DepthwiseConv2D<int16_t>(-11, get_l3_e_depth_filter(), NULL, get_l3_e_depth_relu(), PADDING_SAME, 1, 1, "l3_e_depth")),
              l3_e_compress(Conv2D<int16_t>(-12, get_l3_e_compress_filter(), get_l3_e_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_e_compress")),
              l3_concat("l3_concat"),
              l4_depth(DepthwiseConv2D<int16_t>(-12, get_l4_depth_filter(), NULL, get_l4_depth_leaky_relu(), PADDING_VALID, 1, 1, "l4_depth")),
              l4_compress(Conv2D<int16_t>(-11, get_l4_compress_filter(), get_l4_compress_bias(), NULL, PADDING_VALID, 1, 1, "l4_compress")),
              l5_depth(DepthwiseConv2D<int16_t>(-10, get_l5_depth_filter(), NULL, get_l5_depth_leaky_relu(), PADDING_VALID, 1, 1, "l5_depth")),
              l5_compress(Conv2D<int16_t>(-9, get_l5_compress_filter(), get_l5_compress_bias(), NULL, PADDING_VALID, 1, 1, "l5_compress")) {}
};
```



### Step-4.4 Implement `void build(Tensor<input_t> &input)`

To distinguish `build()` of `Model` and `build()` of `Layer`, let `Model.build()` stands for `build()` of `Model`, `Layer.build()` stands for `build()` of `Layer`.

`Model.build()` is implemented with calling all `Layer.build()`. `Model.build()` is effective when its input shape changes. 

The details about when is `Model.build()` called, please check [Step-5](#Step-5-Run-a-Model)

The details about how to call `Layer.build()` of each kind of Layer, please check the [declaration of them](../include/layer/).

```c++
class MNIST : public Model<int16_t>
{
    // ellipsis menber variables
    // ellipsis construcor function
    
    void build(Tensor<int16_t> &input)
    {
        this->l1.build(input);
        this->l2_depth.build(this->l1.output);
        this->l2_compress.build(this->l2_depth.output);
        this->l3_a_depth.build(this->l2_compress.output);
        this->l3_a_compress.build(this->l3_a_depth.output);
        this->l3_b_depth.build(this->l2_compress.output);
        this->l3_b_compress.build(this->l3_b_depth.output);
        this->l3_c_depth.build(this->l3_b_compress.output);
        this->l3_c_compress.build(this->l3_c_depth.output);
        this->l3_d_depth.build(this->l3_b_compress.output);
        this->l3_d_compress.build(this->l3_d_depth.output);
        this->l3_e_depth.build(this->l3_d_compress.output);
        this->l3_e_compress.build(this->l3_e_depth.output);
        this->l3_concat.build({&this->l3_a_compress.output, &this->l3_c_compress.output, &this->l3_e_compress.output});
        this->l4_depth.build(this->l3_concat.output);
        this->l4_compress.build(this->l4_depth.output);
        this->l5_depth.build(this->l4_compress.output);
        this->l5_compress.build(this->l5_depth.output);

        this->l3_concat.backward();
    }
};
```



### Step-4.5 Implement `void call(Tensor<input_t> &input)`

`Model.call()` is implemented with calling all `Layer.call()`. The details about how to call `Layer.call()` of each kind of Layer, please check the [declaration of them](../include/layer/).


```c++
class MNIST : public Model<int16_t>
{
    // ellipsis menber variables
    // ellipsis construcor function
    // ellipsis build(...)

    void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2_depth.call(this->l1.output);
        this->l1.output.free_element();

        this->l2_compress.call(this->l2_depth.output);
        this->l2_depth.output.free_element();

        this->l3_a_depth.call(this->l2_compress.output);
        // this->l2_compress.output.free_element();

        this->l3_concat.calloc_element(); // calloc a memory for layers concat in future.

        this->l3_a_compress.call(this->l3_a_depth.output);
        this->l3_a_depth.output.free_element();

        this->l3_b_depth.call(this->l2_compress.output);
        this->l2_compress.output.free_element();

        this->l3_b_compress.call(this->l3_b_depth.output);
        this->l3_b_depth.output.free_element();

        this->l3_c_depth.call(this->l3_b_compress.output);
        // this->l3_b_compress.output.free_element();

        this->l3_c_compress.call(this->l3_c_depth.output);
        this->l3_c_depth.output.free_element();

        this->l3_d_depth.call(this->l3_b_compress.output);
        this->l3_b_compress.output.free_element();

        this->l3_d_compress.call(this->l3_d_depth.output);
        this->l3_d_depth.output.free_element();

        this->l3_e_depth.call(this->l3_d_compress.output);
        this->l3_d_compress.output.free_element();

        this->l3_e_compress.call(this->l3_e_depth.output);
        this->l3_e_depth.output.free_element();

        this->l4_depth.call(this->l3_concat.output);
        this->l3_concat.output.free_element();

        this->l4_compress.call(this->l4_depth.output);
        this->l4_depth.output.free_element();

        this->l5_depth.call(this->l4_compress.output);
        this->l4_compress.output.free_element();

        this->l5_compress.call(this->l5_depth.output);
        this->l5_depth.output.free_element();
    }
};
```



## Step-5 Run a Model

- Create an object of Model 

- Run `Model.forward()` for neural-network inference. The progress of `Model.forward()` is:

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

  

**Example:** In [`./main/main.cpp`](./main/app_main.cpp), we create an object of MNIST and call `forward()` to get inference result.

```c++
// model forward
MNIST model;
model.forward(input);
```

