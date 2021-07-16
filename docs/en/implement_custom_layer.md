# Implement Custom Layer Step by Step

Our implemented layers, e.g. Conv2D, DepthwiseConv2D, are derived from base class **Layer** in [`./include/layer/dl_layer_base.hpp`](../../include/layer/dl_layer_base.hpp). As you see, the Layer class only has one member variable `name`, no virtual function needed to be overload. Actually, if it's no use with name, it will not necessary to implement a custom layer by derived from Layer class. However, we recommend deriving for code consistency.

The example here is not a runnable class but for design purpose clarification. For a runnable example, please check the header files in [./include/layer/](../../include/layer/) folder, which contains Conv2D, DepthwiseConv2D, Concat2D, etc implements.

As input and output of layer are Tensor, **it is quite necessary to know the definition about Tensor in [About Type Define](./about_type_define.md/#Tensor)**.

Let's start to implement a custom layer!



## Step-1 Derived from Layer class

Here, we derive a new layer named `MyLayer` from Layer class. Declare member variables, constructor and destructor according to requirements. Do not forget to initialize the constructor of base class. Nothing special here, just follow the C/C++ specification.

```c++
class MyLayer : public Layer
{
private:
    /* private member variables */
public:
    /* public member variables */
    Tensor<int16_t> output; /*<! output of this layer */

    MyLayer(/* arguments */) : Layer(name)
    {
        // initialze anything frozen
    }

    ~MyLayer()
    {
        // destroy
    }
};
```



## Step-2 Implement build()

As a layer, it always has input(one or more) and output. The design purpose of `build()` comes into being due to input and output.

**Update Output Shape**: The output shape is determined by input shape, sometimes coefficient shape as well, e.g., output shape of Conv2D is determined by input shape, filter shape, stride and dilation. In a running application, some configurations of layer like filter shape, stride and dilation are frozen, but input shape is probably variable. Once input shape changed, output shape should be changed accordingly. So `build()` is implemented for the first purpose: update output shape according to input shape.

**Update Input Padding**: For some operation, e.g., Conv2D and DepthwiseConv2D, input probably needs padding. Like output shape, it's determined by input shape, sometimes coefficient shape as well, e.g., input padding of Conv2D is determined by input shape, filter shape, stride, dilation and padding type. So `build()` is implemented for the second purpose: update input padding according to input shape for a layer which need input padded.

We implement `build()` with these two above purpose only by now, but not limited to this. Infer other things from one fact, **all updates based on input could be input in build()**.

**Why build() is needed**: When we [Implement Custom Model](../../tutorial/README.md), all `Layer.build()` are called in `Model.build()`, described in [Step-4 Build a Model](../../tutorial/README.md/#Step-4-Build-a-Model). `Model.build()` is called when input shape is changed, described in [Step-5 Run a Model](../../tutorial/README.md/#Step-5-Run-a-Model). In other word, save `Model.build()` calling when input shape is not changed.

```c++
class MyLayer : public Layer
{
    // ellipsis menber variables
    // ellipsis construcor and destructor

    void build(Tensor<int16_t> &input)
    {
        /* get output_shape according to input shape and other configuration */
        this->output.set_shape(output_shape); // update output_shape

        /* get padding according to input shape and other configuration */
        input.set_padding(this->padding);
    }
};
```



## Step-3 Implement call()

Implement layer inference operation in `call()`. Here are some points to note,

**Call output.[`calloc_element()`](../../include/typedef/dl_variable.hpp/#122)**: Don't forget to apply memory for `output.element`.

**Element Sequence of Tensor**: Input and output are both [`dl::Tensor`](../../include/typedef/dl_variable.hpp). It's very necessary to know the details about Tensor in [About Type Define](./about_type_define.md/#Tensor).

```c++
class MyLayer : public Layer
{
    // ellipsis menber variables
    // ellipsis construcor and destructor
    // ellipsis build(...)

    Tensor<feature_t> &call(Tensor<int16_t> &input, /* other arguments */)
    {
        this->output.calloc_element(); // calloc memory for output.element

        /* implement operation */
        
        return this->output;
    }
};
```

