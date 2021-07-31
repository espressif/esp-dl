# About Variables and Constants

ESP-DL has the following variable and constants:

- **Variable**: [tensors](../../include/typedef/dl_variable.hpp/#15) whose values can be changed
- **Constants**: [filters](../../include/typedef/dl_constant.hpp/#33), [biases](../../include/typedef/dl_constant.hpp/#55), and [activations](../../include/typedef/dl_constant.hpp/#67) whose values are fixed



## Tensor

A tensor is a generalization of matrices to N dimensions. In other words, it could be:

- 0-dimensional, represented as a scalar
- 1-dimensional, represented as a vector
- 2-dimensional, represented as a matrix
- a higher-dimensional structure that is harder to visulize

The number of dimensions and the size of each dimension is known as the shape of a tensor. In ESP-DL, a tensor is the primary data structure. Every input and output of a layer is a tensor.

Sometimes a tensor needs to be padded with zero around its border, so that its shape is amenable to the neural network operation. In this case, the non-zero elements in the tensor may not be continuous. **The incontinuous non-zero elements makes it difficult to customize a layer. Please pay special attention when making customization.**

### Tensor in 1D

To be updated when the API for 1D operations is ready.



### Tensor in 2D

#### Dimension Order

In 2D operations, the input tensor and output tensor of a layer is 2D. Tensor dimensions are ordered in a fixed manner, namely [height, width, channel].

Suppose we have the following shape [5, 3, 4], and the elements of this tensor would be arranged as follows:


   <p align="center">
    <img width="%" src="../../img/tensor_2d_sequence.drawio.png"> 
   </p>


#### Padding

In Conv2D, DepthwiseConv2D, and other types of 2D operations, input tensors probably need to be padded at the left, right, top, and bottom. Different layers might have the same input tensor, but vary in padding requirements. To reduce the times of memory copy (i.e. copying tensor elements from one memory location to another before padding), all input tensors in the memory are padded to the maximum. 

Especially in Concat2D, all input tensors are concatenated. As the figure below shows, A, B, and C are concatenated along the channel dimension (for now the only dimension supported by concatenation in ESP-DL). As input for a specific layer, B is padded at the front and back (probably at the top, bottom, left, and right as well). The elements padded at the front and back of B include meaningful elements of A (at the front of B) and C (at the back of B).

   <p align="center">
    <img width="%" src="../../img/concat_2d.drawio.png"> 
   </p>

The figure below illustrates [tensor's member variables](../../include/typedef/dl_variable.hpp/#22) used for padding:

- `Tensor.element`: a specific point whose position would never change in the tensor, regardless of padding at the front and back
- `Tensor.shape`: the original tensor shape before being padded (the red box in the figure)
- `Tensor.padding`: the padding size of the original tensor. `Tensor.padding` should be greater than or equal to `Layer.padding` in each dimension
    > In [`dl_layer_conv2d.hpp`](../../include/layer/dl_layer_conv2d.hpp), `Layer.padding` is used, and is less than or equal to `Tensor.padding`.

   <p align="center">
    <img width="%" src="../../img/tensor_2d_padding.drawio.png"> 
   </p>
   

#### In Application

**Input tensor:** Suppose that we have an input tensor to be padded by the padding size `Layer.padding`. Then [`Tensor.get_element_ptr(Layer.padding)`](../../include/typedef/dl_variable.hpp/#100) should point to the first padding element, i.e. the first placeholder to pad the input tensor. Note that there might be gaps between the padded output of `Layer.padding` (the blue box in the figure) and the padded output of `Tensor.padding` (the biggest box in the figure).

**Output tensor** Elements in an output tensor are used to store the operation result of a layer, so an output tensor always does not need to be padded. Then `Tensor.get_element_ptr()` should point to the first element of an output tensor without padding (the red box in the figure above).

**Be cautious about movements of the pointer.**

## Filter, Bias and Activation

Unlike a tensor, a filter, bias, and activiation do not need to be padded. The order of these three `elements` is flexible according to specific operations.

For more details, please refer to [`dl_constant.hpp`](../../include/typedef/dl_constant.hpp) or API documentation.
