# 变量与常量介绍 [[English]](../en/about_type_define.md)

ESP-DL 有以下变量和常量：

- **变量（值可以变化）**：[张量](../../include/typedef/dl_variable.hpp/#15)
- **常量（值固定不变）**：[过滤器](../../include/typedef/dl_constant.hpp/#33)、[偏差](../../include/typedef/dl_constant.hpp/#55)和[激活函数](../../include/typedef/dl_constant.hpp/#67)



## 张量

张量是矩阵向更高维度的泛化。也就是说，张量可以是：

- 0 维，表示为标量
- 1 维，表示为向量
- 2 维，表示为矩阵
- 难以想象的多维结构

维数和每个维度的大小即为张量的形状。ESP-DL 的主要数据结构就是张量。一个层的所有输入和输出均为张量。

有时张量的边缘需要用 0 填充，以便用于神经网络操作。此时，张量中的非零元素可能不连续。**定制层时需注意，不连续的非零元素会增加定制层的难度。**

### 一维张量

将在一维操作 API 完成时更新。



### 二维张量

#### 维度顺序

二维操作中，层的输入张量和输出张量均是二维。张量的维度顺序固定，按照[高度，宽度，通道]的顺序排序。

假设张量的形状是 [5, 3, 4]，则张量中的元素应如下排列：


   <p align="center">
    <img width="%" src="../../img/tensor_2d_sequence.drawio.png"> 
   </p>


#### 填充

Conv2D、DepthwiseConv2D 和其他二维操作中，输入张量需要在左、右、上、下进行填充。不同的层可能有相同的输入张量，但填充要求不一。为了减少内存复制（即将张量元素从一个存储位置复制到另一个存储位置）的次数，内存中的所有输入张量会填充到最大。

尤其是 Concat2D，所有输入张量会拼接在一起。如下图所示，A、B 和 C 沿通道维度拼接（目前 ESP-DL 仅支持沿该维度拼接）。B 是某一层的输入张量，前后进行了填充（上、下、左、右可能也同样进行了填充）。B 前后填充的元素包括 A（在 B 前方）和 C（在 B 后方）中的有意义元素。


   <p align="center">
    <img width="%" src="../../img/concat_2d.drawio.png"> 
   </p>

下图展示了填充会用到的[张量成员变量](../../include/typedef/dl_variable.hpp/#22)：

- `Tensor.element`：特定的点，不论是在前方还是后方填充，该点永远指向在下图中的位置
- `Tensor.shape`：填充前的原始张量形状（即下图中的红框）
- `Tensor.padding`：原始张量的填充尺寸。`Tensor.padding` 在每个维度都应大于或等于 `Layer.padding`
    > [`dl_layer_conv2d.hpp`](../../include/layer/dl_layer_conv2d.hpp) 文件中用到了 `Layer.padding`，`Layer.padding` 小于或等于 `Tensor.padding`。

   <p align="center">
    <img width="%" src="../../img/tensor_2d_padding.drawio.png"> 
   </p>
   

#### 应用

**输入张量**：假设有一个输出张量，填充尺寸是 `Layer.padding`。[`Tensor.get_element_ptr(Layer.padding)`](../../include/typedef/dl_variable.hpp/#100) 应指向第一个填充元素，即填充输入张量的第一个占位符。注意，`Layer.padding`（上图中的蓝框）填充后的输出张量和 `Tensor.padding`（上图中最大的方框）填充后的输出张量可能有大小差距。

**输出张量**：输出张量中的元素用于存储层的操作结果，因此输出张量无需填充。`Tensor.get_element_ptr()` 应指向未经填充输出张量（上图中的红框）的第一个元素。在存储层的操作结果时，需注意未经填充输出张量（上图中的红框）和 `Tensor.padding`（上图中最大的方框）之间的区别。

**请注意指针的移动。**

## 过滤器、偏差和激活函数

与张量不同，过滤器、偏差和激活函数无需填充。这三个`元素`的顺序是灵活的，可根据特定操作调整。

更多细节，可参考 [`dl_constant.hpp`](../../include/typedef/dl_constant.hpp) 或 API 文档。
