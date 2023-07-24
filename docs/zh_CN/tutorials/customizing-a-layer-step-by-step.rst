定制层步骤
=============

:link_to_translation:`en:[English]`

Conv2D、DepthwiseConv2D 等 ESP-DL 实现的层由 :project_file:`include/layer/dl_layer_base.hpp` 中的基础层 **Layer** 派生而来。Layer 类只有一个成员变量，即名称 ``name``。如果没有用到 ``name``，可以不必定制 Layer 类的派生层，但为了保持代码一致，我们推荐派生。

本文档中的示例不可运行，仅供参考。如需可运行的示例，请参考 :project:`include/layer` 文件夹中的头文件，其中包括 ``Conv2D``、``DepthwiseConv2D``、``Concat2D`` 等层。

由于层的输入和输出都是张量，**请务必阅读** :term:`张量`，**了解常量的相关内容**。

下面开始定制层吧！

步骤 1：从 Layer 类派生层
-------------------------

从 Layer 类派生一个新层（示例中命名为 ``MyLayer``），并根据要求定义成员变量、构造函数和析构函数。不要忘记初始化基类的构造函数。

.. code:: none

   class MyLayer : public Layer
   {
   private:
       /* private member variables */
   public:
       /* public member variables */
       Tensor<int16_t> output; /*<! output of this layer */

       MyLayer(/* arguments */) : Layer(name)
       {
           // initialize anything frozen
       }

       ~MyLayer()
       {
           // destroy
       }
   };

步骤 2：实现 ``build()``
------------------------

通常一层会有一个或多个输入和一个输出。``build()`` 现有如下作用：

-  **更新输出形状**:

   输出形状由输入形状决定，有时也受系数形状的影响。比如，Conv2D 的输出形状由输入形状、过滤器形状、步幅和扩张决定，但输入形状可能会变化。一旦输入形状改变，输出形状也应有相应改变。``build()`` 的第一个作用是根据输入形状更新输出形状。

-  **更新输入填充**:

   Conv2D、DepthwiseConv2D 等二维卷积层中，输入张量可能需要填充。正如输出形状一样，输入填充也由输入形状决定，有时受系数形状影响。比如，Conv2D 层的输入填充由输入形状、过滤器形状、步幅、扩张和填充类型决定。``build()`` 的第二个作用是根据待填充输入张量的形状更新输入填充。

``build()`` 不仅限于以上两个作用。**所有根据输入所做的更新都可由 build() 实现**。

.. code:: none

   class MyLayer : public Layer
   {
       // ellipsis member variables
       // ellipsis constructor and destructor

       void build(Tensor<int16_t> &input)
       {
           /* get output_shape according to input shape and other configuration */
           this->output.set_shape(output_shape); // update output_shape

           /* get padding according to input shape and other configuration */
           input.set_padding(this->padding);
       }
   };

步骤 3：实现 ``call()``
--------------------------

在 ``call()`` 中实现层推理。请注意：

-  在 :project_file:`include/typedef/dl_variable.hpp` 中，``Tensor.apply_element()``、``Tensor.malloc_element() 或 ``Tensor.calloc_element()`` 均可为 ``output.element`` **分配存储空间**；
-   :term:`张量` **中描述的张量维度顺序**，因为输入和输出均为 :project_file:`include/typedef/dl_variable.hpp`。

.. code:: none

   class MyLayer : public Layer
   {
       // ellipsis member variables
       // ellipsis constructor and destructor
       // ellipsis build(...)

       Tensor<feature_t> &call(Tensor<int16_t> &input, /* other arguments */)
       {
           this->output.calloc_element(); // calloc memory for output.element

           /* implement operation */

           return this->output;
       }
   };
