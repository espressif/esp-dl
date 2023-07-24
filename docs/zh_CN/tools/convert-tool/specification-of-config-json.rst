config.json 配置规范
========================

:link_to_translation:`en:[English]`

``config.json`` 用于保存 ``coefficient.npy`` 文件中浮点数的量化配置。

配置
----

``config.json`` 中的每一项代表一层的配置。以下列代码为例：

.. code:: none

   {
       "l1": {"/* the configuration of layer l1 */"},
       "l2": {"/* the configuration of layer l2 */"},
       "l3": {"/* the configuration of layer l3 */"},
       ...
   }

每项的键 (key) 是 **层名** 。转换工具 ``convert.py`` 根据层名搜索相应的 ``.npy`` 文件。比如，层名为 “l1”，转换工具则会在 “l1_filter.npy” 文件中搜索 l1 的过滤器系数。**config.json 中的层名需和 .npy 文件名中的层名保持一致。**

每项的值是 **层的配置**。请填写表 1 中列出的层配置实参：

.. list-table:: 表 1：层配置实参
    :header-rows: 1
    :widths: 20 15 65
    :align: center

    * - 键
      - 类型
      - 值
    * - "operation"
      - string
      - * "conv2d"
        * "depthwise_conv2d"
        * "fully_connected"
    * - "feature_type"
      - string
      - * "s16" 代表 16 位整数量化，element_width 为 16
        * "s8" 代表 8 位整数量化， element_width 为 8
    * - "filter_exponent"
      - integer
      - * 若填写，则过滤器根据公式量化：value_float = value_int * 2^指数 [1]_
        * 若空置 [2]_，则指数为 log2(max(abs(value_float)) / 2^(element_width - 1))，过滤器会根据公式量化：value_float = value_int * 2^指数
    * - "bias"
      - string
      - * "True" 代表添加偏差
        * "False" 和空置代表不使用偏差
    * - "output_exponent"
      - integer
      - | 输出和偏差根据公式量化：value_float = value_int * 2^指数。
        | 目前，"output_exponent" 仅在转换偏差系数时有效。当使用按层量化时, 必须提供"output_exponent"。如果特定层没有偏差或使用按通道量化时，"output_exponent" 可空置。
    * - "input_exponent"
      - integer
      - | 当使用按通道量化时, 偏差的指数位与输入和过滤器的指数位相关。
        | 如果有偏差时必须提供 "input_exponent" 用于转换偏差系数。如果特定层没有偏差或使用按层量化时，"input_exponent" 可空置。
    * - "activation"
      - dict
      - * 若填写，详见表 2
        * 若空置，则不使用激活函数

.. [1] **指数**：量化时底数相乘的次数。为能更好理解，请阅读 :doc:`量化规范 <../../tools/quantization-toolkit/quantization-specification>`。
.. [2] **空置**：不填写特定实参。




.. list-table:: 表 2：激活函数配置实参
    :header-rows: 1
    :widths: 15 10 75
    :align: center

    * - 键
      - 类型
      - 值
    * - "type"
      - string
      - * "ReLU"
        * "LeakyReLU"
        * "PReLU"
    * - "exponent"
      - integer
      - * 若填写，则激活函数根据公式量化： value_float = value_int * 2^指数
        * 若空置，则指数为 log2(max(abs(value_float)) / 2^(element_width - 1))

示例
----

假设有一个一层模型：

1. 使用 int16 按层量化:
~~~~~~~~~~~~~~~~~~~~~~~

-  层名：“mylayer”
-  operation：Conv2D(input, filter) + bias
-  output_exponent：-10
-  feature_type：s16，即 16 位整数量化
-  激活函数类型：PReLU

config.json 应写作：

.. code:: json

   {
       "mylayer": {
           "operation": "conv2d",
           "feature_type": "s16",
           "bias": "True",
           "output_exponent": -10,
           "activation": {
               "type": "PReLU"
           }
       }
   }

..

   “filter_exponent” 和 “activation” 的 “exponent” 空置。
   必须提供 “output_exponent” 用于转化该层的 bias

2. 使用 int8 按层量化:
~~~~~~~~~~~~~~~~~~~~~~

-  层名：“mylayer”
-  operation：Conv2D(input, filter) + bias
-  output_exponent：-7, 该卷积层结果的指数位
-  feature_type：s8
-  激活函数类型：PReLU

config.json 应写作：

.. code:: json

   {
       "mylayer": {
           "operation": "conv2d",
           "feature_type": "s8",
           "bias": "True",
           "output_exponent": -7,
           "activation": {
               "type": "PReLU"
           }
       }
   }

..

   必须提供 “output_exponent” 用于转化该层的 bias

3. 使用 int8 按通道量化:
~~~~~~~~~~~~~~~~~~~~~~~~

-  层名：“mylayer”
-  operation：Conv2D(input, filter) + bias
-  input_exponent：-7, 该卷积层输入的指数位
-  feature_type：s8
-  激活函数类型：PReLU

config.json 应写作：

.. code:: json

   {
       "mylayer": {
           "operation": "conv2d",
           "feature_type": "s8",
           "bias": "True",
           "input_exponent": -7,
           "activation": {
               "type": "PReLU"
           }
       }
   }

必须提供 “input_exponent” 用于转化该层的 bias

同时，``mylayer_filter.npy``、``mylayer_bias.npy`` 和 ``mylayer_activation.npy`` 需要准备好。
