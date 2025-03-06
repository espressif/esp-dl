如何进行模型推理
====================

:link_to_translation:`en:[English]`

在本教程中，我们将介绍最基本的模型推理流程。:project:`参考例程 <examples/tutorial/how_to_run_model>`

.. contents::
  :local:
  :depth: 2

准备工作
------------

:ref:`安装 ESP_IDF <requirements_esp_idf>`

加载模型
------------

:doc:`如何加载模型 </tutorials/how_to_load_test_profile_model>`

获取模型输入/输出。
---------------------

.. code-block:: cpp

    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;

可以通过 ``get_inputs()`` 和 ``get_outputs()`` api 获得输入/输出的名字和对应的 ``dl::TensorBase``。更多信息，请参阅 :doc:`dl::TensorBase 文档 </api_reference/tensor_api>` 。


.. note::
    
    ESP-DL 的内存管理器会为每个模型的输入/中间结果/输出分配一整块的内存。由于它们共用这部分内存，所以当模型进行推理的时候，后面的结果会覆盖前面的结果。也就是说，``model_input`` 中的数据，在执行完模型推理之后，可能就会被 ``model_output`` 或者其他中间结果所覆盖。

量化输入
-------------

8bit 和 16bit 量化的模型，分别接受 ``int8_t`` 和 ``int16_t`` 类型的输入。``float`` 类型的输入必须先根据 ``exponent`` 量化成对应的整数类型之后才能喂入模型。计算公式：

.. math::

    Q = \text{Clip}\left(\text{Round}\left(\frac{R}{\text{Scale}}\right), \text{MIN}, \text{MAX}\right) \\

.. math::

    \text{Scale} = 2^{\text{Exp}}

其中：

- R 是要量化的浮点数。
- Q 是量化后的整数值，需要在 [MIN, MAX] 范围内进行裁剪。
- MIN 整数最小值，8bit 时，MIN = -128, 16bit 时，MIN = -32768。
- MAX 整数最大值，8bit 时，MAX = 127, 16bit 时，MAX = 32767。


量化单个值
^^^^^^^^^^^^^

.. code-block:: cpp

    float input_v = VALUE;
    // Note that dl::quantize accepts inverse of scale as the second input, so we use DL_RESCALE here.
    int8_t quant_input_v = dl::quantize<int8_t>(input_v, DL_RESCALE(model_input->exponent));


量化 ``dl::TensorBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // assume that input_tensor already contains the float input data.
    dl::TensorBase *input_tensor;
    model_input->assign(input_tensor);


反量化输出
-------------

8bit 和 16bit 量化的模型，分别得到 ``int8_t`` 和 ``int16_t`` 类型的输出。必须根据 ``exponent`` 反量化之后才能得到浮点输出。计算公式：

.. math::

    R' = Q \times \text{Scale}

.. math::

    \text{Scale} = 2^{\text{Exp}}

其中：

- R' 是反量化后恢复的近似浮点值。
- Q 是量化后的整数值。

反量化单个值
^^^^^^^^^^^^^^^

.. code-block:: cpp

    int8_t quant_output_v = VALUE;
    float output_v = dl::dequantize(quant_output_v, DL_SCALE(model_output->exponent));


反量化 ``dl::TensorBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // create a TensorBase filled with 0 of shape [1, 1]
    dl::TensorBase *output_tensor = new dl::TensorBase({1, 1}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    output_tensor->assign(model_output);

模型推理
------------

请参阅： 

- :project:`参考例程 <examples/tutorial/how_to_run_model>`
- :cpp:func:`void dl::Model::run(runtime_mode_t mode)`
- :cpp:func:`void dl::Model::run(TensorBase *input, runtime_mode_t mode)`
- :cpp:func:`void dl::Model::run(std::map<std::string, TensorBase*> &user_inputs, runtime_mode_t mode, std::map<std::string, TensorBase*> user_outputs)`