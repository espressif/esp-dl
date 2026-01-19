如何部署流式模型
================

:link_to_translation:`en:[English]`

时间序列模型如今被应用在许多领域，例如，音频领域。而音频模型在部署时通常有两种模式：

- Offline模式：模型需要一次性接收完整的音频数据（例如整个语音文件），然后进行整体处理。
- Streaming模式：流式模式下，模型逐帧（逐块）接收音频数据，实时处理并输出中间结果。

在本教程中，我们来介绍如何使用 ESP-PPQ 量化流式模型，并使用 ESP-DL 部署量化后的流式模型。

.. contents::
  :local:
  :depth: 2

准备工作
-----------

1. :ref:`安装 ESP_IDF <requirements_esp_idf>`
2. :ref:`安装 ESP_PPQ <requirements_esp_ppq>`

.. _how_to_quantize_streaming_model:

模型量化
-----------

:project:`参考示例 <examples/tutorial/how_to_deploy_streaming_model>`

时间序列模型种类繁多，这里仅以 Temporal Convolutional Network(TCN) 为例，不熟悉的可自行查找资料了解，这里不过多介绍其细节。其它模型需根据自身情况，量体裁衣。

该示例代码中构建了一个 TCN 模型： :project_file:`models.py <examples/tutorial/how_to_deploy_streaming_model/quantize_streaming_model/models.py>` (模型非完整，仅用于演示)。

ESP-PPQ 提供了自动流式转换功能，可以简化创建流式模型的过程。通过 ``auto_streaming=True`` 参数，ESP-PPQ 自动处理流式推理所需的模型转换。

.. note::

   - Offline 模式，模型输入是一段完整数据，input shape 在时间维度上的 size 一般比较大（例如 ``[1, 16, 15]``）。
   - Streaming 模式，模型输入是连续数据，在时间维度上的 size 较小，匹配实时处理的块大小（例如 ``[1, 16, 3]``）。

自动流式转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ESP-PPQ 通过量化过程中的 ``auto_streaming=True`` 参数提供自动流式转换功能。启用此标志后，ESP-PPQ 会自动转换模型以支持流式推理：

1. 分析模型结构以识别适当的分块点
2. 创建内部状态管理以在块之间保持上下文
3. 生成适合流式场景的优化代码

自动流式转换的工作原理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ESP-PPQ 中的自动流式转换会分析模型图，并在关键位置插入 ``StreamingCache`` 节点以实现时间上下文保持。转换过程遵循以下原则：

**1. 算子分类**
   - **支持流式的算子**：需要时间上下文的卷积、池化和转置卷积操作（例如 ``Conv``、``AveragePool``、``MaxPool``、``ConvTranspose``）。
   - **绕过算子**：不需要时间上下文的激活函数、数学运算、量化节点和其他操作（例如 ``Relu``、``Add``、``MatMul``、``LayerNorm``）。

**2. 窗口大小计算**
   对于支持流式的算子，ESP-PPQ 根据以下因素计算所需的缓存窗口大小：
   - Kernel size and dilation rates
   - Padding configuration
   - Stride values

   窗口大小决定了需要缓存多少历史帧才能正确计算当前帧。

**3. StreamingCache 节点插入**
   ESP-PPQ 在支持流式的算子之前插入 ``StreamingCache`` 节点。这些节点：
   - 维护历史帧的滑动窗口缓冲区
   - 调整张量形状以容纳缓存窗口
   - 保留原始操作的量化配置
   - 管理帧轴对齐以进行正确的时间处理

**4. 填充调整**
   对于流式操作，ESP-PPQ 调整填充配置：
   - 移除底部填充以防止前瞻到未来帧
   - 保持对称或仅顶部填充以实现因果处理

**限制和注意事项**
   - 自动转换开箱即用地支持基于卷积的时间操作
   - 自定义操作或复杂的时间依赖关系可能需要手动配置流式表
   - 转换假设时间维度沿轴 1（可通过 ``streaming_table`` 配置）

以下是如何使用自动流式功能的示例：

.. code-block:: python

    # 导出非流式模型
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准步数
        input_shape=INPUT_SHAPE,  # 离线模式的输入形状
        inputs=None,
        target=TARGET,  # 量化目标类型
        num_of_bits=NUM_OF_BITS,  # 量化位数
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # 输出详细日志信息
    )

    # 使用自动转换导出流式模型
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_STEAMING_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
        auto_streaming=True,  # 启用自动流式转换
        streaming_input_shape=[1, 16, 3],  # 流式模式的输入形状
        streaming_table=None,
    )

手动流式缓存配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于 ESP-PPQ 流式转换功能不自动支持的算子（例如 Transpose、Reshape、Slice 等），您可以使用 ``insert_streaming_cache_on_var`` 函数手动插入 StreamingCache 节点。该函数允许您为无法自动插入 streamingCache 的变量指定缓存属性。

``insert_streaming_cache_on_var`` 函数的签名如下：

.. code-block:: python

    def insert_streaming_cache_on_var(
        var_name: str,
        window_size: int,
        op_name: str = None,
        frame_axis: int = 1
    ) -> Dict[str, Any]

参数说明：
- ``var_name``：需要插入流式缓存的变量名称
- ``window_size``：缓存窗口大小（需要缓存的帧数）
- ``op_name``：（可选）与变量关联的算子名称
- ``frame_axis``：（可选）表示时间维度的轴，默认为 1

该函数返回一个包含流式缓存配置的字典，应将其添加到 ``streaming_table`` 列表中并传递给 ``espdl_quantize_torch`` 函数。

使用示例：

.. code-block:: python

    streaming_table = []
    # 为无法自动插入 streamingCache 的变量手动指定缓存属性
    streaming_table.append(
        insert_streaming_cache_on_var("/out_conv/Conv_output_0", output_frame_size - 1)
    )
    streaming_table.append(insert_streaming_cache_on_var("PPQ_Variable_0", 1, "/Slice"))

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_STEAMING_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
        auto_streaming=True,
        streaming_input_shape=[1, 16, 3],
        streaming_table=streaming_table,  # 传递手动配置的流式表
    )

.. _how_to_deploy_streaming_model:

模型部署
------------

:project:`参考示例 <examples/tutorial/how_to_deploy_streaming_model>`, 该示例使用预生成的数据来模拟实时数据流。

.. note::

    基础的模型加载和推理方法，可参考其它文档，这里不再赘述：

    - :doc:`如何加载和测试模型 </tutorials/how_to_load_test_profile_model>`
    - :doc:`如何进行模型推理 </tutorials/how_to_run_model>`

在流式模式下，模型按时间接收数据块，而不是要求一次性获得整个输入。流式模型依次处理这些块，同时在块之间保持内部状态。部署代码负责将输入分解为适当的块并将其馈送到模型。见 :project_file:`app_main.cpp <examples/tutorial/how_to_deploy_streaming_model/test_streaming_model/main/app_main.cpp>` 如下代码块：

.. code-block:: cpp

    dl::TensorBase *run_streaming_model(dl::Model *model, dl::TensorBase *test_input)
    {
        std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
        dl::TensorBase *model_input = model_inputs.begin()->second;
        std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
        dl::TensorBase *model_output = model_outputs.begin()->second;

        if (!test_input) {
            ESP_LOGE(TAG,
                     "Model input doesn't have a corresponding test input. Please enable export_test_values option "
                     "in esp-ppq when export espdl model.");
            return nullptr;
        }

        int test_input_size = test_input->get_bytes();
        uint8_t *test_input_ptr = (uint8_t *)test_input->data;
        int model_input_size = model_input->get_bytes();
        uint8_t *model_input_ptr = (uint8_t *)model_input->data;
        int chunks = test_input_size / model_input_size;
        for (int i = 0; i < chunks; i++) {
            // assign chunk data to model input
            memcpy(model_input_ptr, test_input_ptr + i * model_input_size, model_input_size);
            model->run(model_input);
        }

        return model_output;
    }

这种方法允许模型通过将长序列分解为更小、更易管理的块来高效处理。每个块依次馈送到模型中，内部状态自动维护以确保跨块的连续性。

.. note::

    - 块的数量是根据完整输入大小与流式模型输入大小的比率计算的。
    - ESP-DL 流式模型自动处理内部状态管理，使部署变得简单。
    - 流式模型的输出应与等效离线模型输出的最后部分匹配。
