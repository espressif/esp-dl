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

:project:`参考示例 <examples/tutorial/how_to_quantize_model/quantize_streaming_model>`

如何转换为流式模型
^^^^^^^^^^^^^^^^^^

时间序列模型种类繁多，这里仅以 Temporal Convolutional Network(TCN) 为例，不熟悉的可自行查找资料了解，这里不过多介绍其细节。其它模型需根据自身情况，量体裁衣。

该示例代码中构建了一个 TCN 模型：:project_file:`test_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/test_model.py>`(模型非完整，仅用于演示)。演示模型将一个完整模型拆分为了三个部分： ``TestModel_0``, ``TestModel_1``,  ``TestModel_2``，代表一帧数据按顺序每次仅流经一个模块，数据流为： ``frame_data -> TestModel_0 -> TestModel_1 -> TestModel_2`` 。

.. note::

   这种拆分没有固定范式，根据模型结构自由决定。拆分的模块越多，cpu的负载就越低，但最终计算结果的输出延迟会增加。反之亦然。

对于流式模型，在训练和部署时会有差异：训练时，为了简便，采用 offline 模式；部署时，则换做 streaming 模式，以更好适配实际情形下，接收连续数据。因为这个差异，模型需要新增一个 streaming_forward 函数，对前向逻辑稍作修改，以满足量化部署时的需求。

.. note::

   - Offline 模式，模型输入是一段完整数据，input shape 在时间维度上的 size 一般比较大。
   - Streaming 模式，模型输入是连续数据，由于对前向逻辑做了修改，所以 input shape 在时间维度上的 size 小，一般为上面提及的拆分模块的数量。

如下代码块以 ``TestModel_0`` 为例，含有 forward 和 streaming_forward 两个前向函数。forward用于训练，streaming_forward 用于量化部署。
两者的差异在于 self.layer[1] 的输入 padding 上，这是 TCN 为了满足时间维度 size 一致，在卷积时，需对输入进行 padding。streaming_forward 的改动相当于是以滑动窗口的方式，对输入进行 padding，这时就需要缓存当前时间步之前的数据，并与当前时间步数据拼接，以实现滑动窗口的效果。同时需要将 cache 在模型的输入和输出暴露出来，这样在量化部署时，才可以在 model 层使用 cache。

.. code-block:: python

    def forward(self, input: Tensor) -> Tensor:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        out1 = F.pad(out1, (self.padding, 0), "constant", 0)
        out1 = self.layer[1](out1)
        out2 = self.layer[2](out1)
        output = self.layer[3](out1 * out2) + input
        return output

    def streaming_forward(self, input: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        # input [B, C, T] -> output [B, C, T]
        input = self.prev_conv(input)
        out1 = self.layer[0](input)
        # 1D Depthwise Conv
        assert cache.shape == (out1.size(0), out1.size(1), self.padding), (
            cache.shape,
            (out1.size(0), out1.size(1), self.padding),
        )
        out1 = torch.cat([cache, out1], dim=2)
        # Update cache
        cache = out1[:, :, -self.padding :]

        out1 = self.layer[1](out1)
        out2 = self.layer[2](out1)
        output = self.layer[3](out1 * out2) + input
        return output, cache

最后，由于 pytorch 默认调用 module 的 forward 方法，所以在量化时，需要对 streaming_forward 方法进行封装，使其能够被调用。见 :project_file:`quantize_streaming_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/quantize_streaming_model.py>` 如下代码块：

.. code-block:: python

   class ModelStreamingWrapper(nn.Module):
        """A wrapper for model"""

        def __init__(self, model: nn.Module):
            """
            Args:
            model: A pytorch model.
            """
            super().__init__()
            self.model = model

        def forward(
            self, input: Tensor, cache: Optional[Tensor] = None
        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            """Please see the help information of TestModel_0.streaming_forward"""

            if cache is not None:
                output, new_cache = self.model.streaming_forward(input, cache)
                return output, new_cache
            else:
                output = self.model.streaming_forward(input)
                return output

如何准备校准数据集
^^^^^^^^^^^^^^^^^^

首先校准数据集需要和你的模型输入格式一致，校准数据集需要尽可能覆盖你的模型输入的所有可能情况，以便更好地量化模型。
对于 streaming 模式，输入是 offline 模式的输入在时间维度上的切分片段，如有 cache buffer ，则需要调用模型前向，收集所有输入切分片段对应的 cache 数据。
见 :project_file:`quantize_streaming_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/quantize_streaming_model.py>` 如下代码块：

.. code-block:: python
   
    def load_calibration_dataset(self) -> Iterable:
        if self.streaming:
            data_total = []
            if self.model_config.get("streaming_cache_shape", []):
                caches = []
                caches.append(
                    torch.zeros(size=self.model_config["streaming_cache_shape"][1:])
                )
                if not self.multi_input:
                    for data in self.dataset:
                        # Ensure that the size of the W dimension is divisible by self.streaming_window_size.
                        # Split the input and collect cache data.
                        split_tensors = torch.split(
                            data[0] if isinstance(data, tuple) else data,
                            self.streaming_window_size,
                            dim=1,
                        )
                        for index, split_tensor in enumerate(split_tensors):
                            _, cache = self.model(
                                split_tensor.unsqueeze(0), caches[index].unsqueeze(0)
                            )
                            caches.append(cache.squeeze(0))

                        data_total += [
                            list(pair) for pair in zip(list(split_tensors), caches)
                        ]
                else:
                    # It depends on which inputs of the model require streaming, so multiple inputs have not been added.
                    pass

                return data_total
            else:
                if not self.multi_input:
                    for data in self.dataset:
                        # Ensure that the size of the W dimension is divisible by self.streaming_window_size.
                        # Split the input and collect cache data.
                        split_tensors = torch.split(
                            data[0] if isinstance(data, tuple) else data,
                            self.streaming_window_size,
                            dim=1,
                        )
                        data_total += list(split_tensors)
                else:
                    pass

                return data_total
        else:
            return self.dataset


.. _how_to_deploy_streaming_model:

模型部署
------------

:project:`参考示例 <examples/tutorial/how_to_run_streaming_model>`, 该示例使用预生成的数据来模拟实时数据流。

.. note::

    基础的模型加载和推理方法，可参考其它文档，这里不再赘述：

    - :doc:`如何加载和测试模型 </tutorials/how_to_load_test_profile_model>`
    - :doc:`如何进行模型推理 </tutorials/how_to_run_model>`

streaming 模式下，模型逐帧（逐块）接收数据，实时处理并输出中间结果。即：一帧数据按顺序每次仅流经一个模块。见 :project_file:`app_main.cpp <examples/tutorial/how_to_run_streaming_model/main/app_main.cpp>` 如下代码块：

.. code-block:: cpp

    for (int i = 0; i < TIME_SERIES_LENGTH; i++) {
        one_step_input_tensor->set_element_ptr(const_cast<int8_t *>(&test_inputs[i][0]));
        // Because the first layer of model_0 in the example is conv, so the time series dimension is 1.
        input_tensor->push(one_step_input_tensor, 1);

        if (i < (input_tensor->get_shape()[1] - 1)) {
            // The data is populated to facilitate accuracy testing, as this step is omitted in actual deployment.
            continue;
        } else {
            switch (step_index) {
            case 1:
                output = (*p_model_0)(input_tensor);
                step_index++;
                break;
            case 2:
                output = (*p_model_1)(output);
                step_index++;
                break;
            case 3:
                output = (*p_model_2)(output);
                dl::tool::copy_memory(output_buffer + (i / 3 - 1) * STREAMING_WINDOW_SIZE * TEST_INPUT_CHANNELS,
                                    output->data,
                                    STREAMING_WINDOW_SIZE * TEST_INPUT_CHANNELS);
                step_index = 1;
                break;
            default:
                break;
            }
        }
    }

上面代码块中的如下部分，仅仅是为了在精度测试时，能够准确地对齐 offline 精度。实际部署时，可去除。

.. code-block:: cpp

    if (i < (input_tensor->get_shape()[1] - 1)) {
        // The data is populated to facilitate accuracy testing, as this step is omitted in actual deployment.
        continue;
    }

从上面可以看出，一帧数据，在一个时间步中，仅在一个模块中被处理，循环往复实现了流式处理。

.. note::

    - 帧数据在被 push 到临时 TensorBase 时，需要确保两者数据类型一致。
    - ESP-DL 对于 Conv, GlobalAveragePool, AveragePool, MaxPool, Resize 的输入/输出数据排布要求是 NHWC 或者 NWC，所以在给模型喂数据时，需要根据流式模型第一层算子，调整好输入数据排布。
