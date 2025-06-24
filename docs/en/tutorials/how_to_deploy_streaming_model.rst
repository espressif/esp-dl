How to Deploy Streaming Models
==============================

:link_to_translation:`zh_CN:[中文]`

Time series models are now widely applied in various fields, such as audio processing. Audio models typically have two deployment modes when deployed:

- Offline mode: The model receives the complete audio data (e.g., an entire speech file) at once and processes it as a whole.
- Streaming mode: In streaming mode, the model receives audio data frame by frame (or block by block) in real-time, processes it, and outputs intermediate results.

In this tutorial, we will introduce how to quantize a streaming model using ESP-PPQ and deploy the quantized streaming model with ESP-DL.

.. contents::
  :local:
  :depth: 2

Prerequisites
-------------

1. :ref:`Install ESP-IDF <requirements_esp_idf>`
2. :ref:`Install ESP-PPQ <requirements_esp_ppq>`

.. _how_to_quantize_streaming_model:

Model Quantization
-------------------

:project:`Reference example <examples/tutorial/how_to_quantize_model/quantize_streaming_model>`

How to Convert to a Streaming Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are numerous types of time series models. Here, we take the Temporal Convolutional Network (TCN) as an example. If you are unfamiliar with TCNs, please refer to relevant resources for details; we won't elaborate further. Other models should be customized based on their specific structures.

The example code constructs a TCN model: :project_file:`test_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/test_model.py>` (the model is incomplete and used only for demonstration). The demonstration model splits a complete model into three parts: ``TestModel_0``, ``TestModel_1``, and ``TestModel_2``. This represents that a single frame of data sequentially flows through one module at a time, with the data flow being: ``frame_data -> TestModel_0 -> TestModel_1 -> TestModel_2``.

.. note::

   There is no fixed convention for splitting models; it depends on the model's structure. The more modules you split, the lower the CPU load, but the output latency of the final computation increases. Conversely, fewer splits reduce latency but increase CPU load.

For streaming models, there are differences during training and deployment: during training, the offline mode is used for simplicity; during deployment, the streaming mode is employed to better adapt to real-world scenarios where continuous data is received. Due to this difference, the model needs an additional ``streaming_forward`` function to slightly modify the forward logic to meet the requirements of quantization deployment.

.. note::

   - In offline mode, the model input is a complete data segment, and the input shape typically has a large size along the time dimension.
   - In streaming mode, the model input is continuous data. Since the forward logic is modified, the input shape along the time dimension is smaller, generally equal to the number of split modules mentioned above.

The following code block demonstrates ``TestModel_0`` with both ``forward`` and ``streaming_forward`` functions. ``forward`` is used for training, while ``streaming_forward`` is used for quantization deployment. The difference lies in the padding of the input to ``self.layer[1]``. This is because the TCN requires padding the input during convolution to maintain consistent time dimension sizes. The modification in ``streaming_forward`` implements a sliding window approach for padding, which necessitates caching data from previous time steps and concatenating them with the current data to achieve the sliding window effect. Additionally, the cache must be exposed at the model's input and output to utilize it during quantization deployment.

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

Finally, since PyTorch defaults to calling the ``forward`` method, the ``streaming_forward`` method needs to be wrapped to make it callable during quantization. See :project_file:`quantize_streaming_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/quantize_streaming_model.py>` for the following code block:

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

How to Prepare the Calibration Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration dataset must match the input format of your model. The dataset should cover as many possible input scenarios as possible to ensure better model quantization. For streaming mode, the input is a time-dimension slice of the offline mode's input. If there is a cache buffer, you need to call the model's forward method to collect the corresponding cache data for all input slices. See :project_file:`quantize_streaming_model.py <examples/tutorial/how_to_quantize_model/quantize_streaming_model/quantize_streaming_model.py>` for the following code block:

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

Model Deployment
----------------

:project:`Reference example <examples/tutorial/how_to_run_streaming_model>`, this example uses pre-generated data to simulate a real-time data stream.

.. note::

    For basic model loading and inference methods, please refer to other documents:

    - :doc:`How to Load and Test a Model </tutorials/how_to_load_test_profile_model>`
    - :doc:`How to Perform Model Inference </tutorials/how_to_run_model>`

In streaming mode, the model receives data frame by frame (or block by block), processes it in real-time, and outputs intermediate results. That is, each frame of data sequentially flows through one module at a time. See :project_file:`app_main.cpp <examples/tutorial/how_to_run_streaming_model/main/app_main.cpp>` for the following code block:

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

The following part of the code block is included solely to align the offline accuracy during precision testing. It can be omitted during actual deployment.

.. code-block:: cpp

    if (i < (input_tensor->get_shape()[1] - 1)) {
        // The data is populated to facilitate accuracy testing, as this step is omitted in actual deployment.
        continue;
    }

As shown above, a frame of data is processed in one module per time step, and the loop repeatedly implements streaming processing.

.. note::

    - When pushing frame data to the temporary TensorBase, ensure the data types match.
    - ESP-DL requires the input/output data layout for Conv, GlobalAveragePool, AveragePool, MaxPool, and Resize to be NHWC or NWC. Therefore, adjust the input data layout according to the first operator of the streaming model when feeding data to the model.