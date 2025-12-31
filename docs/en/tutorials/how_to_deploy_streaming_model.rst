How to Deploy Streaming Models
==============================

:link_to_translation:`zh_CN:[中文]`

Time series models are now widely applied in various fields, such as audio processing. Audio models typically have two deployment modes when deployed:

- Offline mode: The model receives the complete audio data (e.g., an entire speech file) at once and processes it as a whole.
- Streaming mode: In streaming mode, the model receives audio data frame by frame (or chunk by chunk) in real-time, processes it, and outputs intermediate results.

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

:project:`Reference example <examples/tutorial/how_to_deploy_streaming_model>`

How to Convert to a Streaming Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are numerous types of time series models. Here, we take the Temporal Convolutional Network (TCN) as an example. If you are unfamiliar with TCNs, please refer to relevant resources for details; we won't elaborate further. Other models should be customized based on their specific structures.

The example code constructs a TCN model: :project_file:`models.py <examples/tutorial/how_to_deploy_streaming_model/quantize_streaming_model/models.py>` (the model is incomplete and used only for demonstration).

ESP-PPQ provides an automatic streaming conversion feature that simplifies the process of creating streaming models. With the ``auto_streaming=True`` parameter, ESP-PPQ automatically handles the model transformation required for streaming inference.

.. note::

   - In offline mode, the model input is a complete data segment, and the input shape typically has a large size along the time dimension (e.g., ``[1, 16, 15]``).
   - In streaming mode, the model input is continuous data with a smaller time dimension, which matches the chunk size for real-time processing (e.g., ``[1, 16, 3]``).

Automatic Streaming Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ESP-PPQ provides an automatic streaming conversion feature via the ``auto_streaming=True`` parameter in the quantization process. When this flag is enabled, ESP-PPQ automatically transforms the model to support streaming inference by:

1. Analyzing the model structure to identify appropriate chunking points
2. Creating internal state management for maintaining context between chunks
3. Generating optimized code suitable for streaming scenarios

How Auto Streaming Conversion Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The automatic streaming conversion in ESP-PPQ analyzes the model graph and inserts ``StreamingCache`` nodes at strategic locations to enable temporal context preservation. The conversion process follows these principles:

**1. Operation Classification**
   - **Streaming-enabled operations**: Convolution, pooling, and transpose convolution operations that require temporal context (e.g., ``Conv``, ``AveragePool``, ``MaxPool``, ``ConvTranspose``).
   - **Bypass operations**: Activation functions, mathematical operations, quantization nodes, and other operations that don't require temporal context (e.g., ``Relu``, ``Add``, ``MatMul``, ``LayerNorm``).

**2. Window Size Calculation**
   For streaming-enabled operations, ESP-PPQ calculates the required cache window size based on:
   - Kernel size and dilation rates
   - Padding configuration
   - Stride values

   The window size determines how many previous frames need to be cached for proper computation of the current frame.

**3. StreamingCache Node Insertion**
   ESP-PPQ inserts ``StreamingCache`` nodes before streaming-enabled operations. These nodes:
   - Maintain a sliding window buffer of historical frames
   - Adjust tensor shapes to accommodate the cache window
   - Preserve quantization configurations from the original operation
   - Manage frame axis alignment for proper temporal processing

**4. Padding Adjustment**
   For streaming operations, ESP-PPQ adjusts padding configurations:
   - Removes bottom padding to prevent look-ahead into future frames
   - Maintains symmetric or top-only padding for causal processing

**Limitations and Considerations**
   - Automatic conversion supports convolution-based temporal operations out-of-the-box
   - Custom operations or complex temporal dependencies may require manual streaming table configuration
   - The conversion assumes the time dimension is along axis 1 (configurable via ``streaming_table``)

Here's an example of how to use the auto streaming feature:

.. code-block:: python

    # Export non-streaming model
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # Number of calibration steps
        input_shape=INPUT_SHAPE,  # Input shape for offline mode
        inputs=None,
        target=TARGET,  # Quantization target type
        num_of_bits=NUM_OF_BITS,  # Number of quantization bits
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )

    # Export streaming model with automatic conversion
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
        auto_streaming=True,  # Enable automatic streaming conversion
        streaming_input_shape=[1, 16, 3],  # Input shape for streaming mode
        streaming_table=None,
    )


.. _how_to_deploy_streaming_model:

Model Deployment
----------------

:project:`Reference example <examples/tutorial/how_to_deploy_streaming_model>`, this example uses pre-generated data to simulate a real-time data stream.

.. note::

    For basic model loading and inference methods, please refer to other documents:

    - :doc:`How to Load and Test a Model </tutorials/how_to_load_test_profile_model>`
    - :doc:`How to Perform Model Inference </tutorials/how_to_run_model>`

In streaming mode, the model receives data in chunks over time rather than requiring the entire input at once. The streaming model processes these chunks sequentially while maintaining internal state between chunks. The deployment code handles splitting the input into appropriate chunks and feeding them to the model. See :project_file:`app_main.cpp <examples/tutorial/how_to_deploy_streaming_model/test_streaming_model/main/app_main.cpp>` for the following code block:

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

This approach allows the model to process long sequences efficiently by breaking them into smaller, manageable chunks. Each chunk is fed to the model sequentially, and the internal state is maintained automatically to ensure continuity across chunks.

.. note::

    - The number of chunks is calculated based on the ratio between the full input size and the streaming model's input size.
    - ESP-DL streaming models handle internal state management automatically, making deployment straightforward.
    - The output from the streaming model should match the final portion of the equivalent offline model's output.