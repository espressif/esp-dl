How to run model
==========================

:link_to_translation:`zh_CN:[中文]`

In this tutorial, we will introduce the most basic model inference process. :project:`example <examples/tutorial/how_to_run_model>`

.. contents::
  :local:
  :depth: 2

Preparation
----------------

:ref:`Install ESP_IDF <requirements_esp_idf>`

Load model
----------------

:doc:`How to load model </tutorials/how_to_load_test_profile_model>`

Get model input/output.
------------------------------

.. code-block:: cpp

    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;

You can get the input/output names and the corresponding ``dl::TensorBase`` with ``get_inputs()`` and ``get_outputs()`` api. For more information, see :doc:`dl::TensorBase documentation </api_reference/tensor_api>`.

.. note::

    ESP-DL's memory manager allocates a whole block of memory for each model's input/intermediate result/output. Since they share this memory, when the model is inferencing, the later results will overwrite the previous results. In other words, the data in ``model_input`` may be overwritten by ``model_output`` or other intermediate results after the model inference is completed.

Quantize Input
--------------------

8-bit and 16-bit quantized models accept inputs of type ``int8_t`` and ``int16_t`` respectively. ``float`` inputs must be quantized to the one of them according to ``exponent`` before being fed into the model. Calculation formula:

.. math::

    Q = \text{Clip}\left(\text{Round}\left(\frac{R}{\text{Scale}}\right), \text{MIN}, \text{MAX}\right) \\

.. math::

    \text{Scale} = 2^{\text{Exp}}

Where:

- R is the floating point number to be quantized.
- Q is the integer value after quantization, which needs to be clipped within the range [MIN, MAX].
- MIN is the minimum integer value, when 8bit, MIN = -128, when 16bit, MIN = -32768.
- MAX is the maximum integer value, when 8bit, MAX = 127, when 16bit, MAX = 32767.

Quantize a single value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    float input_v = VALUE;
    // Note that dl::quantize accepts inverse of scale as the second input, so we use DL_RESCALE here.
    int8_t quant_input_v = dl::quantize<int8_t>(input_v, DL_RESCALE(model_input->exponent));


Quantize ``dl::TensorBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // assume that input_tensor already contains the float input data.
    dl::TensorBase *input_tensor;
    model_input->assign(input_tensor);


Dequantize output
------------------------

8bit and 16bit quantized model, get ``int8_t`` and ``int16_t`` type output respectively. Must be dequantized according to ``exponent`` to get floating point output. Calculation formula:

.. math::

    R' = Q \times \text{Scale}

.. math::

    \text{Scale} = 2^{\text{Exp}}

Where:

- R' is the approximate floating point value recovered after dequantization.
- Q is the integer value after quantization.

Dequantize a single value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    int8_t quant_output_v = VALUE;
    float output_v = dl::dequantize(quant_output_v, DL_SCALE(model_output->exponent));

Dequantize ``dl::TensorBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // create a TensorBase filled with 0 of shape [1, 1]
    dl::TensorBase *output_tensor = new dl::TensorBase({1, 1}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    output_tensor->assign(model_output);

Model Inference
---------------------

See:

- :project:`example <examples/tutorial/how_to_run_model>`
- :cpp:func:`void dl::Model::run(runtime_mode_t mode)`
- :cpp:func:`void dl::Model::run(TensorBase *input, runtime_mode_t mode)`
- :cpp:func:`void dl::Model::run(std::map<std::string, TensorBase*> &user_inputs, runtime_mode_t mode, std::map<std::string, TensorBase*> user_outputs)`