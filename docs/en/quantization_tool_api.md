# Quantization Toolkit API

## Calibrator  Class

### Initialization

```
Calibrator(quantization_bit, granularity, calib_method)
```

**Arguments**
- **quantization_bit** _(integer)_:
  - 8 for int8 quantization.
  - 16 for int16 quantization.
- **granularity** _(string)_:
  - If granularity = 'per-tensor'(default), there will be one exponent per entire tensor.
  - If granularity = 'per-channel', there will be one exponent for each channel of a convolution layer.
- **calib_method** _(string)_:   
  - If calib_method = 'minmax'(default), the threshold is the minimum and maximum values in the calibration dataset.
  - If calib_method = 'entropy', the threshold is derived from Kullback-Leibler divergence (KL divergence).

### *check_model* method

```
Calibrator.check_model(model_proto)
```
Checks the compatibility of your model.

**Argument**
- **model_proto** _(ModelProto)_: An FP32 ONNX model.

**Returns**
- **-1**: The model is incompatible.

### *set_method* method
```
Calibrator.set_method(granularity, calib_method)
```
Configures quantization.

**Arguments**
- **granularity** _(string)_:
  - If granularity = 'per-tensor'(default), there will be one exponent per entire tensor.
  - If granularity = 'per-channel', there will be one exponent for each channel of a convolution layer.
- **calib_method** _(string)_:   
  - If calib_method = 'minmax'(default), the threshold is the minimum and maximum values in the calibration dataset.
  - If calib_method = 'entropy', the threshold is derived from Kullback-Leibler divergence (KL divergence).

### *set_providers* method
```
Calibrator.set_providers(providers)
```
Configures the execution provider of ONNX Runtime.

**Argument**
- **providers** _(list of strings)_: An execution provider in the [list](https://onnxruntime.ai/docs/reference/execution-providers/), for example 'CPUExecutionProvider', and 'CUDAExecutionProvider'.


### *generate_quantization_table* method
```
Calibrator.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
```
Generates the quantization table.

**Arguments**
- **model_proto** _(ModelProto)_: An FP32 ONNX model.
- **calib_dataset** _(ndarray)_: The calibration dataset used to compute the threshold. The larger the dataset, the longer time it takes to generate the quantization table.
- **pickle_file_path** _(string)_: Path of the pickle file that stores the dictionary of quantization parameters.


## Evaluator Class

### Initialization

```
Evaluator(quantization_bit, target_chip)
```
**Arguments**
- **quantization_bit** _(integer)_:
  - 8 for int8 quantization.
  - 16 for int16 quantization.
- **target_chip** _(string)_: 'esp32s3' by default.


### *check_model* method
```
Evaluator.check_model(model_proto)
```
Checks the compatibility of your model.

**Argument**
- **model_proto** _(ModelProto)_: An FP32 ONNX model.

**Returns**
- **-1**: The model is incompatible.

### *set_target_chip* method
```
Evaluator.set_target_chip(target_chip)
```
Configures the chip environment to simulate.

**Argument**
- **target_chip** _(string)_: For now only 'esp32s3' is supported.


### *set_providers* method
```
Evaluator.set_providers(providers)
```
Configures the execution provider of ONNX Runtime.

**Argument**
- **providers** _(list of strings)_: An execution provider in the [list](https://onnxruntime.ai/docs/reference/execution-providers/), for example 'CPUExecutionProvider', and 'CUDAExecutionProvider'.


### *generate_quantized_model* method
```
Evaluator.generate_quantized_model(model_proto, pickle_file_path)
```
Generates the quantized model.

**Arguments**
- **model_proto** _(ModelProto)_: An FP32 ONNX model.
- **pickle_file_path** _(string)_: Path of the pickle file that stores all quantization parameters for the FP32 ONXX model. This pickle file must contain a dictionary of quantization parameters for all input and output nodes in the model graph.


### *evaluate_quantized_model* method
```
Evaluator.evaluate_quantized_model(batch_fp_input, to_float=false)
```
Obtains outputs of the quantized model.

**Arguments**
- **batch_fp_input** _(ndarray)_: Batch of floating-point inputs.
- **to_float** _(bool)_: 
  - False (default): Outputs will be returned directly.
  - True: Outputs will be converted to floating-point values.

**Returns**

A tuple of outputs and output_names:
- **outputs** _(list of ndarray)_: Outputs of the quantized model.
- **output_names** _(list of strings)_: Names of outputs.


