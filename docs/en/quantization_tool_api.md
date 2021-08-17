# Quantization tool APIs
## Calibrator class

**initialization**

```
Calibration(quantization_bit, granularity, method)
```

Arguments
- quantization_bit (int): 8 for int8 quantization, 16 for int16 quantization
- granularity (str):
  - If granularity = 'per-tensor'(default), there will be one scale per entire tensor
  - If granularity = 'per-channel', there will be one scale per slice in the output_channel of the weights of convolutions
- calib_method (str):
  - If calib_method = 'minmax'(default), the min and max values of the layer outputs obtained from the calibration dataset will be used as the threshold
  - If calib_method = 'entropy', the threshold will be derived based on KL divergence


***check_model* method**
```
Calibrator.check_model(model_proto)
```
Check the compatibility of the provided model

Arguments
- model_proto (ModelProto): a fp32 onnx model

Returns
- -1 if the model is incompatible.

***set_method* method**
```
Calibrator.set_method(granularity, calib_method)
```
Set the configuration of quantization

Arguments
- granularity (str): 
  - If granularity = 'per-tensor', there will be one scale per entire tensor
  - If granularity = 'per-channel', there will be one scale per slice in the output_channel of the weights of convolutions
- calib_method (str): 
  - If calib_method = 'minmax', the min and max values of the layer outputs obtained from the calibration dataset will be used as the threshold
  - If calib_method = 'entropy', the threshold will be derived based on KL divergence

***set_providers* method**
```
Calibrator.set_providers(providers)
```
Set onnx runtime execution providers

Arguments
- providers (list of strings): list of execution providers for onnx runtime, eg. 'CPUExecutionProvider', 'CUDAExecutionProvider', more details in https://onnxruntime.ai/docs/reference/execution-providers/

***generate_quantization_table* method**
```
Calibrator.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
```

Arguments
- model_proto (ModelProto): a fp32 onnx model
- calib_dataset (ndarray): calibration dataset, the whole dataset will be used to compute the threshold. The larger the dataset, the longer the time it will take to generate the final table
- pickle_file_path (str): path of the pickle file that stores the dict of quantization parameters


## Evaluator class

**initialization**

```
Evaluator(quantization_bit, target_chip)
```

Arguments
- quantization_bit (int): 8 for int8 quantization, 16 for int16 quantization
- target_chip (str): 'esp32s3' (default)

***check_model* method**
```
Evaluator.check_model(model_proto)
```
Check the compatibility of the provided model

Arguments
- model_proto (ModelProto): a fp32 onnx model

Returns
- -1 if the model is incompatible.

***set_target_chip* method**
```
Evaluator.set_target_chip(target_chip)
```
Set the simulated chip environment

Arguments
- target_chip (str): the chip envirionment to simulate, currently only support 'esp32s3'


***set_providers* method**
```
Evaluator.set_providers(providers)
```
Set onnx runtime execution providers

Arguments
- providers (list of strings): list of execution providers for onnx runtime, eg. 'CPUExecutionProvider', 'CUDAExecutionProvider', more details in  https://onnxruntime.ai/docs/reference/execution-providers/

***generate_quantized_model* method**
```
Evaluator.generate_quantized_model(model_proto, pickle_file_path)
```
Generate the quantized model 

Arguments
- model_proto (ModelProto): a fp32 onnx model
- pickle_file_path (str): path of a pickle file that stores all the quantization parameters for the corresponding fp32 onnx model. The pickle file must contain a dict of quantization paramater for all inputs and outputs of all nodes inside the model graph  

***evaluate_quantized_model* method**
```
Evaluator.evaluate_quantized_model(batch_fp_input, to_float=false)
```
Obtain the outputs of the quantized model
Arguments
- batch_fp_input (ndarray): batch of floating-point inputs
- to_float (bool): 
  - if to_float = False(default): the quantized output will be returned directly
  - if to_float = True: the returned output will be re-scaled to floating point value

Returns
- tuple of outputs and output_names
  - outputs (list of ndarray): a list of outputs of the quantized model, 
  - output_names (list of string): a list of name of outputs
