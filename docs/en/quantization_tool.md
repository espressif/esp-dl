# Quantization tool
This quantization tool is provided for users who want to run the inference of their models on ESP SoCs using [ESP-DL](https://github.com/espressif/esp-dl). 

This tool consists of three independent parts: optimizer, calibrator, evaluator. The optimizer performs graph optimization. The calibrator supports post-training quantization, which means you do not need to re-train your model. The evaluator helps you evaluate the performance of the quantized model. Specification of each part will be given in the following sections. APIs for each part are listed in [quantization_tool_api.md](quantization_tool_api.md).

Before you use the tool, please convert your own model to an [onnx]([GitHub - onnx/onnx: Open standard for machine learning interoperability](https://github.com/onnx/onnx) model. This whole quantization tool runs based on onnx system. More help can be found in appendix.

## Optimizer
The optimizer provides graph optimization to improve the model performance. This process contains redundant node elimination, model structure simplification, model fusion, etc.

The optimizer is mostly using [onnx-optimizer](https://github.com/onnx/optimizer), a tool to perform optimizations on onnx models. You can select the optimization passes from [provided passes](https://github.com/onnx/optimizer/tree/master/onnxoptimizer/passes) to apply on your model. Some additional optimization are also implemented. You can check the source code in [optimizer.py](). It is important to enable graph fusion before quantization, especially batch normalization fusion.

It is recommended to pass your model through the optimizer before you use the calibrator or evaluator. You can use [netron]([GitHub - lutzroeder/netron: Visualizer for neural network, deep learning, and machine learning models](https://github.com/lutzroeder/netron) to view your model structure.


**Python example API**
Enable two types of model fusion to the given model: fuse batch normalization into convolution, fuse add bias into convolution. The batch size of the inputs are set to dynamic for running inference. 

```
model_proto = onnx.load('mnsit.onnx')
model_proto = onnxoptimizer.optimize(model_proto, ['fuse_bn_into_conv', fuse_add_bias_into_conv])
optimized_model = convert_model_batch_to_dynamic(model_proto)
optimized_model_path = 'mnist_optimized.onnx'
onnx.save(new_model,  optimized_model_path)
```

## Calibrator
The calibrator helps convert a floating-point model to a quantized model which meets the requirements of running its quantized reference on ESP SoCs. For more details about the supported quantized operations on chip, please check [quantization_specification.md](quantization_specification.md).

The overall flow for converting a model from FP32 to INT8/INT16 is:
- prepare an fp32 model
- prepare a calibration dataset
- configure the quantization setting
- obtain the quantization parameter

#### FP32 model
The provided fp32 model must be compatible with the existing library. If the model contains operations that are not supported, it cannot be accepted by this calibrator. If your model is incompatible, you will receive the related error messages.
The model will be checked when you try to obtain the quantization parameter, or you can also call *check_model* method to confirm in advance. The normalization of inputs should be excluded from the model for better performance.

#### Calibration dataset
Choosing an appropriate calibration dataset is important for quantization. A good calibration dataset should be representative. You can try on different calibration datasets and compare the performance of the quantized model when using different quantization parameters.

#### Configuration
This tool supports both fully int8 and int16 quantization. The provided configurations for different precision are as below:

int8:
- granularity: 'per-tensor', 'per-channel'
- calibration_method: 'entropy', 'minmax'

int16:
- granularity: 'per-tensor'
- calibration_method: 'entropy', 'minmax'

#### Quantization parameter
The returned quantization table is a list that contains the calculated parameter for all layers.


**Python API example**

Quantize the optimized mnist model to int8 model,  using per-channel quantization, entropy calibration method. The prepared *calib_dataset* is used for calibration. The quantization parameter is saved to the given pickle file.

```
model_proto = onnx.load(optimized_model_path)
calib = Calibrator(8, 'per-channel', 'entropy')
calib.set_providers(['CPUExecutionProvider'])
pickle_file_path = 'mnist_calib.pickle'
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
```

## Evaluator
The evaluator is a tool that simulates the calculation process on chip. You can use this tool to evaluate the performance of the quantized model as it runs on ESP chips.

If the model contains operations that are not supported by ESP-DL library, it cannot be accepted by this evaluator. If your model is incompatible, you will receive the related error message.

If the performance of the quantized model does not satisfy the needs, please consider Quantization Aware Training.


**Python API example**

The int8 mnist model on esp32s3 chip is generated based on the quantization parameter in the given pickle file. The floating-point result is returned.  

```
eva = Evaluator(8, 'esp32s3')
eva.generate_quantized_model(model_proto, pickle_file_path)
outputs = eva.evaluate_quantized_model(test_images, to_float = True)
res = np.argmax(outputs[0])
```

## Example
please refer to [example.py]() for an full example of quantization and evaluation of a mnist model.


## Appendix
[ONNX](https://github.com/onnx/onnx) (Open Neural Network Exchange) is an open source format for AI models. It is widely supported by many frameworks.

Environment Requirements:
- onnx
- onnxruntime
- onnxoptimizer

The following links may provide some help for you to convert to onnx model from other platform.

tensorflow: [GitHub - onnx/tensorflow-onnx: Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx) 

mxnet: https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/deploy/export/onnx.html 

pytorch: [torch.onnx &mdash; PyTorch 1.9.0 documentation](https://pytorch.org/docs/stable/onnx.html)
