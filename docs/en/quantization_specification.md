# Quantization Specification

[Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) converts floating-point values to fixed-point values. This conversion technique can shrink model size, and reduce CPU and hardware accelerator latency without losing accuracy.

For chips such as ESP32-S3, which has relatively limited memory yet up to 7.5 giga multiply-accumulate operations (MAC) per second at 240 MHz, it is necessary to run inferences with a quantized model. You can use the provided [Quantization Toolkit](../../tools/quantization_tool/README.md) to quantize your floating-point model, or deploy your integer model following steps in [Usage of convert.py](../../tools/convert_tool/README.md).

## Full Integer Quantization

All data in the model are quantized to 8-bit or 16-bit integers, including
  - constants weights, biases, and activations
  - variable tensors, such as inputs and outputs of intermediate layers (activations)

Such 8-bit or 16-bit quantization approximates floating-point values using the following formula：

```math
real\_value = int\_value * 2^{\ exponent}
```

### Signed Integer

For 8-bit quantization, `int_value` is represented by a value in the signed **int8** range [-128, 127]. For 16-bit quantization, `int_value` is represented by a value in the signed **int16** range [-32768, 32767].

### Symmetric

All the quantized data are **symmetric**, which means no zero point (bias), so we can avoid the runtime cost of multiplying the zero point with other values.

### Granularity

**Per-tensor (aka per-layer) quantization** means that there will be only one exponent per entire tensor, and all the values within the tensor are quantized by this exponent.

**Per-channel quantization** means that there will be different exponents for each channel of a convolution kernel.

Compared with per-tensor quantization, usually per-channel quantization can achieve higher accuracy on some models. However, it would be more time-consuming. You can simulate inference on chips using the *evaluator* in [Quantization Toolkit](../../tools/quantization_tool/README.md) to see the performance after quantization, and then decide which form of quantization to apply.

For 16-bit quantization, we only support per-tensor quantization to ensure faster computation. For 8-bit quantization, we support both per-tensor and per-channel quantization, allowing a trade-off between performance and speed.


## Quantized Operator Specifications

Below we describe the quantization requirements for our APIs:
```
ADD2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

AVERAGE_POOL_2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

CONCATENATION
  Input ...:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Inputs and output must have the same exponent

CONV_2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8 / int16
    range      : [-127, 127] / [-32767, 32767]
    granularity: {per-channel / per-tensor for int8} / {per-tensor for int16} 
  Input 2 (Bias):
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
    restriction: exponent = output_exponent
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

DEPTHWISE_CONV_2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8 / int16
    range      : [-127, 127] / [-32767, 32767]
    granularity: {per-channel / per-tensor for int8} / {per-tensor for int16} 
  Input 2 (Bias):
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
    restriction: exponent = output_exponent
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

MAX_POOL_2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

MUL2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

SUB2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor

MAX2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Input and output must have the same exponent

MIN2D
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Input and output must have the same exponent

ReLU
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Input and output must have the same exponent

LeakyReLU
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1 (Alpha):
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Input and output must have the same exponent

PReLU
  Input 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  Input 1 (Alpha):
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
  Output 0:
    data_type  : int8 / int16
    range      : [-128, 127] / [-32768, 32767]
    granularity: per-tensor
  restriction: Input and output must have the same exponent

```