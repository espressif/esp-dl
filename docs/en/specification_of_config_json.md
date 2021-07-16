# Specification of config.json

> The config.json helps to convert coefficient.npy in float-point to C/C++ in bit-quantize.



## Specification

Each item of config.json stands for the configuration of one layer. Take the following code as an example.

```json
{
    "l1": {"/* the configuration of layer l1 */"},
    "l2": {"/* the configuration of layer l2 */"},
    "l3": {"/* the configuration of layer l3 */"},
    ...
}
```

The key of each item is the **layer name.** Convert tool will search the related npy files according to the layer name. For example, with a layer named "l1", the tool will search the coefficient of filter by name "l1_filter.npy". **Please keep the layer name and the filename consistent.**



The value of each item is the **layer configuration.** Some arguments about layer configuration, listed in the table below, needed to be filled.

> **dropped** in table means drop this specific argument.
>
> **exponent** in table meas quantization exponent. Read [*About Bit Quantize*](./about_bit_quantize.md) for better understanding.

| Key               | Value                                                        |
| ----------------- | ------------------------------------------------------------ |
| "operation"       | **string**: "conv2d", "depthwise_conv2d"                     |
| "feature_type"    | **string**: "s16" for int16 quantization with element_width = 16, "s8" for int8 quantization with element_width = 8 |
| "filter_exponent" | **integer**: filter is quantized according to equation, value_float = value_int  * 2 ^ exponent<br />**dropped**: exponent is determined according to equation, exponent = log2(max(abs(value_float)) / 2 ^ (element_width - 1)). filter is quantized according to equation above. |
| "bias"            | **string**: "True" for adding bias, "False" and dropped for no bias |
| "output_exponent" | **integer**: both output and bias are quantized according to equation, value_float = value_int * 2 ^ exponent<br />> "output_exponent" is effective for "bias" coefficient converting only by now. So "output_exponent" could be dropped if there is no "bias" in this layer. |
| "activation"      | **dict**: details is in below<br />**dropped**: no activation |

The value of "activation" is the **activation configuration**, listed in the table below, needed to be filled.

| Key        | Value                                                        |
| ---------- | ------------------------------------------------------------ |
| "type"     | **string**: "ReLU", "LeakyReLU", "PReLU"                     |
| "exponent" | **integer**: activation is quantized according to equation, value_float = value_int  * 2 ^ exponent<br />**dropped**: entire range of element is projected to entire quantize range |



## Example

Let's assume that we have a one-layer-model.

layer name: "mylayer"

output_exponent: -10

feature_type: in int16 quantization

implement: PReLU(Conv2D(input, filter) + bias)

The config.json should be written as

```json
{
	"mylayer": {
		"operation": "conv2d",
		"feature_type": "s16",
        "bias": "True",
        "output_exponent": -10,
        "activation": {
            "type": "PReLU"
        }
	}
}
```
> "filter_exponent" is dropped.
>
> "exponent" of "activation" is dropped.


Meanwhile, `mylayer_filter.npy`, `mylayer_bias.npy` and `mylayer_activation.npy` should be ready.
