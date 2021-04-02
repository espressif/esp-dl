# DL-LIB

This is a component which provides APIs for neurual network and some deep-learning application APIs, such as Cat Face Detection, Human Face Detection and Human Face Recognition. It can be used as a component of some project as it doesn't support any interface of periphrals. By default, it works along with ESP-WHO, which is a project-level repository. 

> It's an alpha version for the customers who have received ESP32S3 series chip, 7.2.5 and 7.2.8. We are working on perfecting it.



## Neural Network Support

### Data Type Support

- [x] 16-bit
- [ ] 8-bit

The DL_LIB only supports quantization calculation. Element will be quantized in following rule.
$$
element_{float} * 2^{exponent} = element_{quantized}
$$


### API

|                       | ESP32 | ESP32S2 | ESP32C3 | 7.2.5 | ESP32S3 |
| --------------------- | ----- | ------- | ------- | ----- | ------- |
| Conv2D                |       |         |         |       |         |
| DepthwiseConv2D       |       |         |         |       |         |
| GlobalDepthwiseConv2D |       |         |         |       |         |
| Concat2D              |       |         |         |       |         |
| ReLU                  |       |         |         |       |         |
| LeakyReLU             |       |         |         |       |         |
| PReLU                 |       |         |         |       |         |
|                       |       |         |         |       |         |
|                       |       |         |         |       |         |
|                       |       |         |         |       |         |



## Build Your Own Model

### Step 1: save model to npy files

Save the float point coefficients of model into npy files, layer by layer, e.g.

```python
import numpy
numpy.save(file=f'{root}/{layer_name}_filter.npy', arr=filter)
numpy.save(file=f'{root}/{layer_name}_bias.npy', arr=bias)
```

Write a json file for model configuration. The format is 

```json
{
    "layer_name": {				// must be the same as the corresponding npy file
        "type": "conv2d", 		// "conv2d", "depthwise_conv2d", "global_depthwise_conv2d" only by now
        "filter": -99, 			// exponent of filter. If it equals to -99, the convert tool will select an exponent to project filter to whole quantization range.
        "element_width": 16, 	// quantization element width
        "bias": -10, 			// exponent of bias, which must be equal to output's
        "activation": {
            "type": "relu"		// "relu", "leaky_relu", "prelu" only by now
        }
    }, 
    ... ...
}
```



```json
{
    "b_0": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -10, 
        "activation": {
            "type": "relu"
        }
    }, 
    "b_1_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "b_1_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -8
    }
}
```

