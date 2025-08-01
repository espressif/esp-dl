# Operator Support State

| Supported Targets | ESP32 | ESP32-S3 | ESP32-P4 |
| ----------------- | ----- | -------- | -------- |

## Quantization Strategy

The quantization type of all operators is symmetric quantization. Now ESP-DL supports both 8-bit and 16-bit.
The ``ESP32`` uses [rounding half up](https://simple.wikipedia.org/wiki/Rounding#Round_half_up) as its rounding strategy. Its operator is implemented in C, which is significantly slower compared to the ESP32-S3 and ESP32-P4.

    - When quantizing **ESP32** platform models using **ESP-PPQ**, set the target to ``c``. Because ESP-DL implements its operators in C.
    - When deploying **ESP32** platform models using **ESP-DL**, set the project compilation target to ``esp32``.

The rounding for ``ESP32-S3`` is [rounding half up](https://simple.wikipedia.org/wiki/Rounding#Round_half_up).
The rounding for ``ESP32-P4`` is [rounding half to even](https://simple.wikipedia.org/wiki/Rounding#Round_half_to_even).

## Support Operators

The ESP-DL operator interface is aligned with ONNX. The opset 13 is recommended to export ONNX.
Currently, the following 41 operators have been implemented and tested. Some operators do not implement all functionalities and attributes. Please refer to the restrictions of each operator or [test cases](./tools/ops_test/config/op_cfg.toml) for details.

Most operators maintain the same input/output data layout as ONNX or PyTorch. However, to fully leverage instruction-level acceleration, certain operators such as Conv, GlobalAveragePool, AveragePool, MaxPool, and Resize adopt NHWC or NWC data layouts for their inputs/outputs.
| Operator                                                                                                                                                     | int8     | int16    | Restrictions                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------|------------------------------------------------------------------------|
| Add[(ESP-DL)](esp-dl/dl/module/include/dl_module_add.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Add.html)                                             | &#10004; | &#10004; | Support up to 4D                                                       |
| AveragePool[(ESP-DL)](esp-dl/dl/module/include/dl_module_average_pool.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__AveragePool.html)                    | &#10004; | &#10004; | Support 1d/2d, don't support dilation                                  |
| Clip[(ESP-DL)](esp-dl/dl/module/include/dl_module_clip.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Clip.html)                                          | &#10004; | &#10004; |                                                                        |
| Concat[(ESP-DL)](esp-dl/dl/module/include/dl_module_concat.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Concat.html)                                    | &#10004; | &#10004; |                                                                        |
| Conv[(ESP-DL)](esp-dl/dl/module/include/dl_module_conv.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Conv.html)                                          | &#10004; | &#10004; | Support 1d/2d conv, groups only support 1 or input_channels            |
| Div[(ESP-DL)](esp-dl/dl/module/include/dl_module_div.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Div.html)                                             | &#10004; | &#10004; | Support up to 4D                                                       |
| Elu[(ESP-DL)](esp-dl/dl/module/include/dl_module_elu.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Elu.html)                                             | &#10004; | &#10004; |                                                                        |
| Equal[(ESP-DL)](esp-dl/dl/module/include/dl_module_equal.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Equal.html)                                       | &#10004; | &#10004; |                                                                        |
| Exp[(ESP-DL)](esp-dl/dl/module/include/dl_module_exp.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Exp.html)                                             | &#10004; | &#10004; |                                                                        |
| Flatten[(ESP-DL)](esp-dl/dl/module/include/dl_module_flatten.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Flatten.html)                                 | &#10004; | &#10004; |                                                                        |
| Gather[(ESP-DL)](esp-dl/dl/module/include/dl_module_gather.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Gather.html)                                    | &#10004; | &#10004; |                                                                        |
| Gemm[(ESP-DL)](esp-dl/dl/module/include/dl_module_gemm.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Gemm.html)                                          | &#10004; | &#10004; |                                                                        |
| GlobalAveragePool[(ESP-DL)](esp-dl/dl/module/include/dl_module_global_average_pool.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | &#10004; | &#10004; | Support 1d/2d                                                          |
| Greater[(ESP-DL)](esp-dl/dl/module/include/dl_module_greater.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Greater.html)                                 | &#10004; | &#10004; |                                                                        |
| GreaterOrEqual[(ESP-DL)](esp-dl/dl/module/include/dl_module_greater_or_equal.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html)          | &#10004; | &#10004; |                                                                        |
| HardSigmoid[(ESP-DL)](esp-dl/dl/module/include/dl_module_hard_sigmoid.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html)                    | &#10004; | &#10004; |                                                                        |
| HardSwish[(ESP-DL)](esp-dl/dl/module/include/dl_module_hard_swish.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__HardSwish.html)                          | &#10004; | &#10004; |                                                                        |
| LeakyRelu[(ESP-DL)](esp-dl/dl/module/include/dl_module_leaky_relu.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html)                          | &#10004; | &#10004; |                                                                        |
| Less[(ESP-DL)](esp-dl/dl/module/include/dl_module_less.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Less.html)                                          | &#10004; | &#10004; |                                                                        |
| LessOrEqual[(ESP-DL)](esp-dl/dl/module/include/dl_module_less_or_equal.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__LessOrEqual.html)                   | &#10004; | &#10004; |                                                                        |
| Log[(ESP-DL)](esp-dl/dl/module/include/dl_module_log.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Log.html)                                             | &#10004; | &#10004; |                                                                        |
| MatMul[(ESP-DL)](esp-dl/dl/module/include/dl_module_matmul.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__MatMul.html)                                    | &#10004; | &#10004; | Support up to 4D                                                       |
| MaxPool[(ESP-DL)](esp-dl/dl/module/include/dl_module_max_pool.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__MaxPool.html)                                | &#10004; | &#10004; | Support 1d/2d, don't support dilation                                  |
| Mul[(ESP-DL)](esp-dl/dl/module/include/dl_module_mul.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Mul.html)                                             | &#10004; | &#10004; | Support up to 4D                                                       |
| Pad[(ESP-DL)](esp-dl/dl/module/include/dl_module_pad.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Pad.html)                                             | &#10004; | &#10004; | Do not support wrap mode                                               |
| PRelu[(ESP-DL)](esp-dl/dl/module/include/dl_module_prelu.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__PRelu.html)                                       | &#10004; | &#10004; |                                                                        |
| Relu[(ESP-DL)](esp-dl/dl/module/include/dl_module_relu.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Relu.html)                                          | &#10004; | &#10004; |                                                                        |
| Reshape[(ESP-DL)](esp-dl/dl/module/include/dl_module_reshape.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Reshape.html)                                 | &#10004; | &#10004; |                                                                        |
| Resize[(ESP-DL)](esp-dl/dl/module/include/dl_module_resize.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Resize.html)                                    | &#10004; | &#10006; | support 1d/2d nearest/linear/bilinear, don't support roi and antialias |
| ReverseSequence[(ESP-DL)](esp-dl/dl/module/include/dl_module_reverse_sequence.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html)        | &#10004; | &#10004; |                                                                        |
| Sigmoid[(ESP-DL)](esp-dl/dl/module/include/dl_module_sigmoid.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Sigmoid.html)                                 | &#10004; | &#10004; |                                                                        |
| Slice[(ESP-DL)](esp-dl/dl/module/include/dl_module_slice.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Slice.html)                                       | &#10004; | &#10004; |                                                                        |
| Softmax[(ESP-DL)](esp-dl/dl/module/include/dl_module_softmax.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Softmax.html)                                 | &#10004; | &#10004; | Dtype of output is float32                                             |
| Split[(ESP-DL)](esp-dl/dl/module/include/dl_module_split.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Split.html)                                       | &#10004; | &#10004; |                                                                        |
| Sqrt[(ESP-DL)](esp-dl/dl/module/include/dl_module_sqrt.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Sqrt.html)                                          | &#10004; | &#10004; |                                                                        |
| Squeeze[(ESP-DL)](esp-dl/dl/module/include/dl_module_squeeze.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Squeeze.html)                                 | &#10004; | &#10004; |                                                                        |
| Sub[(ESP-DL)](esp-dl/dl/module/include/dl_module_sub.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Sub.html)                                             | &#10004; | &#10004; | Support up to 4D                                                       |
| Swish[(ESP-DL)](esp-dl/dl/module/include/dl_module_swish.hpp)                                                                                                | &#10004; | &#10004; |                                                                        |
| Tanh[(ESP-DL)](esp-dl/dl/module/include/dl_module_tanh.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Tanh.html)                                          | &#10004; | &#10004; |                                                                        |
| Transpose[(ESP-DL)](esp-dl/dl/module/include/dl_module_transpose.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Transpose.html)                           | &#10004; | &#10004; |                                                                        |
| Unsqueeze[(ESP-DL)](esp-dl/dl/module/include/dl_module_unsqueeze.hpp)[(ONNX)](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html)                           | &#10004; | &#10004; |                                                                        |

Generation Time: 2025-07-10 11:22:12