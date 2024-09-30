#include "test_data.hpp"

#include "dl_base.hpp"
#include "dl_define.hpp"
#include "dl_tool.hpp"

#include "dl_layer_add2d.hpp"
#include "dl_layer_avg_pool2d.hpp"
#include "dl_layer_concat.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_depthwise_conv2d.hpp"
#include "dl_layer_expand_dims.hpp"
#include "dl_layer_flatten.hpp"
#include "dl_layer_fullyconnected.hpp"
#include "dl_layer_global_avg_pool2d.hpp"
#include "dl_layer_global_max_pool2d.hpp"
#include "dl_layer_leakyrelu.hpp"
#include "dl_layer_max2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_min2d.hpp"
#include "dl_layer_mul2d.hpp"
#include "dl_layer_pad.hpp"
#include "dl_layer_prelu.hpp"
#include "dl_layer_relu.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_squeeze.hpp"
#include "dl_layer_sub2d.hpp"
#include "dl_layer_transpose.hpp"

#include "unity.h"
#include <limits.h>

using namespace dl;
using namespace nn;
using namespace layer;
using namespace tool;
using namespace base;
using namespace std;

int exponent0 = 0;
int exponent1 = 0;
int output_exponent = 0;

int height = 3;
int width = 3;
int channel = 16;

bool test_add_layer()
{
    printf("\n add\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Add2D<int8_t> _add2d_s8_1(output_exponent, NULL, "add2d_s8 inplace", true);
    Add2D<int8_t> _add2d_s8_2(output_exponent, NULL, "add2d_s8 no inplace", false);
    Add2D<int16_t> _add2d_s16_1(output_exponent, NULL, "add2d_s16 inplace", true);
    Add2D<int16_t> _add2d_s16_2(output_exponent, NULL, "add2d_s16 no inplace", false);

    _add2d_s8_1.build(input0_s8, input1_s8, true);
    _add2d_s8_2.build(input0_s8, input1_s8, true);
    _add2d_s16_1.build(input0_s16, input1_s16, true);
    _add2d_s16_2.build(input0_s16, input1_s16, true);

    _add2d_s8_1.call(input0_s8, input1_s8);
    _add2d_s8_1.get_output().print({}, "add2d_s8 inplace");
    _add2d_s8_2.call(input0_s8, input1_s8);
    _add2d_s8_2.get_output().print({}, "add2d_s8 no inplace");
    _add2d_s16_1.call(input0_s16, input1_s16);
    _add2d_s16_1.get_output().print({}, "add2d_s16 inplace");
    _add2d_s16_2.call(input0_s16, input1_s16);
    _add2d_s16_2.get_output().print({}, "add2d_s16 no inplace");

    return true;
}

bool test_sub_layer()
{
    printf("\n sub\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Sub2D<int8_t> _sub2d_s8_1(output_exponent, NULL, "sub2d_s8 inplace", true);
    Sub2D<int8_t> _sub2d_s8_2(output_exponent, NULL, "sub2d_s8 no inplace", false);
    Sub2D<int16_t> _sub2d_s16_1(output_exponent, NULL, "sub2d_s16 inplace", true);
    Sub2D<int16_t> _sub2d_s16_2(output_exponent, NULL, "sub2d_s16 no inplace", false);

    _sub2d_s8_1.build(input0_s8, input1_s8, true);
    _sub2d_s8_2.build(input0_s8, input1_s8, true);
    _sub2d_s16_1.build(input0_s16, input1_s16, true);
    _sub2d_s16_2.build(input0_s16, input1_s16, true);

    _sub2d_s8_1.call(input0_s8, input1_s8);
    _sub2d_s8_1.get_output().print({}, "sub2d_s8 inplace");
    _sub2d_s8_2.call(input0_s8, input1_s8);
    _sub2d_s8_2.get_output().print({}, "sub2d_s8 no inplace");
    _sub2d_s16_1.call(input0_s16, input1_s16);
    _sub2d_s16_1.get_output().print({}, "sub2d_s16 inplace");
    _sub2d_s16_2.call(input0_s16, input1_s16);
    _sub2d_s16_2.get_output().print({}, "sub2d_s16 no inplace");

    return true;
}

bool test_mul_layer()
{
    printf("\n mul\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Mul2D<int8_t> _mul2d_s8_1(output_exponent, NULL, "mul2d_s8 inplace", true);
    Mul2D<int8_t> _mul2d_s8_2(output_exponent, NULL, "mul2d_s8 no inplace", false);
    Mul2D<int16_t> _mul2d_s16_1(output_exponent, NULL, "mul2d_s16 inplace", true);
    Mul2D<int16_t> _mul2d_s16_2(output_exponent, NULL, "mul2d_s16 no inplace", false);

    _mul2d_s8_1.build(input0_s8, input1_s8, true);
    _mul2d_s8_2.build(input0_s8, input1_s8, true);
    _mul2d_s16_1.build(input0_s16, input1_s16, true);
    _mul2d_s16_2.build(input0_s16, input1_s16, true);

    _mul2d_s8_1.call(input0_s8, input1_s8);
    _mul2d_s8_1.get_output().print({}, "mul2d_s8 inplace");
    _mul2d_s8_2.call(input0_s8, input1_s8);
    _mul2d_s8_2.get_output().print({}, "mul2d_s8 no inplace");
    _mul2d_s16_1.call(input0_s16, input1_s16);
    _mul2d_s16_1.get_output().print({}, "mul2d_s16 inplace");
    _mul2d_s16_2.call(input0_s16, input1_s16);
    _mul2d_s16_2.get_output().print({}, "mul2d_s16 no inplace");

    return true;
}

bool test_expand_dims_layer()
{
    printf("\n expand_dims\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    ExpandDims<int8_t> _expand_dims_s8_1({0, 1}, "expand_dims_s8 inplace", true);
    ExpandDims<int8_t> _expand_dims_s8_2({0, 1}, "expand_dims_s8 no inplace", false);
    ExpandDims<int16_t> _expand_dims_s16_1({0, 1}, "expand_dims_s16 inplace", true);
    ExpandDims<int16_t> _expand_dims_s16_2({0, 1}, "expand_dims_s16 no inplace", false);

    _expand_dims_s8_1.build(input0_s8, true);
    _expand_dims_s8_2.build(input1_s8, true);
    _expand_dims_s16_1.build(input0_s16, true);
    _expand_dims_s16_2.build(input1_s16, true);

    _expand_dims_s8_2.call(input1_s8);
    _expand_dims_s8_2.get_output().print({}, "expand_dims_s8 no inplace");
    _expand_dims_s8_1.call(input0_s8);
    _expand_dims_s8_1.get_output().print({}, "expand_dims_s8 inplace");
    _expand_dims_s16_2.call(input1_s16);
    _expand_dims_s16_2.get_output().print({}, "expand_dims2d_s16 no inplace");
    _expand_dims_s16_1.call(input0_s16);
    _expand_dims_s16_1.get_output().print({}, "expand_dims_s16 inplace");

    return true;
}

bool test_flatten_layer()
{
    printf("\n flatten\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    Flatten<int8_t> _flatten_s8_1("flatten_s8 inplace", true);
    Flatten<int8_t> _flatten_s8_2("flatten_s8 no inplace", false);
    Flatten<int16_t> _flatten_s16_1("flatten_s16 inplace", true);
    Flatten<int16_t> _flatten_s16_2("flatten_s16 no inplace", false);

    _flatten_s8_1.build(input0_s8, true);
    _flatten_s8_2.build(input1_s8, true);
    _flatten_s16_1.build(input0_s16, true);
    _flatten_s16_2.build(input1_s16, true);

    _flatten_s8_2.call(input1_s8);
    _flatten_s8_2.get_output().print({}, "flatten_s8 no inplace");
    _flatten_s8_1.call(input0_s8);
    _flatten_s8_1.get_output().print({}, "flatten_s8 inplace");
    _flatten_s16_2.call(input1_s16);
    _flatten_s16_2.get_output().print({}, "flatten2d_s16 no inplace");
    _flatten_s16_1.call(input0_s16);
    _flatten_s16_1.get_output().print({}, "flatten_s16 inplace");

    return true;
}

bool test_squeeze_layer()
{
    printf("\n squeeze\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({1, height, width, channel, 1}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({1, height, width, channel, 1}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    Squeeze<int8_t> _squeeze_s8_1(INT32_MAX, "squeeze_s8 inplace", true);
    Squeeze<int8_t> _squeeze_s8_2(-1, "squeeze_s8 no inplace", false);
    Squeeze<int16_t> _squeeze_s16_1(INT32_MAX, "squeeze_s16 inplace", true);
    Squeeze<int16_t> _squeeze_s16_2(-1, "squeeze_s16 no inplace", false);

    _squeeze_s8_1.build(input0_s8, true);
    _squeeze_s8_2.build(input1_s8, true);
    _squeeze_s16_1.build(input0_s16, true);
    _squeeze_s16_2.build(input1_s16, true);

    _squeeze_s8_2.call(input1_s8);
    _squeeze_s8_2.get_output().print({}, "squeeze_s8 no inplace");
    _squeeze_s8_1.call(input0_s8);
    _squeeze_s8_1.get_output().print({}, "squeeze_s8 inplace");
    _squeeze_s16_2.call(input1_s16);
    _squeeze_s16_2.get_output().print({}, "squeeze2d_s16 no inplace");
    _squeeze_s16_1.call(input0_s16);
    _squeeze_s16_1.get_output().print({}, "squeeze_s16 inplace");

    return true;
}

bool test_reshape_layer()
{
    printf("\n reshape\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    Reshape<int8_t> _reshape_s8_1({-1, width * channel}, "reshape_s8 inplace", true);
    Reshape<int8_t> _reshape_s8_2({-1, width * channel}, "reshape_s8 no inplace", false);
    Reshape<int16_t> _reshape_s16_1({-1, width * channel}, "reshape_s16 inplace", true);
    Reshape<int16_t> _reshape_s16_2({-1, width * channel}, "reshape_s16 no inplace", false);

    _reshape_s8_1.build(input0_s8, true);
    _reshape_s8_2.build(input1_s8, true);
    _reshape_s16_1.build(input0_s16, true);
    _reshape_s16_2.build(input1_s16, true);

    _reshape_s8_2.call(input1_s8);
    _reshape_s8_2.get_output().print({}, "reshape_s8 no inplace");
    _reshape_s8_1.call(input0_s8);
    _reshape_s8_1.get_output().print({}, "reshape_s8 inplace");
    _reshape_s16_2.call(input1_s16);
    _reshape_s16_2.get_output().print({}, "reshape2d_s16 no inplace");
    _reshape_s16_1.call(input0_s16);
    _reshape_s16_1.get_output().print({}, "reshape_s16 inplace");

    return true;
}

bool test_transpose_layer()
{
    printf("\n transpose\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    Transpose<int8_t> _transpose_s8_1({-1, 0, 1}, "transpose_s8 inplace", true);
    Transpose<int8_t> _transpose_s8_2({-1, 0, 1}, "transpose_s8 no inplace", false);
    Transpose<int16_t> _transpose_s16_1({-1, 0, 1}, "transpose_s16 inplace", true);
    Transpose<int16_t> _transpose_s16_2({-1, 0, 1}, "transpose_s16 no inplace", false);

    _transpose_s8_1.build(input0_s8, true);
    _transpose_s8_2.build(input1_s8, true);
    _transpose_s16_1.build(input0_s16, true);
    _transpose_s16_2.build(input1_s16, true);

    input0_s8.set_shape({height, width, channel});
    input0_s16.set_shape({height, width, channel});

    _transpose_s8_2.call(input1_s8);
    _transpose_s8_2.get_output().print({}, "transpose_s8 no inplace");
    _transpose_s8_1.call(input0_s8);
    _transpose_s8_1.get_output().print({}, "transpose_s8 inplace");
    _transpose_s16_2.call(input1_s16);
    _transpose_s16_2.get_output().print({}, "transpose2d_s16 no inplace");
    _transpose_s16_1.call(input0_s16);
    _transpose_s16_1.get_output().print({}, "transpose_s16 inplace");

    return true;
}

bool test_pad_layer()
{
    printf("\n pad\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(2);
    Tensor<int8_t> input1_s8(input0_s8, true);
    Tensor<int16_t> input1_s16(input0_s16, true);

    Pad<int8_t> _pad_s8_1({1, 1, 1, 1, 1, 1}, {100, 101, 102, 103, 104, 105}, PADDING_CONSTANT, "pad_s8 constant");
    Pad<int8_t> _pad_s8_2({1, 1, 1, 1, 1, 1}, {100, 101, 102, 103, 104, 105}, PADDING_REFLECT, "pad_s8 reflect");
    Pad<int16_t> _pad_s16_1({1, 1, 1, 1, 1, 1}, {100, 101, 102, 103, 104, 105}, PADDING_CONSTANT, "pad_s16 constant");
    Pad<int16_t> _pad_s16_2({1, 1, 1, 1, 1, 1}, {100, 101, 102, 103, 104, 105}, PADDING_REFLECT, "pad_s16 reflect");

    _pad_s8_1.build(input0_s8, true);
    _pad_s8_2.build(input1_s8, true);
    _pad_s16_1.build(input0_s16, true);
    _pad_s16_2.build(input1_s16, true);

    _pad_s8_1.call(input0_s8);
    _pad_s8_1.get_output().print({}, "pad_s8 constant");
    _pad_s8_2.call(input1_s8);
    _pad_s8_2.get_output().print({}, "pad_s8 reflect");
    _pad_s16_1.call(input0_s16);
    _pad_s16_1.get_output().print({}, "pad2d_s16 constant");
    _pad_s16_2.call(input1_s16);
    _pad_s16_2.get_output().print({}, "pad_s16 reflect");

    return true;
}

bool test_min_layer()
{
    printf("\n min\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Min2D<int8_t> _min2d_s8_1("min2d_s8 inplace", true);
    Min2D<int8_t> _min2d_s8_2("min2d_s8 no inplace", false);
    Min2D<int16_t> _min2d_s16_1("min2d_s16 inplace", true);
    Min2D<int16_t> _min2d_s16_2("min2d_s16 no inplace", false);

    _min2d_s8_1.build(input0_s8, input1_s8, true);
    _min2d_s8_2.build(input0_s8, input1_s8, true);
    _min2d_s16_1.build(input0_s16, input1_s16, true);
    _min2d_s16_2.build(input0_s16, input1_s16, true);

    _min2d_s8_1.call(input0_s8, input1_s8);
    _min2d_s8_1.get_output().print({}, "min2d_s8 inplace");
    _min2d_s8_2.call(input0_s8, input1_s8);
    _min2d_s8_2.get_output().print({}, "min2d_s8 no inplace");
    _min2d_s16_1.call(input0_s16, input1_s16);
    _min2d_s16_1.get_output().print({}, "min2d_s16 inplace");
    _min2d_s16_2.call(input0_s16, input1_s16);
    _min2d_s16_2.get_output().print({}, "min2d_s16 no inplace");

    return true;
}

bool test_max_layer()
{
    printf("\n max\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Max2D<int8_t> _max2d_s8_1("max2d_s8 inplace", true);
    Max2D<int8_t> _max2d_s8_2("max2d_s8 no inplace", false);
    Max2D<int16_t> _max2d_s16_1("max2d_s16 inplace", true);
    Max2D<int16_t> _max2d_s16_2("max2d_s16 no inplace", false);

    _max2d_s8_1.build(input0_s8, input1_s8, true);
    _max2d_s8_2.build(input0_s8, input1_s8, true);
    _max2d_s16_1.build(input0_s16, input1_s16, true);
    _max2d_s16_2.build(input0_s16, input1_s16, true);

    _max2d_s8_1.call(input0_s8, input1_s8);
    _max2d_s8_1.get_output().print({}, "max2d_s8 inplace");
    _max2d_s8_2.call(input0_s8, input1_s8);
    _max2d_s8_2.get_output().print({}, "max2d_s8 no inplace");
    _max2d_s16_1.call(input0_s16, input1_s16);
    _max2d_s16_1.get_output().print({}, "max2d_s16 inplace");
    _max2d_s16_2.call(input0_s16, input1_s16);
    _max2d_s16_2.get_output().print({}, "max2d_s16 no inplace");

    return true;
}

bool test_global_max_pool_layer()
{
    printf("\n global_max_pool\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value({0, 1, 0, 3, 0, 16}, 1);
    input0_s8.set_value({1, 2, 0, 3, 0, 16}, 2);
    input0_s8.set_value({2, 3, 0, 3, 0, 16}, 3);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value({0, 1, 0, 3, 0, 16}, 1);
    input0_s16.set_value({1, 2, 0, 3, 0, 16}, 2);
    input0_s16.set_value({2, 3, 0, 3, 0, 16}, 3);

    GlobalMaxPool2D<int8_t> _global_max_pool_s8_1("global_max_pool_s8");
    GlobalMaxPool2D<int16_t> _global_max_pool_s16_1("global_max_pool_s16");

    _global_max_pool_s8_1.build(input0_s8, true);
    _global_max_pool_s16_1.build(input0_s16, true);

    _global_max_pool_s8_1.call(input0_s8);
    _global_max_pool_s8_1.get_output().print({}, "global_max_pool_s8 constant");
    _global_max_pool_s16_1.call(input0_s16);
    _global_max_pool_s16_1.get_output().print({}, "global_max_pool2d_s16 constant");

    return true;
}

bool test_global_avg_pool_layer()
{
    printf("\n global_avg_pool\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value({0, 1, 0, 3, 0, 16}, 9);
    input0_s8.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s8.set_value({2, 3, 0, 3, 0, 16}, 27);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value({0, 1, 0, 3, 0, 16}, 9);
    input0_s16.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s16.set_value({2, 3, 0, 3, 0, 16}, 27);

    GlobalAveragePool2D<int8_t> _global_avg_pool_s8_1(output_exponent, "global_avg_pool_s8");
    GlobalAveragePool2D<int16_t> _global_avg_pool_s16_1(output_exponent, "global_avg_pool_s16");

    _global_avg_pool_s8_1.build(input0_s8, true);
    _global_avg_pool_s16_1.build(input0_s16, true);

    _global_avg_pool_s8_1.call(input0_s8);
    _global_avg_pool_s8_1.get_output().print({}, "global_avg_pool_s8 constant");
    _global_avg_pool_s16_1.call(input0_s16);
    _global_avg_pool_s16_1.get_output().print({}, "global_avg_pool2d_s16 constant");

    return true;
}

bool test_relu_layer()
{
    printf("\n relu\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s8.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s8.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s8.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s8.set_value({2, 3, 0, 3, 0, 16}, 27);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s16.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s16.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s16.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s16.set_value({2, 3, 0, 3, 0, 16}, 27);

    Relu<int8_t> _relu_s8_1("relu_s8 inplace", true);
    Relu<int8_t> _relu_s8_2("relu_s8 no inplace", false);
    Relu<int16_t> _relu_s16_1("relu_s16 inplace", true);
    Relu<int16_t> _relu_s16_2("relu_s16 no inplace", false);

    _relu_s8_1.build(input0_s8, true);
    _relu_s8_2.build(input0_s8, true);
    _relu_s16_1.build(input0_s16, true);
    _relu_s16_2.build(input0_s16, true);

    _relu_s8_1.call(input0_s8);
    _relu_s8_1.get_output().print({}, "relu_s8 inplace");
    _relu_s8_2.call(input1_s8);
    _relu_s8_2.get_output().print({}, "relu_s8 no inplace");
    _relu_s16_1.call(input0_s16);
    _relu_s16_1.get_output().print({}, "relu_s16 inplace");
    _relu_s16_2.call(input1_s16);
    _relu_s16_2.get_output().print({}, "relu_s16 no inplace");

    return true;
}

bool test_leaky_relu_layer()
{
    printf("\n leaky_relu\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s8.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s8.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s8.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s8.set_value({2, 3, 0, 3, 0, 16}, 27);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s16.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s16.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s16.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s16.set_value({2, 3, 0, 3, 0, 16}, 27);

    LeakyRelu<int8_t> _leaky_relu_s8_1(2, exponent0, "leaky_relu_s8 inplace", true);
    LeakyRelu<int8_t> _leaky_relu_s8_2(2, exponent0, "leaky_relu_s8 no inplace", false);
    LeakyRelu<int16_t> _leaky_relu_s16_1(2, exponent0, "leaky_relu_s16 inplace", true);
    LeakyRelu<int16_t> _leaky_relu_s16_2(2, exponent0, "leaky_relu_s16 no inplace", false);

    _leaky_relu_s8_1.build(input0_s8, true);
    _leaky_relu_s8_2.build(input0_s8, true);
    _leaky_relu_s16_1.build(input0_s16, true);
    _leaky_relu_s16_2.build(input0_s16, true);

    _leaky_relu_s8_1.call(input0_s8);
    _leaky_relu_s8_1.get_output().print({}, "leaky_relu_s8 inplace");
    _leaky_relu_s8_2.call(input1_s8);
    _leaky_relu_s8_2.get_output().print({}, "leaky_relu_s8 no inplace");
    _leaky_relu_s16_1.call(input0_s16);
    _leaky_relu_s16_1.get_output().print({}, "leaky_relu_s16 inplace");
    _leaky_relu_s16_2.call(input1_s16);
    _leaky_relu_s16_2.get_output().print({}, "leaky_relu_s16 no inplace");

    return true;
}

bool test_prelu_layer()
{
    printf("\n prelu\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s8.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s8.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s8.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s8.set_value({2, 3, 0, 3, 0, 16}, 27);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value({0, 1, 0, 3, 0, 16}, -9);
    input0_s16.set_value({1, 2, 0, 3, 0, 16}, 18);
    input0_s16.set_value({2, 3, 0, 3, 0, 16}, -27);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value({0, 1, 0, 3, 0, 16}, 9);
    input1_s16.set_value({1, 2, 0, 3, 0, 16}, -18);
    input1_s16.set_value({2, 3, 0, 3, 0, 16}, 27);
    int8_t alpha_s8[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    int16_t alpha_s16[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    PRelu<int8_t> _prelu_s8_1(alpha_s8, exponent0, "prelu_s8 inplace", true);
    PRelu<int8_t> _prelu_s8_2(alpha_s8, exponent0, "prelu_s8 no inplace", false);
    PRelu<int16_t> _prelu_s16_1(alpha_s16, exponent0, "prelu_s16 inplace", true);
    PRelu<int16_t> _prelu_s16_2(alpha_s16, exponent0, "prelu_s16 no inplace", false);

    _prelu_s8_1.build(input0_s8, true);
    _prelu_s8_2.build(input0_s8, true);
    _prelu_s16_1.build(input0_s16, true);
    _prelu_s16_2.build(input0_s16, true);

    _prelu_s8_1.call(input0_s8);
    _prelu_s8_1.get_output().print({}, "prelu_s8 inplace");
    _prelu_s8_2.call(input1_s8);
    _prelu_s8_2.get_output().print({}, "prelu_s8 no inplace");
    _prelu_s16_1.call(input0_s16);
    _prelu_s16_1.get_output().print({}, "prelu_s16 inplace");
    _prelu_s16_2.call(input1_s16);
    _prelu_s16_2.get_output().print({}, "prelu_s16 no inplace");

    return true;
}

bool test_concat_layer()
{
    printf("\n concat\n");
    Tensor<int8_t> input0_s8;
    input0_s8.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s8.set_value(1);
    Tensor<int8_t> input1_s8;
    input1_s8.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s8.set_value(2);
    Tensor<int16_t> input0_s16;
    input0_s16.set_exponent(exponent0).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input0_s16.set_value(1);
    Tensor<int16_t> input1_s16;
    input1_s16.set_exponent(exponent1).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    input1_s16.set_value(2);

    Concat<int8_t> _concat_s8_1(-1, "concat_s8 inplace");
    Concat<int8_t> _concat_s8_2(-2, "concat_s8 no inplace");
    Concat<int16_t> _concat_s16_1(0, "concat_s16 inplace");
    Concat<int16_t> _concat_s16_2(1, "concat_s16 no inplace");

    _concat_s8_1.build({&input0_s8, &input1_s8}, true);
    _concat_s8_2.build({&input0_s8, &input1_s8}, true);
    _concat_s16_1.build({&input0_s16, &input1_s16}, true);
    _concat_s16_2.build({&input0_s16, &input1_s16}, true);

    _concat_s8_1.call({&input0_s8, &input1_s8});
    _concat_s8_1.get_output().print({}, "concat_s8 inplace");
    _concat_s8_2.call({&input0_s8, &input1_s8});
    _concat_s8_2.get_output().print({}, "concat_s8 no inplace");
    _concat_s16_1.call({&input0_s16, &input1_s16});
    _concat_s16_1.get_output().print({}, "concat_s16 inplace");
    _concat_s16_2.call({&input0_s16, &input1_s16});
    _concat_s16_2.get_output().print({}, "concat_s16 no inplace");

    return true;
}

TEST_CASE("test layer", "[add]")
{
    TEST_ASSERT(test_add_layer());
    TEST_ASSERT(test_sub_layer());
    TEST_ASSERT(test_mul_layer());
    TEST_ASSERT(test_expand_dims_layer());
    TEST_ASSERT(test_flatten_layer());
    TEST_ASSERT(test_squeeze_layer());
    TEST_ASSERT(test_reshape_layer());
    TEST_ASSERT(test_transpose_layer());
    TEST_ASSERT(test_pad_layer());
    TEST_ASSERT(test_min_layer());
    TEST_ASSERT(test_max_layer());
    TEST_ASSERT(test_global_max_pool_layer());
    TEST_ASSERT(test_global_avg_pool_layer());
    TEST_ASSERT(test_relu_layer());
    TEST_ASSERT(test_leaky_relu_layer());
    TEST_ASSERT(test_prelu_layer());
    TEST_ASSERT(test_concat_layer());
    printf("\n\n");
}
