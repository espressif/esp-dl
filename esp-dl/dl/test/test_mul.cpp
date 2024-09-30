#include "test_data.hpp"

#include "dl_base.hpp"
#include "dl_constant.hpp"
#include "dl_define.hpp"
#include "dl_nn_mul2d.hpp"
#include "dl_tool.hpp"
#include "dl_variable.hpp"

#include "unity.h"
#include <limits.h>

using namespace dl;
using namespace nn;
using namespace tool;
using namespace base;
using namespace std;

template <typename feature_t>
void mul2d_c(Tensor<feature_t> &output,
             Tensor<feature_t> &input0,
             Tensor<feature_t> &input1,
             const Activation<feature_t> *const activation)
{
    int height = input0.shape[0]; // inputs and output are the same shape
    int width = input0.shape[1];
    int channel = input0.shape[2];

    feature_t *input0_element = input0.get_element_ptr();
    int input0_y_offset = input0.shape[1] * input0.shape[2];
    int input0_x_offset = input0.shape[2];

    feature_t *input1_element = input1.get_element_ptr();
    int input1_y_offset = input1.shape[1] * input1.shape[2];
    int input1_x_offset = input1.shape[2];

    feature_t *output_element = output.get_element_ptr(); // output
    int output_y_offset = output.shape[1] * output.shape[2];
    int output_x_offset = output.shape[2];

    int buffer = 0;
    int mul_shift = output.exponent - input0.exponent - input1.exponent;

    activation_type_t activation_type = activation ? activation->type : Linear;
    feature_t activation_alpha;
    int activation_shift;
    const feature_t *activation_alpha_ptr;

    switch (activation_type) {
    case ReLU:
        activation_alpha = 0;
        activation_shift = 0;
        activation_alpha_ptr = NULL;
        break;
    case LeakyReLU:
        activation_alpha = activation->element[0];
        activation_shift = -activation->exponent;
        activation_alpha_ptr = NULL;
        break;
    case PReLU:
        activation_alpha = 0;
        activation_alpha_ptr = activation->element;
        activation_shift = -activation->exponent;
        break;
    default:
        activation_alpha = 0;
        activation_alpha_ptr = NULL;
        activation_shift = -1;
        break;
    }

    for (size_t output_y = 0; output_y < height; output_y++) {
        feature_t *input0_11c = input0_element;
        feature_t *input1_11c = input1_element;
        feature_t *output_11c = output_element;

        for (size_t output_x = 0; output_x < width; output_x++) {
            for (size_t output_c = 0; output_c < channel; output_c++) {
                buffer = (int)input0_11c[output_c] * (int)input1_11c[output_c];
                buffer = DL_RIGHT_SHIFT(buffer, mul_shift);
                tool::truncate(output_11c[output_c], buffer);
                if (activation_type == ReLU) {
                    output_11c[output_c] = DL_MAX(0, output_11c[output_c]);
                } else if (activation_type == LeakyReLU) {
                    if (output_11c[output_c] < 0) {
                        buffer = DL_RIGHT_SHIFT((output_11c[output_c] * activation_alpha), activation_shift);
                        tool::truncate(output_11c[output_c], buffer);
                    }
                } else if (activation_type == PReLU) {
                    if (output_11c[output_c] < 0) {
                        buffer =
                            DL_RIGHT_SHIFT((output_11c[output_c] * activation_alpha_ptr[output_c]), activation_shift);
                        tool::truncate(output_11c[output_c], buffer);
                    }
                }
            }
            input0_11c += input0_x_offset;
            input1_11c += input1_x_offset;
            output_11c += output_x_offset;
        }

        input0_element += input0_y_offset;
        input1_element += input1_y_offset;
        output_element += output_y_offset;
    }
}

template <typename feature_t>
Tensor<feature_t> mul2d_c(const int output_exponent,
                          Tensor<feature_t> &input0,
                          Tensor<feature_t> &input1,
                          const Activation<feature_t> *activation,
                          const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
{
    assert(input0.is_same_shape(input1));

    Tensor<feature_t> output;
    output.set_exponent(output_exponent).set_shape(input0.shape).malloc_element();
    mul2d_c(output, input0, input1, activation);

    return output;
}

bool test_mul_s8(int exponent0,
                 int exponent1,
                 int exponent_out,
                 int offset0,
                 int offset1,
                 int height,
                 int width,
                 int channel,
                 int activation_type,
                 bool inplace)
{
    if (!inplace) {
        Tensor<int8_t> input0;
        input0.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent0)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int8_t> input1;
        input1.set_element((int8_t *)&input1_element[offset1])
            .set_exponent(exponent1)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        if (activation_type == 0) {
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, (Activation<int8_t> *)NULL);
            latency.start();
            Tensor<int8_t> output = mul2d<false, int8_t>(
                exponent_out, input0, input1, (Activation<int8_t> *)NULL); //(Activation<int8_t> *)NULL
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 1) { // relu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_relu);
            latency.start();
            Tensor<int8_t> output = mul2d<false, int8_t>(exponent_out, input0, input1, &layer_activation_relu);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 2) { // leakyrelu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_lrelu);
            latency.start();
            Tensor<int8_t> output = mul2d<false, int8_t>(exponent_out, input0, input1, &layer_activation_lrelu);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 3) { // prelu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_prelu);
            latency.start();
            Tensor<int8_t> output = mul2d<false, int8_t>(exponent_out, input0, input1, &layer_activation_prelu);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        }
        return false;
    } else {
        Tensor<int8_t> input0_tmp;
        input0_tmp.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent0)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int8_t> input1_tmp;
        input1_tmp.set_element((int8_t *)&input1_element[offset1])
            .set_exponent(exponent1)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int8_t> input0(input0_tmp, true);
        input0.set_auto_free(true);
        Tensor<int8_t> input1(input1_tmp, true);
        input1.set_auto_free(true);

        // output;
        if (activation_type == 0) {
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, (Activation<int8_t> *)NULL);
            latency.start();
            mul2d<true, int8_t>(exponent_out, input0, input1, (Activation<int8_t> *)NULL); //(Activation<int8_t> *)NULL
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 1) { // relu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_relu);
            latency.start();
            mul2d<true, int8_t>(exponent_out, input0, input1, &layer_activation_relu);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 2) { // leakyrelu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_lrelu);
            latency.start();
            mul2d<true, int8_t>(exponent_out, input0, input1, &layer_activation_lrelu);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 3) { // prelu
            Tensor<int8_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_prelu);
            latency.start();
            mul2d<true, int8_t>(exponent_out, input0, input1, &layer_activation_prelu);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        }
        return false;
    }
}

bool test_mul_s16(int exponent0,
                  int exponent1,
                  int exponent_out,
                  int offset0,
                  int offset1,
                  int height,
                  int width,
                  int channel,
                  int activation_type,
                  bool inplace)
{
    if (!inplace) {
        Tensor<int16_t> input0;
        input0.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent0)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int16_t> input1;
        input1.set_element((int16_t *)&input1_element_s16[offset1])
            .set_exponent(exponent1)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        if (activation_type == 0) {
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, (Activation<int16_t> *)NULL);
            latency.start();
            Tensor<int16_t> output = mul2d<false, int16_t>(
                exponent_out, input0, input1, (Activation<int16_t> *)NULL); //(Activation<int8_t> *)NULL
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 1) { // relu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_relu_s16);
            latency.start();
            Tensor<int16_t> output = mul2d<false, int16_t>(exponent_out, input0, input1, &layer_activation_relu_s16);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 2) { // leakyrelu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_lrelu_s16);
            latency.start();
            Tensor<int16_t> output = mul2d<false, int16_t>(exponent_out, input0, input1, &layer_activation_lrelu_s16);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 3) { // prelu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_prelu_s16);
            latency.start();
            Tensor<int16_t> output = mul2d<false, int16_t>(exponent_out, input0, input1, &layer_activation_prelu_s16);
            latency.end();
            latency.print();

            return output.check_element(output_c.get_element_ptr(), 2, false);
        }
        return false;
    } else {
        Tensor<int16_t> input0_tmp;
        input0_tmp.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent0)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int16_t> input1_tmp;
        input1_tmp.set_element((int16_t *)&input1_element_s16[offset1])
            .set_exponent(exponent1)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int16_t> input0(input0_tmp, true);
        input0.set_auto_free(true);
        Tensor<int16_t> input1(input1_tmp, true);
        input1.set_auto_free(true);

        // output;
        if (activation_type == 0) {
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, (Activation<int16_t> *)NULL);
            latency.start();
            mul2d<true, int16_t>(
                exponent_out, input0, input1, (Activation<int16_t> *)NULL); //(Activation<int8_t> *)NULL
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 1) { // relu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_relu_s16);
            latency.start();
            mul2d<true, int16_t>(exponent_out, input0, input1, &layer_activation_relu_s16);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 2) { // leakyrelu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_lrelu_s16);
            latency.start();
            mul2d<true, int16_t>(exponent_out, input0, input1, &layer_activation_lrelu_s16);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        } else if (activation_type == 3) { // prelu
            Tensor<int16_t> output_c = mul2d_c(exponent_out, input0, input1, &layer_activation_prelu_s16);
            latency.start();
            mul2d<true, int16_t>(exponent_out, input0, input1, &layer_activation_prelu_s16);
            latency.end();
            latency.print();

            return input0.check_element(output_c.get_element_ptr(), 2, false);
        }
        return false;
    }
}

//---------------------------------------------------------------no-inplace------------------------------------------------------------------------------------
TEST_CASE("test no inplace start", "[mul]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

// s8, c = 6
TEST_CASE("test_mul2d, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 0, false));
}

TEST_CASE("test_mul2d_relu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 3, false));
}

// s8, c = 16
TEST_CASE("test_mul2d, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 0, false));
}

TEST_CASE("test_mul2d_relu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 3, false));
}

// s8, c = 35
TEST_CASE("test_mul2d, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 0, false));
}

TEST_CASE("test_mul2d_relu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 3, false));
}

// s16, c = 6
TEST_CASE("test_mul2d, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 0, false));
}

TEST_CASE("test_mul2d_relu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 3, false));
}

// s16, c = 16
TEST_CASE("test_mul2d, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 0, false));
}

TEST_CASE("test_mul2d_relu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 3, false));
}

// s16, c = 35
TEST_CASE("test_mul2d, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 0, false));
}

TEST_CASE("test_mul2d_relu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 1, false));
}

TEST_CASE("test_mul2d_leakyrelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 2, false));
}

TEST_CASE("test_mul2d_prelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 3, false));
}

//---------------------------------------------------------------inplace------------------------------------------------------------------------------------
TEST_CASE("test inplace start", "[min]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

// s8, c = 6
TEST_CASE("test_mul2d, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 0, true));
}

TEST_CASE("test_mul2d_relu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 6, 3, true));
}

// s8, c = 16
TEST_CASE("test_mul2d, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 0, true));
}

TEST_CASE("test_mul2d_relu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 16, 3, true));
}

// s8, c = 35
TEST_CASE("test_mul2d, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 0, true));
}

TEST_CASE("test_mul2d_relu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s8(-8, -8, -8, 4, 7, 5, 7, 35, 3, true));
}

// s16, c = 6
TEST_CASE("test_mul2d, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 0, true));
}

TEST_CASE("test_mul2d_relu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=6", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 6, 3, true));
}

// s16, c = 16
TEST_CASE("test_mul2d, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 0, true));
}

TEST_CASE("test_mul2d_relu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=16", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 16, 3, true));
}

// s16, c = 35
TEST_CASE("test_mul2d, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 0, true));
}

TEST_CASE("test_mul2d_relu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 1, true));
}

TEST_CASE("test_mul2d_leakyrelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 2, true));
}

TEST_CASE("test_mul2d_prelu, c=35", "[mul]")
{
    TEST_ASSERT(test_mul_s16(-8, -8, -8, 4, 7, 5, 7, 35, 3, true));
}
