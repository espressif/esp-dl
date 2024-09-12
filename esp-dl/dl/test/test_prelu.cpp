#include "test_data.hpp"

#include "dl_base.hpp"
#include "dl_constant.hpp"
#include "dl_define.hpp"
#include "dl_nn_prelu.hpp"
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
void prelu_c(Tensor<feature_t> &output,
             Tensor<feature_t> &input,
             const feature_t *activation_element,
             const int activation_exponent)
{
    int height = input.shape[0]; // inputs and output are the same shape
    int width = input.shape[1];
    int channel = input.shape[2];

    feature_t *input_element = input.get_element_ptr();
    int input_y_offset = input.shape[1] * input.shape[2];
    int input_x_offset = input.shape[2];

    feature_t *output_element = output.get_element_ptr(); // output
    int output_y_offset = output.shape[1] * output.shape[2];
    int output_x_offset = output.shape[2];

    int buffer = 0;

    int activation_shift = -activation_exponent;
    const feature_t *activation_alpha_ptr = activation_element;

    for (size_t output_y = 0; output_y < height; output_y++) {
        feature_t *input_11c = input_element;
        feature_t *output_11c = output_element;

        for (size_t output_x = 0; output_x < width; output_x++) {
            for (size_t output_c = 0; output_c < channel; output_c++) {
                output_11c[output_c] = input_11c[output_c];
                if (output_11c[output_c] < 0) {
                    buffer = DL_RIGHT_SHIFT(output_11c[output_c] * activation_alpha_ptr[output_c], activation_shift);
                    tool::truncate(output_11c[output_c], buffer);
                }
            }
            input_11c += input_x_offset;
            output_11c += output_x_offset;
        }

        input_element += input_y_offset;
        output_element += output_y_offset;
    }
}

template <typename feature_t>
Tensor<feature_t> prelu_c(Tensor<feature_t> &input, const feature_t *activation_element, const int activation_exponent)
{
    Tensor<feature_t> output;
    output.set_exponent(input.exponent).set_shape(input.shape).malloc_element();
    prelu_c(output, input, activation_element, activation_exponent);
    return output;
}

bool test_prelu_s8(int exponent, int offset0, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int8_t> input0;
        input0.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int8_t> output_c = prelu_c(input0, layer_activation_prelu_element, exponent);
        latency.start();
        Tensor<int8_t> output = prelu<false, int8_t>(input0, layer_activation_prelu_element, exponent);
        latency.end();
        latency.print();

        return output.check_element(output_c.get_element_ptr(), 2, false);
    } else {
        Tensor<int8_t> input0_tmp;
        input0_tmp.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int8_t> input0(input0_tmp, true);
        input0.set_auto_free(true);

        // output;
        Tensor<int8_t> output_c = prelu_c(input0, layer_activation_prelu_element, exponent);
        latency.start();
        prelu<true, int8_t>(input0, layer_activation_prelu_element, exponent);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

bool test_prelu_s16(int exponent, int offset0, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int16_t> input0;
        input0.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int16_t> output_c = prelu_c(input0, layer_activation_prelu_element_s16, exponent);
        latency.start();
        Tensor<int16_t> output = prelu<false, int16_t>(input0, layer_activation_prelu_element_s16, exponent);
        latency.end();
        latency.print();

        return output.check_element(output_c.get_element_ptr(), 2, false);
    } else {
        Tensor<int16_t> input0_tmp;
        input0_tmp.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int16_t> input0(input0_tmp, true);
        input0.set_auto_free(true);

        // output;
        Tensor<int16_t> output_c = prelu_c(input0, layer_activation_prelu_element_s16, exponent);
        latency.start();
        prelu<true, int16_t>(input0, layer_activation_prelu_element_s16, exponent);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

//---------------------------------------------------------------no-inplace------------------------------------------------------------------------------------
TEST_CASE("test no inplace start", "[prelu]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}
// s8

TEST_CASE("test_prelu_s8, c=6", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 6, false));
}

TEST_CASE("test_prelu_s8, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 16, false));
}

TEST_CASE("test_prelu_s8, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 5, 5, 7, 16, false));
}

TEST_CASE("test_prelu_s8, c=35", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 35, false));
}

// s16

TEST_CASE("test_prelu_s16, c=6", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 6, false));
}

TEST_CASE("test_prelu_s16, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 16, false));
}

TEST_CASE("test_prelu_s16, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 5, 5, 7, 16, false));
}

TEST_CASE("test_prelu_s16, c=35", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 35, false));
}

//---------------------------------------------------------------inplace------------------------------------------------------------------------------------
TEST_CASE("test inplace start", "[prelu]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

// s8

TEST_CASE("test_prelu_s8, c=6", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 6, true));
}

TEST_CASE("test_prelu_s8, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 16, true));
}

TEST_CASE("test_prelu_s8, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 5, 5, 7, 16, true));
}

TEST_CASE("test_prelu_s8, c=35", "[prelu]")
{
    TEST_ASSERT(test_prelu_s8(-8, 0, 5, 7, 35, true));
}

// s16

TEST_CASE("test_prelu_s16, c=6", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 6, true));
}

TEST_CASE("test_prelu_s16, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 16, true));
}

TEST_CASE("test_prelu_s16, c=16", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 5, 5, 7, 16, true));
}

TEST_CASE("test_prelu_s16, c=35", "[prelu]")
{
    TEST_ASSERT(test_prelu_s16(-16, 0, 5, 7, 35, true));
}
