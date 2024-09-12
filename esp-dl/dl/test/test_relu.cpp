#include "test_data.hpp"

#include "dl_base.hpp"
#include "dl_constant.hpp"
#include "dl_define.hpp"
#include "dl_nn_relu.hpp"
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
void relu_c(Tensor<feature_t> &output, Tensor<feature_t> &input)
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

    for (size_t output_y = 0; output_y < height; output_y++) {
        feature_t *input_11c = input_element;
        feature_t *output_11c = output_element;

        for (size_t output_x = 0; output_x < width; output_x++) {
            for (size_t output_c = 0; output_c < channel; output_c++) {
                output_11c[output_c] = DL_MAX(0, input_11c[output_c]);
            }
            input_11c += input_x_offset;
            output_11c += output_x_offset;
        }

        input_element += input_y_offset;
        output_element += output_y_offset;
    }
}

template <typename feature_t>
Tensor<feature_t> relu_c(Tensor<feature_t> &input)
{
    Tensor<feature_t> output;
    output.set_exponent(input.exponent).set_shape(input.shape).malloc_element();
    relu_c(output, input);
    return output;
}

bool test_relu_s8(int exponent, int offset0, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int8_t> input0;
        input0.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int8_t> output_c = relu_c(input0);
        latency.start();
        Tensor<int8_t> output = relu<false, int8_t>(input0);
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
        Tensor<int8_t> output_c = relu_c(input0);
        latency.start();
        relu<true, int8_t>(input0);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

bool test_relu_s16(int exponent, int offset0, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int16_t> input0;
        input0.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int16_t> output_c = relu_c(input0);
        latency.start();
        Tensor<int16_t> output = relu<false, int16_t>(input0);
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
        Tensor<int16_t> output_c = relu_c(input0);
        latency.start();
        relu<true, int16_t>(input0);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

//---------------------------------------------------------------no-inplace------------------------------------------------------------------------------------
TEST_CASE("test no inplace start", "[relu]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

TEST_CASE("test_relu_s8, c=6", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 6, false));
}

TEST_CASE("test_relu_s8, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 16, false));
}

TEST_CASE("test_relu_s8, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 5, 5, 7, 16, false));
}

TEST_CASE("test_relu_s8, c=35", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 35, false));
}

TEST_CASE("test_relu_s16, c=6", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 6, false));
}

TEST_CASE("test_relu_s16, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 16, false));
}

TEST_CASE("test_relu_s16, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 5, 5, 7, 16, false));
}

TEST_CASE("test_relu_s16, c=35", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 35, false));
}

//---------------------------------------------------------------inplace------------------------------------------------------------------------------------
TEST_CASE("test inplace start", "[relu]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

TEST_CASE("test_relu_s8, c=6", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 6, true));
}

TEST_CASE("test_relu_s8, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 16, true));
}

TEST_CASE("test_relu_s8, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 5, 5, 7, 16, true));
}

TEST_CASE("test_relu_s8, c=35", "[relu]")
{
    TEST_ASSERT(test_relu_s8(-8, 0, 5, 7, 35, true));
}

TEST_CASE("test_relu_s16, c=6", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 6, true));
}

TEST_CASE("test_relu_s16, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 16, true));
}

TEST_CASE("test_relu_s16, c=16", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 5, 5, 7, 16, true));
}

TEST_CASE("test_relu_s16, c=35", "[relu]")
{
    TEST_ASSERT(test_relu_s16(-8, 0, 5, 7, 35, true));
}
