#include "test_data.hpp"

#include "dl_base.hpp"
#include "dl_constant.hpp"
#include "dl_define.hpp"
#include "dl_nn_min2d.hpp"
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
void min2d_c(Tensor<feature_t> &output, Tensor<feature_t> &input0, Tensor<feature_t> &input1)
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

    for (size_t output_y = 0; output_y < height; output_y++) {
        feature_t *input0_11c = input0_element;
        feature_t *input1_11c = input1_element;
        feature_t *output_11c = output_element;

        for (size_t output_x = 0; output_x < width; output_x++) {
            for (size_t output_c = 0; output_c < channel; output_c++) {
                output_11c[output_c] = DL_MIN(input0_11c[output_c], input1_11c[output_c]);
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
Tensor<feature_t> min2d_c(Tensor<feature_t> &input0, Tensor<feature_t> &input1)
{
    assert(input0.is_same_shape(input1));
    assert(input0.exponent == input1.exponent);

    Tensor<feature_t> output;
    output.set_exponent(input0.exponent).set_shape(input0.shape).malloc_element();
    min2d_c(output, input0, input1);

    return output;
}

bool test_min_s8(int exponent, int offset0, int offset1, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int8_t> input0;
        input0.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int8_t> input1;
        input1.set_element((int8_t *)&input1_element[offset1])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int8_t> output_c = min2d_c(input0, input1);
        latency.start();
        Tensor<int8_t> output = min2d<false, int8_t>(input0, input1);
        latency.end();
        latency.print();

        return output.check_element(output_c.get_element_ptr(), 2, false);
    } else {
        Tensor<int8_t> input0_tmp;
        input0_tmp.set_element((int8_t *)&input0_element[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int8_t> input1_tmp;
        input1_tmp.set_element((int8_t *)&input1_element[offset1])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int8_t> input0(input0_tmp, true);
        input0.set_auto_free(true);
        Tensor<int8_t> input1(input0_tmp, true);
        input1.set_auto_free(true);

        // output;
        Tensor<int8_t> output_c = min2d_c(input0, input1);
        latency.start();
        min2d<true, int8_t>(input0, input1);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

bool test_min_s16(int exponent, int offset0, int offset1, int height, int width, int channel, bool inplace)
{
    if (!inplace) {
        Tensor<int16_t> input0;
        input0.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int16_t> input1;
        input1.set_element((int16_t *)&input1_element_s16[offset1])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        // output;
        Tensor<int16_t> output_c = min2d_c(input0, input1);
        latency.start();
        Tensor<int16_t> output = min2d<false, int16_t>(input0, input1);
        latency.end();
        latency.print();

        return output.check_element(output_c.get_element_ptr(), 2, false);
    } else {
        Tensor<int16_t> input0_tmp;
        input0_tmp.set_element((int16_t *)&input0_element_s16[offset0])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);

        Tensor<int16_t> input1_tmp;
        input1_tmp.set_element((int16_t *)&input1_element_s16[offset1])
            .set_exponent(exponent)
            .set_shape({height, width, channel})
            .set_auto_free(false);
        Latency latency;

        Tensor<int16_t> input0(input0_tmp, true);
        input0.set_auto_free(true);
        Tensor<int16_t> input1(input1_tmp, true);
        input1.set_auto_free(true);

        // output;
        Tensor<int16_t> output_c = min2d_c(input0, input1);
        latency.start();
        min2d<true, int16_t>(input0, input1);
        latency.end();
        latency.print();

        return input0.check_element(output_c.get_element_ptr(), 2, false);
    }
}

//---------------------------------------------------------------no-inplace------------------------------------------------------------------------------------
TEST_CASE("test no inplace start", "[min]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

TEST_CASE("test_min2d_s8, c=6", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 6, false));
}

TEST_CASE("test_min2d_s8, c=16", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 16, false));
}

TEST_CASE("test_min2d_s8, c=16", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 4, 7, 5, 7, 16, false));
}

TEST_CASE("test_min2d_s8, c=35", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 35, false));
}

// s16
TEST_CASE("test_min2d_s16, c=6", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 6, false));
}

TEST_CASE("test_min2d_s16, c=16", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 16, false));
}

TEST_CASE("test_min2d_s16, c=16", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 4, 7, 5, 7, 16, false));
}

TEST_CASE("test_min2d_s16, c=35", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 35, false));
}

//---------------------------------------------------------------inplace------------------------------------------------------------------------------------
TEST_CASE("test inplace start", "[min]")
{
    TEST_ASSERT(true);
    printf("\n\n");
}

TEST_CASE("test_min2d_s8, c=6", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 6, true));
}

TEST_CASE("test_min2d_s8, c=16", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 16, true));
}

TEST_CASE("test_min2d_s8, c=16", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 4, 7, 5, 7, 16, true));
}

TEST_CASE("test_min2d_s8, c=35", "[min]")
{
    TEST_ASSERT(test_min_s8(-8, 0, 0, 5, 7, 35, true));
}

// s16
TEST_CASE("test_min2d_s16, c=6", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 6, true));
}

TEST_CASE("test_min2d_s16, c=16", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 16, true));
}

TEST_CASE("test_min2d_s16, c=16", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 4, 7, 5, 7, 16, true));
}

TEST_CASE("test_min2d_s16, c=35", "[min]")
{
    TEST_ASSERT(test_min_s16(-8, 0, 0, 5, 7, 35, true));
}
