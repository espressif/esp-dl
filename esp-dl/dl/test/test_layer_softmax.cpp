#include "dl_layer_softmax.hpp"

#include <math.h>
#include <memory>

#include "unity.h"

#include "test_tool.hpp"

template <typename I, typename O, int type>
std::unique_ptr<O[]> softmax_float(
    const int output_exp, I *input_ptr, const int input_exp, const uint32_t loop, const uint32_t channel)
{
    std::unique_ptr<O[]> ref(new O[loop * channel]);
    std::unique_ptr<float[]> buf(new float[channel]);

    float scale = (input_exp > 0) ? (1 << input_exp) : ((float)1.0 / (1 << -input_exp));
    float rescale = (output_exp > 0) ? ((float)1.0 / (1 << output_exp)) : (1 << -output_exp);

    for (size_t i = 0; i < loop; i++) {
        I max_input = input_ptr[0];
        for (size_t j = 1; j < channel; j++) max_input = DL_MAX(max_input, input_ptr[j]);

        float summary = 0.0;
        for (size_t j = 0; j < channel; j++) {
            buf[j] = exp(((float)input_ptr[j] - max_input) * scale);
            summary += buf[j];
        }

        if constexpr (type == QIQO) {
            summary = rescale / summary;
            for (size_t j = 0; j < channel; j++) dl::tool::truncate(ref[i * channel + j], buf[j] * summary);
        } else if constexpr (type == QIFO) {
            summary = 1.0 / summary;
            for (size_t j = 0; j < channel; j++) ref[i * channel + j] = buf[j] * summary;
        }

        input_ptr += channel;
    }

    return ref;
}

template <typename I, typename O, int type, bool inplace>
bool testcase()
{
    // dl::tool::Latency latency;

    int input_exponent = -16;
    int output_exponent = -15;

    int height = 5;
    int width = 1;
    int channel = 7;

    dl::Tensor<I> input;
    input.set_exponent(input_exponent).set_shape({height, width, channel}).malloc_element();
    random_array(input.element, input.get_size());

    // latency.start();
    std::unique_ptr<O[]> ref =
        softmax_float<I, O, type>(output_exponent, input.element, input.exponent, height * width, channel);
    // latency.end();
    // latency.print("float");

    dl::layer::Softmax<I, O, type, inplace> softmax(output_exponent);
    softmax.build(input);
    // latency.start();
    softmax.call(input);
    // latency.end();
    // latency.print("quant");

    return softmax.get_output().check_element(ref.get(), 5, false, INT32_MAX);
}

TEST_CASE("Softmax", "[dl::layer::Softmax]")
{
    bool ans = false;

    ans = testcase<int8_t, int8_t, QIQO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int8_t, int8_t, QIQO, true>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, int16_t, QIQO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, int16_t, QIQO, true>();
    TEST_ASSERT(ans);

    ans = testcase<int8_t, float, QIFO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int8_t, float, QIFO, true>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, float, QIFO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, float, QIFO, true>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, int8_t, QIQO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int16_t, int8_t, QIQO, true>();
    TEST_ASSERT(ans);

    ans = testcase<int8_t, int16_t, QIQO, false>();
    TEST_ASSERT(ans);

    ans = testcase<int8_t, int16_t, QIQO, true>();
    TEST_ASSERT(ans);
}
