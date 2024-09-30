#include "dl_layer_tanh.hpp"

#include <math.h>
#include <memory>

#include "unity.h"

#include "test_tool.hpp"

template <typename I, typename O, int type>
std::unique_ptr<O[]> tanh_float(const int output_exp, I *input_ptr, const int input_exp, const int size)
{
    std::unique_ptr<O[]> ref(new O[size]);

    float scale = DL_SCALE(input_exp + 1);
    float rescale = DL_RESCALE(output_exp);

    for (size_t i = 0; i < size; i++) {
        float temp = exp((float)input_ptr[i] * scale);
        temp = (temp - 1.0f) / (temp + 1.0f);

        if constexpr (type == QIQO)
            dl::tool::truncate(ref[i], temp * rescale);
        else if constexpr (type == QIFO)
            ref[i] = temp;
    }

    return ref;
}

template <typename I, typename O, int type, bool inplace>
bool testcase()
{
    dl::tool::Latency latency;

    int input_exponent = -14;
    int output_exponent = -16;

    int height = 5;
    int width = 6;
    int channel = 7;

    dl::Tensor<I> input;
    input.set_exponent(input_exponent).set_shape({height, width, channel}).set_auto_free(true).malloc_element();
    random_array(input.element, input.get_size());

    latency.start();
    std::unique_ptr<O[]> ref = tanh_float<I, O, type>(output_exponent, input.element, input.exponent, input.get_size());
    latency.end();
    latency.print("float");

    dl::layer::TanH<I, O, type, inplace> tanh(output_exponent);
    tanh.build(input);
    latency.start();
    tanh.call(input);
    latency.end();
    latency.print("quant");

    return tanh.get_output().check_element(ref.get(), 2, false, 100);
}

TEST_CASE("TanH", "[dl::layer::TanH]")
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
