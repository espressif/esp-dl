#include "dl_base_conv2d.hpp"
#include "dl_base_isa.hpp"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "unity.h"
#include <algorithm>
#include <cstring>
#include <vector>

#if CONFIG_IDF_TARGET_ESP32P4 && CONFIG_PIE_V2_BOOST
namespace {
using Args = dl::base::ArgsType<int16_t>;
using Kernel = void (*)(int16_t *, int16_t *, void *);

// The pre-tiling spatial shell with the unchanged P4 kernels is the oracle.
// This deliberately never calls conv2d(), whose dispatcher is under test.
void reference_conv(Args &args)
{
    const bool aligned = args.input_channel % 8 == 0 && args.output_channel % 8 == 0 &&
        !(reinterpret_cast<uintptr_t>(args.input_element) & 15) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 15);
    const Kernel kernels[2][2][4] = {{{dl_esp32p4_s16_unaligned_conv2d_11cn,
                                       dl_esp32p4_s16_unaligned_conv2d_11cn_relu,
                                       dl_esp32p4_s16_unaligned_conv2d_11cn_bias,
                                       dl_esp32p4_s16_unaligned_conv2d_11cn_bias_relu},
                                      {dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn,
                                       dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_relu,
                                       dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_bias,
                                       dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_bias_relu}},
                                     {{dl_esp32p4_s16_conv2d_11cn,
                                       dl_esp32p4_s16_conv2d_11cn_relu,
                                       dl_esp32p4_s16_conv2d_11cn_bias,
                                       dl_esp32p4_s16_conv2d_11cn_bias_relu},
                                      {dl_esp32p4_s16_conv2d_per_channel_11cn,
                                       dl_esp32p4_s16_conv2d_per_channel_11cn_relu,
                                       dl_esp32p4_s16_conv2d_per_channel_11cn_bias,
                                       dl_esp32p4_s16_conv2d_per_channel_11cn_bias_relu}}};
    const int index = (args.bias_element ? 2 : 0) + (args.activation_type == dl::ReLU ? 1 : 0);
    dl::base::ImplFunc_t<int16_t, int16_t> kernel = kernels[aligned][args.mac_shift == INT_MIN][index];
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    dl::base::conv_operation_shell<int16_t, int64_t>(args, kernel, kernel, nullptr, nullptr, nullptr);
}

int16_t *allocate(size_t elements)
{
    auto *result = static_cast<int16_t *>(heap_caps_aligned_alloc(16, elements * sizeof(int16_t), MALLOC_CAP_SPIRAM));
    TEST_ASSERT_NOT_NULL(result);
    return result;
}

uint32_t next_random(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}
} // namespace

TEST_CASE("P4 S16 pointwise tiling matches untiled convolution", "[pointwise]")
{
    constexpr size_t capacity = 65536;
    int16_t *input = allocate(capacity), *output = allocate(capacity), *reference = allocate(capacity);
    int16_t *weights = allocate(512 * 136 + 32);
    uint32_t random = 42;
    int cases = 0;
    auto check = [&](int h,
                     int w,
                     int ci,
                     int co,
                     int offset,
                     bool per_channel,
                     bool bias_on,
                     bool relu,
                     int pattern,
                     int stride,
                     dl::runtime_mode_t mode) {
        std::fill(output, output + capacity, 0x4567);
        std::fill(reference, reference + capacity, 0x4567);
        for (int i = 0; i < h * w * ci + 32; ++i) {
            input[i] = pattern ? (i % 3 ? INT16_MIN : INT16_MAX) : static_cast<int16_t>(next_random(random) >> 16);
        }
        for (int i = 0; i < ci * co + 32; ++i) {
            weights[i] = pattern ? (i % 5 ? 0 : INT16_MAX) : static_cast<int16_t>((next_random(random) >> 20) - 2048);
        }
        const int oh = (h - 1) / stride + 1, ow = (w - 1) / stride + 1;
        dl::TensorBase in({1, h, w, ci}, input + offset, 0, dl::DATA_TYPE_INT16, false);
        dl::TensorBase out({1, oh, ow, co}, output + 16 + offset, 12, dl::DATA_TYPE_INT16, false);
        dl::TensorBase ref({1, oh, ow, co}, reference + 16 + offset, 12, dl::DATA_TYPE_INT16, false);
        std::vector<int> exponents(co);
        for (int i = 0; i < co; ++i) exponents[i] = i % 3;
        auto *filter = per_channel ? new dl::TensorBase({1, 1, ci, co}, weights, exponents, dl::DATA_TYPE_INT16, false)
                                   : new dl::TensorBase({1, 1, ci, co}, weights, 0, dl::DATA_TYPE_INT16, false);
        dl::TensorBase bias({co}, nullptr, 0, dl::DATA_TYPE_INT64);
        auto *bias_data = static_cast<int64_t *>(bias.get_element_ptr());
        for (int i = 0; i < co; ++i) bias_data[i] = static_cast<int32_t>(next_random(random)) / 8;
        bias.reset_bias_layout(dl::QUANT_TYPE_SYMM_16BIT, false);
        std::vector<int> pads{0, 0, 0, 0};
        dl::base::ConvOpArgs<int16_t> expected(&ref,
                                               &in,
                                               pads,
                                               filter,
                                               {stride, stride},
                                               {1, 1},
                                               1,
                                               bias_on ? &bias : nullptr,
                                               relu ? dl::ReLU : dl::Linear,
                                               nullptr,
                                               mode);
        dl::base::ConvOpArgs<int16_t> actual(&out,
                                             &in,
                                             pads,
                                             filter,
                                             {stride, stride},
                                             {1, 1},
                                             1,
                                             bias_on ? &bias : nullptr,
                                             relu ? dl::ReLU : dl::Linear,
                                             nullptr,
                                             mode);
        for (size_t i = 0; i < expected.size(); ++i) reference_conv(expected.get_args(i));
        for (size_t i = 0; i < actual.size(); ++i) dl::base::conv2d<int16_t, int32_t, int64_t>(&actual.get_args(i));
        // Includes sentinels and untouched parts of the destination allocation.
        TEST_ASSERT_EQUAL_INT16_ARRAY(reference, output, capacity);
        ++cases;
        delete filter;
    };
    const int shapes[][2] = {{1, 1}, {1, 2}, {1, 31}, {1, 32}, {1, 33}, {5, 7}, {17, 7}};
    for (const auto &shape : shapes)
        for (int ci : {8, 24, 64, 128, 256, 512})
            for (int co : {8, 24, 64, 128, 136})
                for (int offset : {0, 1})
                    for (bool per_channel : {false, true})
                        for (int flags = 0; flags < 4; ++flags)
                            for (int pattern = 0; pattern < 2; ++pattern) {
                                check(shape[0],
                                      shape[1],
                                      ci,
                                      co,
                                      offset,
                                      per_channel,
                                      flags & 1,
                                      flags & 2,
                                      pattern,
                                      1,
                                      dl::RUNTIME_MODE_SINGLE_CORE);
                            }
    for (int stride : {1, 2})
        for (int ci : {128, 256, 512})
            for (bool per_channel : {false, true})
                for (int flags = 0; flags < 4; ++flags) {
                    // Exercise the descriptors used by the dual-core module scheduler.
                    check(17, 7, ci, 136, 0, per_channel, flags & 1, flags & 2, 0, stride, dl::RUNTIME_MODE_MULTI_CORE);
                }
    heap_caps_free(input);
    heap_caps_free(output);
    heap_caps_free(reference);
    heap_caps_free(weights);
    TEST_ASSERT_TRUE(heap_caps_check_integrity_all(true));
    printf("pointwise: %d exact comparisons, including output sentinels\n", cases);
}

TEST_CASE("P4 S16 pointwise tiling benchmark", "[pointwise_perf]")
{
    const int shapes[][4] = {{80, 80, 64, 64},
                             {40, 40, 128, 128},
                             {40, 40, 128, 256},
                             {20, 20, 256, 256},
                             {20, 20, 512, 256},
                             {20, 20, 512, 128},
                             {1, 49, 256, 136}};
    for (const auto &shape : shapes) {
        const int h = shape[0], w = shape[1], ci = shape[2], co = shape[3];
        auto *input = allocate(h * w * ci + 32), *weights = allocate(ci * co + 32);
        auto *output = allocate(h * w * co + 32), *reference = allocate(h * w * co + 32);
        uint32_t random = 42;
        for (int i = 0; i < h * w * ci + 32; ++i) input[i] = next_random(random) >> 20;
        for (int i = 0; i < ci * co + 32; ++i) weights[i] = static_cast<int16_t>((next_random(random) >> 20) - 2048);
        dl::TensorBase in({1, h, w, ci}, input, 0, dl::DATA_TYPE_INT16, false);
        dl::TensorBase out({1, h, w, co}, output, 12, dl::DATA_TYPE_INT16, false);
        dl::TensorBase ref({1, h, w, co}, reference, 12, dl::DATA_TYPE_INT16, false);
        dl::TensorBase filter({1, 1, ci, co}, weights, 0, dl::DATA_TYPE_INT16, false);
        std::vector<int> pads{0, 0, 0, 0};
        dl::base::ConvOpArgs<int16_t> a(
            &out, &in, pads, &filter, {1, 1}, {1, 1}, 1, nullptr, dl::Linear, nullptr, dl::RUNTIME_MODE_SINGLE_CORE);
        dl::base::ConvOpArgs<int16_t> b(
            &ref, &in, pads, &filter, {1, 1}, {1, 1}, 1, nullptr, dl::Linear, nullptr, dl::RUNTIME_MODE_SINGLE_CORE);
        reference_conv(b.get_args(0));
        std::vector<int64_t> elapsed[2];
        for (int repeat = 0; repeat < 10; ++repeat) {
            for (int order = 0; order < 2; ++order) {
                const int mode = (repeat + order) % 2;
                const int64_t start = esp_timer_get_time();
                if (mode)
                    dl::base::conv2d<int16_t, int32_t, int64_t>(&a.get_args(0));
                else
                    // Both timed implementations write to the same address so
                    // cache-set placement cannot favor one allocation.
                    reference_conv(a.get_args(0));
                const int64_t duration = esp_timer_get_time() - start;
                if (repeat)
                    elapsed[mode].push_back(duration);
                TEST_ASSERT_EQUAL_INT16_ARRAY(reference, output, h * w * co);
            }
        }
        for (int repeat = 0; repeat < 9; ++repeat) {
            printf("pointwise_sample: h=%d w=%d ci=%d co=%d repeat=%d reference_us=%lld tiled_us=%lld\n",
                   h,
                   w,
                   ci,
                   co,
                   repeat,
                   elapsed[0][repeat],
                   elapsed[1][repeat]);
        }
        for (auto &values : elapsed) std::sort(values.begin(), values.end());
        printf("pointwise_perf: h=%d w=%d ci=%d co=%d reference_us=%lld tiled_us=%lld repeats=9\n",
               h,
               w,
               ci,
               co,
               elapsed[0][4],
               elapsed[1][4]);
        heap_caps_free(input);
        heap_caps_free(weights);
        heap_caps_free(output);
        heap_caps_free(reference);
    }
    TEST_ASSERT_TRUE(heap_caps_check_integrity_all(true));
}
#endif
