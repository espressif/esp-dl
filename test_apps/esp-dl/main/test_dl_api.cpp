#include "dl_model_base.hpp"
#include "dl_module_add.hpp"
#include "dl_module_creator.hpp"
#include "dl_module_layer_normalization.hpp"
#include "dl_module_lut.hpp"
#include "dl_module_relu.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "fbs_loader.hpp"
#include "unity.h"
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>
static const char *TAG = "TEST DL MODEL";

using namespace dl;

uint8_t key[16] = {0x8a, 0x7f, 0xc9, 0x61, 0xe4, 0xe6, 0xff, 0x0a, 0xd2, 0x64, 0x36, 0x95, 0x28, 0x75, 0xae, 0x4a};

TEST_CASE("Test dl model API: load()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: load()");
    int internal_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    Model *model = new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, key);
    delete model;
    module::ModuleCreator::get_instance()->clear();

    int internal_ram_size_second = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    fbs::FbsModel *fbs_model = fbs_loader->load(0, key);
    Model *model2 = new Model(fbs_model);
    delete model2;
    delete fbs_loader;
    delete fbs_model;
    module::ModuleCreator::get_instance()->clear();

    int internal_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG,
             "internal ram size before: %d, second: %d, end:%d",
             internal_ram_size_before,
             internal_ram_size_second,
             internal_ram_size_end);
    ESP_LOGI(TAG, "psram size before: %d, end:%d", psram_size_before, psram_size_end);
    TEST_ASSERT_EQUAL(true, internal_ram_size_before - internal_ram_size_second < 1000);
    TEST_ASSERT_EQUAL(true, internal_ram_size_second == internal_ram_size_end);
    TEST_ASSERT_EQUAL(true, psram_size_before == psram_size_end);
}

TEST_CASE("Test dl model API: profile()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: run()");
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

    model->profile();
    model->minimize();
    model->profile_memory();
    delete model;

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before <= total_ram_size_end);
}

TEST_CASE("Test dl model API: run()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: run()");
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    dl::tool::Latency latency;
    for (int i = 0; i < 15; i++) {
        model = new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, (i % 2) * 100000);

        latency.start();
        model->run();
        latency.end();
        printf("run:%ld ms\n", latency.get_period() / 1000);

        latency.start();
        model->run(3, 0);
        model->run(3, 1);
        model->run(3, 2);
        latency.end();
        printf("run stage by stage:%ld ms\n", latency.get_period() / 1000);
        delete model;
    }

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

// Helper: fill model inputs with data from saved buffers, then run
static void fill_inputs_and_run(Model *model, const std::map<std::string, std::pair<uint8_t *, size_t>> &saved_inputs)
{
    std::map<std::string, TensorBase *> &model_inputs = model->get_inputs();
    for (auto &pair : model_inputs) {
        const std::string &name = pair.first;
        TensorBase *tensor = pair.second;
        auto it = saved_inputs.find(name);
        if (it != saved_inputs.end()) {
            memcpy(tensor->get_element_ptr(), it->second.first, it->second.second);
        }
    }
    model->run();
}

TEST_CASE("Test dl model API: reset()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: reset()");

    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    std::map<std::string, std::pair<uint8_t *, size_t>> saved_input_data;
    std::map<std::string, TensorBase *> &model_inputs = model->get_inputs();
    for (auto &pair : model_inputs) {
        const std::string &input_name = pair.first;
        TensorBase *input_tensor = pair.second;
        input_tensor->rand();
        size_t bytes = input_tensor->get_bytes();
        uint8_t *buf = new uint8_t[bytes];
        memcpy(buf, input_tensor->get_element_ptr(), bytes);
        saved_input_data.emplace(input_name, std::make_pair(buf, bytes));
    }

    // First run with random inputs
    fill_inputs_and_run(model, saved_input_data);
    std::map<std::string, TensorBase *> &outputs = model->get_outputs();
    std::vector<uint8_t *> output_data_first_list;
    std::vector<size_t> output_bytes_list;
    int idx = 0;
    for (auto &pair : outputs) {
        TensorBase *tensor = pair.second;
        size_t bytes = tensor->get_bytes();
        uint8_t *first = new uint8_t[bytes];
        memcpy(first, tensor->get_element_ptr(), bytes);
        output_data_first_list.push_back(first);
        output_bytes_list.push_back(bytes);
        idx++;
    }

    // Second run without reset
    fill_inputs_and_run(model, saved_input_data);

    model->reset();

    // Third run after reset with same inputs - output should match first run
    fill_inputs_and_run(model, saved_input_data);

    idx = 0;
    for (auto &pair : outputs) {
        TensorBase *tensor = pair.second;
        size_t bytes = output_bytes_list[idx];
        const void *expected = output_data_first_list[idx];
        const void *actual = tensor->get_element_ptr();
        TEST_ASSERT_EQUAL_MESSAGE(0,
                                  memcmp(expected, actual, bytes),
                                  "Third run output does not match first run - reset() may not have cleared state");
        idx++;
    }

    for (auto p : output_data_first_list) {
        delete[] p;
    }
    output_data_first_list.clear();
    output_bytes_list.clear();

    for (auto &p : saved_input_data) {
        delete[] p.second.first;
    }
    saved_input_data.clear();
    delete model;
}

TEST_CASE("Test dl module API: run()", "[api]")
{
    ESP_LOGI(TAG, "Test dl module API: run()");
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    TensorBase *input1 = new TensorBase({1, 3, 64, 64}, nullptr, 0, DATA_TYPE_INT8);
    TensorBase *input2 = new TensorBase({1, 1, 1, 64}, nullptr, 0, DATA_TYPE_INT8);
    TensorBase *output = new TensorBase({1, 3, 64, 64}, nullptr, 0, DATA_TYPE_INT8);

    // input2 stays zero so the Add check below cannot saturate int8.
    input1->rand();

    // single input and single output
    module::Module *relu_op = new module::Relu("relu", MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    relu_op->run(input1, output);
    for (int i = 0; i < output->get_size(); i++) {
        int8_t in = input1->get_element<int8_t>(i);
        int8_t out = output->get_element<int8_t>(i);
        if (in > 0) {
            TEST_ASSERT_EQUAL(true, in == out);
        } else {
            TEST_ASSERT_EQUAL(true, out == 0);
        }
    }

    // multiple inputs and multiple outputs
    module::Module *add_op = new module::Add("add", MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    add_op->run({input1, input2}, {output});

    for (int i = 0; i < output->get_size(); i++) {
        int8_t in1 = input1->get_element<int8_t>(i);
        int8_t in2 = input2->get_element<int8_t>(i % 64);
        int8_t out = output->get_element<int8_t>(i);
        TEST_ASSERT_EQUAL(true, in1 + in2 == out);
    }

    delete input1;
    delete input2;
    delete output;
    delete relu_op;
    delete add_op;

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

/**
 * @brief Fill a tensor with finite, deterministic values.
 *
 * TensorBase::rand() may produce NaN or Inf bit patterns for float tensors.
 */
static void fill_test_pattern(TensorBase *tensor, uint32_t seed)
{
    uint32_t state = seed;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };

    const int n = tensor->get_size();
    if (tensor->dtype == DATA_TYPE_FLOAT) {
        float *ptr = static_cast<float *>(tensor->get_element_ptr());
        for (int i = 0; i < n; i++) {
            ptr[i] = static_cast<float>(static_cast<int32_t>(next() >> 21) - 1024) / 256.0f;
        }
    } else if (tensor->dtype == DATA_TYPE_INT16) {
        int16_t *ptr = static_cast<int16_t *>(tensor->get_element_ptr());
        for (int i = 0; i < n; i++) {
            ptr[i] = static_cast<int16_t>(next() >> 16);
        }
    } else {
        int8_t *ptr = static_cast<int8_t *>(tensor->get_element_ptr());
        for (int i = 0; i < n; i++) {
            ptr[i] = static_cast<int8_t>(next() >> 24);
        }
    }
}

/**
 * @brief True when every element of the tensor holds the same value.
 *
 * This rejects comparisons made meaningless by constant or saturated output.
 */
static bool tensor_is_uniform(TensorBase *tensor)
{
    const uint8_t *ptr = static_cast<const uint8_t *>(tensor->get_element_ptr());
    const int elem_bytes = tensor->get_bytes() / tensor->get_size();
    for (int i = 1; i < tensor->get_size(); i++) {
        if (memcmp(ptr, ptr + i * elem_bytes, elem_bytes) != 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Compare single-core and dual-core LayerNormalization outputs.
 *
 * Also covers even and odd row splits, unaligned rows, in-place mode, and
 * aliased input/output buffers. Model::test() itself only runs single-core.
 */
TEST_CASE("Test dl module API: LayerNormalization dual core", "[api]")
{
    ESP_LOGI(TAG, "Test dl module API: LayerNormalization dual core");

    struct LNShape {
        int outer; ///< rows, i.e. the dimension the split divides
        int inner; ///< normalized length
    };
    // Covers even and odd splits, an unaligned row, and an unsplittable row.
    const LNShape shapes[] = {{8, 64}, {7, 64}, {5, 193}, {1, 64}};

    struct LNDtype {
        quant_type_t quant;
        dtype_t dtype;
        int act_exponent;    ///< X and Y scale
        int affine_exponent; ///< gamma and beta scale
    };
    // Keep results inside the output range so saturation cannot hide errors.
    const LNDtype dtypes[] = {
        {QUANT_TYPE_SYMM_8BIT, DATA_TYPE_INT8, -5, -7},
        {QUANT_TYPE_SYMM_16BIT, DATA_TYPE_INT16, -13, -15},
        {QUANT_TYPE_FLOAT32, DATA_TYPE_FLOAT, 0, 0},
    };

    // Exclude persistent worker-task allocation from the heap check.
    {
        TensorBase *warmup_x = new TensorBase({4, 16}, nullptr, -5, DATA_TYPE_INT8);
        TensorBase *warmup_affine = new TensorBase({16}, nullptr, -7, DATA_TYPE_INT8);
        TensorBase *warmup_y = new TensorBase({4, 16}, nullptr, -5, DATA_TYPE_INT8);
        module::Module *warmup_ln =
            new module::LayerNormalization(-1, 1e-5f, 0, "ln_warmup", MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
        warmup_ln->run({warmup_x, warmup_affine, warmup_affine}, {warmup_y}, RUNTIME_MODE_MULTI_CORE);
        delete warmup_ln;
        delete warmup_x;
        delete warmup_affine;
        delete warmup_y;
    }

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    // Test both module allocation modes and actual x/y buffer aliasing.
    const module_inplace_t inplace_modes[] = {MODULE_NON_INPLACE, MODULE_INPLACE_CHANGED_BUFFER};

    for (const LNShape &shape : shapes) {
        for (const LNDtype &dt : dtypes) {
            for (module_inplace_t inplace : inplace_modes) {
                for (bool alias_output : {false, true}) {
                    const std::vector<int> x_shape = {shape.outer, shape.inner};
                    const std::vector<int> affine_shape = {shape.inner};

                    TensorBase *x_source = new TensorBase(x_shape, nullptr, dt.act_exponent, dt.dtype);
                    TensorBase *gamma = new TensorBase(affine_shape, nullptr, dt.affine_exponent, dt.dtype);
                    TensorBase *beta = new TensorBase(affine_shape, nullptr, dt.affine_exponent, dt.dtype);
                    fill_test_pattern(x_source, 0x12345678);
                    fill_test_pattern(gamma, 0x9e3779b9);
                    fill_test_pattern(beta, 0x85ebca6b);

                    // Start both runs from identical, independent inputs.
                    TensorBase *in_serial = new TensorBase(x_shape, nullptr, dt.act_exponent, dt.dtype);
                    TensorBase *in_dual = new TensorBase(x_shape, nullptr, dt.act_exponent, dt.dtype);
                    memcpy(in_serial->get_element_ptr(), x_source->get_element_ptr(), x_source->get_bytes());
                    memcpy(in_dual->get_element_ptr(), x_source->get_element_ptr(), x_source->get_bytes());

                    TensorBase *y_serial =
                        alias_output ? in_serial : new TensorBase(x_shape, nullptr, dt.act_exponent, dt.dtype);
                    TensorBase *y_dual =
                        alias_output ? in_dual : new TensorBase(x_shape, nullptr, dt.act_exponent, dt.dtype);

                    module::Module *ln = new module::LayerNormalization(-1, 1e-5f, 0, "ln", inplace, dt.quant);
                    ln->run({in_serial, gamma, beta}, {y_serial}, RUNTIME_MODE_SINGLE_CORE);
                    ln->run({in_dual, gamma, beta}, {y_dual}, RUNTIME_MODE_MULTI_CORE);

                    TEST_ASSERT_EQUAL(false, tensor_is_uniform(y_serial));
                    TEST_ASSERT_EQUAL_MEMORY(
                        y_serial->get_element_ptr(), y_dual->get_element_ptr(), y_serial->get_bytes());

                    delete ln;
                    delete x_source;
                    delete gamma;
                    delete beta;
                    if (!alias_output) {
                        delete y_serial;
                        delete y_dual;
                    }
                    delete in_serial;
                    delete in_dual;
                }
            }
        }
    }

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

/**
 * @brief Verify normalization over shape[axis:].
 *
 * A positive axis on a rank-4 tensor must match an equivalent 2D fold.
 * esp-ppq only exports last-axis LayerNormalization, so this is tested through
 * the module API.
 */
TEST_CASE("Test dl module API: LayerNormalization multi-axis", "[api]")
{
    ESP_LOGI(TAG, "Test dl module API: LayerNormalization multi-axis");
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    // POD avoids allocations that would outlive the heap check.
    struct AxisCase {
        int shape[4]; ///< rank-4 input
        int axis;     ///< positive axis into that shape
        int outer;    ///< prod(shape[:axis]), the rows it has to fold to
        int inner;    ///< prod(shape[axis:]), the length it has to reduce
    };
    const AxisCase cases[] = {
        {{2, 3, 4, 16}, 2, 6, 64},  ///< reduce the last two axes
        {{2, 3, 4, 16}, 1, 2, 192}, ///< reduce the last three
        {{2, 3, 4, 16}, 3, 24, 16}, ///< last axis only, spelled positively
    };
    const struct {
        quant_type_t quant;
        dtype_t dtype;
        int act_exponent;
        int affine_exponent;
    } dtypes[] = {
        {QUANT_TYPE_SYMM_8BIT, DATA_TYPE_INT8, -5, -7},
        {QUANT_TYPE_SYMM_16BIT, DATA_TYPE_INT16, -13, -15},
        {QUANT_TYPE_FLOAT32, DATA_TYPE_FLOAT, 0, 0},
    };

    for (const AxisCase &ac : cases) {
        for (const auto &dt : dtypes) {
            const std::vector<int> nd_shape(ac.shape, ac.shape + 4);
            const std::vector<int> folded = {ac.outer, ac.inner};
            const std::vector<int> affine_shape = {ac.inner};

            TensorBase *x_nd = new TensorBase(nd_shape, nullptr, dt.act_exponent, dt.dtype);
            TensorBase *x_2d = new TensorBase(folded, nullptr, dt.act_exponent, dt.dtype);
            TensorBase *gamma = new TensorBase(affine_shape, nullptr, dt.affine_exponent, dt.dtype);
            TensorBase *beta = new TensorBase(affine_shape, nullptr, dt.affine_exponent, dt.dtype);
            TensorBase *y_nd = new TensorBase(nd_shape, nullptr, dt.act_exponent, dt.dtype);
            TensorBase *y_2d = new TensorBase(folded, nullptr, dt.act_exponent, dt.dtype);

            fill_test_pattern(x_nd, 0x2545f491);
            memcpy(x_2d->get_element_ptr(), x_nd->get_element_ptr(), x_nd->get_bytes());
            fill_test_pattern(gamma, 0x9e3779b9);
            fill_test_pattern(beta, 0x85ebca6b);

            module::Module *ln_nd =
                new module::LayerNormalization(ac.axis, 1e-5f, 0, "ln_nd", MODULE_NON_INPLACE, dt.quant);
            module::Module *ln_2d = new module::LayerNormalization(-1, 1e-5f, 0, "ln_2d", MODULE_NON_INPLACE, dt.quant);
            ln_nd->run({x_nd, gamma, beta}, {y_nd}, RUNTIME_MODE_SINGLE_CORE);
            ln_2d->run({x_2d, gamma, beta}, {y_2d}, RUNTIME_MODE_SINGLE_CORE);

            TEST_ASSERT_EQUAL(false, tensor_is_uniform(y_nd));
            TEST_ASSERT_EQUAL_MEMORY(y_2d->get_element_ptr(), y_nd->get_element_ptr(), y_nd->get_bytes());

            delete ln_nd;
            delete ln_2d;
            delete x_nd;
            delete x_2d;
            delete gamma;
            delete beta;
            delete y_nd;
            delete y_2d;
        }
    }

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

TEST_CASE("Test dl module API: StreamingCache reset()", "[api]")
{
    ESP_LOGI(TAG, "Test dl module API: StreamingCache reset()");
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    TensorBase *input = new TensorBase({1, 2, 4, 4}, nullptr, 0, DATA_TYPE_INT8);
    TensorBase *output = new TensorBase({1, 3, 4, 4}, nullptr, 0, DATA_TYPE_INT8);

    module::Module *streaming_cache_op =
        new module::StreamingCache("streaming_cache", 2, 1, MODULE_NON_INPLACE, QUANT_TYPE_NONE);

    // First run
    memset(input->get_element_ptr(), 5, input->get_bytes());
    streaming_cache_op->run(input, output);

    int8_t *output_data_first = new int8_t[output->get_size()];
    memcpy(output_data_first, output->get_element_ptr(), output->get_bytes());

    // Second run without reset - cache should be updated
    memset(input->get_element_ptr(), 10, input->get_bytes());
    streaming_cache_op->run(input, output);

    int8_t *output_data_second = new int8_t[output->get_size()];
    memcpy(output_data_second, output->get_element_ptr(), output->get_bytes());

    // Verify that outputs are different (cache was updated)
    TEST_ASSERT_EQUAL(true, memcmp(output_data_first, output_data_second, output->get_bytes()) != 0);

    // Reset cache
    streaming_cache_op->reset();

    // Third run after reset - output should match first run pattern (cache cleared)
    memset(input->get_element_ptr(), 5, input->get_bytes());
    streaming_cache_op->run(input, output);

    int8_t *output_data_after_reset = new int8_t[output->get_size()];
    memcpy(output_data_after_reset, output->get_element_ptr(), output->get_bytes());

    TEST_ASSERT_EQUAL(true, memcmp(output_data_first, output_data_after_reset, output->get_bytes()) == 0);

    // Verify that first frame has zeros (from reset cache)
    // for (int i = 0; i < 16; i++) {
    //     TEST_ASSERT_EQUAL(0, output_data_after_reset[i]);
    // }

    delete input;
    delete output;
    delete streaming_cache_op;
    delete[] output_data_first;
    delete[] output_data_second;
    delete[] output_data_after_reset;

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

TEST_CASE("Test dl module API: LUT runtime modes", "[api]")
{
    struct LutCase {
        const char *name;
        quant_type_t quant;
        dtype_t dtype;
        int table_size;
    };
    const LutCase cases[] = {
        {"int8", QUANT_TYPE_SYMM_8BIT, DATA_TYPE_INT8, 256},
        {"int16_step16", QUANT_TYPE_SYMM_16BIT, DATA_TYPE_INT16, 4097},
        {"int16_step1", QUANT_TYPE_SYMM_16BIT, DATA_TYPE_INT16, 65537},
    };
    // Sizes cover:
    //   - tiny inputs that cannot be split at all (1..15)
    //   - the first size at which INT16 / INT8 splits succeed (16 / 32)
    //   - odd sizes whose dual-core tail is not a multiple of the SIMD chunk
    //     (17, 33, 65, 129, 4097, 8193)
    //   - INT16 step=1 AUTO threshold crossing (< 2048 vs >= 2048)
    //   - INT8 / INT16 step>=2 AUTO threshold crossing (< 8192 vs >= 8192)
    const int sizes[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 1011, 1024, 2048, 4097, 8192, 8193};

    for (const LutCase &cfg : cases) {
        TensorBase *table = new TensorBase({cfg.table_size}, nullptr, 0, cfg.dtype);
        if (cfg.dtype == DATA_TYPE_INT8) {
            int8_t *ptr = static_cast<int8_t *>(table->get_element_ptr());
            for (int i = 0; i < cfg.table_size; i++) {
                ptr[i] = static_cast<int8_t>(i * 109 + 37);
            }
        } else {
            int16_t *ptr = static_cast<int16_t *>(table->get_element_ptr());
            for (int i = 0; i < cfg.table_size; i++) {
                ptr[i] = static_cast<int16_t>(i * 109 + 37);
            }
        }
        const int step_val = cfg.dtype == DATA_TYPE_INT16 ? 65536 / (cfg.table_size - 1) : 0;

        for (int size : sizes) {
            TensorBase *input = new TensorBase({size}, nullptr, 0, cfg.dtype);
            TensorBase *single = new TensorBase({size}, nullptr, 0, cfg.dtype);
            TensorBase *multi = new TensorBase({size}, nullptr, 0, cfg.dtype);
            TensorBase *automatic = new TensorBase({size}, nullptr, 0, cfg.dtype);
            TensorBase *reference = new TensorBase({size}, nullptr, 0, cfg.dtype);

            if (cfg.dtype == DATA_TYPE_INT8) {
                int8_t *ptr = static_cast<int8_t *>(input->get_element_ptr());
                for (int i = 0; i < size; i++) {
                    ptr[i] = static_cast<int8_t>(i * 73 + 19);
                }
                // Independent reference: call the base kernel directly on the
                // raw buffers. If the module were to mis-wire size/step/table,
                // this reference would still be correct.
                dl::base::lut_s8(static_cast<int8_t *>(reference->get_element_ptr()),
                                 ptr,
                                 size,
                                 static_cast<int8_t *>(table->get_element_ptr()));
            } else {
                int16_t *ptr = static_cast<int16_t *>(input->get_element_ptr());
                for (int i = 0; i < size; i++) {
                    ptr[i] = static_cast<int16_t>(i * 1009 + 313);
                }
                dl::base::lut_s16_nearest_neighbor(static_cast<int16_t *>(reference->get_element_ptr()),
                                                   ptr,
                                                   size,
                                                   static_cast<int16_t *>(table->get_element_ptr()),
                                                   step_val);
            }

            module::LUT lut("lut", MODULE_NON_INPLACE, cfg.quant);
            lut.run({input, table}, {single}, RUNTIME_MODE_SINGLE_CORE);
            lut.run({input, table}, {multi}, RUNTIME_MODE_MULTI_CORE);
            lut.run({input, table}, {automatic}, RUNTIME_MODE_AUTO);

            const int bytes = single->get_bytes();
            TEST_ASSERT_EQUAL_MEMORY(reference->get_element_ptr(), single->get_element_ptr(), bytes);
            TEST_ASSERT_EQUAL_MEMORY(reference->get_element_ptr(), multi->get_element_ptr(), bytes);
            TEST_ASSERT_EQUAL_MEMORY(reference->get_element_ptr(), automatic->get_element_ptr(), bytes);

            delete input;
            delete single;
            delete multi;
            delete automatic;
            delete reference;
        }
        delete table;
    }
}

TEST_CASE("Test PDL3 package: verify integrity", "[api]")
{
    ESP_LOGI(TAG, "Test PDL3 package: verify integrity");

    // The "model" partition is packed in the PDL3 format by default, verify it directly.
    fbs::FbsLoader loader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

    int model_num = loader.get_model_num();
    ESP_LOGI(TAG, "model_num: %d", model_num);
    TEST_ASSERT_GREATER_THAN(0, model_num);

    if (loader.get_model_format() != fbs::FBS_FILE_FORMAT_PDL3) {
        ESP_LOGI(TAG, "model format is not PDL3");
        return;
    }

    uint32_t package_size = loader.get_package_size();
    ESP_LOGI(TAG, "package_size: %lu", (unsigned long)package_size);
    TEST_ASSERT_GREATER_THAN(0, package_size);

    char ver[16] = {0};
    TEST_ASSERT_EQUAL(ESP_OK, loader.get_package_version(ver, sizeof(ver)));
    ESP_LOGI(TAG, "package_version: %s", ver);
    // package_version is set by the build (see main/CMakeLists.txt).
    TEST_ASSERT_EQUAL_STRING("v1.0.0", ver);

    // The digest stored in the header must match the one recomputed from the package bytes.
    uint8_t stored[32];
    uint8_t calc[32];
    TEST_ASSERT_EQUAL(ESP_OK, loader.get_package_sha256(stored));
    TEST_ASSERT_EQUAL(ESP_OK, loader.calc_package_sha256(calc));
    TEST_ASSERT_EQUAL(0, memcmp(stored, calc, 32));

    TEST_ASSERT_EQUAL(true, loader.verify_package_sha256());
}

TEST_CASE("Test PDL3 package: non-PDL3 returns unsupported", "[api]")
{
    ESP_LOGI(TAG, "Test PDL3 package: non-PDL3 returns unsupported");

    // A non-PDL3 (EDL2) buffer should make all PDL3-specific interfaces report unsupported/default.
    std::vector<uint8_t> buf(64, 0);
    memcpy(buf.data(), "PDL2", 4);

    fbs::FbsLoader loader((const char *)buf.data(), fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    TEST_ASSERT_EQUAL_UINT32(0, loader.get_package_size());

    char ver[16] = {0};
    TEST_ASSERT_EQUAL(ESP_ERR_NOT_SUPPORTED, loader.get_package_version(ver, sizeof(ver)));

    uint8_t sha[32];
    TEST_ASSERT_EQUAL(ESP_ERR_NOT_SUPPORTED, loader.get_package_sha256(sha));
    TEST_ASSERT_EQUAL(ESP_ERR_NOT_SUPPORTED, loader.calc_package_sha256(sha));
    TEST_ASSERT_EQUAL(false, loader.verify_package_sha256());
}
