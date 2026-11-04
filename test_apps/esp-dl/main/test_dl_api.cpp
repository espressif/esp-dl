#include "dl_model_base.hpp"
#include "dl_model_context.hpp"
#include "dl_module_add.hpp"
#include "dl_module_attention.hpp"
#include "dl_module_creator.hpp"
#include "dl_module_inverse_sigmoid.hpp"
#include "dl_module_layer_normalization.hpp"
#include "dl_module_lut.hpp"
#include "dl_module_relu.hpp"
#include "dl_module_rotary_embedding.hpp"
#include "dl_module_softmax.hpp"
#include "dl_module_streaming_cache.hpp"
#include "dl_module_top_k.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "fbs_loader.hpp"
#include "unity.h"
#include <cmath>
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

TEST_CASE("Test dl model context: root allocation rollback and arena reset", "[api]")
{
    ModelContext context;
    constexpr size_t root_bytes = 256;
#if CONFIG_SPIRAM
    TEST_ASSERT_TRUE(context.root_alloc(0, root_bytes));
    void *const old_root = context.get_psram_root();
    const size_t largest_free = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    TEST_ASSERT_FALSE(context.root_alloc(0, largest_free + 16));
    TEST_ASSERT_EQUAL_PTR(old_root, context.get_psram_root());
#else
    TEST_ASSERT_TRUE(context.root_alloc(root_bytes, 0));
    void *const old_root = context.get_internal_root();
    const size_t largest_free = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    TEST_ASSERT_FALSE(context.root_alloc(largest_free + 16, 0));
    TEST_ASSERT_EQUAL_PTR(old_root, context.get_internal_root());
#endif

    context.m_variables.push_back(new TensorBase({1}, old_root, 0, DATA_TYPE_INT8, false));
    const WorkspaceBindingId binding_id = context.create_workspace_binding("reset-check");
    context.bind_workspace(binding_id, old_root, root_bytes);
    context.reset_variable_arena();
    TEST_ASSERT_NULL(context.m_variables[0]);
    TEST_ASSERT_EQUAL_INT(0, context.get_workspace_binding_count());
    TEST_ASSERT_NULL(context.get_psram_root());
    TEST_ASSERT_NULL(context.get_internal_root());
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

TEST_CASE("Test TopK sorted attribute", "[topk]")
{
    {
        const int8_t input_data[] = {1, 8, 2, 7, 3, 6, 4, 5};
        TensorBase input({1, 8}, input_data, 0, DATA_TYPE_INT8, false);

        TensorBase sorted_values({1, 4}, nullptr, 0, DATA_TYPE_INT8);
        TensorBase sorted_indices({1, 4}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK sorted_topk(4, -1, 1, 1);
        sorted_topk.run({&input}, {&sorted_values, &sorted_indices});
        const int8_t expected_sorted_values[] = {8, 7, 6, 5};
        const int64_t expected_sorted_indices[] = {1, 3, 5, 7};
        for (int i = 0; i < 4; i++) {
            TEST_ASSERT_EQUAL_INT8(expected_sorted_values[i], sorted_values.get_element_ptr<int8_t>()[i]);
            TEST_ASSERT_EQUAL_INT(static_cast<int>(expected_sorted_indices[i]),
                                  static_cast<int>(sorted_indices.get_element_ptr<int64_t>()[i]));
        }

        TensorBase int32_values({1, 3}, nullptr, 0, DATA_TYPE_INT8);
        TensorBase int32_indices({1, 3}, nullptr, 0, DATA_TYPE_INT32);
        const int64_t k_data[] = {3};
        TensorBase k_tensor({1}, k_data, 0, DATA_TYPE_INT64, false);
        module::TopK int32_topk(3, -1, 1, 1, nullptr, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
        int32_topk.run({&input, &k_tensor}, {&int32_values, &int32_indices});
        for (int i = 0; i < 3; i++) {
            TEST_ASSERT_EQUAL_INT8(expected_sorted_values[i], int32_values.get_element_ptr<int8_t>()[i]);
            TEST_ASSERT_EQUAL_INT(static_cast<int>(expected_sorted_indices[i]),
                                  int32_indices.get_element_ptr<int32_t>()[i]);
        }

        TensorBase heap_values({1, 4}, nullptr, 0, DATA_TYPE_INT8);
        TensorBase heap_indices({1, 4}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK heap_topk(4, -1, 1, 0);
        heap_topk.run({&input}, {&heap_values, &heap_indices});
        TEST_ASSERT_EQUAL_INT8(5, heap_values.get_element_ptr<int8_t>()[0]);
        TEST_ASSERT_EQUAL_INT(7, static_cast<int>(heap_indices.get_element_ptr<int64_t>()[0]));

        TensorBase counting_values({1, 5}, nullptr, 0, DATA_TYPE_INT8);
        TensorBase counting_indices({1, 5}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK counting_topk(5, -1, 1, 0);
        counting_topk.run({&input}, {&counting_values, &counting_indices});
        const int8_t expected_counting_values[] = {8, 7, 6, 4, 5};
        const int64_t expected_counting_indices[] = {1, 3, 5, 6, 7};
        for (int i = 0; i < 5; i++) {
            TEST_ASSERT_EQUAL_INT8(expected_counting_values[i], counting_values.get_element_ptr<int8_t>()[i]);
            TEST_ASSERT_EQUAL_INT(static_cast<int>(expected_counting_indices[i]),
                                  static_cast<int>(counting_indices.get_element_ptr<int64_t>()[i]));
        }
    }

    {
        int16_t input16_data[32];
        for (int i = 0; i < 32; i++) {
            input16_data[i] = i;
        }
        TensorBase input16({1, 32}, input16_data, 0, DATA_TYPE_INT16, false);
        TensorBase nth_values({1, 20}, nullptr, 0, DATA_TYPE_INT16);
        TensorBase nth_indices({1, 20}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK nth_topk(20, -1, 1, 0);
        nth_topk.run({&input16}, {&nth_values, &nth_indices});
        bool nth_seen[32] = {};
        for (int i = 0; i < 20; i++) {
            const int index = static_cast<int>(nth_indices.get_element_ptr<int64_t>()[i]);
            TEST_ASSERT_TRUE(index >= 12 && index < 32);
            TEST_ASSERT_FALSE(nth_seen[index]);
            TEST_ASSERT_EQUAL_INT(index, nth_values.get_element_ptr<int16_t>()[i]);
            nth_seen[index] = true;
        }

        TensorBase full_values({1, 25}, nullptr, 0, DATA_TYPE_INT16);
        TensorBase full_indices({1, 25}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK full_topk(25, -1, 1, 0);
        full_topk.run({&input16}, {&full_values, &full_indices});
        bool full_seen[32] = {};
        for (int i = 0; i < 25; i++) {
            const int index = static_cast<int>(full_indices.get_element_ptr<int64_t>()[i]);
            TEST_ASSERT_TRUE(index >= 7 && index < 32);
            TEST_ASSERT_FALSE(full_seen[index]);
            TEST_ASSERT_EQUAL_INT(index, full_values.get_element_ptr<int16_t>()[i]);
            full_seen[index] = true;
        }
    }

    {
        int16_t tie16_data[32];
        for (int i = 0; i < 32; i++) {
            tie16_data[i] = static_cast<int16_t>((i % 4) * 10);
        }
        TensorBase tie16_input({1, 32}, tie16_data, 0, DATA_TYPE_INT16, false);
        TensorBase full_sorted_values({1, 25}, nullptr, 0, DATA_TYPE_INT16);
        TensorBase full_sorted_indices({1, 25}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK full_sorted_topk(25, -1, 1, 1);
        full_sorted_topk.run({&tie16_input}, {&full_sorted_values, &full_sorted_indices});
        int pos = 0;
        for (int bucket = 3; bucket >= 0 && pos < 25; bucket--) {
            for (int idx = bucket; idx < 32 && pos < 25; idx += 4) {
                TEST_ASSERT_EQUAL_INT16(bucket * 10, full_sorted_values.get_element_ptr<int16_t>()[pos]);
                TEST_ASSERT_EQUAL_INT(idx, static_cast<int>(full_sorted_indices.get_element_ptr<int64_t>()[pos]));
                pos++;
            }
        }
    }

    {
        // k == 1 with inner > 1 (axis = 1 on [1, 4, 3]): exercises in-place argbest accumulation and tie-breaking.
        const int8_t argbest_data[] = {
            3,
            9,
            1,
            7,
            9,
            4,
            7,
            2,
            8,
            5,
            9,
            8,
        };
        TensorBase argbest_input({1, 4, 3}, argbest_data, 0, DATA_TYPE_INT8, false);
        TensorBase argbest_values({1, 1, 3}, nullptr, 0, DATA_TYPE_INT8);
        TensorBase argbest_indices({1, 1, 3}, nullptr, 0, DATA_TYPE_INT64);
        module::TopK argbest_topk(1, 1, 1, 1);
        argbest_topk.run({&argbest_input}, {&argbest_values, &argbest_indices});
        const int8_t expected_argbest_values[] = {7, 9, 8};
        const int expected_argbest_indices[] = {1, 0, 2};
        for (int i = 0; i < 3; i++) {
            TEST_ASSERT_EQUAL_INT8(expected_argbest_values[i], argbest_values.get_element_ptr<int8_t>()[i]);
            TEST_ASSERT_EQUAL_INT(expected_argbest_indices[i],
                                  static_cast<int>(argbest_indices.get_element_ptr<int64_t>()[i]));
        }
    }

    {
        module::TopK invalid_k_topk(10, -1, 1, 1);
        std::vector<std::vector<int>> shapes = {{1, 4}};
        TEST_ASSERT_TRUE(invalid_k_topk.get_output_shape(shapes).empty());

        const int8_t in_data[] = {1, 2, 3, 4};
        int8_t val_data[] = {9, 9};
        int64_t idx_data[] = {7, 7};
        TensorBase in_tensor({1, 4}, in_data, 0, DATA_TYPE_INT8, false);
        TensorBase val_tensor({1, 2}, val_data, 0, DATA_TYPE_INT8, false);
        TensorBase idx_tensor({1, 2}, idx_data, 0, DATA_TYPE_INT64, false);
        invalid_k_topk.run({&in_tensor}, {&val_tensor, &idx_tensor});
        for (int i = 0; i < 2; i++) {
            TEST_ASSERT_EQUAL_INT8(0, val_data[i]);
            TEST_ASSERT_EQUAL_INT(0, static_cast<int>(idx_data[i]));
        }

        // Aliasing input X and output Values must be rejected without corrupting X.
        module::TopK valid_topk(2, -1, 1, 1);
        int8_t aliased_in[] = {4, 3, 2, 1};
        int64_t aliased_idx[] = {9, 9};
        TensorBase aliased_in_tensor({1, 4}, aliased_in, 0, DATA_TYPE_INT8, false);
        TensorBase aliased_val_tensor({1, 2}, aliased_in, 0, DATA_TYPE_INT8, false);
        TensorBase aliased_idx_tensor({1, 2}, aliased_idx, 0, DATA_TYPE_INT64, false);
        valid_topk.run({&aliased_in_tensor}, {&aliased_val_tensor, &aliased_idx_tensor});
        TEST_ASSERT_EQUAL_INT8(4, aliased_in[0]);
        TEST_ASSERT_EQUAL_INT8(3, aliased_in[1]);
        TEST_ASSERT_EQUAL_INT(0, static_cast<int>(aliased_idx[0]));
        TEST_ASSERT_EQUAL_INT(0, static_cast<int>(aliased_idx[1]));

        // Values and Indices sharing one buffer must also be rejected, and the
        // whole shared span must be zeroed: the int32 Indices view is wider than
        // the int8 Values view, so zeroing only one view would leave stale bytes.
        int32_t shared_output_buffer[] = {0x11111111, 0x22222222};
        TensorBase shared_val_tensor({1, 2}, shared_output_buffer, 0, DATA_TYPE_INT8, false);
        TensorBase shared_idx_tensor({1, 2}, shared_output_buffer, 0, DATA_TYPE_INT32, false);
        valid_topk.run({&aliased_in_tensor}, {&shared_val_tensor, &shared_idx_tensor});
        TEST_ASSERT_EQUAL_INT(0, shared_output_buffer[0]);
        TEST_ASSERT_EQUAL_INT(0, shared_output_buffer[1]);
    }
}

TEST_CASE("Test InverseSigmoid runtime contract", "[inverse_sigmoid]")
{
    const float input_data[] = {-0.5f, 0.5f, 1.5f};
    float output_data[] = {123.0f, 123.0f, 123.0f};
    TensorBase input({3}, input_data, 0, DATA_TYPE_FLOAT, false);
    TensorBase output({3}, output_data, 0, DATA_TYPE_FLOAT, false);

    module::InverseSigmoid valid("inverse_sigmoid", 0.1f, MODULE_NON_INPLACE, QUANT_TYPE_FLOAT32);
    valid.run(&input, &output);
    const float expected_bound = logf(10.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -expected_bound, output_data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output_data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_bound, output_data[2]);

    for (float &value : output_data) {
        value = 123.0f;
    }
    module::InverseSigmoid default_quant("default_quant", 0.1f);
    default_quant.run(&input, &output);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -expected_bound, output_data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output_data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_bound, output_data[2]);

    float inplace_data[] = {-0.5f, 0.5f, 1.5f};
    TensorBase inplace_tensor({3}, inplace_data, 0, DATA_TYPE_FLOAT, false);
    module::InverseSigmoid inplace_op("inplace_op", 0.1f, MODULE_INPLACE_CHANGED_BUFFER, QUANT_TYPE_FLOAT32);
    inplace_op.run(&inplace_tensor, &inplace_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -expected_bound, inplace_data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, inplace_data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_bound, inplace_data[2]);

    for (float &value : output_data) {
        value = 123.0f;
    }
    module::InverseSigmoid invalid_eps("invalid_eps", 0.0f, MODULE_NON_INPLACE, QUANT_TYPE_FLOAT32);
    std::vector<std::vector<int>> shapes = {{3}};
    TEST_ASSERT_TRUE(invalid_eps.get_output_shape(shapes).empty());
    invalid_eps.run(&input, &output);
    for (float value : output_data) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, value);
    }

    for (float &value : output_data) {
        value = 123.0f;
    }
    module::InverseSigmoid wrong_quant_type("wrong_quant_type", 0.1f, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    wrong_quant_type.run(&input, &output);
    for (float value : output_data) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, value);
    }

    for (float &value : output_data) {
        value = 123.0f;
    }
    valid.run({&input, &input}, {&output});
    for (float value : output_data) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, value);
    }

    float mismatched_shape_data[] = {123.0f, 123.0f, 123.0f};
    TensorBase mismatched_shape_output({1, 3}, mismatched_shape_data, 0, DATA_TYPE_FLOAT, false);
    valid.run(&input, &mismatched_shape_output);
    for (float value : mismatched_shape_data) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, value);
    }

    std::vector<std::vector<int>> non_positive_shapes = {{0, 3}};
    TEST_ASSERT_TRUE(valid.get_output_shape(non_positive_shapes).empty());
}

static __attribute__((noinline)) void test_rope_s8_int_min_and_contract()
{
    // Pure INT8 SIMD (D=32, half=16) with x2 == INT8_MIN (-128), cos == 0, sin == 127:
    // exact -(-128) * 127 = +16256; shifted right by 7 with HALF_EVEN gives +127.
    alignas(16) int8_t x_s8[32] = {0};
    alignas(16) int8_t cos_s8[16] = {0};
    alignas(16) int8_t sin_s8[16] = {0};
    alignas(16) int8_t y_s8[32] = {0};
    for (int k = 0; k < 16; ++k) {
        x_s8[16 + k] = -128;
        sin_s8[k] = 127;
    }
    TensorBase x_tensor({1, 1, 1, 32}, x_s8, -7, DATA_TYPE_INT8, false);
    TensorBase cos_tensor({1, 1, 16}, cos_s8, -7, DATA_TYPE_INT8, false);
    TensorBase sin_tensor({1, 1, 16}, sin_s8, -7, DATA_TYPE_INT8, false);
    TensorBase y_tensor({1, 1, 1, 32}, y_s8, -7, DATA_TYPE_INT8, false);
    module::RotaryEmbedding rope_s8("rope_s8", 0, 0, 0, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    rope_s8.run({&x_tensor, &cos_tensor, &sin_tensor}, {&y_tensor});
    for (int k = 0; k < 16; ++k) {
        TEST_ASSERT_EQUAL_INT8(127, y_s8[k]);
        TEST_ASSERT_EQUAL_INT8(0, y_s8[16 + k]);
    }

    // Output Y aliasing cos_cache must be rejected without zeroing cos_cache.
    cos_s8[0] = 42;
    TensorBase aliased_y({1, 1, 1, 32}, cos_s8, -7, DATA_TYPE_INT8, false);
    rope_s8.run({&x_tensor, &cos_tensor, &sin_tensor}, {&aliased_y});
    TEST_ASSERT_EQUAL_INT8(42, cos_s8[0]);
}

static __attribute__((noinline)) void test_rope_quant_type_mismatch()
{
    // quant_type mismatch (S8 module fed FLOAT tensors) must zero Y.
    float xf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cf[2] = {1.0f, 1.0f};
    float sf[2] = {0.0f, 0.0f};
    float yf[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    TensorBase xf_t({1, 1, 1, 4}, xf, 0, DATA_TYPE_FLOAT, false);
    TensorBase cf_t({1, 1, 2}, cf, 0, DATA_TYPE_FLOAT, false);
    TensorBase sf_t({1, 1, 2}, sf, 0, DATA_TYPE_FLOAT, false);
    TensorBase yf_t({1, 1, 1, 4}, yf, 0, DATA_TYPE_FLOAT, false);
    module::RotaryEmbedding rope_s8("rope_s8", 0, 0, 0, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    rope_s8.run({&xf_t, &cf_t, &sf_t}, {&yf_t});
    for (float v : yf) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, v);
    }
}

static __attribute__((noinline)) void test_rope_s16_int_min()
{
    // Pure INT16 SIMD (D=16, half=8) with x2 == INT16_MIN (-32768), cos == 0, sin == 32767:
    // exact -(-32768) * 32767 >> 15 == +32767.
    alignas(16) int16_t x_s16[16] = {0};
    alignas(16) int16_t cos_s16[8] = {0};
    alignas(16) int16_t sin_s16[8] = {0};
    alignas(16) int16_t y_s16[16] = {0};
    for (int k = 0; k < 8; ++k) {
        x_s16[8 + k] = -32768;
        sin_s16[k] = 32767;
    }
    TensorBase x_tensor({1, 1, 1, 16}, x_s16, -15, DATA_TYPE_INT16, false);
    TensorBase cos_tensor({1, 1, 8}, cos_s16, -15, DATA_TYPE_INT16, false);
    TensorBase sin_tensor({1, 1, 8}, sin_s16, -15, DATA_TYPE_INT16, false);
    TensorBase y_tensor({1, 1, 1, 16}, y_s16, -15, DATA_TYPE_INT16, false);
    module::RotaryEmbedding rope_s16("rope_s16", 0, 0, 0, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_16BIT);
    rope_s16.run({&x_tensor, &cos_tensor, &sin_tensor}, {&y_tensor});
    for (int k = 0; k < 8; ++k) {
        TEST_ASSERT_EQUAL_INT16(32767, y_s16[k]);
        TEST_ASSERT_EQUAL_INT16(0, y_s16[8 + k]);
    }
}

TEST_CASE("Test RotaryEmbedding runtime contract and INT_MIN SIMD", "[rotary_embedding]")
{
    test_rope_s8_int_min_and_contract();
    test_rope_quant_type_mismatch();
    test_rope_s16_int_min();
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

TEST_CASE("Test dl module API: Attention runtime modes", "[api]")
{
    constexpr int H = 4;
    constexpr int Lq = 17;
    constexpr int Lk = 19;
    constexpr int D = 16;
    const std::vector<int> q_shape = {1, H, Lq, D};
    const std::vector<int> kv_shape = {1, H, Lk, D};
    std::vector<int8_t> q_data(static_cast<size_t>(H * Lq * D));
    std::vector<int8_t> k_data(static_cast<size_t>(H * Lk * D));
    std::vector<int8_t> v_data(static_cast<size_t>(H * Lk * D));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<int8_t>(static_cast<int>((i * 13 + 7) % 31) - 15);
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<int8_t>(static_cast<int>((i * 17 + 3) % 29) - 14);
        v_data[i] = static_cast<int8_t>(static_cast<int>((i * 19 + 11) % 63) - 31);
    }

    const auto run = [&](runtime_mode_t mode) {
        TensorBase q(q_shape, q_data.data(), -6, DATA_TYPE_INT8);
        TensorBase k(kv_shape, k_data.data(), -6, DATA_TYPE_INT8);
        TensorBase v(kv_shape, v_data.data(), -5, DATA_TYPE_INT8);
        TensorBase output(q_shape, nullptr, -5, DATA_TYPE_INT8);
        TEST_ASSERT_NOT_NULL(output.data);
        module::Attention attention(
            "attention", false, 0.f, false, AttentionLayout::BHND, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
        attention.run({&q, &k, &v}, {&output}, mode);
        const int8_t *data = static_cast<const int8_t *>(output.data);
        return std::vector<int8_t>(data, data + output.get_size());
    };

    const std::vector<int8_t> single = run(RUNTIME_MODE_SINGLE_CORE);
    const std::vector<int8_t> multi = run(RUNTIME_MODE_MULTI_CORE);
    const std::vector<int8_t> automatic = run(RUNTIME_MODE_AUTO);
    TEST_ASSERT_EQUAL_INT8_ARRAY(single.data(), multi.data(), static_cast<int>(single.size()));
    TEST_ASSERT_EQUAL_INT8_ARRAY(single.data(), automatic.data(), static_cast<int>(single.size()));

    TEST_ASSERT_NULL(module::Attention::deserialize(nullptr, "attention"));

    TensorBase q(q_shape, q_data.data(), -6, DATA_TYPE_INT8);
    TensorBase k(kv_shape, k_data.data(), -6, DATA_TYPE_INT8);
    TensorBase v(kv_shape, v_data.data(), -5, DATA_TYPE_INT8);
    TensorBase output(q_shape, nullptr, -5, DATA_TYPE_INT8);
    module::Attention attention(
        "attention", false, 0.f, false, AttentionLayout::BHND, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);

    std::memset(output.data, 0x5a, output.get_bytes());
    attention.run({&q, &k}, {&output}, RUNTIME_MODE_SINGLE_CORE);
    for (int i = 0; i < output.get_size(); ++i) {
        TEST_ASSERT_EQUAL_INT8(0, output.get_element_ptr<int8_t>()[i]);
    }

    std::memset(output.data, 0x5a, output.get_bytes());
    attention.run({&q, &k, &v, &q, &k}, {&output}, RUNTIME_MODE_SINGLE_CORE);
    for (int i = 0; i < output.get_size(); ++i) {
        TEST_ASSERT_EQUAL_INT8(0, output.get_element_ptr<int8_t>()[i]);
    }
}

// Run int16 softmax on one pair. output is float.
static void run_softmax_i16_pair(const int16_t input_data[2], int exponent, float output_data[2])
{
    output_data[0] = 123.f;
    output_data[1] = 123.f;
    TensorBase input({2}, input_data, exponent, DATA_TYPE_INT16, false);
    TensorBase output({2}, output_data, 0, DATA_TYPE_FLOAT, false);
    module::Softmax softmax("softmax_i16", -1, MODULE_NON_INPLACE, QUANT_TYPE_SYMM_16BIT);
    softmax.run(&input, &output);
}

TEST_CASE("Test Softmax int16 exp table cutoff", "[api]")
{
    // exponent -9: hi_step = 256 * 2^-9 = 0.5, hi_len = int(88 / 0.5) + 2 = 178.
    // A full 257-entry table would make hi >= hi_len unreachable (max hi is 255).
    // The wide_range ops_test case only hopes calibration lands here; this case
    // fixes the exponent and the index.
    const int16_t beyond[] = {32767, -12801}; // n = 45568, hi = 178 >= 178
    const int16_t inside[] = {32767, -12545}; // n = 45312, hi = 177, last stored index
    const int16_t span[] = {32767, -32768};   // n = 65535, hi = 255
    float output[2];

    // exp(-89) is still a subnormal, so a zero here is the table cutoff, not expf.
    TEST_ASSERT_TRUE(expf(-89.f) > 0.f);
    run_softmax_i16_pair(beyond, -9, output);
    TEST_ASSERT_TRUE(output[0] == 1.f);
    TEST_ASSERT_TRUE(output[1] == 0.f);

    // The previous high index is stored. 1 + exp(-88.5) rounds to 1, so the
    // second output is exp(-88.5) itself.
    const float kept = expf(-88.5f);
    TEST_ASSERT_TRUE(kept > 0.f);
    run_softmax_i16_pair(inside, -9, output);
    TEST_ASSERT_TRUE(output[0] == 1.f);
    TEST_ASSERT_TRUE(output[1] > 0.f);
    TEST_ASSERT_TRUE(fabsf(output[1] - kept) / kept < 1e-5f);

    // Same codes at exponent -10: hi_len stays 257, hi = 178 is in range, and
    // the second output must not be forced to zero.
    const float fine = expf(-44.5f);
    TEST_ASSERT_TRUE(fine > 0.f);
    run_softmax_i16_pair(beyond, -10, output);
    TEST_ASSERT_TRUE(output[0] == 1.f);
    TEST_ASSERT_TRUE(output[1] > 0.f);
    TEST_ASSERT_TRUE(fabsf(output[1] - fine) / fine < 1e-5f);

    // Full int16 span. exp(-128) is already 0, so this matches a reference
    // softmax; the pair above is what fails if the cutoff is not applied.
    run_softmax_i16_pair(span, -9, output);
    TEST_ASSERT_TRUE(output[0] == 1.f);
    TEST_ASSERT_TRUE(output[1] == 0.f);
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
