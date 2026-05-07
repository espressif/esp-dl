#include "dl_model_base.hpp"
#include "dl_module_add.hpp"
#include "dl_module_creator.hpp"
#include "dl_module_relu.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <cstring>
#include <map>
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
        int8_t in2 = input1->get_element<int8_t>(i % 64);
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
