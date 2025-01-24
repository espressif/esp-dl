#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "fbs_loader.hpp"
#include "unity.h"
#include <type_traits>

static const char *TAG = "TEST_ESPDL_MODEL";

// using namespace fbs;
using namespace dl;

void compare_test_outputs(Model *model, std::map<std::string, TensorBase *> infer_outputs)
{
    if (!model) {
        return;
    }

    fbs::FbsModel *fbs_model_instance = model->get_fbs_model();
    fbs_model_instance->load_map();
    int i = 0;
    for (auto iter = infer_outputs.begin(); iter != infer_outputs.end(); iter++) {
        ESP_LOGI(TAG, "output index: %d, name: %s", i++, iter->first.c_str());
        std::string infer_output_name = iter->first;
        TensorBase *infer_output = iter->second;
        if (infer_output) {
            TensorBase *ground_truth_tensor = fbs_model_instance->get_test_output_tensor(infer_output_name);
            TEST_ASSERT_EQUAL_MESSAGE(true, ground_truth_tensor != nullptr, "The test output tensor is not found");
            if (ground_truth_tensor->get_dtype() == DATA_TYPE_INT16 ||
                ground_truth_tensor->get_dtype() == DATA_TYPE_UINT16) {
                // The int16 quantization cannot be fully aligned, and there may be rounding errors of +-1.
                TEST_ASSERT_EQUAL_MESSAGE(true,
                                          infer_output->equal(ground_truth_tensor, 1 + 1e-5, true),
                                          "The output tensor is not equal to the ground truth");
            } else {
                TEST_ASSERT_EQUAL_MESSAGE(true,
                                          infer_output->equal(ground_truth_tensor, 1e-5, true),
                                          "The output tensor is not equal to the ground truth");
            }

            delete ground_truth_tensor;
        }
    }
}

std::map<std::string, TensorBase *> get_graph_test_inputs(Model *model)
{
    std::map<std::string, TensorBase *> test_inputs;

    if (!model) {
        return test_inputs;
    }

    fbs::FbsModel *parser_instance = model->get_fbs_model();
    parser_instance->load_map();
    std::map<std::string, TensorBase *> graph_inputs = model->get_inputs();
    for (auto graph_inputs_iter = graph_inputs.begin(); graph_inputs_iter != graph_inputs.end(); graph_inputs_iter++) {
        std::string input_name = graph_inputs_iter->first;
        TensorBase *test_input = parser_instance->get_test_input_tensor(input_name);
        if (test_input) {
            test_inputs.emplace(input_name, test_input);
        }
    }

    return test_inputs;
}

std::map<std::string, TensorBase *> get_graph_user_outputs(Model *model)
{
    std::map<std::string, TensorBase *> user_outputs;

    if (!model) {
        return user_outputs;
    }

    fbs::FbsModel *parser_instance = model->get_fbs_model();
    parser_instance->load_map();
    std::vector<std::string> test_outputs_name = parser_instance->get_test_outputs_name();

    for (int i = 0; i < test_outputs_name.size(); i++) {
        TensorBase *output = model->get_intermediate(test_outputs_name[i]);
        if (output) {
            TensorBase *user_output = new TensorBase(
                output->get_shape(), nullptr, output->get_exponent(), output->get_dtype(), true, output->get_caps());
            user_outputs.emplace(test_outputs_name[i], user_output);
        }
    }
    return user_outputs;
}

TEST_CASE("Test espdl model", "[dl_model]")
{
    ESP_LOGI(TAG, "get into app_main");
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    int internal_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);

    fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    if (!fbs_loader) {
        ESP_LOGE(TAG, "Can not find any models from partition: %s", "model");
        return;
    }
    int model_num = fbs_loader->get_model_num();
    int model_run_time = 0;
    ESP_LOGI(TAG, "model_num = %d\n", model_num);
    dl::tool::Latency latency;
    for (int i = 0; i < model_num; i++) {
        fbs::FbsModel *fbs_model = fbs_loader->load(i);
        Model *model = new Model(fbs_model);
        std::map<std::string, TensorBase *> graph_test_inputs = get_graph_test_inputs(model);
        std::map<std::string, TensorBase *> graph_user_outputs = get_graph_user_outputs(model);
        model->print();
        latency.start();
        model->run(graph_test_inputs, RUNTIME_MODE_SINGLE_CORE, graph_user_outputs);
        latency.end();
        model_run_time += latency.get_period();
        ESP_LOGI(TAG, "model index:%d  run time:%d us\n", i, latency.get_period());

        ::compare_test_outputs(model, graph_user_outputs.empty() ? model->get_outputs() : graph_user_outputs);
        for (auto graph_test_inputs_iter = graph_test_inputs.begin(); graph_test_inputs_iter != graph_test_inputs.end();
             graph_test_inputs_iter++) {
            if (graph_test_inputs_iter->second) {
                delete graph_test_inputs_iter->second;
            }
        }
        graph_test_inputs.clear();

        for (auto graph_user_outputs_iter = graph_user_outputs.begin();
             graph_user_outputs_iter != graph_user_outputs.end();
             graph_user_outputs_iter++) {
            if (graph_user_outputs_iter->second) {
                delete graph_user_outputs_iter->second;
            }
        }
        graph_user_outputs.clear();

        delete model;
        delete fbs_model;
    }

    delete fbs_loader;
    dl::module::ModuleCreator *module_creator = dl::module::ModuleCreator::get_instance();
    module_creator->clear();

    int total_ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    int internal_ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "total run time: %d us, ", model_run_time);
    ESP_LOGI(TAG, "total ram consume: %d B, ", (total_ram_size_before - total_ram_size_after));
    ESP_LOGI(TAG, "internal ram consume: %d B, ", (internal_ram_size_before - internal_ram_size_after));
    ESP_LOGI(TAG, "psram consume: %d B\n", (psram_size_before - psram_size_after));
    TEST_ASSERT_EQUAL(psram_size_before, psram_size_after);
    ESP_LOGI(TAG, "exit app_main");
}
