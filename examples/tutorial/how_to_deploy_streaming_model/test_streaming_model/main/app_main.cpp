#include "dl_model_base.hpp"
#include <cmath>

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");
static const char *TAG = "streaming_model_example";

dl::TensorBase *run_streaming_model(dl::Model *model, dl::TensorBase *test_input)
{
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;

    if (!test_input) {
        ESP_LOGE(TAG,
                 "Model input doesn't have a corresponding test input. Please enable export_test_values option "
                 "in esp-ppq when export espdl model.");
        return nullptr;
    }

    int test_input_size = test_input->get_bytes();
    uint8_t *test_input_ptr = (uint8_t *)test_input->data;
    int model_input_size = model_input->get_bytes();
    uint8_t *model_input_ptr = (uint8_t *)model_input->data;
    int chunks = test_input_size / model_input_size;
    for (int i = 0; i < chunks; i++) {
        // assign chunk data to model input
        memcpy(model_input_ptr, test_input_ptr + i * model_input_size, model_input_size);
        model->run(model_input);
    }

    return model_output;
}

extern "C" void app_main(void)
{
    // Run non-streaming model
    dl::Model *model = new dl::Model((const char *)model_espdl, "model.espdl", fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    dl::TensorBase *output = model->get_outputs().begin()->second;
    fbs::FbsModel *fbs_model = model->get_fbs_model();
    fbs_model->load_map();
    std::string input_name = model->get_inputs().begin()->first;
    dl::TensorBase *test_input = fbs_model->get_test_input_tensor(input_name);
    model->test();
    model->print();

    // Run streaming model with the same input
    dl::Model *streaming_model =
        new dl::Model((const char *)model_espdl, "streaming_model.espdl", fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    dl::TensorBase *streaming_output = run_streaming_model(streaming_model, test_input);
    streaming_model->print();

    // Compare the last step outputs
    uint8_t *output_data = (uint8_t *)output->data;
    int offset = output->get_bytes() - streaming_output->get_bytes();
    dl::TensorBase *output_sliced = new dl::TensorBase(
        streaming_output->get_shape(), output_data + offset, output->get_exponent(), output->get_dtype());

    if (streaming_output->equal(output_sliced)) {
        ESP_LOGI(TAG, "Streaming model output matches non-streaming model output.");
    } else {
        ESP_LOGE(TAG, "Streaming model output does not match non-streaming model output.");
    }

    delete model;
    delete streaming_model;
    delete output_sliced;
}
