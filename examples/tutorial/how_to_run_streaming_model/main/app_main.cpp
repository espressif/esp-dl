#include "dl_model_base.hpp"
#include "dl_tool.hpp"
#include <cmath>
#include <vector>
#if CONFIG_IDF_TARGET_ESP32P4
#include "test_data_esp32p4.hpp"
#elif CONFIG_IDF_TARGET_ESP32S3
#include "test_data_esp32s3.hpp"
#endif
#include "streaming_model.hpp"

static const char *TAG = "APP_MAIN";

extern "C" void app_main(void)
{
    StreamingModel *p_model_0 = new StreamingModel("model_0_ishap_1_64_36_kshap_3_s8_streaming.espdl" /*model_name*/,
                                                   "input.1" /*input_name*/,
                                                   "42" /*output_name*/,
                                                   "cache" /*input_cache_name*/,
                                                   "31" /*output_cache_name*/);

    StreamingModel *p_model_1 = new StreamingModel("model_1_ishap_1_64_36_kshap_3_s8_streaming.espdl" /*model_name*/,
                                                   "input.1" /*input_name*/,
                                                   "7" /*output_name*/,
                                                   "" /*input_cache_name*/,
                                                   "" /*output_cache_name*/);

    StreamingModel *p_model_2 = new StreamingModel("model_2_ishap_1_64_36_kshap_3_s8_streaming.espdl" /*model_name*/,
                                                   "input.1" /*input_name*/,
                                                   "32" /*output_name*/,
                                                   "cache" /*input_cache_name*/,
                                                   "27" /*output_cache_name*/);

    dl::TensorBase *input_tensor = new dl::TensorBase(p_model_0->get_inputs()->get_shape() /*shape*/,
                                                      nullptr /*element*/,
                                                      p_model_0->get_inputs()->get_exponent(),
                                                      p_model_0->get_inputs()->get_dtype());

    // Because the first layer of model_0 in the example is conv, so the data layout follows NWC format.
    dl::TensorBase *one_step_input_tensor =
        new dl::TensorBase({1, 1, p_model_0->get_inputs()->get_shape()[2]} /*shape*/,
                           test_inputs /*element*/,
                           p_model_0->get_inputs()->get_exponent(),
                           p_model_0->get_inputs()->get_dtype(),
                           false);

    // The output shape of the model is consistent with the input.
    int test_outputs_size = 1 * (TIME_SERIES_LENGTH - STREAMING_WINDOW_SIZE) * TEST_INPUT_CHANNELS;
    int8_t *output_buffer = new int8_t[test_outputs_size];
    dl::TensorBase *output = nullptr;
    int step_index = 1;

    for (int i = 0; i < TIME_SERIES_LENGTH; i++) {
        one_step_input_tensor->set_element_ptr(const_cast<int8_t *>(&test_inputs[i][0]));
        // Because the first layer of model_0 in the example is conv, so the time series dimension is 1.
        input_tensor->push(one_step_input_tensor, 1);

        if (i < (input_tensor->get_shape()[1] - 1)) {
            // The data is populated to facilitate accuracy testing, as this step is omitted in actual deployment.
            continue;
        } else {
            switch (step_index) {
            case 1:
                output = (*p_model_0)(input_tensor);
                step_index++;
                break;
            case 2:
                output = (*p_model_1)(output);
                step_index++;
                break;
            case 3:
                output = (*p_model_2)(output);
                dl::tool::copy_memory(output_buffer + (i / 3 - 1) * STREAMING_WINDOW_SIZE * TEST_INPUT_CHANNELS,
                                      output->data,
                                      STREAMING_WINDOW_SIZE * TEST_INPUT_CHANNELS);
                step_index = 1;
                break;
            default:
                break;
            }
        }
    }

    bool pass = true;
    for (int i = 0; i < test_outputs_size; i++) {
        if (output_buffer[i] != test_outputs[i]) {
            pass = false;
            ESP_LOGE(TAG,
                     "Inconsistent output, index: %d, output_buffer: %d, test_outputs: %d",
                     i,
                     output_buffer[i],
                     test_outputs[i]);
        }
    }

    delete[] output_buffer;
    delete one_step_input_tensor;
    delete input_tensor;
    delete p_model_2;
    delete p_model_1;
    delete p_model_0;

    ESP_LOGW(TAG, "Streaming test pass: %s", pass ? "true" : "false");
}
