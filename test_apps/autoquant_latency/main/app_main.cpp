/**
 * Latency-only benchmark for AutoQuant: loads the model from the "model" flash partition,
 * runs inference in a loop, prints lines "run:<ms> ms" for host-side parsing (see auto_quant/latency.py).
 * No Unity, no model->test() (no export_test_values required).
 */
#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "esp_log.h"

using namespace dl;

static const char *TAG = "autoquant_latency";

extern "C" void app_main(void)
{
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

    dl::tool::Latency latency;
    constexpr int kRuns = 10;
    int64_t results[kRuns];

    for (int i = 0; i < kRuns; i++) {
        latency.start();
        model->run();
        latency.end();
        results[i] = latency.get_period();
    }

    for (int i = 0; i < kRuns; i++) {
        printf("run:%ld ms\n", (long)(results[i] / 1000));
    }

    delete model;
    module::ModuleCreator::get_instance()->clear();
    ESP_LOGI(TAG, "done");
}
