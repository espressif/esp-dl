#include "dl_model_base.hpp"

extern "C" void app_main(void)
{
    // dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    // Keep parameter in flash, saves PSRAM/internal RAM, lower performance.
    dl::Model *model =
        new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);

    // Use test inputs and test outputs embedded in model when exported with esp-ppq to test if inference result is
    // correct.
    ESP_ERROR_CHECK(model->test());

    // print summary in module topological order
    model->profile();
    // print summary in module latency decreasing order
    // model->profile(true);

    // profile() is the combination of profile_memroy() and profile_module().
    // model->profile_memory();
    // model->profile_module();
    // model->profile_module(true);
    delete model;
}
