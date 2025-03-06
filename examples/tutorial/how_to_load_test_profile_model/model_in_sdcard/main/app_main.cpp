#include "dl_model_base.hpp"
#include "bsp/esp-bsp.h"

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(bsp_sdcard_mount());
    dl::Model *model = new dl::Model("/sdcard/model.espdl", fbs::MODEL_LOCATION_IN_SDCARD);

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
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
}
