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
    ESP_LOGI(TAG, "model_num = %d\n", model_num);
    dl::tool::Latency latency;
    for (int i = 0; i < model_num; i++) {
        fbs::FbsModel *fbs_model = fbs_loader->load(i);
        Model *model = new Model(fbs_model);
        model->print();
        TEST_ASSERT_EQUAL(ESP_OK, model->test());
        // model->print_module_info(model->get_module_info(), true);

        delete model;
        delete fbs_model;
    }

    delete fbs_loader;
    dl::module::ModuleCreator *module_creator = dl::module::ModuleCreator::get_instance();
    module_creator->clear();

    int total_ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    int internal_ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "total ram consume: %d B, ", (total_ram_size_before - total_ram_size_after));
    ESP_LOGI(TAG, "internal ram consume: %d B, ", (internal_ram_size_before - internal_ram_size_after));
    ESP_LOGI(TAG, "psram consume: %d B\n", (psram_size_before - psram_size_after));
    TEST_ASSERT_EQUAL(psram_size_before, psram_size_after);
    ESP_LOGI(TAG, "exit app_main");
}
