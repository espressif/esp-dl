#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "fbs_loader.hpp"
#include "unity.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <type_traits>

static const char *TAG = "TEST_ESPDL_MODEL";

// using namespace fbs;
using namespace dl;

// The CI target-test runners are multi-concurrent: one host drives several
// boards of the same chip and hands out a free USB slot per job, so a given
// test shard is measured on an arbitrary board every pipeline. Board-to-board
// spread (flash/PSRAM part, chip revision) is a systematic offset of a few
// percent, which is enough to trip the perf gate on its own. Stamping the
// factory eFuse MAC into every BENCH record lets the host side keep one
// baseline per physical board instead of one per chip family.
static std::string bench_board_id()
{
    // 8 bytes: esp_efuse_mac_get_default() writes an EUI-64 on targets with
    // 802.15.4 support. Only the 48-bit MAC-48 prefix is needed to tell the
    // boards apart.
    uint8_t mac[8] = {0};
    if (esp_efuse_mac_get_default(mac) != ESP_OK) {
        return "unknown";
    }
    char buffer[13];
    snprintf(buffer, sizeof(buffer), "%02x%02x%02x%02x%02x%02x", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return std::string(buffer);
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
    ESP_LOGI(TAG, "model_num = %d\n", model_num);
    std::string board_id = bench_board_id();
    for (int i = 0; i < model_num; i++) {
        fbs::FbsModel *fbs_model = fbs_loader->load(i);
        Model *model = new Model(fbs_model);
        model->print();
        TEST_ASSERT_EQUAL(ESP_OK, model->test());
        // Measurement methodology: warmup runs bring the flash/PSRAM caches to a
        // steady state, then each timed sample is a full model run. On-chip noise
        // (cache misses, FreeRTOS tick preemption, PSRAM refresh, ...) can only
        // ever ADD time to a sample, so the MINIMUM sample converges to the true
        // compute time and is far more stable run-to-run than the median. The
        // benchmark gate therefore compares min-to-min; median/mean are logged
        // for reference only.
        constexpr int kWarmup = 4;
        constexpr int kIters = 12;
        uint32_t samples[kIters];
        uint64_t total_us = 0;
        for (int n = 0; n < kWarmup; n++) {
            model->run();
        }
        for (int n = 0; n < kIters; n++) {
            int64_t start_us = esp_timer_get_time();
            model->run();
            samples[n] = static_cast<uint32_t>(esp_timer_get_time() - start_us);
            total_us += samples[n];
        }
        std::sort(samples, samples + kIters);
        float min_us = samples[0];
        float median_us = (samples[kIters / 2 - 1] + samples[kIters / 2]) / 2.0f;
        // All cases exported from ONNX share the same graph name (e.g. "main_graph"),
        // so use the packed entry name (the per-case model file name) to keep every
        // benchmark record distinguishable. Fall back to the graph name for non-packed models.
        std::string bench_name = fbs_loader->get_model_name(i);
        if (bench_name.empty()) {
            bench_name = fbs_model->get_model_name();
        }
        ESP_LOGI(TAG,
                 "BENCH name=%s board=%s iters=%d min_us=%.3f median_us=%.3f mean_us=%.3f",
                 bench_name.c_str(),
                 board_id.c_str(),
                 kIters,
                 min_us,
                 median_us,
                 total_us / static_cast<float>(kIters));

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
