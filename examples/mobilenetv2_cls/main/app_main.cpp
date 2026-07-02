#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "imagenet_cls.hpp"
#if CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD
#include "bsp/esp-bsp.h"
#endif

extern const uint8_t cat_jpg_start[] asm("_binary_cat_jpg_start");
extern const uint8_t cat_jpg_end[] asm("_binary_cat_jpg_end");
const char *TAG = "mobilenetv2_cls";

static constexpr int WARMUP_RUNS = 1;
static constexpr int BENCH_RUNS = 10;

static void benchmark(ImageNetCls *cls, const dl::image::img_t &img, dl::runtime_mode_t mode, const char *mode_name)
{
    for (int i = 0; i < WARMUP_RUNS; i++) {
        cls->run(img, mode);
    }

    int64_t total_us = 0;
    std::vector<dl::cls::result_t> results;
    for (int i = 0; i < BENCH_RUNS; i++) {
        int64_t start = esp_timer_get_time();
        auto &res = cls->run(img, mode);
        total_us += esp_timer_get_time() - start;
        results = res;
        printf("run %d\n", i);
        fflush(stdout);
    }

    float avg_ms = (float)total_us / BENCH_RUNS / 1000.0f;
    ESP_LOGI(TAG, "===== %s: avg inference latency %.3f ms over %d runs =====", mode_name, avg_ms, BENCH_RUNS);
    for (const auto &res : results) {
        ESP_LOGI(TAG, "category: %s, score: %f", res.cat_name, res.score);
    }
}

extern "C" void app_main(void)
{
#if CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)cat_jpg_start, .data_len = (size_t)(cat_jpg_end - cat_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ImageNetCls *cls = new ImageNetCls();

    benchmark(cls, img, dl::RUNTIME_MODE_SINGLE_CORE, "single-core");
    benchmark(cls, img, dl::RUNTIME_MODE_MULTI_CORE, "multi-core");

    delete cls;
    heap_caps_free(img.data);

#if CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
