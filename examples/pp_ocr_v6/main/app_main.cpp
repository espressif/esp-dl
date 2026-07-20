#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "pp_ocr_v6.hpp"
#if CONFIG_PP_OCR_V6_MODEL_IN_SDCARD
#include "bsp/esp-bsp.h"
#endif

static const char *TAG = "pp_ocr_v6";

extern const uint8_t pp_ocr_v6_jpg_start[] asm("_binary_pp_ocr_v6_jpg_start");
extern const uint8_t pp_ocr_v6_jpg_end[] asm("_binary_pp_ocr_v6_jpg_end");

static void log_ocr_result(const pp_ocr_v6::OCRResult &res)
{
    ESP_LOGI(TAG,
             "text=\"%s\", score=%.4f, box=[%d,%d %d,%d %d,%d %d,%d], det_score=%.4f",
             res.text.c_str(),
             res.score,
             res.box.points[0],
             res.box.points[1],
             res.box.points[2],
             res.box.points[3],
             res.box.points[4],
             res.box.points[5],
             res.box.points[6],
             res.box.points[7],
             res.box.score);
}

extern "C" void app_main(void)
{
#if CONFIG_PP_OCR_V6_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)pp_ocr_v6_jpg_start,
                                      .data_len = (size_t)(pp_ocr_v6_jpg_end - pp_ocr_v6_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    if (!img.data) {
        ESP_LOGE(TAG, "Failed to decode embedded JPEG");
#if CONFIG_PP_OCR_V6_MODEL_IN_SDCARD
        ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
        return;
    }

    auto *ocr = new pp_ocr_v6::PPOCRV6();
    auto results = ocr->run(img);
    for (const auto &res : results) {
        log_ocr_result(res);
    }
    ESP_LOGI(TAG, "OCR results: %u", static_cast<unsigned>(results.size()));

    delete ocr;
    heap_caps_free(img.data);
#if CONFIG_PP_OCR_V6_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
