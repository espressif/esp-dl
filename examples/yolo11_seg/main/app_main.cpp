#include "coco_seg.hpp"
#include "dl_image.hpp"
#include "esp_log.h"
#include "bsp/esp-bsp.h"

extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[] asm("_binary_bus_jpg_end");
const char *TAG = "yolo11n-seg";

static constexpr float k_mask_alpha = 0.5f;
static constexpr const char *k_result_bmp_path = "/sdcard/yolo11_seg_result.bmp";

static std::vector<uint8_t> get_class_color_rgb(int category)
{
    static const uint8_t palette[][3] = {
        {230, 25, 75},   {60, 180, 75},   {255, 225, 25},  {0, 130, 200},   {245, 130, 48},
        {145, 30, 180},  {70, 240, 240},  {240, 50, 230},  {210, 245, 60},  {250, 190, 190},
        {0, 128, 128},   {230, 190, 255}, {170, 110, 40},  {255, 250, 200}, {128, 0, 0},
        {170, 255, 195}, {128, 128, 0},   {255, 215, 180}, {0, 0, 128},     {128, 128, 128},
    };
    const size_t n = sizeof(palette) / sizeof(palette[0]);
    return {palette[category % n][0], palette[category % n][1], palette[category % n][2]};
}

static void blend_instance_mask(dl::image::img_t &img, const dl::detect::result_t &res, float alpha)
{
    if (res.mask.empty()) {
        return;
    }

    int x1 = res.box[0];
    int y1 = res.box[1];
    int x2 = res.box[2];
    int y2 = res.box[3];
    int box_w = x2 - x1;
    int box_h = y2 - y1;
    if (box_w <= 0 || box_h <= 0) {
        return;
    }

    auto color = get_class_color_rgb(res.category);
    uint8_t *data = static_cast<uint8_t *>(img.data);
    int row_step = img.row_step();
    float inv_alpha = 1.f - alpha;

    for (int oy = 0; oy < box_h; oy++) {
        int y = y1 + oy;
        if (y < 0 || y >= (int)img.height) {
            continue;
        }
        for (int ox = 0; ox < box_w; ox++) {
            if (!res.mask[oy * box_w + ox]) {
                continue;
            }
            int x = x1 + ox;
            if (x < 0 || x >= (int)img.width) {
                continue;
            }
            uint8_t *px = data + y * row_step + x * 3;
            px[0] = (uint8_t)(px[0] * inv_alpha + color[0] * alpha);
            px[1] = (uint8_t)(px[1] * inv_alpha + color[1] * alpha);
            px[2] = (uint8_t)(px[2] * inv_alpha + color[2] * alpha);
        }
    }
}

static void draw_segmentation_results(dl::image::img_t &img, const std::list<dl::detect::result_t> &results)
{
    for (const auto &res : results) {
        blend_instance_mask(img, res, k_mask_alpha);
    }

    for (const auto &res : results) {
        int x1 = res.box[0];
        int y1 = res.box[1];
        int x2 = res.box[2];
        int y2 = res.box[3];
        if (x2 <= x1 + 1 || y2 <= y1 + 1) {
            continue;
        }
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min((int)img.width - 1, x2);
        y2 = std::min((int)img.height - 1, y2);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        auto color = get_class_color_rgb(res.category);
        dl::image::draw_hollow_rectangle(img, x1, y1, x2, y2, color, 2);
    }
}

static bool mount_sdcard_for_app(bool required)
{
#if CONFIG_COCO_SEG_MODEL_IN_SDCARD
    (void)required;
    ESP_ERROR_CHECK(bsp_sdcard_mount());
    ESP_LOGI(TAG, "SD card mounted (model location: sdcard)");
    return true;
#else
    esp_err_t ret = bsp_sdcard_mount();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "SD card mounted (model location: flash, export BMP enabled)");
        return true;
    }
    if (required) {
        ESP_LOGE(TAG, "SD card mount failed: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGW(TAG, "SD card mount failed: %s, inference continues without BMP export", esp_err_to_name(ret));
    }
    return false;
#endif
}

static void unmount_sdcard_if_mounted(bool mounted)
{
    if (mounted) {
        ESP_ERROR_CHECK(bsp_sdcard_unmount());
    }
}

extern "C" void app_main(void)
{
    bool sdcard_mounted = mount_sdcard_for_app(/*required=*/false);

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)bus_jpg_start, .data_len = (size_t)(bus_jpg_end - bus_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    COCOSeg *segment = new COCOSeg();
    auto &seg_results = segment->run(img);

    for (const auto &res : seg_results) {
        int box_w = res.box[2] - res.box[0];
        int box_h = res.box[3] - res.box[1];
        int mask_pixels = 0;
        for (uint8_t v : res.mask) {
            mask_pixels += v > 0;
        }
        ESP_LOGI(TAG,
                 "[category: %d, score: %f, x1: %d, y1: %d, x2: %d, y2: %d, mask_pixels: %d, box_area: %d]",
                 res.category,
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3],
                 mask_pixels,
                 box_w * box_h);
    }

    draw_segmentation_results(img, seg_results);

    if (sdcard_mounted) {
        if (dl::image::write_bmp(img, k_result_bmp_path) == ESP_OK) {
            ESP_LOGI(TAG, "Visualization saved to %s", k_result_bmp_path);
        } else {
            ESP_LOGE(TAG, "Failed to save visualization to %s", k_result_bmp_path);
        }
    } else {
        ESP_LOGI(TAG,
                 "Segmentation overlay drawn in memory (mask alpha=%.1f), SD card not available for BMP export",
                 k_mask_alpha);
    }

    delete segment;
    heap_caps_free(img.data);

    unmount_sdcard_if_mounted(sdcard_mounted);
}
