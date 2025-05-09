#include "dl_image.hpp"
#include "unity.h"
#include "bsp/esp-bsp.h"

extern const uint8_t color_320x240_jpg_start[] asm("_binary_color_320x240_jpg_start");
extern const uint8_t color_320x240_jpg_end[] asm("_binary_color_320x240_jpg_end");
extern const uint8_t color_405x540_jpg_start[] asm("_binary_color_405x540_jpg_start");
extern const uint8_t color_405x540_jpg_end[] asm("_binary_color_405x540_jpg_end");
extern const uint8_t gray_320x240_jpg_start[] asm("_binary_gray_320x240_jpg_start");
extern const uint8_t gray_320x240_jpg_end[] asm("_binary_gray_320x240_jpg_end");

using namespace dl::image;

TEST_CASE("Test sw decode/encode", "[dl_image]")
{
    auto ret = bsp_sdcard_mount();
    bool mount = (ret == ESP_OK);
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    int64_t start, end;
    img_t img;
    jpeg_img_t encoded_img;

    // test decode rgb888
    start = esp_timer_get_time();
    img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);
    end = esp_timer_get_time();
    printf("sw_decode_rgb888_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/sw_decode_rgb888_color_405x540.bmp", DL_IMAGE_CAP_RGB_SWAP));
    }
    // test encode rgb888
    start = esp_timer_get_time();
    encoded_img = sw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("sw_encode_rgb888_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/sw_encode_rgb888_color_405x540.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test decode rgb565LE
    start = esp_timer_get_time();
    img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    end = esp_timer_get_time();
    printf("sw_decode_rgb565_le_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/sw_decode_rgb565_le_color_405x540.bmp", DL_IMAGE_CAP_RGB_SWAP));
    }
    // test_encode rgb565LE
    start = esp_timer_get_time();
    encoded_img = sw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("sw_encode_rgb565_le_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/sw_encode_rgb565_le_color_405x540.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test decode rgb565BE
    start = esp_timer_get_time();
    img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    end = esp_timer_get_time();
    printf("sw_decode_rgb565_be_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img,
                                  "/sdcard/sw_decode_rgb565_be_color_405x540.bmp",
                                  DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN));
    }
    // test_encode rgb565BE
    start = esp_timer_get_time();
    encoded_img = sw_encode_jpeg(img, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    end = esp_timer_get_time();
    printf("sw_encode_rgb565_be_color_405x540: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/sw_encode_rgb565_be_color_405x540.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

#if CONFIG_IDF_TARGET_ESP32P4
    // test_encode gray
    jpeg_img = {.data = (void *)gray_320x240_jpg_start,
                .data_len = (size_t)(gray_320x240_jpg_end - gray_320x240_jpg_start)};
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_GRAY);
    start = esp_timer_get_time();
    encoded_img = sw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("sw_encode_gray_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/sw_encode_gray_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);
#endif

    if (mount) {
        ESP_ERROR_CHECK(bsp_sdcard_unmount());
    }
}

#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
TEST_CASE("Test hw decode/encode", "[dl_image]")
{
    auto ret = bsp_sdcard_mount();
    bool mount = (ret == ESP_OK);
    jpeg_img_t jpeg_img = {.data = (void *)color_320x240_jpg_start,
                           .data_len = (size_t)(color_320x240_jpg_end - color_320x240_jpg_start)};
    int64_t start, end;
    img_t img;
    jpeg_img_t encoded_img;

    // test decode rgb888 RGB
    start = esp_timer_get_time();
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);
    end = esp_timer_get_time();
    printf("hw_decode_rgb888rgb_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/hw_decode_rgb888rgb_color_320x240.bmp", DL_IMAGE_CAP_RGB_SWAP));
    }
    // test encode rgb888 RGB
    start = esp_timer_get_time();
    encoded_img = hw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("hw_encode_rgb888rgb_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/hw_encode_rgb888rgb_color_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test decode rgb888 BGR
    start = esp_timer_get_time();
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888, DL_IMAGE_CAP_RGB_SWAP);
    end = esp_timer_get_time();
    printf("hw_decode_rgb888bgr_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/hw_decode_rgb888bgr_color_320x240.bmp"));
    }
    // test encode rgb888 BGR
    start = esp_timer_get_time();
    encoded_img = hw_encode_jpeg(img, DL_IMAGE_CAP_RGB_SWAP);
    end = esp_timer_get_time();
    printf("hw_encode_rgb888bgr_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/hw_encode_rgb888bgr_color_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test decode rgb565LE
    start = esp_timer_get_time();
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    end = esp_timer_get_time();
    printf("hw_decode_rgb565_le_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/hw_decode_rgb565_le_color_320x240.bmp", DL_IMAGE_CAP_RGB_SWAP));
    }
    // test_encode rgb565LE
    start = esp_timer_get_time();
    encoded_img = hw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("hw_encode_rgb565_le_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/hw_encode_rgb565_le_color_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test decode rgb565BE
    start = esp_timer_get_time();
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    end = esp_timer_get_time();
    printf("hw_decode_rgb565_be_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img,
                                  "/sdcard/hw_decode_rgb565_be_color_320x240.bmp",
                                  DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN));
    }
    // test_encode rgb565BE
    start = esp_timer_get_time();
    encoded_img = hw_encode_jpeg(img, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    end = esp_timer_get_time();
    printf("hw_encode_rgb565_be_color_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/hw_encode_rgb565_be_color_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    // test_encode gray
    jpeg_img = {.data = (void *)gray_320x240_jpg_start,
                .data_len = (size_t)(gray_320x240_jpg_end - gray_320x240_jpg_start)};
    start = esp_timer_get_time();
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_GRAY);
    end = esp_timer_get_time();
    printf("hw_decode_gray_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_bmp(img, "/sdcard/hw_decode_gray_320x240.bmp"));
    }
    start = esp_timer_get_time();
    encoded_img = hw_encode_jpeg(img);
    end = esp_timer_get_time();
    printf("hw_encode_gray_320x240: %.2fms\n", (end - start) / 1000.f);
    if (mount) {
        ESP_ERROR_CHECK(write_jpeg(encoded_img, "/sdcard/hw_encode_gray_320x240.jpg"));
    }
    // cleanup
    heap_caps_free(img.data);
    heap_caps_free(encoded_img.data);

    if (mount) {
        ESP_ERROR_CHECK(bsp_sdcard_unmount());
    }
}
#endif

TEST_CASE("Test BMP", "[dl_image][ignore]")
{
    ESP_ERROR_CHECK(bsp_sdcard_mount());

    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);
    ESP_ERROR_CHECK(write_bmp(img, "/sdcard/color_405x540.bmp", DL_IMAGE_CAP_RGB_SWAP));
    auto bmp_img = read_bmp("/sdcard/color_405x540.bmp");
    ESP_ERROR_CHECK(write_bmp(bmp_img, "/sdcard/color_405x540_.bmp"));
    heap_caps_free(img.data);
    heap_caps_free(bmp_img.data);

#if CONFIG_IDF_TARGET_ESP32P4
    jpeg_img = {.data = (void *)gray_320x240_jpg_start,
                .data_len = (size_t)(gray_320x240_jpg_end - gray_320x240_jpg_start)};
    img = hw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_GRAY);
    ESP_ERROR_CHECK(write_bmp(img, "/sdcard/gray_320x240.bmp"));
    bmp_img = read_bmp("/sdcard/gray_320x240.bmp");
    ESP_ERROR_CHECK(write_bmp(bmp_img, "/sdcard/gray_320x240_.bmp"));
    heap_caps_free(img.data);
    heap_caps_free(bmp_img.data);
#endif

    ESP_ERROR_CHECK(bsp_sdcard_unmount());
}
