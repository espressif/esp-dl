#include "dl_image.hpp"
#include "unity.h"
#include <functional>

extern const uint8_t color_320x240_jpg_start[] asm("_binary_color_320x240_jpg_start");
extern const uint8_t color_320x240_jpg_end[] asm("_binary_color_320x240_jpg_end");
extern const uint8_t color_405x540_jpg_start[] asm("_binary_color_405x540_jpg_start");
extern const uint8_t color_405x540_jpg_end[] asm("_binary_color_405x540_jpg_end");
extern const uint8_t gray_320x240_jpg_start[] asm("_binary_gray_320x240_jpg_start");
extern const uint8_t gray_320x240_jpg_end[] asm("_binary_gray_320x240_jpg_end");

using namespace dl::image;
extern bool mount;

TEST_CASE("Test resize", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_caps(DL_IMAGE_CAP_RGB_SWAP);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/resize.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop resize", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_src_img_crop_area({30, 70, 130, 170});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/crop_resize.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding black bg", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_dst_img_border({30, 70, 30, 70});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/resize_padding_black_bg.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding bg same", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    // rgb 120, 120, 120
    std::vector<uint8_t> color{0xcf, 0x7b};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value(color);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/resize_padding_bg_same.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding bg diff", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    // rgb 248, 0, 0
    std::vector<uint8_t> color{0x00, 0xf8};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value(color);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/resize_padding_bg_diff.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_caps(DL_IMAGE_CAP_RGB_SWAP).set_warp_affine_matrix(M);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/warp_affine.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop warp affine", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_warp_affine_matrix(M)
        .set_src_img_crop_area({30, 70, 130, 170});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/crop_warp_affine.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding black bg", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_warp_affine_matrix(M)
        .set_dst_img_border({30, 70, 30, 70});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/warp_affine_padding_black_bg.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding bg same", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    // rgb 120, 120, 120
    std::vector<uint8_t> color{0xcf, 0x7b};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_warp_affine_matrix(M)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value(color);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/warp_affine_padding_bg_same.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding bg diff", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    // rgb 248, 0, 0
    std::vector<uint8_t> color{0x00, 0xf8};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_caps(DL_IMAGE_CAP_RGB_SWAP)
        .set_warp_affine_matrix(M)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value(color);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp_base(dst_img, "/sdcard/warp_affine_padding_bg_diff.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}
