#include "dl_image.hpp"
#include "unity.h"
#include <format>
#include <functional>

extern const uint8_t color_320x240_jpg_start[] asm("_binary_color_320x240_jpg_start");
extern const uint8_t color_320x240_jpg_end[] asm("_binary_color_320x240_jpg_end");
extern const uint8_t color_405x540_jpg_start[] asm("_binary_color_405x540_jpg_start");
extern const uint8_t color_405x540_jpg_end[] asm("_binary_color_405x540_jpg_end");
extern const uint8_t gray_320x240_jpg_start[] asm("_binary_gray_320x240_jpg_start");
extern const uint8_t gray_320x240_jpg_end[] asm("_binary_gray_320x240_jpg_end");

using namespace dl::image;
extern bool mount;

TEST_CASE("Test crop+cvt_color", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_src_img_crop_area({30, 60, 254, 284});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/crop_cvt_color.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop+cvt_color+make_color_border", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_src_img_crop_area({30, 60, 30 + 200, 60 + 200})
        .set_dst_img_border({5, 19, 14, 10})
        .set_bg_value({255, 0, 0});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/crop_cvt_color_make_color_border.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop+cvt_color+make_border", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_src_img_crop_area({30, 60, 30 + 200, 60 + 200})
        .set_dst_img_border({5, 19, 14, 10});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/crop_cvt_color_make_border.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/resize.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop resize", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT),
                     .width = 224,
                     .height = 224,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_src_img_crop_area({30, 70, 130, 170});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/crop_resize.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding black bg", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_dst_img_border({30, 70, 30, 70});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/resize_padding_black_bg.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding bg same", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value({120, 120, 120});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/resize_padding_bg_same.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test resize padding bg diff", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
    img_t dst_img = {.data = heap_caps_malloc(324 * 324 * 3, MALLOC_CAP_DEFAULT),
                     .width = 324,
                     .height = 324,
                     .pix_type = DL_IMAGE_PIX_TYPE_RGB888};

    ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value({255, 0, 0});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/resize_padding_bg_diff.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
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
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_warp_affine_matrix(M);
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/warp_affine.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test crop warp affine", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
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
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_warp_affine_matrix(M).set_src_img_crop_area(
        {30, 70, 130, 170});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/crop_warp_affine.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding black bg", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
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
    transformer.set_src_img(src_img).set_dst_img(dst_img).set_warp_affine_matrix(M).set_dst_img_border(
        {30, 70, 30, 70});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/warp_affine_padding_black_bg.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding bg same", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
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
        .set_warp_affine_matrix(M)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value({120, 120, 120});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/warp_affine_padding_bg_same.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test warp affine padding bg diff", "[dl_image]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t src_img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB565LE);
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
        .set_warp_affine_matrix(M)
        .set_dst_img_border({30, 70, 30, 70})
        .set_bg_value({255, 0, 0});
    int64_t start = esp_timer_get_time();
    transformer.transform();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    if (mount) {
        write_bmp(dst_img, "/sdcard/warp_affine_padding_bg_diff.bmp");
    }
    heap_caps_free(src_img.data);
    heap_caps_free(dst_img.data);
}

TEST_CASE("Test cvt/resize", "[dl_image][ignore]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);
    std::vector<pix_cvt_t> pix_cvts = {
        DL_IMAGE_PIX_CVT_RGB8882RGB888,     DL_IMAGE_PIX_CVT_RGB8882BGR888,     DL_IMAGE_PIX_CVT_RGB8882GRAY,
        DL_IMAGE_PIX_CVT_RGB8882RGB565LE,   DL_IMAGE_PIX_CVT_RGB8882RGB565BE,   DL_IMAGE_PIX_CVT_RGB8882BGR565LE,
        DL_IMAGE_PIX_CVT_RGB8882BGR565BE,   DL_IMAGE_PIX_CVT_BGR8882RGB888,     DL_IMAGE_PIX_CVT_BGR8882BGR888,
        DL_IMAGE_PIX_CVT_BGR8882GRAY,       DL_IMAGE_PIX_CVT_BGR8882RGB565LE,   DL_IMAGE_PIX_CVT_BGR8882RGB565BE,
        DL_IMAGE_PIX_CVT_BGR8882BGR565LE,   DL_IMAGE_PIX_CVT_BGR8882BGR565BE,   DL_IMAGE_PIX_CVT_GRAY2GRAY,
        DL_IMAGE_PIX_CVT_RGB565LE2RGB888,   DL_IMAGE_PIX_CVT_RGB565LE2BGR888,   DL_IMAGE_PIX_CVT_RGB565LE2GRAY,
        DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE, DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE, DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE,
        DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE, DL_IMAGE_PIX_CVT_RGB565BE2RGB888,   DL_IMAGE_PIX_CVT_RGB565BE2BGR888,
        DL_IMAGE_PIX_CVT_RGB565BE2GRAY,     DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE, DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE,
        DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE, DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE, DL_IMAGE_PIX_CVT_BGR565LE2RGB888,
        DL_IMAGE_PIX_CVT_BGR565LE2BGR888,   DL_IMAGE_PIX_CVT_BGR565LE2GRAY,     DL_IMAGE_PIX_CVT_BGR565LE2RGB565LE,
        DL_IMAGE_PIX_CVT_BGR565LE2RGB565BE, DL_IMAGE_PIX_CVT_BGR565LE2BGR565LE, DL_IMAGE_PIX_CVT_BGR565LE2BGR565BE,
        DL_IMAGE_PIX_CVT_BGR565BE2RGB888,   DL_IMAGE_PIX_CVT_BGR565BE2BGR888,   DL_IMAGE_PIX_CVT_BGR565BE2GRAY,
        DL_IMAGE_PIX_CVT_BGR565BE2RGB565LE, DL_IMAGE_PIX_CVT_BGR565BE2RGB565BE, DL_IMAGE_PIX_CVT_BGR565BE2BGR565LE,
        DL_IMAGE_PIX_CVT_BGR565BE2BGR565BE};

    ImageTransformer T1;
    T1.set_src_img(img);
    ImageTransformer T2;

    // i = 0 for cvt
    // i = 1 for resize
    for (int i = 0; i < 2; i++) {
        for (auto pix_cvt : pix_cvts) {
            pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
            pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
            img_t src_img = {
                .data = heap_caps_malloc(img.width * img.height * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT),
                .width = img.width,
                .height = img.height,
                .pix_type = src_pix_type};
            T1.set_dst_img(src_img).transform();
            uint16_t dst_width = (i == 0) ? img.width : 224;
            uint16_t dst_height = (i == 0) ? img.height : 224;
            img_t dst_img = {
                .data = heap_caps_malloc(dst_width * dst_height * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT),
                .width = dst_width,
                .height = dst_height,
                .pix_type = dst_pix_type};
            T2.set_src_img(src_img).set_dst_img(dst_img).transform();
            std::string file_name = (i == 0)
                ? std::format("/sdcard/cvt_{}2{}.bmp", pix_type2str(src_pix_type), pix_type2str(dst_pix_type))
                : std::format("/sdcard/resize_{}2{}.bmp", pix_type2str(src_pix_type), pix_type2str(dst_pix_type));
            write_bmp(dst_img, file_name.c_str());
            heap_caps_free(src_img.data);
            heap_caps_free(dst_img.data);
        }
    }
    heap_caps_free(img.data);
}

#if CONFIG_SOC_PPA_SUPPORTED
TEST_CASE("Test ppa resize", "[dl_image][ignore]")
{
    jpeg_img_t jpeg_img = {.data = (void *)color_405x540_jpg_start,
                           .data_len = (size_t)(color_405x540_jpg_end - color_405x540_jpg_start)};
    img_t img = sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);
    std::vector<pix_type_t> pix_types = {DL_IMAGE_PIX_TYPE_RGB888,
                                         DL_IMAGE_PIX_TYPE_BGR888,
                                         DL_IMAGE_PIX_TYPE_RGB565LE,
                                         DL_IMAGE_PIX_TYPE_RGB565BE,
                                         DL_IMAGE_PIX_TYPE_BGR565LE,
                                         DL_IMAGE_PIX_TYPE_BGR565BE};

    ppa_client_handle_t ppa_srm_handle = register_ppa_srm_client();

    ImageTransformer T;
    T.set_src_img(img);

    for (auto src_pix_type : pix_types) {
        for (auto dst_pix_type : pix_types) {
            img_t src_img = {
                .data = heap_caps_malloc(img.width * img.height * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT),
                .width = img.width,
                .height = img.height,
                .pix_type = src_pix_type};
            T.set_dst_img(src_img).transform();
            img_t dst_img = {.data = alloc_ppa_outbuf(224 * 224 * get_pix_byte_size(dst_pix_type)),
                             .width = 224,
                             .height = 224,
                             .pix_type = dst_pix_type};
            auto ret = resize_ppa(src_img, dst_img, ppa_srm_handle);
            if (ret == ESP_OK && mount) {
                write_bmp(dst_img,
                          std::format("/sdcard/ppa_{}2{}.bmp", pix_type2str(src_pix_type), pix_type2str(dst_pix_type))
                              .c_str());
            }
            heap_caps_free(src_img.data);
            heap_caps_free(dst_img.data);
        }
    }
    heap_caps_free(img.data);
    ESP_ERROR_CHECK(ppa_unregister_client(ppa_srm_handle));
}
#endif
