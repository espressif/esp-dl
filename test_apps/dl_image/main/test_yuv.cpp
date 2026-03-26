#include "dl_image.hpp"
#include "sdkconfig.h"
#include "unity.h"
#include <cstdint>
#include <format>
#include <functional>

using namespace dl::image;

void test_crop_cvt(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, const std::vector<int> &crop_area)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    uint16_t crop_w = crop_area[2] - crop_area[0];
    uint16_t crop_h = crop_area[3] - crop_area[1];
    img_t dst = {.data = heap_caps_malloc(crop_w * crop_h * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = crop_w,
                 .height = crop_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(crop_w * crop_h * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = crop_w,
                    .height = crop_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_src_img_crop_area(crop_area);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_src_img_crop_area(crop_area);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, crop_w * crop_h * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("cvt_crop correctness", "[dl_image][ignore]")
{
    // !odd_left && !odd_right
    test_crop_cvt(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 60, 47});
    // odd_left & !odd_right
    test_crop_cvt(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 60, 47});
    // !odd_left & odd_right
    test_crop_cvt(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 61, 47});
    // odd_left & odd_right
    test_crop_cvt(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 61, 47});
}

void test_cvt_border(pix_cvt_t pix_cvt,
                     uint16_t w,
                     uint16_t h,
                     const std::vector<int> &border,
                     const std::array<uint8_t, 3> &border_color)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    uint16_t border_w = w + border[2] + border[3];
    uint16_t border_h = h + border[0] + border[1];
    img_t dst = {.data = heap_caps_malloc(border_w * border_h * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = border_w,
                 .height = border_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(border_w * border_h * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = border_w,
                    .height = border_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_dst_img_border(border).set_bg_value(border_color);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_dst_img_border(border).set_bg_value(border_color);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, border_w * border_h * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("cvt_border correctness", "[dl_image][ignore]")
{
    test_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {7, 27, 20, 5}, {12, 12, 12});
    test_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {7, 27, 20, 5}, {12, 12, 1});
}

void test_crop_cvt_border(pix_cvt_t pix_cvt,
                          uint16_t w,
                          uint16_t h,
                          const std::vector<int> &crop_area,
                          const std::vector<int> &border,
                          const std::array<uint8_t, 3> &border_color)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    uint16_t crop_border_w = crop_area[2] - crop_area[0] + border[2] + border[3];
    uint16_t crop_border_h = crop_area[3] - crop_area[1] + border[0] + border[1];
    img_t dst = {.data = heap_caps_malloc(crop_border_w * crop_border_h * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = crop_border_w,
                 .height = crop_border_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(crop_border_w * crop_border_h * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = crop_border_w,
                    .height = crop_border_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_src_img_crop_area(crop_area).set_dst_img_border(border).set_bg_value(
        border_color);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_src_img_crop_area(crop_area).set_dst_img_border(border).set_bg_value(
        border_color);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, crop_border_w * crop_border_h * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("crop_cvt_border correctness", "[dl_image][ignore]")
{
    // !odd_left && !odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 60, 47}, {7, 27, 20, 5}, {12, 12, 12});
    // odd_left & !odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 60, 47}, {7, 27, 20, 5}, {12, 12, 12});
    // !odd_left & odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 61, 47}, {7, 27, 20, 5}, {12, 12, 12});
    // odd_left & odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 61, 47}, {7, 27, 20, 5}, {12, 12, 12});
    // !odd_left && !odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 60, 47}, {7, 27, 20, 5}, {12, 12, 1});
    // odd_left & !odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 60, 47}, {7, 27, 20, 5}, {12, 12, 1});
    // !odd_left & odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {20, 7, 61, 47}, {7, 27, 20, 5}, {12, 12, 1});
    // odd_left & odd_right
    test_crop_cvt_border(DL_IMAGE_PIX_CVT_YUYV2RGB888, 240, 240, {21, 7, 61, 47}, {7, 27, 20, 5}, {12, 12, 1});
}

void test_resize(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, uint16_t resized_w, uint16_t resized_h)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    img_t dst = {.data = heap_caps_malloc(resized_h * resized_w * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = resized_w,
                 .height = resized_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(resized_h * resized_w * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = resized_w,
                    .height = resized_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, resized_h * resized_w * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("resize correctness", "[dl_image][ignore]")
{
    test_resize(DL_IMAGE_PIX_CVT_YUYV2RGB888, 512, 512, 240, 240);
}

void test_crop_resize(pix_cvt_t pix_cvt,
                      uint16_t w,
                      uint16_t h,
                      uint16_t resized_w,
                      uint16_t resized_h,
                      const std::vector<int> &crop_area)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    img_t dst = {.data = heap_caps_malloc(resized_h * resized_w * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = resized_w,
                 .height = resized_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(resized_h * resized_w * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = resized_w,
                    .height = resized_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_src_img_crop_area(crop_area);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_src_img_crop_area(crop_area);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, resized_h * resized_w * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("crop resize correctness", "[dl_image][ignore]")
{
    test_crop_resize(DL_IMAGE_PIX_CVT_YUYV2RGB888, 512, 512, 240, 240, {5, 30, 100, 500});
}

void test_crop_resize_border(pix_cvt_t pix_cvt,
                             uint16_t w,
                             uint16_t h,
                             uint16_t resized_w,
                             uint16_t resized_h,
                             const std::vector<int> &crop_area,
                             const std::vector<int> &border,
                             const std::array<uint8_t, 3> &border_color)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    uint16_t resized_border_w = resized_w + border[2] + border[3];
    uint16_t resized_border_h = resized_h + border[0] + border[1];
    img_t dst = {.data = heap_caps_malloc(resized_border_w * resized_border_h * get_pix_byte_size(dst_pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = resized_border_w,
                 .height = resized_border_h,
                 .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {.data = heap_caps_malloc(resized_border_w * resized_border_h * get_pix_byte_size(dst_pix_type),
                                             MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                    .width = resized_border_w,
                    .height = resized_border_h,
                    .pix_type = dst_pix_type};
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_src_img_crop_area(crop_area).set_dst_img_border(border).set_bg_value(
        border_color);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_src_img_crop_area(crop_area).set_dst_img_border(border).set_bg_value(
        border_color);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(
        dst.data, dst_gt.data, resized_border_w * resized_border_h * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("crop resize border correctness", "[dl_image][ignore]")
{
    test_crop_resize_border(
        DL_IMAGE_PIX_CVT_YUYV2RGB888, 512, 512, 240, 240, {5, 30, 100, 500}, {2, 5, 7, 17}, {255, 255, 255});
    test_crop_resize_border(
        DL_IMAGE_PIX_CVT_YUYV2RGB888, 512, 512, 240, 240, {5, 30, 100, 500}, {2, 5, 7, 17}, {255, 0, 0});
}

void test_warp_affine(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, uint16_t dst_w, uint16_t dst_h)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    img_t dst = {
        .data = heap_caps_malloc(dst_w * dst_h * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
        .width = dst_w,
        .height = dst_h,
        .pix_type = dst_pix_type};
    img_t tmp = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    img_t dst_gt = {
        .data = heap_caps_malloc(dst_w * dst_h * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
        .width = dst_w,
        .height = dst_h,
        .pix_type = dst_pix_type};

    dl::math::Matrix<float> M(2, 3);
    M.array[0][0] = 0.866;
    M.array[0][1] = -0.5;
    M.array[0][2] = 0;
    M.array[1][0] = 0.5;
    M.array[1][1] = 0.866;
    M.array[1][2] = 0;

    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst).set_warp_affine_matrix(M);
    T.transform();

    ImageTransformer T1;
    T1.set_src_img(src).set_dst_img(tmp);
    T1.transform();
    ImageTransformer T2;
    T2.set_src_img(tmp).set_dst_img(dst_gt).set_warp_affine_matrix(M);
    T2.transform();

    TEST_ASSERT_EQUAL_MEMORY(dst.data, dst_gt.data, dst_w * dst_h * get_pix_byte_size(dst_pix_type));

    heap_caps_free(src.data);
    heap_caps_free(dst.data);
    heap_caps_free(tmp.data);
    heap_caps_free(dst_gt.data);
}

TEST_CASE("warp affine correctness", "[dl_image][ignore]")
{
    test_warp_affine(DL_IMAGE_PIX_CVT_YUYV2RGB888, 512, 512, 240, 240);
}
