#include "dl_image.hpp"
#include "sdkconfig.h"
#include "unity.h"
#include <cstdint>
#include <format>
#include <functional>

using namespace dl::image;

static std::vector<pix_cvt_t> pix_cvts = {DL_IMAGE_PIX_CVT_RGB8882RGB888,
                                          DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB8882BGR888,
                                          DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB8882GRAY,
                                          DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB8882RGB565LE,
                                          DL_IMAGE_PIX_CVT_RGB8882RGB565BE,
                                          DL_IMAGE_PIX_CVT_RGB8882BGR565LE,
                                          DL_IMAGE_PIX_CVT_RGB8882BGR565BE,
                                          DL_IMAGE_PIX_CVT_RGB8882HSV,
                                          DL_IMAGE_PIX_CVT_RGB8882HSV_MASK,
                                          DL_IMAGE_PIX_CVT_BGR8882RGB888,
                                          DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR8882BGR888,
                                          DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR8882GRAY,
                                          DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR8882RGB565LE,
                                          DL_IMAGE_PIX_CVT_BGR8882RGB565BE,
                                          DL_IMAGE_PIX_CVT_BGR8882BGR565LE,
                                          DL_IMAGE_PIX_CVT_BGR8882BGR565BE,
                                          DL_IMAGE_PIX_CVT_BGR8882HSV,
                                          DL_IMAGE_PIX_CVT_BGR8882HSV_MASK,
                                          DL_IMAGE_PIX_CVT_GRAY2GRAY,
                                          DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565LE2RGB888,
                                          DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565LE2BGR888,
                                          DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565LE2GRAY,
                                          DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE,
                                          DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE,
                                          DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE,
                                          DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE,
                                          DL_IMAGE_PIX_CVT_RGB565LE2HSV,
                                          DL_IMAGE_PIX_CVT_RGB565LE2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_RGB565BE2RGB888,
                                          DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565BE2BGR888,
                                          DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565BE2GRAY,
                                          DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE,
                                          DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE,
                                          DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE,
                                          DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE,
                                          DL_IMAGE_PIX_CVT_RGB565BE2HSV,
                                          DL_IMAGE_PIX_CVT_RGB565BE2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_BGR565LE2RGB888,
                                          DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565LE2BGR888,
                                          DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565LE2GRAY,
                                          DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565LE2RGB565LE,
                                          DL_IMAGE_PIX_CVT_BGR565LE2RGB565BE,
                                          DL_IMAGE_PIX_CVT_BGR565LE2BGR565LE,
                                          DL_IMAGE_PIX_CVT_BGR565LE2BGR565BE,
                                          DL_IMAGE_PIX_CVT_BGR565LE2HSV,
                                          DL_IMAGE_PIX_CVT_BGR565LE2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_BGR565BE2RGB888,
                                          DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565BE2BGR888,
                                          DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565BE2GRAY,
                                          DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_BGR565BE2RGB565LE,
                                          DL_IMAGE_PIX_CVT_BGR565BE2RGB565BE,
                                          DL_IMAGE_PIX_CVT_BGR565BE2BGR565LE,
                                          DL_IMAGE_PIX_CVT_BGR565BE2BGR565BE,
                                          DL_IMAGE_PIX_CVT_BGR565BE2HSV,
                                          DL_IMAGE_PIX_CVT_BGR565BE2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_HSV2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_YUYV2RGB888,
                                          DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_YUYV2BGR888,
                                          DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_YUYV2GRAY,
                                          DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_YUYV2RGB565LE,
                                          DL_IMAGE_PIX_CVT_YUYV2RGB565BE,
                                          DL_IMAGE_PIX_CVT_YUYV2BGR565LE,
                                          DL_IMAGE_PIX_CVT_YUYV2BGR565BE,
                                          DL_IMAGE_PIX_CVT_YUYV2HSV,
                                          DL_IMAGE_PIX_CVT_YUYV2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_YUYV2YUYV,
                                          DL_IMAGE_PIX_CVT_YUYV2UYVY,
                                          DL_IMAGE_PIX_CVT_UYVY2RGB888,
                                          DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT8,
                                          DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT16,
                                          DL_IMAGE_PIX_CVT_UYVY2BGR888,
                                          DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT8,
                                          DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT16,
                                          DL_IMAGE_PIX_CVT_UYVY2GRAY,
                                          DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT8,
                                          DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT16,
                                          DL_IMAGE_PIX_CVT_UYVY2RGB565LE,
                                          DL_IMAGE_PIX_CVT_UYVY2RGB565BE,
                                          DL_IMAGE_PIX_CVT_UYVY2BGR565LE,
                                          DL_IMAGE_PIX_CVT_UYVY2BGR565BE,
                                          DL_IMAGE_PIX_CVT_UYVY2HSV,
                                          DL_IMAGE_PIX_CVT_UYVY2HSV_MASK,
                                          DL_IMAGE_PIX_CVT_UYVY2YUYV,
                                          DL_IMAGE_PIX_CVT_UYVY2UYVY};

void test_cvt_perf(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, int times)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(src_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = src_pix_type};
    img_t dst = {.data =
                     heap_caps_malloc(h * w * get_pix_byte_size(dst_pix_type), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = dst_pix_type};
    assert(src.data);
    assert(dst.data);
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst);
    if (dl::image::is_pix_type_quant(dst_pix_type)) {
        int pix_byte_size = dl::image::get_pix_byte_size(dst_pix_type);
        if (pix_byte_size == 3) {
            T.set_norm_quant_param({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -5, 8);
        } else if (pix_byte_size == 6) {
            T.set_norm_quant_param({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -13, 16);
        } else if (pix_byte_size == 1) {
            T.set_norm_quant_param({123.675}, {58.395}, -5, 8);
        } else if (pix_byte_size == 2) {
            T.set_norm_quant_param({123.675}, {58.395}, -13, 16);
        }
    }
    if (dst_pix_type == dl::image::DL_IMAGE_PIX_TYPE_HSV_MASK) {
        T.set_hsv_thr({2, 10, 10}, {100, 255, 255});
    }

    int64_t max = 0, min = std::numeric_limits<int64_t>::max(), avg = 0;
    for (int i = 0; i < times; i++) {
        int64_t start = esp_timer_get_time();
        T.transform<false>();
        int64_t end = esp_timer_get_time();
        int64_t interval = end - start;
        if (interval > max) {
            max = interval;
        }
        if (interval < min) {
            min = interval;
        }
        avg += interval;
    }
    avg /= times;
    printf("%s2%s, [%d, %d]\n", pix_type2str(src_pix_type).c_str(), pix_type2str(dst_pix_type).c_str(), w, h);
    printf("avg: %lld, max: %lld, min: %lld\n", avg, max, min);
    heap_caps_free(src.data);
    heap_caps_free(dst.data);
}

void test_resize_perf(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, uint16_t resized_w, uint16_t resized_h, int times)
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
                 .width = resized_h,
                 .height = resized_w,
                 .pix_type = dst_pix_type};
    assert(src.data);
    assert(dst.data);
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst);
    if (dl::image::is_pix_type_quant(dst_pix_type)) {
        int pix_byte_size = dl::image::get_pix_byte_size(dst_pix_type);
        if (pix_byte_size == 3) {
            T.set_norm_quant_param({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -5, 8);
        } else if (pix_byte_size == 6) {
            T.set_norm_quant_param({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -13, 16);
        } else if (pix_byte_size == 1) {
            T.set_norm_quant_param({123.675}, {58.395}, -5, 8);
        } else if (pix_byte_size == 2) {
            T.set_norm_quant_param({123.675}, {58.395}, -13, 16);
        }
    }
    if (dst_pix_type == dl::image::DL_IMAGE_PIX_TYPE_HSV_MASK) {
        T.set_hsv_thr({2, 10, 10}, {100, 255, 255});
    }

    int64_t max = 0, min = std::numeric_limits<int64_t>::max(), avg = 0;
    for (int i = 0; i < times; i++) {
        int64_t start = esp_timer_get_time();
        T.transform<false>();
        int64_t end = esp_timer_get_time();
        int64_t interval = end - start;
        if (interval > max) {
            max = interval;
        }
        if (interval < min) {
            min = interval;
        }
        avg += interval;
    }
    avg /= times;
    printf("%s2%s, [%d, %d]\n", pix_type2str(src_pix_type).c_str(), pix_type2str(dst_pix_type).c_str(), w, h);
    printf("avg: %lld, max: %lld, min: %lld\n", avg, max, min);
    heap_caps_free(src.data);
    heap_caps_free(dst.data);
}

TEST_CASE("perf", "[dl_image][ignore]")
{
    for (auto pix_cvt : pix_cvts) {
        test_resize_perf(pix_cvt, 1280, 720, 240, 240, 10);
    }
    for (auto pix_cvt : pix_cvts) {
        test_cvt_perf(pix_cvt, 240, 240, 10);
    }
}
