#include "dl_define.hpp"
#include "dl_image.hpp"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "unity.h"
#include <cassert>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <vector>

using namespace dl::image;

#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
static constexpr bool k_has_simd = true;
#else
static constexpr bool k_has_simd = false;
#endif

struct perf_us_t {
    int64_t avg;
    int64_t min;
    int64_t max;
};

static std::string pix_name(pix_type_t pix_type)
{
    std::string name = pix_type2str(pix_type);
    constexpr char prefix[] = "DL_IMAGE_PIX_TYPE_";
    if (name.starts_with(prefix)) {
        name.erase(0, sizeof(prefix) - 1);
    }
    return name;
}

static img_t alloc_img(uint16_t w, uint16_t h, pix_type_t pix_type)
{
    img_t img = {.data = heap_caps_malloc(static_cast<size_t>(h) * w * get_pix_byte_size(pix_type),
                                          MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD),
                 .width = w,
                 .height = h,
                 .pix_type = pix_type};
    assert(img.data);
    return img;
}

static void apply_dst_params(ImageTransformer &T, pix_type_t dst_pix_type)
{
    if (is_pix_type_quant(dst_pix_type)) {
        int pix_byte_size = get_pix_byte_size(dst_pix_type);
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
    if (dst_pix_type == DL_IMAGE_PIX_TYPE_HSV_MASK) {
        T.set_hsv_thr({2, 10, 10}, {100, 255, 255});
    }
}

// One untimed call drops kernel / xy-map setup from the measured samples.
static perf_us_t measure_transform(ImageTransformer &T, bool simd, int times)
{
    T.transform(simd);
    int64_t max_us = 0;
    int64_t min_us = std::numeric_limits<int64_t>::max();
    int64_t sum_us = 0;
    for (int i = 0; i < times; ++i) {
        int64_t start = esp_timer_get_time();
        T.transform(simd);
        int64_t us = esp_timer_get_time() - start;
        if (us > max_us) {
            max_us = us;
        }
        if (us < min_us) {
            min_us = us;
        }
        sum_us += us;
    }
    return {.avg = sum_us / times, .min = min_us, .max = max_us};
}

static void print_perf(const char *op,
                       pix_type_t src_pix_type,
                       pix_type_t dst_pix_type,
                       uint16_t src_w,
                       uint16_t src_h,
                       uint16_t dst_w,
                       uint16_t dst_h,
                       const perf_us_t &scalar,
                       const perf_us_t *simd)
{
    printf("%s\n",
           std::format("{:<6}  {:<14} -> {:<14}  {:>4}x{:<3} -> {:>3}x{:<3}",
                       op,
                       pix_name(src_pix_type),
                       pix_name(dst_pix_type),
                       src_w,
                       src_h,
                       dst_w,
                       dst_h)
               .c_str());
    printf("%s\n",
           std::format("  scalar  avg {:8} us  min {:8}  max {:8}", scalar.avg, scalar.min, scalar.max).c_str());
    if (simd) {
        double speedup = simd->avg > 0 ? static_cast<double>(scalar.avg) / static_cast<double>(simd->avg) : 0.0;
        printf(
            "%s\n",
            std::format("  simd    avg {:8} us  min {:8}  max {:8}  {:6.2f}x", simd->avg, simd->min, simd->max, speedup)
                .c_str());
    }
}

static void bench_transform(const char *op,
                            ImageTransformer &T,
                            pix_type_t src_pix_type,
                            pix_type_t dst_pix_type,
                            uint16_t src_w,
                            uint16_t src_h,
                            uint16_t dst_w,
                            uint16_t dst_h,
                            int times)
{
    perf_us_t scalar = measure_transform(T, false, times);
    if (k_has_simd) {
        perf_us_t simd = measure_transform(T, true, times);
        print_perf(op, src_pix_type, dst_pix_type, src_w, src_h, dst_w, dst_h, scalar, &simd);
    } else {
        print_perf(op, src_pix_type, dst_pix_type, src_w, src_h, dst_w, dst_h, scalar, nullptr);
    }
}

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
                                          DL_IMAGE_PIX_CVT_UYVY2HSV_MASK};

void test_cvt_perf(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, int times)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = alloc_img(w, h, src_pix_type);
    img_t dst = alloc_img(w, h, dst_pix_type);
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst);
    apply_dst_params(T, dst_pix_type);
    bench_transform("cvt", T, src_pix_type, dst_pix_type, w, h, w, h, times);
    heap_caps_free(src.data);
    heap_caps_free(dst.data);
}

void test_resize_perf(pix_cvt_t pix_cvt, uint16_t w, uint16_t h, uint16_t resized_w, uint16_t resized_h, int times)
{
    pix_type_t src_pix_type = (pix_type_t)(pix_cvt >> 16);
    pix_type_t dst_pix_type = (pix_type_t)(pix_cvt & 0xffff);
    img_t src = alloc_img(w, h, src_pix_type);
    img_t dst = alloc_img(resized_w, resized_h, dst_pix_type);
    ImageTransformer T;
    T.set_src_img(src).set_dst_img(dst);
    apply_dst_params(T, dst_pix_type);
    bench_transform("resize", T, src_pix_type, dst_pix_type, w, h, resized_w, resized_h, times);
    heap_caps_free(src.data);
    heap_caps_free(dst.data);
}

TEST_CASE("perf", "[dl_image][ignore]")
{
    printf("image transform perf, target=%s, %s\n",
           CONFIG_IDF_TARGET,
           k_has_simd ? "scalar = transform(false), simd = transform(true), speedup = scalar/simd"
                      : "scalar = transform(false)");
    printf("---- resize ----\n");
    for (auto pix_cvt : pix_cvts) {
        test_resize_perf(pix_cvt, 1280, 720, 240, 240, 10);
    }
    printf("---- cvt ----\n");
    for (auto pix_cvt : pix_cvts) {
        test_cvt_perf(pix_cvt, 240, 240, 10);
    }
}
