#include "dl_image_color.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_process.hpp"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "unity.h"
#include "utils.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

#if CONFIG_IDF_TARGET_ESP32P4
#define FLAG_STR_0b0000 "_dst_align"
#define FLAG_STR_0b0010 "_dst_align_border_diff"
#define FLAG_STR_0b0011 "_dst_align_border_same"
#define FLAG_STR_0b0100 "_dst_align_crop"
#define FLAG_STR_0b0110 "_dst_align_crop_border_diff"
#define FLAG_STR_0b0111 "_dst_align_crop_border_same"
#define FLAG_STR_0b1000 "_dst_unalign"
#define FLAG_STR_0b1010 "_dst_unalign_border_diff"
#define FLAG_STR_0b1011 "_dst_unalign_border_same"
#define FLAG_STR_0b1100 "_dst_unalign_crop"
#define FLAG_STR_0b1110 "_dst_unalign_crop_border_diff"
#define FLAG_STR_0b1111 "_dst_unalign_crop_border_same"
#define ALIGN_TAG_0b0000 "dst_align"
#define ALIGN_TAG_0b0010 "dst_align"
#define ALIGN_TAG_0b0011 "dst_align"
#define ALIGN_TAG_0b0100 "dst_align"
#define ALIGN_TAG_0b0110 "dst_align"
#define ALIGN_TAG_0b0111 "dst_align"
#define ALIGN_TAG_0b1000 "dst_unalign"
#define ALIGN_TAG_0b1010 "dst_unalign"
#define ALIGN_TAG_0b1011 "dst_unalign"
#define ALIGN_TAG_0b1100 "dst_unalign"
#define ALIGN_TAG_0b1110 "dst_unalign"
#define ALIGN_TAG_0b1111 "dst_unalign"

#define FLAG_STR(x) FLAG_STR_##x
#define ALIGN_TAG(x) ALIGN_TAG_##x

#define TEST_RESIZE_NN(_name, _src_pix_type, _dst_pix_type, _caps, _flag)                         \
    TEST_CASE("resize_nn_" _name FLAG_STR(_flag), "[resize_nn][" ALIGN_TAG(_flag) "][" _name "]") \
    {                                                                                             \
        assert_resize_nn_func(_src_pix_type, _dst_pix_type, _caps, _flag);                        \
    }

void assert_resize_nn_func(bool dst_align,
                           uint16_t src_width,
                           uint16_t src_height,
                           uint16_t dst_width,
                           uint16_t dst_height,
                           dl::image::pix_type_t src_pix_type,
                           dl::image::pix_type_t dst_pix_type,
                           uint32_t caps,
                           const std::vector<int> &crop_area = {},
                           const std::vector<int> &border = {},
                           const std::vector<uint8_t> &border_value = {},
                           bool border_src_color_space = false)
{
    void *input, *output, *output_real, *output_gt;
    int src_img_byte = src_width * src_height * dl::image::get_pix_byte_size(src_pix_type);

    dst_width = border.empty() ? dst_width : (dst_width + border[2] + border[3]);
    dst_height = border.empty() ? dst_height : (dst_height + border[0] + border[1]);
    int dst_img_byte = dst_width * dst_height * dl::image::get_pix_byte_size(dst_pix_type);

    output_gt = heap_caps_malloc(dst_img_byte, MALLOC_CAP_DEFAULT);

    input = heap_caps_malloc(src_img_byte, MALLOC_CAP_DEFAULT);
    fill_random_value<uint8_t>((uint8_t *)input, src_img_byte, 0, 255);

    if (dst_align) {
        output_real = heap_caps_aligned_alloc(16, dst_img_byte, MALLOC_CAP_DEFAULT);
        output = output_real;
        assert(is_align(output));
    } else {
        output_real = heap_caps_aligned_alloc(16, dst_img_byte + 16, MALLOC_CAP_DEFAULT);
        uint8_t offset = get_random_value<uint8_t>(1, 15);
        output = (void *)((uint8_t *)output_real + offset);
        assert(!is_align(output));
    }

    dl::image::img_t src_img = {.data = input, .width = src_width, .height = src_height, .pix_type = src_pix_type};
    dl::image::img_t dst_img_gt = {
        .data = output_gt, .width = dst_width, .height = dst_height, .pix_type = dst_pix_type};
    dl::image::img_t dst_img = {.data = output, .width = dst_width, .height = dst_height, .pix_type = dst_pix_type};

    dl::image::ImageTransformer transformer;
    transformer.set_src_img(src_img)
        .set_dst_img(dst_img)
        .set_dst_img_border(border)
        .set_bg_value(border_value, border_src_color_space)
        .set_src_img_crop_area(crop_area);
    if (dl::image::is_pix_type_quant(dst_pix_type)) {
        int pix_byte_size = dl::image::get_pix_byte_size(dst_pix_type);
        if (pix_byte_size == 3) {
            transformer.set_norm_quant_param(
                {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -5, dl::image::NormQuantWrapper::INT8_QUANT);
        } else if (pix_byte_size == 6) {
            transformer.set_norm_quant_param(
                {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, -13, dl::image::NormQuantWrapper::INT16_QUANT);
        } else if (pix_byte_size == 1) {
            transformer.set_norm_quant_param({123.675}, {58.395}, -5, dl::image::NormQuantWrapper::INT8_QUANT);
        } else if (pix_byte_size == 2) {
            transformer.set_norm_quant_param({123.675}, {58.395}, -13, dl::image::NormQuantWrapper::INT16_QUANT);
        }
    }

    transformer.transform<true>();

    int64_t start = esp_timer_get_time();
    transformer.transform<true>();
    int64_t end = esp_timer_get_time();
    printf("%lld\n", end - start);

    transformer.set_dst_img(dst_img_gt);
    start = esp_timer_get_time();
    transformer.transform<false>();
    end = esp_timer_get_time();
    printf("%lld\n", end - start);

    // for (int i = 0; i < 16; i++) {
    //     printf("0x%02x, ", ((uint8_t *)input)[i]);
    //     if ((i + 1) % 16 == 0) {
    //         printf("\n");
    //     }
    // }
    // for (int i = 0; i < 16; i++) {
    //     printf("0x%02x, ", ((uint8_t *)output_gt)[i]);
    //     if ((i + 1) % 16 == 0) {
    //         printf("\n");
    //     }
    // }
    // for (int i = 0; i < 16; i++) {
    //     printf("0x%02x, ", ((uint8_t *)output)[i]);
    //     if ((i + 1) % 16 == 0) {
    //         printf("\n");
    //     }
    // }

    TEST_ASSERT_EQUAL_MEMORY(output, output_gt, dst_img_byte);
    heap_caps_free(input);
    heap_caps_free(output_real);
    heap_caps_free(output_gt);
}

// flag
// dst_unalign(3) crop(2) border(1) border_same(0)
void assert_resize_nn_func(dl::image::pix_type_t src_pix_type,
                           dl::image::pix_type_t dst_pix_type,
                           uint32_t caps,
                           uint8_t flag)
{
    uint16_t src_width = get_random_value<uint16_t>(1, 2000);
    uint16_t src_height = get_random_value<uint16_t>(1, 1000);
    bool dst_align = !(flag & 0b1000);
    std::vector<int> crop_area = {};
    std::vector<int> border = {};
    std::vector<uint8_t> border_value = {};
    if (flag & 0b100) {
        crop_area.resize(4);
        crop_area[0] = get_random_value<uint16_t>(0, src_width);
        crop_area[1] = get_random_value<uint16_t>(0, src_height);
        crop_area[2] = get_random_value<uint16_t>(0, src_width);
        crop_area[3] = get_random_value<uint16_t>(0, src_height);
        while (crop_area[0] == crop_area[2]) {
            crop_area[2] = get_random_value<uint16_t>(0, src_width);
        }
        while (crop_area[1] == crop_area[3]) {
            crop_area[3] = get_random_value<uint16_t>(0, src_height);
        }
        if (crop_area[2] < crop_area[0]) {
            std::swap(crop_area[2], crop_area[0]);
        }
        if (crop_area[3] < crop_area[1]) {
            std::swap(crop_area[3], crop_area[1]);
        }
        printf("crop_area: [%d, %d, %d, %d]\n", crop_area[0], crop_area[1], crop_area[2], crop_area[3]);
    }
    uint16_t crop_width = crop_area.empty() ? src_width : (crop_area[2] - crop_area[0]);
    uint16_t crop_height = crop_area.empty() ? src_height : (crop_area[3] - crop_area[1]);
    uint16_t dst_width = get_random_value<uint16_t>(1, 2000);
    uint16_t dst_height = get_random_value<uint16_t>(1, 1000);
    if (dst_align) {
        dst_width = dl::image::align_up(dst_width, 16);
    }
    while (dst_width == crop_width) {
        dst_width = get_random_value<uint16_t>(1, 2000);
        if (dst_align) {
            dst_width = dl::image::align_up(dst_width, 16);
        }
    }
    while (dst_height == crop_height) {
        dst_height = get_random_value<uint16_t>(1, 2000);
    }
    printf("[%d, %d]->[%d, %d]\n", src_width, src_height, dst_width, dst_height);
    if (flag & 0b10) {
        border.resize(4);
        fill_random_value<int>((int *)(border.data()), 4, 0, 100);
        if (flag & 0b1 || dl::image::get_pix_byte_size(dst_pix_type) == 1) {
            uint8_t v = get_random_value<uint8_t>(0, 255);
            if (v % 2 == 0) {
                int pix_byte_size = dl::image::get_pix_byte_size(dst_pix_type);
                border_value.resize(pix_byte_size);
                for (int i = 0; i < pix_byte_size; i++) {
                    border_value[i] = v;
                }
            }
        } else {
            int pix_byte_size = dl::image::get_pix_byte_size(dst_pix_type);
            border_value.resize(pix_byte_size);
            while (std::all_of(border_value.begin() + 1, border_value.end(), [&border_value](const auto &v) {
                return v == border_value[0];
            })) {
                fill_random_value<uint8_t>(border_value.data(), pix_byte_size, 0, 255);
            }
        }
        if (dst_align) {
            border[2] = dl::image::align_up(border[2], 16);
            border[3] = dl::image::align_up(border[3], 16);
        }
        printf("border: [%d, %d, %d, %d]\n", border[0], border[1], border[2], border[3]);
        if (border_value.empty()) {
            printf("empty_border\n");
        } else {
            printf("border_value: [");
            for (int i = 0; i < border_value.size(); i++) {
                printf("%d, ", border_value[i]);
            }
            printf("]\n");
        }
    }
    assert_resize_nn_func(dst_align,
                          src_width,
                          src_height,
                          dst_width,
                          dst_height,
                          src_pix_type,
                          dst_pix_type,
                          caps,
                          crop_area,
                          border,
                          border_value);
}

TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0000)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0010)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0011)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0100)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0110)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0111)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1000)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1010)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1011)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1100)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1110)
TEST_RESIZE_NN("rgb5652rgb565", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1111)

TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2bgr565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2bgr565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BYTE_SWAP |
                   dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0000)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0010)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0011)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0100)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0110)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0111)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1000)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1010)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1011)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1100)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1110)
TEST_RESIZE_NN("rgb565le2rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2rgb888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565be2bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0000)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0011)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0100)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0111)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1000)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1011)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1100)
TEST_RESIZE_NN("rgb565le2gray", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr565le2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("bgr565be2gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0000)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0010)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0011)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0100)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0110)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b0111)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1000)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1010)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1011)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1100)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1110)
TEST_RESIZE_NN("rgb8882rgb888", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 0, 0b1111)

TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb8882bgr888",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0000)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0010)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0011)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0100)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0110)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b0111)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1000)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1010)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1011)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1100)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1110)
TEST_RESIZE_NN("rgb8882rgb565le", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB565, 0, 0b1111)

TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("bgr8882rgb565le",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("bgr8882rgb565be",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0000)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0011)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0100)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0111)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1000)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1011)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1100)
TEST_RESIZE_NN("rgb8882gray", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1111)

TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr8882gray",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0000)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0011)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0100)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b0111)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1000)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1011)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1100)
TEST_RESIZE_NN("gray2gray", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY, 0, 0b1111)

TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0010)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0110)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1010)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1110)
TEST_RESIZE_NN(
    "rgb565le2rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2rgb888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565be2bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb565le2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr565le2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("bgr565be2gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0010)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0110)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1010)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1110)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8, 0, 0b1111)

TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb8882bgr888_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb8882gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1111)

TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr8882gray_qint8",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0000)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0011)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0100)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b0111)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1000)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1011)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1100)
TEST_RESIZE_NN("gray2gray_qint8", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT8, 0, 0b1111)

TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0000)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0010)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0011)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0100)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0110)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b0111)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1000)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1010)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1011)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1100)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1110)
TEST_RESIZE_NN("rgb565le2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               0,
               0b1111)

TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2rgb888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565le2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb565be2bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb565le2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr565le2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("bgr565be2gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP | dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0010)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0110)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1010)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1110)
TEST_RESIZE_NN(
    "rgb8882rgb888_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16, 0, 0b1111)

TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("rgb8882bgr888_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_RGB888_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0000)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0011)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0100)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0111)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1000)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1011)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1100)
TEST_RESIZE_NN(
    "rgb8882gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1111)

TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr8882gray_qint16",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0000)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0011)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0100)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b0111)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1000)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1011)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1100)
TEST_RESIZE_NN(
    "gray2gray_qint16", dl::image::DL_IMAGE_PIX_TYPE_GRAY, dl::image::DL_IMAGE_PIX_TYPE_GRAY_QINT16, 0, 0b1111)

TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0000)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0010)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0011)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0100)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0110)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0111)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1000)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1010)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1011)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1100)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1110)
TEST_RESIZE_NN("rgb8882hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB888, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1111)

TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("bgr8882hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB888,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0000)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0010)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0011)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0100)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0110)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b0111)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1000)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1010)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1011)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1100)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1110)
TEST_RESIZE_NN("rgb565le2hsv", dl::image::DL_IMAGE_PIX_TYPE_RGB565, dl::image::DL_IMAGE_PIX_TYPE_HSV, 0, 0b1111)

TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0000)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0010)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0011)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0100)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0110)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b0111)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1000)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1010)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1011)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1100)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1110)
TEST_RESIZE_NN("rgb565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN,
               0b1111)

TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("bgr565le2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)

TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0000)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0010)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0011)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0100)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0110)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b0111)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1000)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1010)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1011)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1100)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1110)
TEST_RESIZE_NN("bgr565be2hsv",
               dl::image::DL_IMAGE_PIX_TYPE_RGB565,
               dl::image::DL_IMAGE_PIX_TYPE_HSV,
               dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN | dl::image::DL_IMAGE_CAP_RGB_SWAP,
               0b1111)
#endif
