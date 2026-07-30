#include "pp_ocr_v6_image_preprocessor.hpp"
#include "dl_math_matrix.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

static const char *TAG = "pp_ocr_v6";

namespace pp_ocr_v6 {
static uint8_t sample_channel(const dl::image::img_t &img, int x, int y, int c)
{
    x = std::clamp(x, 0, static_cast<int>(img.width) - 1);
    y = std::clamp(y, 0, static_cast<int>(img.height) - 1);
    return static_cast<uint8_t *>(img.data)[(y * img.width + x) * 3 + c];
}

static void cubic_weights(float x, float coeffs[4])
{
    constexpr float A = -0.75f;
    coeffs[0] = ((A * (x + 1.0f) - 5.0f * A) * (x + 1.0f) + 8.0f * A) * (x + 1.0f) - 4.0f * A;
    coeffs[1] = ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
    float x1 = 1.0f - x;
    coeffs[2] = ((A + 2.0f) * x1 - (A + 3.0f)) * x1 * x1 + 1.0f;
    coeffs[3] = 1.0f - coeffs[0] - coeffs[1] - coeffs[2];
}

static float sample_cubic_channel(const dl::image::img_t &img, float x, float y, int c)
{
    int ix = static_cast<int>(std::floor(x));
    int iy = static_cast<int>(std::floor(y));
    float wx[4];
    float wy[4];
    cubic_weights(x - ix, wx);
    cubic_weights(y - iy, wy);

    float value = 0.0f;
    for (int ky = 0; ky < 4; ++ky) {
        for (int kx = 0; kx < 4; ++kx) {
            value += wy[ky] * wx[kx] * sample_channel(img, ix + kx - 1, iy + ky - 1, c);
        }
    }
    return std::clamp(value, 0.0f, 255.0f);
}

static float tensor_inv_scale(const dl::TensorBase *tensor)
{
    int exp = tensor->exponent;
    return exp > 0 ? 1.0f / (1 << exp) : static_cast<float>(1 << -exp);
}

template <typename QuantType>
static void resize_norm_img_impl(const dl::image::img_t &img, dl::TensorBase *model_input, float inv_scale)
{
    const int img_h = model_input->shape[1];
    const int img_w = model_input->shape[2];
    const int h = img.height;
    const int w = img.width;
    const float ratio = static_cast<float>(w) / static_cast<float>(h);
    int resized_w = 0;
    if (std::ceil(static_cast<float>(img_h) * ratio) > img_w) {
        resized_w = img_w;
    } else {
        resized_w = static_cast<int>(std::ceil(static_cast<float>(img_h) * ratio));
    }
    resized_w = std::max(1, resized_w);

    QuantType *dst = static_cast<QuantType *>(model_input->data);
    std::fill(dst, dst + img_w * img_h * 3, dl::quantize<QuantType>(0.0f, inv_scale));

    const float inv_scale_x = static_cast<float>(w) / resized_w;
    const float inv_scale_y = static_cast<float>(h) / img_h;
    for (int y = 0; y < img_h; ++y) {
        float src_y = (y + 0.5f) * inv_scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        float wy = src_y - y0;
        for (int x = 0; x < resized_w; ++x) {
            float src_x = (x + 0.5f) * inv_scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            float wx = src_x - x0;
            for (int c = 0; c < 3; ++c) {
                float v00 = sample_channel(img, x0, y0, c);
                float v01 = sample_channel(img, x0 + 1, y0, c);
                float v10 = sample_channel(img, x0, y0 + 1, c);
                float v11 = sample_channel(img, x0 + 1, y0 + 1, c);
                float value =
                    (1.0f - wx) * (1.0f - wy) * v00 + wx * (1.0f - wy) * v01 + (1.0f - wx) * wy * v10 + wx * wy * v11;
                value = std::clamp(value, 0.0f, 255.0f);
                value = value / 255.0f;
                value -= 0.5f;
                value /= 0.5f;
                int dst_c = 2 - c;
                dst[(y * img_w + x) * 3 + dst_c] = dl::quantize<QuantType>(value, inv_scale);
            }
        }
    }
}

std::array<Point, 4> text_box_to_points(const TextBox &box)
{
    std::array<Point, 4> points = {};
    for (int i = 0; i < 4; ++i) {
        points[i] = {static_cast<float>(box.points[i * 2]), static_cast<float>(box.points[i * 2 + 1])};
    }
    return order_points_clockwise(points);
}

void bilinear_letterbox_top_left(
    const dl::image::img_t &src, const dl::image::img_t &dst, int resize_w, int resize_h, uint8_t bg)
{
    if (!src.data || !dst.data) {
        return;
    }
    std::memset(dst.data, bg, static_cast<std::size_t>(dst.width) * dst.height * 3);
    const uint8_t *src_data = static_cast<const uint8_t *>(src.data);
    uint8_t *dst_data = static_cast<uint8_t *>(dst.data);
    int src_w = src.width;
    int src_h = src.height;
    float inv_scale_x = static_cast<float>(src_w) / resize_w;
    float inv_scale_y = static_cast<float>(src_h) / resize_h;

    for (int y = 0; y < resize_h; ++y) {
        float src_y = (y + 0.5f) * inv_scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        float wy = src_y - y0;
        int y0c = std::clamp(y0, 0, src_h - 1);
        int y1c = std::clamp(y0 + 1, 0, src_h - 1);
        float wy1 = 1.0f - wy;
        for (int x = 0; x < resize_w; ++x) {
            float src_x = (x + 0.5f) * inv_scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            float wx = src_x - x0;
            int x0c = std::clamp(x0, 0, src_w - 1);
            int x1c = std::clamp(x0 + 1, 0, src_w - 1);
            float wx1 = 1.0f - wx;
            float w00 = wx1 * wy1;
            float w01 = wx * wy1;
            float w10 = wx1 * wy;
            float w11 = wx * wy;
            const uint8_t *p00 = src_data + (y0c * src_w + x0c) * 3;
            const uint8_t *p01 = src_data + (y0c * src_w + x1c) * 3;
            const uint8_t *p10 = src_data + (y1c * src_w + x0c) * 3;
            const uint8_t *p11 = src_data + (y1c * src_w + x1c) * 3;
            uint8_t *out = dst_data + (y * dst.width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float v = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
                out[c] = static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));
            }
        }
    }
}

float estimate_crop_aspect_ratio(const TextBox &box)
{
    auto pts = text_box_to_points(box);
    float w_top = euclidean_distance(pts[0], pts[1]);
    float w_bot = euclidean_distance(pts[3], pts[2]);
    float h_left = euclidean_distance(pts[0], pts[3]);
    float h_right = euclidean_distance(pts[1], pts[2]);
    float w = std::max(w_top, w_bot);
    float h = std::max(h_left, h_right);
    if (w < 1e-3f || h < 1e-3f) {
        return 0.0f;
    }
    if (h / w >= 1.5f) {
        std::swap(w, h);
    }
    return w / h;
}

dl::image::img_t get_rotate_crop_image(const dl::image::img_t &img, const TextBox &box)
{
    auto points = text_box_to_points(box);
    float width_top = euclidean_distance(points[0], points[1]);
    float width_bottom = euclidean_distance(points[3], points[2]);
    float height_left = euclidean_distance(points[0], points[3]);
    float height_right = euclidean_distance(points[1], points[2]);
    int crop_w = std::max(1, static_cast<int>(std::max(width_top, width_bottom)));
    int crop_h = std::max(1, static_cast<int>(std::max(height_left, height_right)));

    dl::math::Matrix<float> src(4, 2);
    src.array[0][0] = 0;
    src.array[0][1] = 0;
    src.array[1][0] = crop_w;
    src.array[1][1] = 0;
    src.array[2][0] = crop_w;
    src.array[2][1] = crop_h;
    src.array[3][0] = 0;
    src.array[3][1] = crop_h;

    dl::math::Matrix<float> dst(4, 2);
    for (int i = 0; i < 4; ++i) {
        dst.array[i][0] = points[i].x;
        dst.array[i][1] = points[i].y;
    }
    dl::math::Matrix<float> M = dl::math::get_perspective_transform(src, dst);

    uint8_t *data = static_cast<uint8_t *>(heap_caps_malloc(crop_w * crop_h * 3, MALLOC_CAP_DEFAULT));
    if (!data) {
        ESP_LOGE(TAG, "Failed to allocate OCR crop.");
        return {};
    }

    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            float denom = M.array[2][0] * x + M.array[2][1] * y + M.array[2][2];
            float src_x = (M.array[0][0] * x + M.array[0][1] * y + M.array[0][2]) / denom;
            float src_y = (M.array[1][0] * x + M.array[1][1] * y + M.array[1][2]) / denom;
            for (int c = 0; c < 3; ++c) {
                float value = sample_cubic_channel(img, src_x, src_y, c);
                data[(y * crop_w + x) * 3 + c] = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
            }
        }
    }

    if (static_cast<float>(crop_h) / crop_w >= 1.5f) {
        int rotated_w = crop_h;
        int rotated_h = crop_w;
        uint8_t *rotated = static_cast<uint8_t *>(heap_caps_malloc(rotated_w * rotated_h * 3, MALLOC_CAP_DEFAULT));
        if (rotated) {
            for (int y = 0; y < crop_h; ++y) {
                for (int x = 0; x < crop_w; ++x) {
                    int rx = y;
                    int ry = crop_w - 1 - x;
                    for (int c = 0; c < 3; ++c) {
                        rotated[(ry * rotated_w + rx) * 3 + c] = data[(y * crop_w + x) * 3 + c];
                    }
                }
            }
            heap_caps_free(data);
            data = rotated;
            crop_w = rotated_w;
            crop_h = rotated_h;
        }
    }

    return {.data = data,
            .width = static_cast<uint16_t>(crop_w),
            .height = static_cast<uint16_t>(crop_h),
            .pix_type = img.pix_type};
}

bool resize_norm_img(const dl::image::img_t &img, dl::TensorBase *model_input)
{
    if (!img.data || !model_input || model_input->shape.size() != 4 || model_input->shape[3] != 3) {
        return false;
    }
    if (img.width <= 0 || img.height <= 0) {
        return false;
    }

    float inv_scale = tensor_inv_scale(model_input);
    if (model_input->get_dtype() == dl::DATA_TYPE_INT8) {
        resize_norm_img_impl<int8_t>(img, model_input, inv_scale);
        return true;
    }
    if (model_input->get_dtype() == dl::DATA_TYPE_INT16) {
        resize_norm_img_impl<int16_t>(img, model_input, inv_scale);
        return true;
    }

    ESP_LOGE(TAG, "Unsupported recognition input dtype: %d.", model_input->get_dtype());
    return false;
}

} // namespace pp_ocr_v6
