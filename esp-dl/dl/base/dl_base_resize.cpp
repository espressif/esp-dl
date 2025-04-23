#include "dl_base_resize.hpp"

#include "dl_base_isa.hpp"

namespace dl {
namespace base {
template <typename feature_t>
inline void resize_nearest_2x2_c1(feature_t *output_ptr, feature_t *input_ptr, void *args_ptr)
{
    resizeArgsType<feature_t> *args_ptr_t = reinterpret_cast<resizeArgsType<feature_t> *>(args_ptr);
    feature_t *output_ptr_0_0 = output_ptr;
    feature_t *output_ptr_0_1 = output_ptr + args_ptr_t->output_x_offset;
    feature_t *output_ptr_1_0 = output_ptr + args_ptr_t->output_y_offset;
    feature_t *output_ptr_1_1 = output_ptr_1_0 + args_ptr_t->output_x_offset;

    for (int i = 0; i < args_ptr_t->input_channel; i++) {
        feature_t output_value =
            tool::round((float)(*input_ptr++) * args_ptr_t->output_scale / (1 << args_ptr_t->output_shift));
        *(output_ptr_0_0++) = output_value;
        *(output_ptr_0_1++) = output_value;
        *(output_ptr_1_0++) = output_value;
        *(output_ptr_1_1++) = output_value;
    }
}

template <typename feature_t>
inline void resize_nearest_c1(feature_t *output_ptr, feature_t *input_ptr, void *args_ptr)
{
    resizeArgsType<feature_t> *args_ptr_t = reinterpret_cast<resizeArgsType<feature_t> *>(args_ptr);
    for (int i = 0; i < args_ptr_t->input_channel; i++) {
        *(output_ptr++) =
            tool::round((float)(*input_ptr++) * args_ptr_t->output_scale / (1 << args_ptr_t->output_shift));
    }
}

inline void load_resize_nearest_2x2_c1_s8(ImplFunc_t<int8_t, int8_t> &impl_func, const resizeArgsType<int8_t> &args)
{
#if CONFIG_ESP32P4_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_esp32p4_s8_resize_nearest_2x2_c1;
    } else {
        impl_func = dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1;
    }
#elif CONFIG_TIE728_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_tie728_s8_resize_nearest_2x2_c1;
    } else {
        impl_func = dl_tie728_s8_unaligned_resize_nearest_2x2_c1;
    }
#else
    impl_func = resize_nearest_2x2_c1<int8_t>;
#endif
}

inline void load_resize_nearest_c1_s8(ImplFunc_t<int8_t, int8_t> &impl_func, const resizeArgsType<int8_t> &args)
{
#if CONFIG_ESP32P4_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_esp32p4_s8_resize_nearest_c1;
    } else {
        impl_func = dl_esp32p4_s8_unaligned_resize_nearest_c1;
    }
#elif CONFIG_TIE728_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_tie728_s8_resize_nearest_c1;
    } else {
        impl_func = dl_tie728_s8_unaligned_resize_nearest_c1;
    }
#else
    impl_func = resize_nearest_c1<int8_t>;
#endif
}

// It's for non cache.
void linear_coeffs(
    int out_x, int &in_x, float &ratio_0, float &ratio_1, int in_length, float scale_inv, int align_corners)
{
    float fx = 0.f;
    if (align_corners) {
        fx = out_x * scale_inv;
    } else {
        // Aligned with PyTorch, the `coordinate_transformation_mode` of `linear` is "half_pixel".
        fx = (out_x + 0.5f) * scale_inv - 0.5f;
    }

    in_x = static_cast<int>(floorf(fx));
    fx -= in_x;

    if (in_x < 0) {
        in_x = 0;
        fx = 0.f;
    }
    if (in_x >= in_length - 1) {
        in_x = in_length - 2;
        fx = 1.f;
    }
    ratio_0 = 1.f - fx;
    ratio_1 = fx;
}

// It's for cache in_x coordinates + in_x ratio.
void linear_coeffs(int out_length, int in_length, int *in_xp, float *ratio, float scale_inv, int align_corners)
{
    int in_x;
    float ratio_0, ratio_1;
    for (int out_x = 0; out_x < out_length; out_x++) {
        linear_coeffs(out_x, in_x, ratio_0, ratio_1, in_length, scale_inv, align_corners);
        in_xp[out_x] = in_x;
        ratio[out_x * 2] = ratio_0;
        ratio[out_x * 2 + 1] = ratio_1;
    }
}

template <typename feature_t>
inline void resize_linear_c1(const resizeArgsType<feature_t> &args)
{
    int in_x, in_y;
    float x_ratio_0, x_ratio_1, y_ratio_0, y_ratio_1;
    float input_scale = DL_SCALE(args.input_exponent);
    float output_inv_scale = DL_RESCALE(args.output_exponent);

    if (args.dims == 3) {
        // 1d linear resize
        int prev_in_x = -2;
        float *cols0 = args.cache;
        float *cols1 = cols0 + args.input_channel;

        for (int x = 0; x < args.output_width; x++) {
            linear_coeffs(x, in_x, x_ratio_0, x_ratio_1, args.input_width, args.scale_w_inv, args.align_corners);

            if (in_x == prev_in_x) {
                // reuse all cols
            } else if (in_x == prev_in_x + 1) {
                // wresize one col
                float *cols0_use = cols0;
                cols0 = cols1;
                cols1 = cols0_use;

                feature_t *input_x1 = args.input_element + (in_x + 1) * args.input_channel;
                for (int c = 0; c < args.input_channel; c++) {
                    cols1[c] = dequantize(input_x1[c], input_scale);
                }
            } else {
                // wresize two cols
                feature_t *input_x0 = args.input_element + in_x * args.input_channel;
                feature_t *input_x1 = args.input_element + (in_x + 1) * args.input_channel;

                for (int c = 0; c < args.input_channel; c++) {
                    cols0[c] = dequantize(input_x0[c], input_scale);
                    cols1[c] = dequantize(input_x1[c], input_scale);
                }
            }
            prev_in_x = in_x;

            feature_t *output_x = args.output_element + x * args.input_channel;
            for (int c = 0; c < args.input_channel; c++) {
                output_x[c] = quantize<feature_t>(cols0[c] * x_ratio_0 + cols1[c] * x_ratio_1, output_inv_scale);
            }
        }
    } else if (args.dims == 4) {
        // 2d linear resize
        int prev_in_y = -2;
        int *in_xp = reinterpret_cast<int *>(args.cache);
        float *ratio = args.cache + args.output_width;
        float *rows0 = args.cache + args.output_width + args.output_width * 2;
        float *rows1 = rows0 + args.output_width * args.input_channel;
        // Cache the calculation results of the width in advance to avoid repeated calculations.
        linear_coeffs(args.output_width, args.input_width, in_xp, ratio, args.scale_w_inv, args.align_corners);

        for (int y = 0; y < args.output_height; y++) {
            linear_coeffs(y, in_y, y_ratio_0, y_ratio_1, args.input_height, args.scale_h_inv, args.align_corners);

            if (in_y == prev_in_y) {
                // reuse all rows
            } else if (in_y == prev_in_y + 1) {
                // hresize one row
                float *rows0_use = rows0;
                rows0 = rows1;
                rows1 = rows0_use;

                feature_t *input_y1 = args.input_element + (in_y + 1) * args.input_width * args.input_channel;
                for (int x = 0; x < args.output_width; x++) {
                    in_x = in_xp[x];
                    x_ratio_0 = ratio[x * 2];
                    x_ratio_1 = ratio[x * 2 + 1];
                    feature_t *input_x0_y1 = input_y1 + in_x * args.input_channel;
                    feature_t *input_x1_y1 = input_y1 + (in_x + 1) * args.input_channel;
                    float *rows1_tmp = rows1 + x * args.input_channel;

                    for (int c = 0; c < args.input_channel; c++) {
                        rows1_tmp[c] = dequantize(input_x0_y1[c], input_scale) * x_ratio_0 +
                            dequantize(input_x1_y1[c], input_scale) * x_ratio_1;
                    }
                }
            } else {
                // hresize two rows
                feature_t *input_y0 = args.input_element + in_y * args.input_width * args.input_channel;
                feature_t *input_y1 = args.input_element + (in_y + 1) * args.input_width * args.input_channel;
                for (int x = 0; x < args.output_width; x++) {
                    in_x = in_xp[x];
                    x_ratio_0 = ratio[x * 2];
                    x_ratio_1 = ratio[x * 2 + 1];
                    feature_t *input_x0_y0 = input_y0 + in_x * args.input_channel;
                    feature_t *input_x1_y0 = input_y0 + (in_x + 1) * args.input_channel;
                    feature_t *input_x0_y1 = input_y1 + in_x * args.input_channel;
                    feature_t *input_x1_y1 = input_y1 + (in_x + 1) * args.input_channel;
                    float *rows0_tmp = rows0 + x * args.input_channel;
                    float *rows1_tmp = rows1 + x * args.input_channel;

                    for (int c = 0; c < args.input_channel; c++) {
                        rows0_tmp[c] = dequantize(input_x0_y0[c], input_scale) * x_ratio_0 +
                            dequantize(input_x1_y0[c], input_scale) * x_ratio_1;
                        rows1_tmp[c] = dequantize(input_x0_y1[c], input_scale) * x_ratio_0 +
                            dequantize(input_x1_y1[c], input_scale) * x_ratio_1;
                    }
                }
            }
            prev_in_y = in_y;

            feature_t *output_y = args.output_element + y * args.output_width * args.input_channel;
            for (int x = 0; x < args.output_width; x++) {
                feature_t *output_x_y = output_y + x * args.input_channel;
                float *rows0_tmp = rows0 + x * args.input_channel;
                float *rows1_tmp = rows1 + x * args.input_channel;
                for (int c = 0; c < args.input_channel; c++) {
                    output_x_y[c] =
                        quantize<feature_t>(rows0_tmp[c] * y_ratio_0 + rows1_tmp[c] * y_ratio_1, output_inv_scale);
                }
            }
        }
    }
}

template <typename feature_t>
void resize_operation_shell(const resizeArgsType<feature_t> &args, ImplFunc_t<feature_t, feature_t> resize_impl_func)
{
    feature_t *input_ptr = args.input_element;
    feature_t *output_ptr = args.output_element;

    if (args.resize_mode == RESIZE_NEAREST) {
        if (args.scale_h == 2 && args.scale_w == 2) {
            for (int i = 0; i < args.input_height; i++) {
                for (int j = 0; j < args.input_width; j++) {
                    resize_impl_func(output_ptr, input_ptr, (void *)(&args));
                    input_ptr += args.input_channel;
                    output_ptr += args.input_channel * 2;
                }
                output_ptr += args.input_channel * 2 * args.input_width;
            }
        } else {
            // support 1d/2d nearest mode
            float scale_h_inv = args.scale_h_inv;
            float scale_w_inv = args.scale_w_inv;
            // Aligned with PyTorch, the `coordinate_transformation_mode` of `nearest` is "asymmetric".
            for (int y = 0; y < args.output_height; y++) {
                int in_y = std::min((int)(y * scale_h_inv), (args.input_height - 1));
                feature_t *input_y_ptr = input_ptr + in_y * args.input_width * args.input_channel;
                feature_t *out_y_ptr = output_ptr + y * args.output_width * args.input_channel;

                for (int x = 0; x < args.output_width; x++) {
                    int in_x = std::min((int)(x * scale_w_inv), (args.input_width - 1));
                    resize_impl_func(
                        out_y_ptr + x * args.input_channel, input_y_ptr + in_x * args.input_channel, (void *)(&args));
                }
            }
        }
    } else if (args.resize_mode == RESIZE_LINEAR) {
        // Linear does not support instruction acceleration.
        resize_linear_c1(args);
    } else {
        ESP_LOGE("resize", "Don't support this mode: %d.", args.resize_mode);
    }

    return;
}

template <>
void resize<int8_t>(void *args_ptr)
{
    const resizeArgsType<int8_t> &args = *((resizeArgsType<int8_t> *)args_ptr);
    ImplFunc_t<int8_t, int8_t> impl_func;
    if (args.resize_mode == RESIZE_NEAREST) {
        if (args.scale_h == 2 && args.scale_w == 2) {
            load_resize_nearest_2x2_c1_s8(impl_func, args);
        } else {
            // 3d or other 4d
            load_resize_nearest_c1_s8(impl_func, args);
        }
    }
    resize_operation_shell<int8_t>(args, impl_func);
}

template <>
void resize<int16_t>(void *args_ptr)
{
    // const resizeArgsType<int16_t> &args = *((resizeArgsType<int16_t> *)args_ptr);
    // if (args.resize_mode == RESIZE_NEAREST){
    //     if (args.scale_h == 2 && args.scale_w == 2){
    //         ImplFunc_t<int16_t, int16_t> impl_func;
    //         resize_c_impl_func_s16_t c_impl_func = NULL;
    //         load_resize_nearest_2x2_c1_s16(impl_func, c_impl_func, args);
    //         resize_operation_shell<int16_t>(args, impl_func, c_impl_func);
    //     }
    // }
}

} // namespace base
} // namespace dl
