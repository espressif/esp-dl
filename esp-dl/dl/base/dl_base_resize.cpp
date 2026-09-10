#include "dl_base_resize.hpp"

#include "dl_base_isa.hpp"

namespace dl {
namespace base {

// dl::dequantize and dl::quantize are only explicitly instantiated in
// dl_tensor_base.cpp, so calling them costs a real call per channel. The
// bilinear inner loop does up to five of those per output element, which is
// why they are repeated here; the arithmetic is identical.
template <typename feature_t>
inline float dequant_inline(feature_t input, float scale)
{
    return static_cast<float>(input) * scale;
}

template <typename feature_t>
inline feature_t quant_inline(float input, float inv_scale)
{
    int output = tool::round(input * inv_scale);
    if constexpr (sizeof(feature_t) == 1) {
        output = DL_CLIP(output, DL_QUANT8_MIN, DL_QUANT8_MAX);
    } else {
        output = DL_CLIP(output, DL_QUANT16_MIN, DL_QUANT16_MAX);
    }
    return static_cast<feature_t>(output);
}

// Fixed-point format for the bilinear taps. Q14 rather than Q15 because a
// ratio of exactly 1.0 occurs whenever linear_coeffs clamps at a border, and
// 1<<15 does not fit the int16 scalar operand that ESP.VSMULAS.S16.QACC reads;
// 1<<14 does, and measures the same as Q15 against an fp64 reference (both
// stay inside the +-1 LSB the .espdl gate allows, on every shape tested).
constexpr int kBilinearQBits = 14;
constexpr int32_t kBilinearQOne = 1 << kBilinearQBits;

// tool::shift_and_round is only instantiated in dl_tool.cpp, so it costs a
// call per output element. Same arithmetic, same rounding mode.
inline int32_t shift_round_inline(int32_t value, int shift)
{
    if (shift <= 0) {
        return value << -shift;
    }
#if CONFIG_ROUND_HALF_EVEN_ENABLED
    int32_t shifted = value >> shift;
    const int32_t remainder = value & ((static_cast<int32_t>(1) << shift) - 1);
    const int32_t half = static_cast<int32_t>(1) << (shift - 1);
    if (remainder > half || (remainder == half && (shifted & 1) != 0)) {
        shifted += 1;
    }
    return shifted;
#else
    return (value + (static_cast<int32_t>(1) << (shift - 1))) >> shift;
#endif
}

// Quantise one interpolation ratio to Q14. linear_coeffs guarantees
// ratio in [0, 1], so the clamp only absorbs the rounding at the ends.
inline int32_t bilinear_weight_q14(float ratio)
{
    int32_t w = tool::round(ratio * static_cast<float>(kBilinearQOne));
    return DL_CLIP(w, 0, kBilinearQOne);
}

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
#if CONFIG_PIE_V2_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_esp32p4_s8_resize_nearest_2x2_c1;
    } else {
        impl_func = dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1;
    }
#elif CONFIG_PIE_V1_BOOST
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
#if CONFIG_PIE_V2_BOOST
    if (args.input_channel % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        impl_func = dl_esp32p4_s8_resize_nearest_c1;
    } else {
        impl_func = dl_esp32p4_s8_unaligned_resize_nearest_c1;
    }
#elif CONFIG_PIE_V1_BOOST
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

// 2d bilinear for int8 features, in Q14 fixed point. The bilinear weights are
// separable, so the two interpolation stages collapse into one 4-tap integer
// dot product:
//   out = round((W00*q00 + W10*q10 + W01*q01 + W11*q11) >> shift)
// That dequantizes nothing, rounds once instead of twice, and needs no
// interpolated rows on the side - the float path below caches
// output_width*channel*2 floats (~121KB at 80x192) purely to avoid redoing the
// horizontal pass for every output row that maps to the same source pair.
//
// int8 only. The gate is an absolute +-1 LSB, so an int16 range needs the taps
// 8 bits finer than an int8 one for the same margin: Q14 measures 0 LSB
// against an fp64 reference for int8 but 2 LSB for int16, and int16 only gets
// back inside the gate at Q16. Q16 is one step too far - the scalar operand of
// VSMULAS.S16.QACC is an int16 half-word, which is what caps the taps at Q14
// in the first place (Q15 already overflows it, a border ratio of 1.0 being
// 1<<15), and the accumulator would reach 2^31. There is no width that clears
// both, so int16 stays on the float path, where a 24-bit fp32 mantissa covers
// 16-bit data comfortably.
#if CONFIG_PIE_V2_BOOST
// Mirrors the layout dl_esp32p4_s8_resize_linear_taps reads.
struct alignas(16) resizeLinearTapsType {
    int16_t w[8];
    const int8_t *p[4];
    int32_t c_div_16;
    int32_t shift;
};
#endif

inline void resize_linear_2d_q14(const resizeArgsType<int8_t> &args)
{
    int in_x, in_y;
    float x_ratio_0, x_ratio_1, y_ratio_0, y_ratio_1;

    int *in_xp = reinterpret_cast<int *>(args.cache);
    int *x_weight = in_xp + args.output_width;
    for (int x = 0; x < args.output_width; x++) {
        linear_coeffs(x, in_x, x_ratio_0, x_ratio_1, args.input_width, args.scale_w_inv, args.align_corners);
        in_xp[x] = in_x;
        const int32_t w0 = bilinear_weight_q14(x_ratio_0);
        x_weight[x * 2] = w0;
        x_weight[x * 2 + 1] = kBilinearQOne - w0;
    }

    // The four taps sum to 1<<kBilinearQBits, so the accumulator peaks at
    // (1<<kBilinearQBits) * 127 and stays well inside int32.
    const int shift = kBilinearQBits - (args.input_exponent - args.output_exponent);
    const int in_y_stride = args.input_width * args.input_channel;
    const int channel = args.input_channel;

#if CONFIG_PIE_V2_BOOST
    // The PIE kernel does 16 channels per iteration off 16-byte aligned
    // loads. channel % 16 == 0 also makes every in_x * channel offset a
    // multiple of 16, so checking the two base pointers covers all four
    // source vectors and the output. A runtime shift below 1 cannot be fed
    // to srcmb, which treats it as an unsigned amount.
    const bool use_simd = channel % 16 == 0 && shift > 0 && shift < 32 &&
        !(reinterpret_cast<uintptr_t>(args.input_element) & 15) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 15);
    resizeLinearTapsType taps;
    if (use_simd) {
        taps.w[4] = DL_QUANT8_MAX;
        taps.w[5] = DL_QUANT8_MIN;
        taps.w[6] = 0;
        taps.w[7] = 0;
        taps.c_div_16 = channel / 16;
        taps.shift = shift;
    }
#endif

    for (int y = 0; y < args.output_height; y++) {
        linear_coeffs(y, in_y, y_ratio_0, y_ratio_1, args.input_height, args.scale_h_inv, args.align_corners);
        const int32_t wy0 = bilinear_weight_q14(y_ratio_0);
        const int32_t wy1 = kBilinearQOne - wy0;

        int8_t *input_y0 = args.input_element + in_y * in_y_stride;
        int8_t *input_y1 = input_y0 + in_y_stride;
        int8_t *output_y = args.output_element + y * args.output_width * channel;

        for (int x = 0; x < args.output_width; x++) {
            const int32_t wx0 = x_weight[x * 2];
            const int32_t wx1 = x_weight[x * 2 + 1];
            // Round each product back down to Q(kBilinearQBits), then push the
            // residue onto the largest tap so the four still sum to exactly 1
            // and a flat neighbourhood reproduces its own value.
            int32_t w[4] = {(wx0 * wy0 + (kBilinearQOne >> 1)) >> kBilinearQBits,
                            (wx1 * wy0 + (kBilinearQOne >> 1)) >> kBilinearQBits,
                            (wx0 * wy1 + (kBilinearQOne >> 1)) >> kBilinearQBits,
                            (wx1 * wy1 + (kBilinearQOne >> 1)) >> kBilinearQBits};
            int largest = 0;
            for (int i = 1; i < 4; i++) {
                if (w[i] > w[largest]) {
                    largest = i;
                }
            }
            w[largest] += kBilinearQOne - (w[0] + w[1] + w[2] + w[3]);

            const int x0_offset = in_xp[x] * channel;
            int8_t *input_x0_y0 = input_y0 + x0_offset;
            int8_t *input_x1_y0 = input_x0_y0 + channel;
            int8_t *input_x0_y1 = input_y1 + x0_offset;
            int8_t *input_x1_y1 = input_x0_y1 + channel;
            int8_t *output_x_y = output_y + x * channel;

#if CONFIG_PIE_V2_BOOST
            if (use_simd) {
                taps.w[0] = static_cast<int16_t>(w[0]);
                taps.w[1] = static_cast<int16_t>(w[1]);
                taps.w[2] = static_cast<int16_t>(w[2]);
                taps.w[3] = static_cast<int16_t>(w[3]);
                taps.p[0] = input_x0_y0;
                taps.p[1] = input_x1_y0;
                taps.p[2] = input_x0_y1;
                taps.p[3] = input_x1_y1;
                dl_esp32p4_s8_resize_linear_taps(output_x_y, &taps);
                continue;
            }
#endif
            for (int c = 0; c < channel; c++) {
                const int32_t acc = w[0] * static_cast<int32_t>(input_x0_y0[c]) +
                    w[1] * static_cast<int32_t>(input_x1_y0[c]) + w[2] * static_cast<int32_t>(input_x0_y1[c]) +
                    w[3] * static_cast<int32_t>(input_x1_y1[c]);
                output_x_y[c] =
                    static_cast<int8_t>(DL_CLIP(shift_round_inline(acc, shift), DL_QUANT8_MIN, DL_QUANT8_MAX));
            }
        }
    }
}

// 2d bilinear in fp32, kept for int16 features. Caches the two horizontally
// interpolated rows so an output row landing on the same source pair does not
// repeat the horizontal pass.
template <typename feature_t>
inline void resize_linear_2d_float(const resizeArgsType<feature_t> &args)
{
    int in_x, in_y;
    float x_ratio_0, x_ratio_1, y_ratio_0, y_ratio_1;
    float input_scale = DL_SCALE(args.input_exponent);
    float output_inv_scale = DL_RESCALE(args.output_exponent);

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
                    rows1_tmp[c] = dequant_inline(input_x0_y1[c], input_scale) * x_ratio_0 +
                        dequant_inline(input_x1_y1[c], input_scale) * x_ratio_1;
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
                    rows0_tmp[c] = dequant_inline(input_x0_y0[c], input_scale) * x_ratio_0 +
                        dequant_inline(input_x1_y0[c], input_scale) * x_ratio_1;
                    rows1_tmp[c] = dequant_inline(input_x0_y1[c], input_scale) * x_ratio_0 +
                        dequant_inline(input_x1_y1[c], input_scale) * x_ratio_1;
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
                    quant_inline<feature_t>(rows0_tmp[c] * y_ratio_0 + rows1_tmp[c] * y_ratio_1, output_inv_scale);
            }
        }
    }
}

template <typename feature_t>
inline void resize_linear_c1(const resizeArgsType<feature_t> &args)
{
    int in_x;
    float x_ratio_0, x_ratio_1;
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
                    cols1[c] = dequant_inline(input_x1[c], input_scale);
                }
            } else {
                // wresize two cols
                feature_t *input_x0 = args.input_element + in_x * args.input_channel;
                feature_t *input_x1 = args.input_element + (in_x + 1) * args.input_channel;

                for (int c = 0; c < args.input_channel; c++) {
                    cols0[c] = dequant_inline(input_x0[c], input_scale);
                    cols1[c] = dequant_inline(input_x1[c], input_scale);
                }
            }
            prev_in_x = in_x;

            feature_t *output_x = args.output_element + x * args.input_channel;
            for (int c = 0; c < args.input_channel; c++) {
                output_x[c] = quant_inline<feature_t>(cols0[c] * x_ratio_0 + cols1[c] * x_ratio_1, output_inv_scale);
            }
        }
    } else if (args.dims == 4) {
        if constexpr (sizeof(feature_t) == 1) {
            resize_linear_2d_q14(args);
        } else {
            resize_linear_2d_float(args);
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
