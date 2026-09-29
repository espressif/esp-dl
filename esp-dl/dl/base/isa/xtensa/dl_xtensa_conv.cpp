#include "dl_define_private.hpp"

#if CONFIG_XTENSA_MAC16_BOOST
#include "dl_base_conv_args.hpp"
#include "dl_c_conv_common.hpp"
#include "dl_xtensa_conv.hpp"

#include <algorithm>

namespace dl {
namespace base {

// Layouts are read by dl_xtensa_s16_conv2d_1x1.S and dl_xtensa_s16_depthwise_conv2d.S.
struct XtensaConvParams {
    int32_t loops;
    uint32_t init_lo;
    int32_t init_hi;
    int32_t sar;
    int32_t relu_lo;
    uint32_t *wq;
};
struct XtensaDwParams {
    int32_t taps;
    int32_t tap_stride;
    int32_t sar;
    int32_t relu_lo;
};

extern "C" {
void dl_xtensa_s16_conv2d_1x1(
    const void *w, const int16_t *const *xs, int n_px, const XtensaConvParams *prm, int16_t *const *out, int out_off);
void dl_xtensa_w8a16_conv2d_1x1(
    const void *w, const int16_t *const *xs, int n_px, const XtensaConvParams *prm, int16_t *const *out, int out_off);
void dl_xtensa_s16_depthwise_conv2d(
    const int16_t *x, const uint32_t *wt, int pairs, const XtensaDwParams *prm, int16_t *out, const int32_t *init);
}

// Exact while |init| + C * 2^30 < 2^39 (|x * w| <= 2^30, also for the 256x int8 weights) and the
// shifted accumulator fits 32 bits (sar >= 8). Output channels whose bias breaks the first bound
// are computed in int64.
bool conv2d_1x1_s16_xtensa(
    const ConvArgsType &args, int filt_bytes, void *input_ptr, void *output_ptr, int height, int width)
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const int shift = args.mac_shift;
    const int sar = filt_bytes == 1 ? shift + 8 : shift;
    const int64_t init_max = ((int64_t)1 << 39) - 1 - ((int64_t)C << 30);
    if (C % 4 || shift < 0 || sar < 8 || sar > 31 || init_max < 0 || ((uintptr_t)input_ptr & 3) ||
        ((uintptr_t)args.filter_element & 3)) {
        return false;
    }
    const int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    const int64_t *bias = static_cast<const int64_t *>(args.bias_element);
    const uint64_t lim = filt_bytes == 1 ? init_max >> 8 : init_max;
    const bool relu = args.activation_type == ReLU;

    const int K = C / 2;
    const int words = K + 1;
    const int pixels = height * width;
    const int tile = std::max(1, std::min({pixels, 16, 4096 / (words * 4)}));
    const size_t x_bytes = (size_t)tile * words * sizeof(uint32_t);
    const size_t w_bytes = filt_bytes == 1 ? (size_t)K * sizeof(uint32_t) : 0;
    char *scratch = static_cast<char *>(conv_c_scratch_alloc(x_bytes + w_bytes + (size_t)tile * 2 * sizeof(void *)));
    if (!scratch) {
        return false;
    }
    uint32_t *xbuf = reinterpret_cast<uint32_t *>(scratch);
    uint32_t *wq = reinterpret_cast<uint32_t *>(scratch + x_bytes);
    const int16_t **xs = reinterpret_cast<const int16_t **>(scratch + x_bytes + w_bytes);
    int16_t **out = reinterpret_cast<int16_t **>(scratch + x_bytes + w_bytes) + tile;

    XtensaConvParams prm;
    prm.loops = C / 4 - 1;
    prm.sar = sar;
    prm.relu_lo = relu ? 0 : INT32_MIN;
    prm.wq = wq;
    auto kernel = filt_bytes == 1 ? dl_xtensa_w8a16_conv2d_1x1 : dl_xtensa_s16_conv2d_1x1;
    for (int p0 = 0; p0 < pixels; p0 += tile) {
        const int n = std::min(tile, pixels - p0);
        // Input words Y_j = (x[2j], x[2j - 1]), see dl_xtensa_s16_conv2d_1x1.S.
        for (int p = 0; p < n; p++) {
            const int y = (p0 + p) / width;
            const int x = (p0 + p) % width;
            const uint32_t *in =
                reinterpret_cast<const uint32_t *>(static_cast<const int16_t *>(input_ptr) +
                                                   y * args.input_stride_y_offset + x * args.input_stride_x_offset);
            uint32_t *xp = xbuf + (size_t)p * words;
            uint32_t prev = 0;
            for (int j = 0; j < K; j++) {
                const uint32_t cur = in[j];
                xp[j] = (cur << 16) | (prev >> 16);
                prev = cur;
            }
            xp[K] = prev >> 16;
            xs[p] = reinterpret_cast<const int16_t *>(xp);
            out[p] = static_cast<int16_t *>(output_ptr) + y * args.output_y_offset + x * args.output_x_offset;
        }
        for (int oc = 0; oc < N; oc++) {
            const int64_t b = (bias ? bias[oc] : 0) + half;
            const void *w = static_cast<const char *>(args.filter_element) + (size_t)oc * C * filt_bytes;
            if ((uint64_t)b + lim > 2 * lim) {
                for (int p = 0; p < n; p++) {
                    const int16_t *xv = static_cast<const int16_t *>(input_ptr) +
                        (p0 + p) / width * args.input_stride_y_offset + (p0 + p) % width * args.input_stride_x_offset;
                    int64_t acc = b - half;
                    for (int i = 0; i < C; i++) {
                        acc += (int32_t)xv[i] *
                            (filt_bytes == 1 ? (int32_t)static_cast<const int8_t *>(w)[i]
                                             : (int32_t)static_cast<const int16_t *>(w)[i]);
                    }
                    out[p][oc] = conv_c_requant_s16(acc, half, shift, relu);
                }
                continue;
            }
            const int64_t init = filt_bytes == 1 ? b * 256 : b;
            prm.init_lo = (uint32_t)init;
            prm.init_hi = (int32_t)(init >> 32);
            kernel(w, xs, n, &prm, out, oc * (int)sizeof(int16_t));
        }
    }
    heap_caps_free(scratch);
    return true;
}

// Taps along a single axis only. Exact while |bias + half| + taps * 2^30 < 2^39 and the shifted
// accumulator fits 32 bits (shift >= 8).
bool depthwise_conv2d_s16_xtensa(const ConvArgsType &args, void *input_ptr, void *output_ptr, int height, int width)
{
    const int C = args.input_channel;
    const int taps = args.filter_height * args.filter_width;
    const int shift = args.mac_shift;
    const int tap_stride = args.filter_width == 1 ? args.input_dilation_y_offset : args.input_dilation_x_offset;
    const int64_t init_max = ((int64_t)1 << 39) - 1 - ((int64_t)taps << 30);
    if ((args.filter_width != 1 && args.filter_height != 1) || (C & 1) || shift < 8 || shift > 31 || init_max < 0 ||
        ((uintptr_t)input_ptr & 3) || (tap_stride & 1) || (args.input_stride_x_offset & 1) ||
        (args.input_stride_y_offset & 1)) {
        return false;
    }
    const int pairs = C / 2;
    uint32_t *wt = static_cast<uint32_t *>(
        heap_caps_malloc(((size_t)taps * pairs + 2 * C) * sizeof(uint32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!wt) {
        return false;
    }
    int32_t *init = reinterpret_cast<int32_t *>(wt + taps * pairs);
    const int64_t half = (int64_t)1 << (shift - 1);
    const int64_t *bias = static_cast<const int64_t *>(args.bias_element);
    for (int c = 0; c < C; c++) {
        const int64_t b = (bias ? bias[c] : 0) + half;
        if (b > init_max || b < -init_max) {
            heap_caps_free(wt);
            return false;
        }
        init[2 * c] = (int32_t)b;
        init[2 * c + 1] = (int32_t)(b >> 32);
    }
    const uint16_t *filter = static_cast<const uint16_t *>(args.filter_element);
    for (int j = 0; j < pairs; j++) {
        for (int t = 0; t < taps; t++) {
            wt[j * taps + t] = filter[t * C + 2 * j] | (uint32_t)filter[t * C + 2 * j + 1] << 16;
        }
    }
    const XtensaDwParams prm = {
        taps, tap_stride * (int)sizeof(int16_t), shift, args.activation_type == ReLU ? 0 : INT32_MIN};
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dl_xtensa_s16_depthwise_conv2d(static_cast<const int16_t *>(input_ptr) + y * args.input_stride_y_offset +
                                               x * args.input_stride_x_offset,
                                           wt,
                                           pairs,
                                           &prm,
                                           static_cast<int16_t *>(output_ptr) + y * args.output_y_offset +
                                               x * args.output_x_offset,
                                           init);
        }
    }
    heap_caps_free(wt);
    return true;
}

} // namespace base
} // namespace dl
#endif
