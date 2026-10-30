#pragma once

#include "dl_base_conv_args.hpp"
#include "dl_define.hpp"
#include "dl_kernel.hpp"
#include "esp_log.h"
#include <climits>
#include <cstdint>
#include <vector>

namespace dl {
namespace base {

/** Pack Conv.yml features. Must match cmake/compile_finalize.py pack_conv_key(). */
uint16_t dl_conv_pack_key(int dtype, int group_dw, int kshape, int bias, int act, int per_ch, int c_align);

/** Look up intern names for an old-model Conv. Missing slots are nullptr. */
bool dl_conv_select(uint16_t key, const char **slot0, const char **slot1, const char **slot2);

/** Build the key from runtime args (old models). group_dw: 0=Conv, 1=depthwise.
 * Pointers are assumed 16-byte aligned; c_align is channel-width only. */
inline uint16_t dl_conv_key_from_args(const ConvArgsType &args, int group_dw, quant_type_t quant)
{
    int dtype = 0; // s8
    if (quant == QUANT_TYPE_SYMM_W8A16) {
        dtype = 2;
    } else if (quant == QUANT_TYPE_SYMM_16BIT) {
        dtype = 1;
    }
    int kshape = 2; // hw
    if (args.filter_height == 1 && args.filter_width == 1) {
        kshape = 0;
    } else if (args.filter_height == 3 && args.filter_width == 3) {
        kshape = 1;
    }
    int act = (int)args.activation_type;
    if (act < 0 || act > 3) {
        act = 0;
    }
    int c_align = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
    int lanes = (dtype == 1) ? 8 : 16; // s16: 8; s8/w8a16: 16
    if (dtype == 2) {
        lanes = 8;
    }
    c_align = (args.input_channel % lanes == 0) && (args.output_channel % lanes == 0);
#endif
    return dl_conv_pack_key(
        dtype, group_dw ? 1 : 0, kshape, args.bias_element ? 1 : 0, act, args.mac_shift == INT_MIN ? 1 : 0, c_align);
}

/** Old EDL: pack args → intern names → lookup into fn[3] (body, border, tail). */
inline bool dl_conv_bind_from_args(const ConvArgsType &args, int group_dw, quant_type_t q, dl_kernel_erased_t fn[3])
{
    uint16_t key = dl_conv_key_from_args(args, group_dw, q);
    const char *s0 = nullptr;
    const char *s1 = nullptr;
    const char *s2 = nullptr;
    if (!dl_conv_select(key, &s0, &s1, &s2)) {
        return false;
    }
    fn[0] = dl_kernel_lookup(s0);
    fn[1] = dl_kernel_lookup(s1);
    fn[2] = dl_kernel_lookup(s2);
    if (!fn[0]) {
        return false;
    }
#if !(CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST)
    if (!fn[2]) {
        return false;
    }
#endif
    return true;
}

/** If body (and C tail) already bound, no-op. Else packed-select once. */
inline bool dl_conv_ensure_kernels(
    dl_kernel_erased_t fn[3], const ConvArgsType &args, int group_dw, quant_type_t q, const char *tag)
{
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
    if (fn[0]) {
        return true;
    }
#else
    if (fn[0] && fn[2]) {
        return true;
    }
#endif
    if (!dl_conv_bind_from_args(args, group_dw, q, fn)) {
        ESP_LOGE(tag, "no Conv kernel for packed key 0x%04x", dl_conv_key_from_args(args, group_dw, q));
        return false;
    }
    return true;
}

} // namespace base
} // namespace dl
