#pragma once

#include <stdint.h>

extern "C" {
void dl_riscv_w8a16_mac_3x4(
    const int8_t *f0, const int8_t *f1, const int8_t *f2, const int16_t *xp, const int16_t *xe, int32_t *acc);
void dl_riscv_s16_mac_2x4(const int16_t *f0, const int16_t *f1, const int16_t *xp, const int16_t *xe, int32_t *sh);
}
