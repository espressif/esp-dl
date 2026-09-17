/* Sum eight int16 channels across rows.
 * a0: aligned input; a1: row stride in bytes, a multiple of 16.
 * a2: row count, 1 to 65536; a3: aligned int32 output[8].
 * The caller must keep each sum within the int32 range.
 */
#include "sdkconfig.h"
#if CONFIG_IDF_TARGET_ESP32P4
    .text
    .balign 4
    .global dl_esp32p4_reduce_inner_s16
    .type dl_esp32p4_reduce_inner_s16, @function
dl_esp32p4_reduce_inner_s16:
    addi sp, sp, -64
    mv a4, sp
    li a5, 1
    sh a5, 0(a4)
    esp.vldbc.16.ip q1, a4, 0
    esp.zero.qacc
1:
    esp.vld.128.xp q0, a0, a1
    esp.vmulas.s16.qacc q0, q1
    addi a2, a2, -1
    bnez a2, 1b
    mv a4, sp
    esp.st.qacc.l.l.128.ip a4, 16
    esp.st.qacc.l.h.128.ip a4, 16
    esp.st.qacc.h.l.128.ip a4, 16
    esp.st.qacc.h.h.128.ip a4, 16
    mv a4, sp
    esp.vld.128.ip q2, a4, 16
    esp.vld.128.ip q3, a4, 16
    esp.vld.128.ip q4, a4, 16
    esp.vld.128.ip q5, a4, 16
    esp.vunzip.32 q2, q3
    esp.vunzip.32 q4, q5
    esp.vst.128.ip q2, a3, 16
    esp.vst.128.ip q4, a3, 16
    addi sp, sp, 64
    ret
    .size dl_esp32p4_reduce_inner_s16, .-dl_esp32p4_reduce_inner_s16
#endif
