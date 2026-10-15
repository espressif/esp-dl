#include "sdkconfig.h"
#include "esp_idf_version.h"

#if (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 5, 2))
.macro vldbc_8_ip q, a, n
    esp.vldbc.8.ip \q, \a, \n * 4
.endm

.macro vldbc_16_ip q, a, n
    esp.vldbc.16.ip \q, \a, \n * 2
.endm
#else
.macro vldbc_8_ip q, a, n
    esp.vldbc.8.ip \q, \a, \n
.endm

.macro vldbc_16_ip q, a, n
    esp.vldbc.16.ip \q, \a, \n
.endm
#endif

.extern vsl_8_1
.extern vsl_8_2
.extern vsl_8_3
.extern vsl_8_4
.extern vsl_8_5
.extern vsl_8_6
.extern vsl_8_7
.extern vsr_u8_1_mask
.extern vsr_u8_2_mask
.extern vsr_u8_3_mask
.extern vsr_u8_4_mask
.extern vsr_u8_5_mask
.extern vsr_u8_6_mask
.extern vsr_u8_7_mask
.extern vsl_16_1
.extern vsl_16_2
.extern vsl_16_3
.extern vsl_16_4
.extern vsl_16_5
.extern vsl_16_6
.extern vsl_16_7
.extern vsl_16_8
.extern vsl_16_9
.extern vsl_16_10
.extern vsl_16_11
.extern vsl_16_12
.extern vsl_16_13
.extern vsl_16_14
.extern vsl_16_15
.extern rgb5652rgb888_mask_tab
.extern rgb8882rgb565_mask_tab
.extern rgb8882gray_const_tab
.extern hsv_round_delta
.extern hsv_h_range

.macro vsl_8 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        la \a_tmp, vsl_8_1
    .elseif \n == 2
        la \a_tmp, vsl_8_2
    .elseif \n == 3
        la \a_tmp, vsl_8_3
    .elseif \n == 4
        la \a_tmp, vsl_8_4
    .elseif \n == 5
        la \a_tmp, vsl_8_5
    .elseif \n == 6
        la \a_tmp, vsl_8_6
    .elseif \n == 7
        la \a_tmp, vsl_8_7
    .else
        .error "unsupported n"
    .endif
    esp.vldbc.8.ip \q_tmp, \a_tmp, 0
#if (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 5, 2)) || defined(CONFIG_ESP32P4_SELECTS_REV_LESS_V3)
    esp.vsld.8 \qd, \qs, \q_tmp
#else
    esp.vsld.8 \qd, \qs, \q_tmp, trunc
#endif
.endm

.macro vsl_16 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        la \a_tmp, vsl_16_1
    .elseif \n == 2
        la \a_tmp, vsl_16_2
    .elseif \n == 3
        la \a_tmp, vsl_16_3
    .elseif \n == 4
        la \a_tmp, vsl_16_4
    .elseif \n == 5
        la \a_tmp, vsl_16_5
    .elseif \n == 6
        la \a_tmp, vsl_16_6
    .elseif \n == 7
        la \a_tmp, vsl_16_7
    .elseif \n == 8
        la \a_tmp, vsl_16_8
    .elseif \n == 9
        la \a_tmp, vsl_16_9
    .elseif \n == 10
        la \a_tmp, vsl_16_10
    .elseif \n == 11
        la \a_tmp, vsl_16_11
    .elseif \n == 12
        la \a_tmp, vsl_16_12
    .elseif \n == 13
        la \a_tmp, vsl_16_13
    .elseif \n == 14
        la \a_tmp, vsl_16_14
    .elseif \n == 15
        la \a_tmp, vsl_16_15
    .else
        .error "unsupported n"
    .endif
    esp.vldbc.16.ip \q_tmp, \a_tmp, 0
    esp.vsld.16 \qd, \qs, \q_tmp
.endm

.macro vsr_u8 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        la \a_tmp, vsr_u8_1_mask
    .elseif \n == 2
        la \a_tmp, vsr_u8_2_mask
    .elseif \n == 3
        la \a_tmp, vsr_u8_3_mask
    .elseif \n == 4
        la \a_tmp, vsr_u8_4_mask
    .elseif \n == 5
        la \a_tmp, vsr_u8_5_mask
    .elseif \n == 6
        la \a_tmp, vsr_u8_6_mask
    .elseif \n == 7
        la \a_tmp, vsr_u8_7_mask
    .else
        .error "unsupported n"
    .endif
    esp.vldbc.8.ip \q_tmp, \a_tmp, 0
    li \a_tmp, \n
    esp.movx.w.sar \a_tmp
    esp.vsr.u32 \qd, \qs
    esp.andq \qd, \qd, \q_tmp
.endm

.macro vlut_32_8 qd0, qd1, q_idx, a_lut
    esp.ldxq.32 \qd0, \q_idx, \a_lut, 0, 0
    esp.ldxq.32 \qd0, \q_idx, \a_lut, 1, 1
    esp.ldxq.32 \qd0, \q_idx, \a_lut, 2, 2
    esp.ldxq.32 \qd0, \q_idx, \a_lut, 3, 3
    esp.ldxq.32 \qd1, \q_idx, \a_lut, 0, 4
    esp.ldxq.32 \qd1, \q_idx, \a_lut, 1, 5
    esp.ldxq.32 \qd1, \q_idx, \a_lut, 2, 6
    esp.ldxq.32 \qd1, \q_idx, \a_lut, 3, 7
.endm

.macro vlut_16_8 qd, q_idx_8, a_lut, q_tmp
    vlut_32_8 \qd, \q_tmp, \q_idx_8, \a_lut
    esp.vunzip.16 \qd, \q_tmp
.endm

.macro vlut_16_16 qd0, qd1, q_idx_16, a_lut, q_tmp0, q_tmp1
    esp.vext.u8 \q_tmp0, \q_tmp1, \q_idx_16
    vlut_32_8 \qd0, \qd1, \q_tmp0, \a_lut
    esp.vunzip.16 \qd0, \qd1
    vlut_32_8 \qd1, \q_tmp0, \q_tmp1, \a_lut
    esp.vunzip.16 \qd1, \q_tmp0
.endm

.macro vlut_16_16_ip qd0, qd1, q_idx_16, a_lut, q_tmp0, q_tmp1, imm
    vlut_16_16 \qd0, \qd1, \q_idx_16, \a_lut, \q_tmp0, \q_tmp1
    addi \a_lut, \a_lut, \imm
.endm

.macro vlut_8_16 qd, q_idx_16, a_lut, q_tmp0, q_tmp1, q_tmp2
    vlut_16_16 \qd, \q_tmp0, \q_idx_16, \a_lut, \q_tmp1, \q_tmp2
    esp.vunzip.8 \qd, \q_tmp0
.endm

.macro vlut_8_16_ip qd, q_idx_16, a_lut, q_tmp0, q_tmp1, q_tmp2, imm
    vlut_8_16 \qd, \q_idx_16, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
    addi \a_lut, \a_lut, \imm
.endm

.macro norm_quant_1chn_8 q0, a_lut, q_tmp0, q_tmp1, q_tmp2
    vlut_8_16 \q0, \q0, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro norm_quant_1chn_16 qd0, qd1_qs0, a_lut, q_tmp0, q_tmp1
    vlut_16_16 \qd0, \qd1_qs0, \qd1_qs0, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro norm_quant_3chn_8 q0, q1, q2, a_lut, q_tmp0, q_tmp1, q_tmp2
    vlut_8_16_ip \q0, \q0, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2, 1024
    vlut_8_16_ip \q1, \q1, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2, 1024
    vlut_8_16_ip \q2, \q2, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2, -2048
.endm

.macro norm_quant_3chn_16 qd0, qd1, qd2, qd3_qs0, qd4_qs1, qd5_qs2, a_lut, q_tmp0, q_tmp1
    vlut_16_16_ip \qd0, \qd1, \qd3_qs0, \a_lut, \q_tmp0, \q_tmp1, 1024
    vlut_16_16_ip \qd2, \qd3_qs0, \qd4_qs1, \a_lut, \q_tmp0, \q_tmp1, 1024
    vlut_16_16_ip \qd4_qs1, \qd5_qs2, \qd5_qs2, \a_lut, \q_tmp0, \q_tmp1, -2048
.endm

.macro rgb5652rgb888 qd0, qd1, qd2, qs0, qs1, a_tmp0, a_tmp1
    la \a_tmp0, rgb5652rgb888_mask_tab
    vldbc_8_ip \qd0, \a_tmp0, 2
    esp.andq \qd0, \qs0, \qd0
    vsr_u8 \qd0, \qd0, \qd1, \a_tmp1, 3

    vsl_8 \qd1, \qs1, \qd2, \a_tmp1, 5
    esp.vadd.u8 \qd1, \qd0, \qd1

    vsl_8 \qd2, \qs0, \qd0, \a_tmp1, 3

    vldbc_8_ip \qd0, \a_tmp0, 0
    esp.andq \qd0, \qs1, \qd0
.endm

.macro rgb565le2rgb888 qd0, qd1, qd2, qs0, qs1, a_tmp0, a_tmp1
    rgb5652rgb888 \qd0, \qd1, \qd2, \qs0, \qs1, \a_tmp0, \a_tmp1
.endm

.macro rgb565be2rgb888 qd0, qd1, qd2, qs0, qs1, a_tmp0, a_tmp1
    rgb5652rgb888 \qd0, \qd1, \qd2, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb565le2bgr888 qd0, qd1, qd2, qs0, qs1, a_tmp0, a_tmp1
    rgb5652rgb888 \qd2, \qd1, \qd0, \qs0, \qs1, \a_tmp0, \a_tmp1
.endm

.macro rgb565be2bgr888 qd0, qd1, qd2, qs0, qs1, a_tmp0, a_tmp1
    rgb5652rgb888 \qd2, \qd1, \qd0, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb565le2rgb888_qint8 qd0, qd1, qd2, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp
    rgb565le2rgb888 \qd0, \qd1, \qd2, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \qs0, \qs1, \q_tmp
.endm

.macro rgb565be2rgb888_qint8 qd0, qd1, qd2, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp
    rgb565be2rgb888 \qd0, \qd1, \qd2, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \qs0, \qs1, \q_tmp
.endm

.macro rgb565le2bgr888_qint8 qd0, qd1, qd2, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp
    rgb565le2bgr888 \qd0, \qd1, \qd2, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \qs0, \qs1, \q_tmp
.endm

.macro rgb565be2bgr888_qint8 qd0, qd1, qd2, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp
    rgb565be2bgr888 \qd0, \qd1, \qd2, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \qs0, \qs1, \q_tmp
.endm

.macro rgb565le2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_lut, qs0, qs1, a_tmp0, a_tmp1
    rgb565le2rgb888 \qd3, \qd4, \qd5, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \qs0, \qs1
.endm

.macro rgb565be2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_lut, qs0, qs1, a_tmp0, a_tmp1
    rgb565be2rgb888 \qd3, \qd4, \qd5, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \qs0, \qs1
.endm

.macro rgb565le2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_lut, qs0, qs1, a_tmp0, a_tmp1
    rgb565le2bgr888 \qd3, \qd4, \qd5, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \qs0, \qs1
.endm

.macro rgb565be2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_lut, qs0, qs1, a_tmp0, a_tmp1
    rgb565be2bgr888 \qd3, \qd4, \qd5, \qs0, \qs1, \a_tmp0, \a_tmp1
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \qs0, \qs1
.endm

.macro rgb8882rgb565 qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    la \a_tmp0, rgb8882rgb565_mask_tab
    vldbc_8_ip \qd0, \a_tmp0, 1
    esp.andq \qs0, \qs0, \qd0
    vsr_u8 \qs2, \qs2, \qd0, \a_tmp1, 3

    vldbc_8_ip \qd0, \a_tmp0, 1
    esp.andq \qd0, \qs1, \qd0
    vsl_8 \qd0, \qd0, \qd1, \a_tmp1, 3
    esp.vadd.u8 \qd0, \qd0, \qs2

    vldbc_8_ip \qd1, \a_tmp0, 0
    esp.andq \qd1, \qs1, \qd1
    vsr_u8 \qd1, \qd1, \qs1, \a_tmp1, 5
    esp.vadd.u8 \qd1, \qd1, \qs0
.endm

.macro rgb8882rgb565le qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    rgb8882rgb565 \qd0, \qd1, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1
.endm

.macro bgr8882rgb565le qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    rgb8882rgb565 \qd0, \qd1, \qs2, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb8882rgb565be qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    rgb8882rgb565 \qd1, \qd0, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1
.endm

.macro bgr8882rgb565be qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    rgb8882rgb565 \qd1, \qd0, \qs2, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb8882gray qd, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    la \a_tmp, rgb8882gray_const_tab
    esp.ldqa.u16.128.ip \a_tmp, 16

    vldbc_16_ip \q_tmp0, \a_tmp, 2
    vldbc_16_ip \q_tmp1, \a_tmp, 2
    vldbc_16_ip \q_tmp2, \a_tmp, 2
    vldbc_16_ip \q_tmp3, \a_tmp, -22

    esp.vext.u8 \qd, \qs0, \qs0
    esp.vmulas.u16.qacc \qd, \q_tmp0
    esp.vext.u8 \qd, \qs1, \qs1
    esp.vmulas.u16.qacc \qd, \q_tmp1
    esp.vext.u8 \qd, \qs2, \qs2
    esp.vmulas.u16.qacc \qd, \q_tmp2
    esp.srcmb.u16.q.qacc \qd, \q_tmp3, 0

    esp.ldqa.u16.128.ip \a_tmp, 0
    esp.vmulas.u16.qacc \qs0, \q_tmp0
    esp.vmulas.u16.qacc \qs1, \q_tmp1
    esp.vmulas.u16.qacc \qs2, \q_tmp2
    esp.srcmb.u16.q.qacc \qs0, \q_tmp3, 0

    esp.vunzip.8 \qd, \qs0
.endm

.macro bgr8882gray qd, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882gray \qd, \qs2, \qs1, \qs0, \a_tmp, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
.endm

.macro rgb8882gray_qint8 qd, a_lut, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882gray \qd, \qs0, \qs1, \qs2, \a_tmp, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro bgr8882gray_qint8 qd, a_lut, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr8882gray \qd, \qs0, \qs1, \qs2, \a_tmp, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro rgb8882gray_qint16 qd0, qd1, a_lut, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2
    rgb8882gray \qd1, \qs0, \qs1, \qs2, \a_tmp, \qd0, \q_tmp0, \q_tmp1, \q_tmp2
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro bgr8882gray_qint16 qd0, qd1, a_lut, qs0, qs1, qs2, a_tmp, q_tmp0, q_tmp1, q_tmp2
    bgr8882gray \qd1, \qs0, \qs1, \qs2, \a_tmp, \qd0, \q_tmp0, \q_tmp1, \q_tmp2
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro rgb565le2gray qd, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882gray \qd, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565be2gray qd, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2gray \qd, \qs1, \qs0, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
.endm

.macro bgr565le2gray qd, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882gray \qd, \q_tmp2, \q_tmp1, \q_tmp0, \a_tmp0, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565be2gray qd, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    bgr565le2gray \qd, \qs1, \qs0, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
.endm

.macro rgb565le2gray_qint8 qd, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2gray \qd, \qs0, \qs1, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro rgb565be2gray_qint8 qd, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2gray \qd, \qs0, \qs1, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro bgr565le2gray_qint8 qd, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    bgr565le2gray \qd, \qs0, \qs1, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro bgr565be2gray_qint8 qd, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    bgr565be2gray \qd, \qs0, \qs1, \a_tmp0, \a_tmp1, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \q_tmp4
    norm_quant_1chn_8 \qd, \a_lut, \q_tmp0, \q_tmp1, \q_tmp2
.endm

.macro rgb565le2gray_qint16 qd0, qd1, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb565le2gray \qd1, \qs0, \qs1, \a_tmp0, \a_tmp1, \qd0, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro rgb565be2gray_qint16 qd0, qd1, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb565be2gray \qd1, \qs0, \qs1, \a_tmp0, \a_tmp1, \qd0, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro bgr565le2gray_qint16 qd0, qd1, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr565le2gray \qd1, \qs0, \qs1, \a_tmp0, \a_tmp1, \qd0, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro bgr565be2gray_qint16 qd0, qd1, a_lut, qs0, qs1, a_tmp0, a_tmp1, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr565be2gray \qd1, \qs0, \qs1, \a_tmp0, \a_tmp1, \qd0, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
    norm_quant_1chn_16 \qd0, \qd1, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro rgb5652bgr565 qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    la \a_tmp0, rgb5652rgb888_mask_tab
    vldbc_8_ip \qd0, \a_tmp0, 1
    esp.andq \qd0, \qs0, \qd0
    vsl_8 \qs0, \qs0, \qd1, \a_tmp1, 3
    vldbc_8_ip \qd1, \a_tmp0, 0
    esp.andq \qd1, \qs1, \qd1
    esp.vadd.u8 \qd1, \qd1, \qs0
    vsr_u8 \qs1, \qs1, \qs0, \a_tmp1, 3
    esp.vadd.u8 \qd0, \qd0, \qs1
.endm

.macro rgb565le2bgr565le qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd0 \qd1 \qs0 \qs1 \a_tmp0 \a_tmp1
.endm

.macro rgb565be2bgr565be qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd1 \qd0 \qs1 \qs0 \a_tmp0 \a_tmp1
.endm

.macro rgb565le2bgr565be qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd1 \qd0 \qs0 \qs1 \a_tmp0 \a_tmp1
.endm

.macro rgb565be2bgr565le qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd0 \qd1 \qs1 \qs0 \a_tmp0 \a_tmp1
.endm

.macro mul_u32xu8_round_and_shift qd, q_u32, q_u8, q_round_delta, a_tmp0, a_tmp1, sel4_q_u8
    esp.movi.32.a \q_u32, \a_tmp0, 0
    esp.movi.8.a \q_u8, \a_tmp1, 0 + 4 * \sel4_q_u8
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 0

    esp.movi.32.a \q_u32, \a_tmp0, 1
    esp.movi.8.a \q_u8, \a_tmp1, 1 + 4 * \sel4_q_u8
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 1

    esp.movi.32.a \q_u32, \a_tmp0, 2
    esp.movi.8.a \q_u8, \a_tmp1, 2 + 4 * \sel4_q_u8
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 2

    esp.movi.32.a \q_u32, \a_tmp0, 3
    esp.movi.8.a \q_u8, \a_tmp1, 3 + 4 * \sel4_q_u8
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 3

    esp.vadd.u32 \qd, \qd, \q_round_delta
    esp.vsr.u32 \qd, \qd
.endm

.macro cal_s qd, q_vmax, q_diff, q_round_delta, a_sdiv_lut, q_tmp0, q_tmp1, a_tmp0, a_tmp1
    li \a_tmp0, 12
    esp.movx.w.sar \a_tmp0
    esp.vext.u8 \q_vmax, \q_tmp0, \q_vmax

    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 0, 0
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 1, 1
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 2, 2
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 3, 3
    mul_u32xu8_round_and_shift \qd, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, 0

    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 0, 4
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 1, 5
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 2, 6
    esp.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 3, 7
    mul_u32xu8_round_and_shift \q_vmax, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, 1

    esp.vunzip.16 \qd, \q_vmax

    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 0, 0
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 1, 1
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 2, 2
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 3, 3
    mul_u32xu8_round_and_shift \q_vmax, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, 2

    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 0, 4
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 1, 5
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 2, 6
    esp.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 3, 7
    mul_u32xu8_round_and_shift \q_tmp0, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, 3

    esp.vunzip.16 \q_vmax, \q_tmp0
    esp.vunzip.8 \qd, \q_vmax
.endm

.macro mul_s32xs32_round_and_shift qd, qs0, qs1, q_round_delta, a_tmp0, a_tmp1
    esp.movi.32.a \qs0, \a_tmp0, 0
    esp.movi.32.a \qs1, \a_tmp1, 0
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 0

    esp.movi.32.a \qs0, \a_tmp0, 1
    esp.movi.32.a \qs1, \a_tmp1, 1
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 1

    esp.movi.32.a \qs0, \a_tmp0, 2
    esp.movi.32.a \qs1, \a_tmp1, 2
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 2

    esp.movi.32.a \qs0, \a_tmp0, 3
    esp.movi.32.a \qs1, \a_tmp1, 3
    mul \a_tmp0, \a_tmp0, \a_tmp1
    esp.movi.32.q \qd, \a_tmp0, 3

    esp.vadd.s32 \qd, \qd, \q_round_delta
    esp.vsr.s32 \qd, \qd
.endm

.macro cal_h_helper qd, q_h, q_diff, q_round_delta, a_hdiv_lut, q_tmp0, q_tmp1, a_tmp0, a_tmp1
    esp.vext.s16 \q_h, \q_tmp0, \q_h

    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 0, 0
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 1, 1
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 2, 2
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 3, 3
    mul_s32xs32_round_and_shift \qd, \q_h, \q_tmp1, \q_round_delta, \a_tmp0, \a_tmp1

    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 0, 4
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 1, 5
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 2, 6
    esp.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 3, 7
    mul_s32xs32_round_and_shift \q_h, \q_tmp0, \q_tmp1, \q_round_delta, \a_tmp0, \a_tmp1

    esp.vunzip.16 \qd, \q_h
.endm

.macro rgb8882hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1
    addi sp, sp, -144
    addi \a_tmp0, sp, 0
    esp.vmax.u8 \qd0, \qs0, \qs1
    esp.vmax.u8 \qd0, \qd0, \qs2
    esp.vmin.u8 \qd2, \qs0, \qs1
    esp.vmin.u8 \qd2, \qd2, \qs2
    esp.vsub.u8 \qd2, \qd0, \qd2
    esp.vst.128.ip \qd0, \a_tmp0, 16
    esp.vst.128.ip \qd2, \a_tmp0, 16

    esp.vcmp.eq.u8 \q_tmp0, \qd0, \qs0
    esp.vst.128.ip \q_tmp0, \a_tmp0, 16
    esp.notq \q_tmp0, \q_tmp0
    esp.vcmp.eq.u8 \q_tmp1, \qd0, \qs1
    esp.andq \q_tmp1, \q_tmp1, \q_tmp0
    esp.vst.128.ip \q_tmp1, \a_tmp0, 16
    esp.notq \q_tmp1, \q_tmp1
    esp.andq \q_tmp1, \q_tmp0, \q_tmp1
    esp.vst.128.ip \q_tmp1, \a_tmp0, 16

    esp.vst.128.ip \qs0, \a_tmp0, 16
    esp.vst.128.ip \qs1, \a_tmp0, 16
    esp.vst.128.ip \qs2, \a_tmp0, 16

    la \a_tmp1, rgb8882hsv_const_tab
    esp.vldbc.32.ip \q_tmp0, \a_tmp1, 0

    cal_s \qd1, \qd0, \qd2, \q_tmp0, \a_sdiv_lut, \q_tmp1, \qs0, \a_tmp2, \a_tmp3
    esp.vst.128.ip \qd1, \a_tmp0, -48

    esp.vldext.u8.ip \qs0, \q_tmp0, \a_tmp0, 16
    esp.vldext.u8.ip \qs1, \q_tmp0, \a_tmp0, 16
    esp.vldext.u8.ip \qs2, \q_tmp0, \a_tmp0, -96

    # diff
    esp.vldext.u8.ip \qd1, \q_tmp0, \a_tmp0, 16

    # mask1
    esp.vldext.s8.ip \q_tmp0, \q_tmp1, \a_tmp0, 16
    esp.vsub.s16 \q_tmp1, \qs1, \qs2
    esp.andq \qd0, \q_tmp0, \q_tmp1

    # mask2
    esp.vldext.s8.ip \q_tmp0, \q_tmp1, \a_tmp0, 16
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 1
    esp.vsub.s16 \q_tmp1, \qs2, \qs0
    esp.vadd.s16 \q_tmp1, \q_tmp1, \qd2
    esp.andq \q_tmp0, \q_tmp0, \q_tmp1
    esp.vadd.s16 \qd0, \qd0, \q_tmp0

    # mask3
    esp.vldext.s8.ip \q_tmp0, \q_tmp1, \a_tmp0, 16
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 2
    esp.vsub.s16 \q_tmp1, \qs0, \qs1
    esp.vadd.s16 \q_tmp1, \q_tmp1, \qd2
    esp.andq \q_tmp0, \q_tmp0, \q_tmp1
    esp.vadd.s16 \qd0, \qd0, \q_tmp0

    esp.vldbc.32.ip \q_tmp0, \a_tmp1, 0
    cal_h_helper \qd2, \qd0, \qd1, \q_tmp0, \a_hdiv_lut, \q_tmp1, \qs0, \a_tmp2, \a_tmp3

    esp.vldext.u8.ip \q_tmp0, \qs0, \a_tmp0, 16
    esp.vldext.u8.ip \q_tmp0, \qs1, \a_tmp0, 16
    esp.vldext.u8.ip \q_tmp0, \qs2, \a_tmp0, -32
    esp.vst.128.ip \qd2, \a_tmp0, -64

    # diff
    esp.vldext.u8.ip \q_tmp0, \qd1, \a_tmp0, 16

    # mask1
    esp.vldext.s8.ip \q_tmp1, \q_tmp0, \a_tmp0, 16
    esp.vsub.s16 \q_tmp1, \qs1, \qs2
    esp.andq \qd0, \q_tmp0, \q_tmp1

    # mask2
    esp.vldext.s8.ip \q_tmp1, \q_tmp0, \a_tmp0, 16
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 1
    esp.vsub.s16 \q_tmp1, \qs2, \qs0
    esp.vadd.s16 \q_tmp1, \q_tmp1, \qd2
    esp.andq \q_tmp0, \q_tmp0, \q_tmp1
    esp.vadd.s16 \qd0, \qd0, \q_tmp0

    # mask3
    esp.vldext.s8.ip \q_tmp1, \q_tmp0, \a_tmp0, 16
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 2
    esp.vsub.s16 \q_tmp1, \qs0, \qs1
    esp.vadd.s16 \q_tmp1, \q_tmp1, \qd2
    esp.andq \q_tmp0, \q_tmp0, \q_tmp1
    esp.vadd.s16 \qd0, \qd0, \q_tmp0

    esp.vldbc.32.ip \q_tmp0, \a_tmp1, 4
    cal_h_helper \qs1, \qd0, \qd1, \q_tmp0, \a_hdiv_lut, \q_tmp1, \qs0, \a_tmp2, \a_tmp3

    vldbc_16_ip \q_tmp0, \a_tmp1, 0
    esp.vld.128.ip \qs0, \a_tmp0, 48
    esp.zero.q \q_tmp1
    esp.vcmp.lt.s16 \qd0, \qs0, \q_tmp1
    esp.andq \qd0, \qd0, \q_tmp0
    esp.vadd.s16 \qd0, \qd0, \qs0

    esp.vcmp.lt.s16 \qd1, \qs1, \q_tmp1
    esp.andq \qd1, \qd1, \q_tmp0
    esp.vadd.s16 \qd1, \qd1, \qs1
    esp.vunzip.8 \qd0, \qd1
    esp.vld.128.ip \qd1, \a_tmp0, -128
    esp.vld.128.ip \qd2, \a_tmp0, 0

    addi sp, sp, 144
.endm

.macro bgr8882hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \qs2, \qs1, \qs0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp0, \q_tmp1
.endm

.macro rgb565le2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro rgb565be2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs1, \qs0, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro bgr565le2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \q_tmp2, \q_tmp1, \q_tmp0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro bgr565be2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs1, \qs0, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \q_tmp2, \q_tmp1, \q_tmp0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro hsv2hsv_mask qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp, h_across_zero
    # h < h_min
    vldbc_8_ip \q_tmp, \a_hsv_min_addr, 1
    esp.vcmp.lt.u8 \qd, \qs0, \q_tmp
    # h > h_max
    vldbc_8_ip \q_tmp, \a_hsv_max_addr, 1
    esp.vcmp.gt.u8 \q_tmp, \qs0, \q_tmp
    .if \h_across_zero == 0
        esp.orq \qd, \qd, \q_tmp
    .elseif \h_across_zero == 1
        esp.andq \qd, \qd, \q_tmp
    .else
        .error "h_across_zero can only be 0/1."
    .endif
    # s < s_min
    vldbc_8_ip \q_tmp, \a_hsv_min_addr, 1
    esp.vcmp.lt.u8 \q_tmp, \qs1, \q_tmp
    esp.orq \qd, \qd, \q_tmp
    # s > s_max
    vldbc_8_ip \q_tmp, \a_hsv_max_addr, 1
    esp.vcmp.gt.u8 \q_tmp, \qs1, \q_tmp
    esp.orq \qd, \qd, \q_tmp
    # v < v_min
    vldbc_8_ip \q_tmp, \a_hsv_min_addr, -2
    esp.vcmp.lt.u8 \q_tmp, \qs2, \q_tmp
    esp.orq \qd, \qd, \q_tmp
    # v > v_max
    vldbc_8_ip \q_tmp, \a_hsv_max_addr, -2
    esp.vcmp.gt.u8 \q_tmp, \qs2, \q_tmp
    esp.orq \qd, \qd, \q_tmp
    esp.notq \qd, \qd
.endm

.macro hsv2hsv_mask0 qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp
    hsv2hsv_mask \qd, \a_hsv_min_addr, \a_hsv_max_addr, \qs0, \qs1, \qs2, \q_tmp, 0
.endm

.macro hsv2hsv_mask1 qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp
    hsv2hsv_mask \qd, \a_hsv_min_addr, \a_hsv_max_addr, \qs0, \qs1, \qs2, \q_tmp, 1
.endm

.macro rgb8882hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask0 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
.endm

.macro rgb8882hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask1 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
.endm

.macro bgr8882hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask0 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
.endm

.macro bgr8882hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask1 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3
.endm

.macro rgb565le2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565le2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565be2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565be2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565le2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565le2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565be2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565be2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

# byte_off is 0 for the pixel address, 1 for the next byte (UYVY Y).
.macro resize_nn_load_1chn_128 qd, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, byte_off
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 0
    esp.movi.8.q \qd, \a_tmp1, 1
    esp.movi.8.q \qd, \a_tmp2, 2
    esp.movi.8.q \qd, \a_tmp3, 3

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 4
    esp.movi.8.q \qd, \a_tmp1, 5
    esp.movi.8.q \qd, \a_tmp2, 6
    esp.movi.8.q \qd, \a_tmp3, 7

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 8
    esp.movi.8.q \qd, \a_tmp1, 9
    esp.movi.8.q \qd, \a_tmp2, 10
    esp.movi.8.q \qd, \a_tmp3, 11

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 12
    esp.movi.8.q \qd, \a_tmp1, 13
    esp.movi.8.q \qd, \a_tmp2, 14
    esp.movi.8.q \qd, \a_tmp3, 15
.endm

.macro resize_nn_load_store_1chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, byte_off
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    sb \a_tmp0, 0(\dst_addr)
    sb \a_tmp1, 1(\dst_addr)
    sb \a_tmp2, 2(\dst_addr)
    sb \a_tmp3, 3(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    sb \a_tmp0, 4(\dst_addr)
    sb \a_tmp1, 5(\dst_addr)
    sb \a_tmp2, 6(\dst_addr)
    sb \a_tmp3, 7(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    sb \a_tmp0, 8(\dst_addr)
    sb \a_tmp1, 9(\dst_addr)
    sb \a_tmp2, 10(\dst_addr)
    sb \a_tmp3, 11(\dst_addr)

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, \byte_off(\a_tmp0)
    lb \a_tmp1, \byte_off(\a_tmp1)
    lb \a_tmp2, \byte_off(\a_tmp2)
    lb \a_tmp3, \byte_off(\a_tmp3)
    sb \a_tmp0, 12(\dst_addr)
    sb \a_tmp1, 13(\dst_addr)
    sb \a_tmp2, 14(\dst_addr)
    sb \a_tmp3, 15(\dst_addr)
    addi \dst_addr, \dst_addr, 16
.endm

.macro resize_nn_load_2chn_128 qd0, qd1, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    esp.movi.16.q \qd0, \a_tmp0, 0
    esp.movi.16.q \qd0, \a_tmp1, 1
    esp.movi.16.q \qd0, \a_tmp2, 2
    esp.movi.16.q \qd0, \a_tmp3, 3

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    esp.movi.16.q \qd0, \a_tmp0, 4
    esp.movi.16.q \qd0, \a_tmp1, 5
    esp.movi.16.q \qd0, \a_tmp2, 6
    esp.movi.16.q \qd0, \a_tmp3, 7

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    esp.movi.16.q \qd1, \a_tmp0, 0
    esp.movi.16.q \qd1, \a_tmp1, 1
    esp.movi.16.q \qd1, \a_tmp2, 2
    esp.movi.16.q \qd1, \a_tmp3, 3

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    esp.movi.16.q \qd1, \a_tmp0, 4
    esp.movi.16.q \qd1, \a_tmp1, 5
    esp.movi.16.q \qd1, \a_tmp2, 6
    esp.movi.16.q \qd1, \a_tmp3, 7
.endm

# One output pixel. a_addr is the pixel byte address. a_base is the row pointer.
# Parity is (a_addr - a_base) & 2. Y is read before a_addr is aligned to the pair.
# lbu zero-extends into an s16 lane. YUYV memory is Y0 U Y1 V. Y is at the pixel address.
.macro resize_nn_yuyv_lane q_y, q_u, q_v, a_addr, a_scratch, a_base, lane
    lbu \a_scratch, 0(\a_addr)
    esp.movi.16.q \q_y, \a_scratch, \lane
    sub \a_scratch, \a_addr, \a_base
    andi \a_scratch, \a_scratch, 2
    sub \a_addr, \a_addr, \a_scratch
    lbu \a_scratch, 1(\a_addr)
    esp.movi.16.q \q_u, \a_scratch, \lane
    lbu \a_scratch, 3(\a_addr)
    esp.movi.16.q \q_v, \a_scratch, \lane
.endm

# UYVY memory is U Y0 V Y1. Y is at the pixel address + 1.
.macro resize_nn_uyvy_lane q_y, q_u, q_v, a_addr, a_scratch, a_base, lane
    lbu \a_scratch, 1(\a_addr)
    esp.movi.16.q \q_y, \a_scratch, \lane
    sub \a_scratch, \a_addr, \a_base
    andi \a_scratch, \a_scratch, 2
    sub \a_addr, \a_addr, \a_scratch
    lbu \a_scratch, 0(\a_addr)
    esp.movi.16.q \q_u, \a_scratch, \lane
    lbu \a_scratch, 2(\a_addr)
    esp.movi.16.q \q_v, \a_scratch, \lane
.endm

# 8 pixels. yuv_lane is resize_nn_yuyv_lane or resize_nn_uyvy_lane.
# offsets advances 32 bytes. q_y/q_u/q_v hold 8 zero-extended s16 values.
# a_base receives the row pointer from q_tmp0 and stays live across the lanes.
.macro resize_nn_load_yuv_8 q_y, q_u, q_v, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, a_scratch, a_base, yuv_lane
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.movi.32.a \q_tmp0, \a_base, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp0, \a_scratch, \a_base, 0
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp1, \a_scratch, \a_base, 1
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp2, \a_scratch, \a_base, 2
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp3, \a_scratch, \a_base, 3

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp0, \a_scratch, \a_base, 4
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp1, \a_scratch, \a_base, 5
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp2, \a_scratch, \a_base, 6
    \yuv_lane \q_y, \q_u, \q_v, \a_tmp3, \a_scratch, \a_base, 7
.endm

.macro resize_nn_load_store_2chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    sh \a_tmp0, 0(\dst_addr)
    sh \a_tmp1, 2(\dst_addr)
    sh \a_tmp2, 4(\dst_addr)
    sh \a_tmp3, 6(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    sh \a_tmp0, 8(\dst_addr)
    sh \a_tmp1, 10(\dst_addr)
    sh \a_tmp2, 12(\dst_addr)
    sh \a_tmp3, 14(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    sh \a_tmp0, 16(\dst_addr)
    sh \a_tmp1, 18(\dst_addr)
    sh \a_tmp2, 20(\dst_addr)
    sh \a_tmp3, 22(\dst_addr)

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lh \a_tmp0, 0(\a_tmp0)
    lh \a_tmp1, 0(\a_tmp1)
    lh \a_tmp2, 0(\a_tmp2)
    lh \a_tmp3, 0(\a_tmp3)
    sh \a_tmp0, 24(\dst_addr)
    sh \a_tmp1, 26(\dst_addr)
    sh \a_tmp2, 28(\dst_addr)
    sh \a_tmp3, 30(\dst_addr)
    addi \dst_addr, \dst_addr, 32
.endm

.macro resize_nn_load_3chn_128 qd0, qd1, qd2, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, a_tmp4, a_tmp5, a_tmp6, a_tmp7
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp1, 0
    esp.movi.32.a \q_tmp1, \a_tmp3, 1
    esp.movi.32.a \q_tmp1, \a_tmp5, 2
    esp.movi.32.a \q_tmp1, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    esp.movi.16.q \qd0, \a_tmp0, 0
    esp.movi.8.q  \qd0, \a_tmp1, 2
    esp.movi.8.q  \qd0, \a_tmp2, 3
    esp.movi.16.q \qd0, \a_tmp3, 2
    esp.movi.16.q \qd0, \a_tmp4, 3
    esp.movi.8.q  \qd0, \a_tmp5, 8
    esp.movi.8.q  \qd0, \a_tmp6, 9
    esp.movi.16.q \qd0, \a_tmp7, 5

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp1, 0
    esp.movi.32.a \q_tmp2, \a_tmp3, 1
    esp.movi.32.a \q_tmp2, \a_tmp5, 2
    esp.movi.32.a \q_tmp2, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    esp.movi.16.q \qd0, \a_tmp0, 6
    esp.movi.8.q  \qd0, \a_tmp1, 14
    esp.movi.8.q  \qd0, \a_tmp2, 15
    esp.movi.16.q \qd1, \a_tmp3, 0
    esp.movi.16.q \qd1, \a_tmp4, 1
    esp.movi.8.q  \qd1, \a_tmp5, 4
    esp.movi.8.q  \qd1, \a_tmp6, 5
    esp.movi.16.q \qd1, \a_tmp7, 3

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp1, 0
    esp.movi.32.a \q_tmp1, \a_tmp3, 1
    esp.movi.32.a \q_tmp1, \a_tmp5, 2
    esp.movi.32.a \q_tmp1, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    esp.movi.16.q \qd1, \a_tmp0, 4
    esp.movi.8.q  \qd1, \a_tmp1, 10
    esp.movi.8.q  \qd1, \a_tmp2, 11
    esp.movi.16.q \qd1, \a_tmp3, 6
    esp.movi.16.q \qd1, \a_tmp4, 7
    esp.movi.8.q  \qd2, \a_tmp5, 0
    esp.movi.8.q  \qd2, \a_tmp6, 1
    esp.movi.16.q \qd2, \a_tmp7, 1

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp1, 0
    esp.movi.32.a \q_tmp2, \a_tmp3, 1
    esp.movi.32.a \q_tmp2, \a_tmp5, 2
    esp.movi.32.a \q_tmp2, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    esp.movi.16.q \qd2, \a_tmp0, 2
    esp.movi.8.q  \qd2, \a_tmp1, 6
    esp.movi.8.q  \qd2, \a_tmp2, 7
    esp.movi.16.q \qd2, \a_tmp3, 4
    esp.movi.16.q \qd2, \a_tmp4, 5
    esp.movi.8.q  \qd2, \a_tmp5, 12
    esp.movi.8.q  \qd2, \a_tmp6, 13
    esp.movi.16.q \qd2, \a_tmp7, 7
.endm

.macro resize_nn_load_store_3chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, a_tmp4, a_tmp5, a_tmp6, a_tmp7
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp1, 0
    esp.movi.32.a \q_tmp1, \a_tmp3, 1
    esp.movi.32.a \q_tmp1, \a_tmp5, 2
    esp.movi.32.a \q_tmp1, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    sh \a_tmp0, 0(\dst_addr)
    sb \a_tmp1, 2(\dst_addr)
    sb \a_tmp2, 3(\dst_addr)
    sh \a_tmp3, 4(\dst_addr)
    sh \a_tmp4, 6(\dst_addr)
    sb \a_tmp5, 8(\dst_addr)
    sb \a_tmp6, 9(\dst_addr)
    sh \a_tmp7, 10(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp1, 0
    esp.movi.32.a \q_tmp2, \a_tmp3, 1
    esp.movi.32.a \q_tmp2, \a_tmp5, 2
    esp.movi.32.a \q_tmp2, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    sh \a_tmp0, 12(\dst_addr)
    sb \a_tmp1, 14(\dst_addr)
    sb \a_tmp2, 15(\dst_addr)
    sh \a_tmp3, 16(\dst_addr)
    sh \a_tmp4, 18(\dst_addr)
    sb \a_tmp5, 20(\dst_addr)
    sb \a_tmp6, 21(\dst_addr)
    sh \a_tmp7, 22(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp1, 0
    esp.movi.32.a \q_tmp1, \a_tmp3, 1
    esp.movi.32.a \q_tmp1, \a_tmp5, 2
    esp.movi.32.a \q_tmp1, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    sh \a_tmp0, 24(\dst_addr)
    sb \a_tmp1, 26(\dst_addr)
    sb \a_tmp2, 27(\dst_addr)
    sh \a_tmp3, 28(\dst_addr)
    sh \a_tmp4, 30(\dst_addr)
    sb \a_tmp5, 32(\dst_addr)
    sb \a_tmp6, 33(\dst_addr)
    sh \a_tmp7, 34(\dst_addr)

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp1, 0
    esp.movi.32.a \q_tmp2, \a_tmp3, 1
    esp.movi.32.a \q_tmp2, \a_tmp5, 2
    esp.movi.32.a \q_tmp2, \a_tmp7, 3
    lh \a_tmp0, 0(\a_tmp1)
    lb \a_tmp1, 2(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp3)
    lh \a_tmp3, 1(\a_tmp3)
    lh \a_tmp4, 0(\a_tmp5)
    lb \a_tmp5, 2(\a_tmp5)
    lb \a_tmp6, 0(\a_tmp7)
    lh \a_tmp7, 1(\a_tmp7)
    sh \a_tmp0, 36(\dst_addr)
    sb \a_tmp1, 38(\dst_addr)
    sb \a_tmp2, 39(\dst_addr)
    sh \a_tmp3, 40(\dst_addr)
    sh \a_tmp4, 42(\dst_addr)
    sb \a_tmp5, 44(\dst_addr)
    sb \a_tmp6, 45(\dst_addr)
    sh \a_tmp7, 46(\dst_addr)
    addi \dst_addr, \dst_addr, 48
.endm

# 8 deltas from one U and one V. q_tmp receives dR, q_v receives dG, q_u receives dB.
# q_cof is a coefficient scratch and is dead on exit. a_shift holds 14.
# a_tmp points at the 255 half in yuv2rgb888_const_tab.
.macro yuv2rgb888_delta q_u, q_v, q_tmp, q_cof, a_tmp, a_shift
    la \a_tmp, yuv2rgb888_const_tab
    esp.ldqa.s16.128.ip \a_tmp, 16
    vldbc_16_ip \q_tmp, \a_tmp, 2
    esp.vadd.s16 \q_u, \q_u, \q_tmp
    esp.vadd.s16 \q_v, \q_v, \q_tmp
    vldbc_16_ip \q_tmp, \a_tmp, -18
    esp.vmulas.s16.qacc \q_v, \q_tmp
    esp.srcmb.s16.qacc \q_tmp, \a_shift, 0

    esp.ldqa.s16.128.ip \a_tmp, 32
    vldbc_16_ip \q_cof, \a_tmp, 2
    esp.vmulas.s16.qacc \q_u, \q_cof
    vldbc_16_ip \q_cof, \a_tmp, -34
    esp.vmulas.s16.qacc \q_v, \q_cof
    esp.srcmb.s16.qacc \q_v, \a_shift, 0

    esp.ldqa.s16.128.ip \a_tmp, 48
    vldbc_16_ip \q_cof, \a_tmp, 2
    esp.vmulas.s16.qacc \q_u, \q_cof
    esp.srcmb.s16.qacc \q_u, \a_shift, 0
.endm

# Clamp six s16 QRs to 0..255. q_zero and q_255 are free and receive the bounds.
# a_255 already points at the 255 half in yuv2rgb888_const_tab.
.macro yuv2rgb888_clamp6 q0, q1, q2, q3, q4, q5, q_zero, q_255, a_255
    esp.zero.q \q_zero
    vldbc_16_ip \q_255, \a_255, 0
    esp.vmax.s16 \q0, \q0, \q_zero
    esp.vmax.s16 \q1, \q1, \q_zero
    esp.vmax.s16 \q2, \q2, \q_zero
    esp.vmax.s16 \q3, \q3, \q_zero
    esp.vmax.s16 \q4, \q4, \q_zero
    esp.vmax.s16 \q5, \q5, \q_zero
    esp.vmin.s16 \q0, \q0, \q_255
    esp.vmin.s16 \q1, \q1, \q_255
    esp.vmin.s16 \q2, \q2, \q_255
    esp.vmin.s16 \q3, \q3, \q_255
    esp.vmin.s16 \q4, \q4, \q_255
    esp.vmin.s16 \q5, \q5, \q_255
.endm

# 16 output pixels, no stack spill. The eight QRs are distinct.
# q_r/q_g/q_b end as planar packed R/G/B. q_y is both groups' Y and then zero.
# q_r1/q_g1/q_b1 are the high halves. q_255 is the address temp and then 255.
# Group 1 gather aliases its q_tmp1 with q_y.
# The gather borrows a_shift as the row base, then puts 14 back before each delta.
.macro resize_nn_yuv2rgb888_16 q_r, q_g, q_b, base_addr, offsets, yuv_lane, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_load_yuv_8 \q_y, \q_b, \q_g, \base_addr, \offsets, \q_r, \q_r1, \q_255, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \a_shift, \yuv_lane
    li \a_shift, 14
    yuv2rgb888_delta \q_b, \q_g, \q_r, \q_r1, \a_scratch, \a_shift
    esp.vadd.s16 \q_r, \q_y, \q_r
    esp.vadd.s16 \q_g, \q_y, \q_g
    esp.vadd.s16 \q_b, \q_y, \q_b

    resize_nn_load_yuv_8 \q_y, \q_b1, \q_g1, \base_addr, \offsets, \q_r1, \q_y, \q_255, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \a_shift, \yuv_lane
    li \a_shift, 14
    yuv2rgb888_delta \q_b1, \q_g1, \q_r1, \q_255, \a_scratch, \a_shift
    esp.vadd.s16 \q_r1, \q_y, \q_r1
    esp.vadd.s16 \q_g1, \q_y, \q_g1
    esp.vadd.s16 \q_b1, \q_y, \q_b1

    yuv2rgb888_clamp6 \q_b, \q_g, \q_r, \q_r1, \q_b1, \q_g1, \q_y, \q_255, \a_scratch

    esp.vunzip.8 \q_r, \q_r1
    esp.vunzip.8 \q_g, \q_g1
    esp.vunzip.8 \q_b, \q_b1
.endm

# qd0/qd1/qd2 are planar output-order channels. The eight QRs are distinct.
# On exit only qd0/qd1/qd2 are live. BGR feeds qd2, qd1, qd0 in as R, G, B.
.macro resize_nn_yuyv2rgb888 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd0, \qd1, \qd2, \base_addr, \offsets, resize_nn_yuyv_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_uyvy2rgb888 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd0, \qd1, \qd2, \base_addr, \offsets, resize_nn_uyvy_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_yuyv2bgr888 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd2, \qd1, \qd0, \base_addr, \offsets, resize_nn_yuyv_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_uyvy2bgr888 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd2, \qd1, \qd0, \base_addr, \offsets, resize_nn_uyvy_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

# Quant temps are q_y, q_r1, q_255. They stay off qd0/qd1/qd2.
.macro resize_nn_yuyv2rgb888_qint8 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuyv2rgb888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_uyvy2rgb888_qint8 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_uyvy2rgb888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_yuyv2bgr888_qint8 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuyv2bgr888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_uyvy2bgr888_qint8 qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_uyvy2bgr888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

# qd0/qd1, qd2/qd3, qd4/qd5 are the three channels' low/high halves.
# Planar RGB is written into qd3/qd4/qd5, then consumed as the quant indices.
# qd0..qd5, q_tmp0, q_tmp1 are eight distinct QRs.
.macro resize_nn_yuyv2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_yuyv2rgb888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_uyvy2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_uyvy2rgb888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_yuyv2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_yuyv2bgr888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_uyvy2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_uyvy2bgr888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

# qd0/qd1 are the 565 halves. qs0/qs1/qs2 stay planar RGB.
# The bgr565 packer swaps R and B. q_t0..q_t2 complete the gather temps.
# a_shift and a_g0 are the packer temps.
.macro resize_nn_yuyv2rgb565le qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2rgb565be qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2bgr565le qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2bgr565be qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2rgb565le qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2rgb565be qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2bgr565le qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2bgr565be qd0, qd1, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

# qd0/qd1/qd2 are H/S/V. qs0/qs1/qs2 are planar RGB.
.macro resize_nn_yuyv2hsv qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \qd2
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1
.endm

.macro resize_nn_uyvy2hsv qd0, qd1, qd2, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \qd2
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1
.endm

# qd is the mask byte. qs0/qs1/qs2 are planar RGB. q_t0..q_t2 are H/S/V and q_t3 is scratch.
# a_t0..a_t3 may alias the gather ARs. a_min/a_max stay live across rgb8882hsv.
.macro resize_nn_yuyv2hsv_mask0 qd, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_min, a_max, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_uyvy2hsv_mask0 qd, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_min, a_max, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_yuyv2hsv_mask1 qd, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_min, a_max, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_uyvy2hsv_mask1 qd, base_addr, offsets, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, a_min, a_max, a_t0, a_t1, a_t2, a_t3, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_t3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

# qd0/qd1/qd2 end as planar packed R/G/B. qs0 = y1, qs1 = y2, qs2 = u, qs3 = v.
# qd0 is the coefficient scratch, then y1+dR. a_shift already holds 14.
.macro yuyv2rgb888 qd0, qd1, qd2, qs0, qs1, qs2, qs3, a_tmp, a_shift, q_tmp
    yuv2rgb888_delta \qs2, \qs3, \q_tmp, \qd0, \a_tmp, \a_shift
    esp.vadd.s16 \qd0, \qs0, \q_tmp
    esp.vadd.s16 \q_tmp, \qs1, \q_tmp
    esp.vadd.s16 \qd1, \qs0, \qs3
    esp.vadd.s16 \qs3, \qs1, \qs3
    esp.vadd.s16 \qd2, \qs0, \qs2
    esp.vadd.s16 \qs2, \qs1, \qs2

    yuv2rgb888_clamp6 \qd0, \q_tmp, \qd1, \qs3, \qd2, \qs2, \qs0, \qs1, \a_tmp

    esp.vzip.16 \qd0, \q_tmp
    esp.vunzip.8 \qd0, \q_tmp
    esp.vzip.16 \qd1, \qs3
    esp.vunzip.8 \qd1, \qs3
    esp.vzip.16 \qd2, \qs2
    esp.vunzip.8 \qd2, \qs2
.endm


# 16 contiguous pixels. q_y1/q_y2/q_u/q_v are YUYV byte positions and are distinct.
# vext reads q_y1/q_y2 and writes them back as the low half.
.macro cvt_load_yuyv16 q_y1, q_y2, q_u, q_v, a_src
    esp.vld.128.ip \q_y1, \a_src, 16
    esp.vld.128.ip \q_y2, \a_src, 16
    esp.vunzip.8 \q_y1, \q_y2
    esp.vunzip.8 \q_y1, \q_y2
    esp.vext.u8 \q_y1, \q_u, \q_y1
    esp.vext.u8 \q_y2, \q_v, \q_y2
.endm

.macro cvt_yuyv2rgb888 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_tmp, a_shift, q_tmp
    cvt_load_yuyv16 \q_y1, \q_y2, \q_u, \q_v, \a_src
    yuyv2rgb888 \qd0, \qd1, \qd2, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
.endm

.macro cvt_uyvy2rgb888 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_tmp, a_shift, q_tmp
    cvt_load_yuyv16 \q_y1, \q_y2, \q_u, \q_v, \a_src
    yuyv2rgb888 \qd0, \qd1, \qd2, \q_u, \q_v, \q_y1, \q_y2, \a_tmp, \a_shift, \q_tmp
.endm

.macro cvt_yuyv2bgr888 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_tmp, a_shift, q_tmp
    cvt_yuyv2rgb888 \qd2, \qd1, \qd0, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
.endm

.macro cvt_uyvy2bgr888 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_tmp, a_shift, q_tmp
    cvt_uyvy2rgb888 \qd2, \qd1, \qd0, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
.endm

# Quant temps are q_y1, q_y2, q_tmp. They are dead after the rgb888 result.
.macro cvt_yuyv2rgb888_qint8 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_lut, a_tmp, a_shift, q_tmp
    cvt_yuyv2rgb888 \qd0, \qd1, \qd2, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y1, \q_y2, \q_tmp
.endm

.macro cvt_uyvy2rgb888_qint8 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_lut, a_tmp, a_shift, q_tmp
    cvt_uyvy2rgb888 \qd0, \qd1, \qd2, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y1, \q_y2, \q_tmp
.endm

.macro cvt_yuyv2bgr888_qint8 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_lut, a_tmp, a_shift, q_tmp
    cvt_yuyv2bgr888 \qd0, \qd1, \qd2, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y1, \q_y2, \q_tmp
.endm

.macro cvt_uyvy2bgr888_qint8 qd0, qd1, qd2, a_src, q_y1, q_y2, q_u, q_v, a_lut, a_tmp, a_shift, q_tmp
    cvt_uyvy2bgr888 \qd0, \qd1, \qd2, \a_src, \q_y1, \q_y2, \q_u, \q_v, \a_tmp, \a_shift, \q_tmp
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y1, \q_y2, \q_tmp
.endm

# Planar u8 lands in qd3/qd4/qd5. Load uses qd0/qd1/qd2/q_tmp1. q_tmp0 is the yuv scratch.
.macro cvt_yuyv2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_src, a_lut, a_tmp, a_shift, q_tmp0, q_tmp1
    cvt_yuyv2rgb888 \qd3, \qd4, \qd5, \a_src, \qd0, \qd1, \qd2, \q_tmp1, \a_tmp, \a_shift, \q_tmp0
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro cvt_uyvy2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_src, a_lut, a_tmp, a_shift, q_tmp0, q_tmp1
    cvt_uyvy2rgb888 \qd3, \qd4, \qd5, \a_src, \qd0, \qd1, \qd2, \q_tmp1, \a_tmp, \a_shift, \q_tmp0
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro cvt_yuyv2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_src, a_lut, a_tmp, a_shift, q_tmp0, q_tmp1
    cvt_yuyv2bgr888 \qd3, \qd4, \qd5, \a_src, \qd0, \qd1, \qd2, \q_tmp1, \a_tmp, \a_shift, \q_tmp0
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro cvt_uyvy2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, a_src, a_lut, a_tmp, a_shift, q_tmp0, q_tmp1
    cvt_uyvy2bgr888 \qd3, \qd4, \qd5, \a_src, \qd0, \qd1, \qd2, \q_tmp1, \a_tmp, \a_shift, \q_tmp0
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

# qd0 is the yuv scratch and the low 565 half. qd1 is q_v, then the high 565 half.
.macro cvt_yuyv2rgb565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_yuyv2rgb565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_yuyv2bgr565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_yuyv2bgr565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_uyvy2rgb565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_uyvy2rgb565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_uyvy2bgr565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

.macro cvt_uyvy2bgr565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift, a_extra
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_extra
.endm

# qd0/qd1/qd2 are H/S/V and also q_y1/q_y2/q_u. q_tmp0 is q_v.
.macro cvt_yuyv2hsv qd0, qd1, qd2, a_src, a_sdiv, a_hdiv, a_const, a_shift, a_t0, a_t1, a_t2, qr, qg, qb, q_tmp0, q_tmp1
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \qd0, \qd1, \qd2, \q_tmp0, \a_const, \a_shift, \q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \qr, \qg, \qb, \a_const, \a_t0, \a_t1, \a_t2, \q_tmp0, \q_tmp1
.endm

.macro cvt_uyvy2hsv qd0, qd1, qd2, a_src, a_sdiv, a_hdiv, a_const, a_shift, a_t0, a_t1, a_t2, qr, qg, qb, q_tmp0, q_tmp1
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \qd0, \qd1, \qd2, \q_tmp0, \a_const, \a_shift, \q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \qr, \qg, \qb, \a_const, \a_t0, \a_t1, \a_t2, \q_tmp0, \q_tmp1
.endm

# qd is the mask byte and the yuv scratch. q_h/q_s/q_v/q_scratch are the YUV lanes, then H/S/V/scratch.
.macro cvt_yuyv2hsv_mask0 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_const, a_shift, a_t0, a_t1, a_t2, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_yuyv2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_const, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_uyvy2hsv_mask0 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_const, a_shift, a_t0, a_t1, a_t2, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_uyvy2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_const, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_yuyv2hsv_mask1 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_const, a_shift, a_t0, a_t1, a_t2, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_yuyv2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_const, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_uyvy2hsv_mask1 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_const, a_shift, a_t0, a_t1, a_t2, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_uyvy2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \qs0, \qs1, \qs2, \a_t0, \a_t1, \a_t2, \a_const, \q_h, \q_s, \q_v, \q_scratch
.endm
