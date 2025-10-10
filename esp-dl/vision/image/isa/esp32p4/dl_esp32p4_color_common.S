#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ < 150000)
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
    esp.vsld.8 \qd, \qs, \q_tmp
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

.macro resize_nn_load_1chn_128 qd, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 0
    esp.movi.8.q \qd, \a_tmp1, 1
    esp.movi.8.q \qd, \a_tmp2, 2
    esp.movi.8.q \qd, \a_tmp3, 3

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 4
    esp.movi.8.q \qd, \a_tmp1, 5
    esp.movi.8.q \qd, \a_tmp2, 6
    esp.movi.8.q \qd, \a_tmp3, 7

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 8
    esp.movi.8.q \qd, \a_tmp1, 9
    esp.movi.8.q \qd, \a_tmp2, 10
    esp.movi.8.q \qd, \a_tmp3, 11

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    esp.movi.8.q \qd, \a_tmp0, 12
    esp.movi.8.q \qd, \a_tmp1, 13
    esp.movi.8.q \qd, \a_tmp2, 14
    esp.movi.8.q \qd, \a_tmp3, 15
.endm

.macro resize_nn_load_store_1chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    esp.vldbc.32.ip \q_tmp0, \base_addr, 0
    esp.vld.128.ip \q_tmp1, \offsets, 16
    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    sb \a_tmp0, 0(\dst_addr)
    sb \a_tmp1, 1(\dst_addr)
    sb \a_tmp2, 2(\dst_addr)
    sb \a_tmp3, 3(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    sb \a_tmp0, 4(\dst_addr)
    sb \a_tmp1, 5(\dst_addr)
    sb \a_tmp2, 6(\dst_addr)
    sb \a_tmp3, 7(\dst_addr)

    esp.vadd.u32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    esp.movi.32.a \q_tmp1, \a_tmp0, 0
    esp.movi.32.a \q_tmp1, \a_tmp1, 1
    esp.movi.32.a \q_tmp1, \a_tmp2, 2
    esp.movi.32.a \q_tmp1, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
    sb \a_tmp0, 8(\dst_addr)
    sb \a_tmp1, 9(\dst_addr)
    sb \a_tmp2, 10(\dst_addr)
    sb \a_tmp3, 11(\dst_addr)

    esp.vadd.u32 \q_tmp2, \q_tmp0, \q_tmp2
    esp.movi.32.a \q_tmp2, \a_tmp0, 0
    esp.movi.32.a \q_tmp2, \a_tmp1, 1
    esp.movi.32.a \q_tmp2, \a_tmp2, 2
    esp.movi.32.a \q_tmp2, \a_tmp3, 3
    lb \a_tmp0, 0(\a_tmp0)
    lb \a_tmp1, 0(\a_tmp1)
    lb \a_tmp2, 0(\a_tmp2)
    lb \a_tmp3, 0(\a_tmp3)
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
