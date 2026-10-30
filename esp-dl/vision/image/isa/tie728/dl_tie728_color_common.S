.macro vsl_8 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        .literal .Lvsl_8_mask_addr\@, vsl_8_1_mask
    .elseif \n == 2
        .literal .Lvsl_8_mask_addr\@, vsl_8_2_mask
    .elseif \n == 3
        .literal .Lvsl_8_mask_addr\@, vsl_8_3_mask
    .elseif \n == 4
        .literal .Lvsl_8_mask_addr\@, vsl_8_4_mask
    .elseif \n == 5
        .literal .Lvsl_8_mask_addr\@, vsl_8_5_mask
    .elseif \n == 6
        .literal .Lvsl_8_mask_addr\@, vsl_8_6_mask
    .elseif \n == 7
        .literal .Lvsl_8_mask_addr\@, vsl_8_7_mask
    .else
        .error "unsupported n"
    .endif
    l32r \a_tmp, .Lvsl_8_mask_addr\@
    ee.vldbc.8.ip \q_tmp, \a_tmp, 0
    movi \a_tmp, \n
    wsr.sar \a_tmp
    ee.vsl.32 \qd, \qs
    ee.andq \qd, \qd, \q_tmp
.endm

.macro vsl_16 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        .literal .Lvsl_16_mask_addr\@, vsl_16_1_mask
    .elseif \n == 2
        .literal .Lvsl_16_mask_addr\@, vsl_16_2_mask
    .elseif \n == 3
        .literal .Lvsl_16_mask_addr\@, vsl_16_3_mask
    .elseif \n == 4
        .literal .Lvsl_16_mask_addr\@, vsl_16_4_mask
    .elseif \n == 5
        .literal .Lvsl_16_mask_addr\@, vsl_16_5_mask
    .elseif \n == 6
        .literal .Lvsl_16_mask_addr\@, vsl_16_6_mask
    .elseif \n == 7
        .literal .Lvsl_16_mask_addr\@, vsl_16_7_mask
    .elseif \n == 8
        .literal .Lvsl_16_mask_addr\@, vsl_16_8_mask
    .elseif \n == 9
        .literal .Lvsl_16_mask_addr\@, vsl_16_9_mask
    .elseif \n == 10
        .literal .Lvsl_16_mask_addr\@, vsl_16_10_mask
    .elseif \n == 11
        .literal .Lvsl_16_mask_addr\@, vsl_16_11_mask
    .elseif \n == 12
        .literal .Lvsl_16_mask_addr\@, vsl_16_12_mask
    .elseif \n == 13
        .literal .Lvsl_16_mask_addr\@, vsl_16_13_mask
    .elseif \n == 14
        .literal .Lvsl_16_mask_addr\@, vsl_16_14_mask
    .elseif \n == 15
        .literal .Lvsl_16_mask_addr\@, vsl_16_15_mask
    .else
        .error "unsupported n"
    .endif
    l32r \a_tmp, .Lvsl_16_mask_addr\@
    ee.vldbc.16.ip \q_tmp, \a_tmp, 0
    movi \a_tmp, \n
    wsr.sar \a_tmp
    ee.vsl.32 \qd, \qs
    ee.andq \qd, \qd, \q_tmp
.endm

.macro vsr_u8 qd, qs, q_tmp, a_tmp, n
    .if \n == 1
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_1_mask
    .elseif \n == 2
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_2_mask
    .elseif \n == 3
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_3_mask
    .elseif \n == 4
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_4_mask
    .elseif \n == 5
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_5_mask
    .elseif \n == 6
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_6_mask
    .elseif \n == 7
        .literal .Lvsr_u8_mask_addr\@, vsr_u8_7_mask
    .else
        .error "unsupported n"
    .endif
    l32r \a_tmp, .Lvsr_u8_mask_addr\@
    ee.vldbc.8.ip \q_tmp, \a_tmp, 0
    movi \a_tmp, \n
    wsr.sar \a_tmp
    ee.vsr.32 \qd, \qs
    ee.andq \qd, \qd, \q_tmp
.endm

.macro vunzipt_8_helper q0, q1, q2, q_tmp
    mv.qr \q_tmp, \q1
    ee.srci.2q \q2, \q1, 7
    ee.vzip.8 \q0, \q1
    ee.vzip.8 \q_tmp, \q2
.endm

.macro vunzipt_8 q0, q1, q2, q_tmp
    .rept 2
    vunzipt_8_helper \q0, \q1, \q2, \q_tmp
    vunzipt_8_helper \q0, \q1, \q_tmp, \q2
    .endr
.endm

.macro vzipt_8 q0, q1, q2, q_tmp
    .rept 4
    mv.qr \q_tmp, \q1
    ee.vunzip.8 \q0, \q_tmp
    ee.vunzip.8 \q1, \q2
    ee.srci.2q \q_tmp, \q1, 7
    .endr
.endm

.macro vunzipt_16_helper q0, q1, q2, q_tmp
    mv.qr \q_tmp, \q2
    ee.slci.2q \q2, \q1, 7
    ee.vzip.16 \q0, \q2
    ee.vzip.16 \q1, \q_tmp
.endm

.macro vunzipt_16 q0, q1, q2, q_tmp
    vunzipt_16_helper \q0, \q1, \q2, \q_tmp
    vunzipt_16_helper \q0, \q2, \q_tmp, \q1
    vunzipt_16_helper \q0, \q_tmp, \q1, \q2
.endm

.macro vzipt_16 q0, q1, q2, q_tmp
    .rept 3
    mv.qr \q_tmp, \q1
    ee.vunzip.16 \q0, \q_tmp
    ee.vunzip.16 \q1, \q2
    ee.srci.2q \q_tmp, \q1, 7
    .endr
.endm

.macro vld_1chn_unaligned q0, src_ptr, q_tmp
    ee.ld.128.usar.ip \q0, \src_ptr, 16
    ee.vld.128.ip \q_tmp, \src_ptr, 0
    ee.src.q \q0, \q0, \q_tmp
.endm

.macro vld_2chn_unaligned q0, q1, src_ptr, q_tmp
    ee.ld.128.usar.ip \q0, \src_ptr, 16
    ee.vld.128.ip \q1, \src_ptr, 16
    ee.vld.128.ip \q_tmp, \src_ptr, 0
    ee.src.q \q0, \q0, \q1
    ee.src.q \q1, \q1, \q_tmp
.endm

.macro vld_3chn_unaligned q0, q1, q2, src_ptr, q_tmp
    ee.ld.128.usar.ip \q0, \src_ptr, 16
    ee.vld.128.ip \q1, \src_ptr, 16
    ee.vld.128.ip \q2, \src_ptr, 16
    ee.vld.128.ip \q_tmp, \src_ptr, 0
    ee.src.q \q0, \q0, \q1
    ee.src.q \q1, \q1, \q2
    ee.src.q \q2, \q2, \q_tmp
.endm

.macro vst_unaligned_init dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
    ee.ld.128.usar.ip \q_tmp, \dst_ptr, 0
    rur.sar_byte \a_dst_sar
    ee.src.q \q_prev, \q_prev, \q_tmp
    neg \a_dst_sar, \a_dst_sar
    addi \a_dst_sar, \a_dst_sar, 16
    wur.sar_byte \a_dst_sar
.ifnc "\a_stack", "none"
    ee.vst.128.ip \q_prev, \a_stack, 0
.endif
.endm

.macro vst_1chn_unaligned q0, dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
    wur.sar_byte \a_dst_sar
.ifnc "\a_stack", "none"
    ee.vld.128.ip \q_prev, \a_stack, 0
.endif
    ee.src.q.qup \q_tmp, \q_prev, \q0
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
.ifnc "\a_stack", "none"
    ee.vst.128.ip \q_prev, \a_stack, 0
.endif
.endm

.macro vst_2chn_unaligned q0, q1, dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
    wur.sar_byte \a_dst_sar
.ifnc "\a_stack", "none"
    ee.vld.128.ip \q_prev, \a_stack, 0
.endif
    ee.src.q.qup \q_tmp, \q_prev, \q0
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q1
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
.ifnc "\a_stack", "none"
    ee.vst.128.ip \q_prev, \a_stack, 0
.endif
.endm

.macro vst_3chn_unaligned q0, q1, q2, dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
    wur.sar_byte \a_dst_sar
.ifnc "\a_stack", "none"
    ee.vld.128.ip \q_prev, \a_stack, 0
.endif
    ee.src.q.qup \q_tmp, \q_prev, \q0
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q1
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q2
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
.ifnc "\a_stack", "none"
    ee.vst.128.ip \q_prev, \a_stack, 0
.endif
.endm

.macro vst_6chn_unaligned q0, q1, q2, q3, q4, q5, dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
    wur.sar_byte \a_dst_sar
.ifnc "\a_stack", "none"
    ee.vld.128.ip \q_prev, \a_stack, 0
.endif
    ee.src.q.qup \q_tmp, \q_prev, \q0
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q1
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q2
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q3
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q4
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
    ee.src.q.qup \q_tmp, \q_prev, \q5
    ee.vst.128.ip \q_tmp, \dst_ptr, 16
.ifnc "\a_stack", "none"
    ee.vst.128.ip \q_prev, \a_stack, 0
.endif
.endm

.macro vst_unaligned_flush dst_ptr, a_dst_sar, q_prev, q_tmp, a_stack=none
.ifnc "\a_stack", "none"
    ee.vld.128.ip \q_prev, \a_stack, 0
.endif
    ee.ld.128.usar.ip \q_tmp, \dst_ptr, 0
    ee.src.q \q_tmp, \q_tmp, \q_tmp
    wur.sar_byte \a_dst_sar
    ee.src.q \q_tmp, \q_prev, \q_tmp
    ee.vst.128.ip \q_tmp, \dst_ptr, 0
.endm


.macro vlut_32_8 qd0, qd1, q_idx, a_lut
    ee.ldxq.32 \qd1, \q_idx, \a_lut, 0, 4
    ee.ldxq.32 \qd1, \q_idx, \a_lut, 1, 5
    ee.ldxq.32 \qd1, \q_idx, \a_lut, 2, 6
    ee.ldxq.32 \qd1, \q_idx, \a_lut, 3, 7
    ee.ldxq.32 \qd0, \q_idx, \a_lut, 3, 3
    ee.ldxq.32 \qd0, \q_idx, \a_lut, 2, 2
    ee.ldxq.32 \qd0, \q_idx, \a_lut, 1, 1
    ee.ldxq.32 \qd0, \q_idx, \a_lut, 0, 0
.endm

.macro vlut_16_8 qd, q_idx_8, a_lut, q_tmp
    vlut_32_8 \qd, \q_tmp, \q_idx_8, \a_lut
    ee.vunzip.16 \qd, \q_tmp
.endm

.macro vlut_16_16 qd0, qd1, q_idx_16, a_lut, q_tmp0, q_tmp1
    ee.zero.q \q_tmp1
    ee.vzip.8 \q_idx_16, \q_tmp1
    vlut_32_8 \qd0, \q_tmp0, \q_idx_16, \a_lut
    ee.vunzip.16 \qd0, \q_tmp0
    vlut_32_8 \qd1, \q_tmp0, \q_tmp1, \a_lut
    ee.vunzip.16 \qd1, \q_tmp0
.endm

.macro vlut_16_16_ip qd0, qd1, q_idx_16, a_lut, q_tmp0, q_tmp1, imm
    vlut_16_16 \qd0, \qd1, \q_idx_16, \a_lut, \q_tmp0, \q_tmp1
    addi \a_lut, \a_lut, \imm
.endm

.macro vlut_8_16 qd, q_idx_16, a_lut, q_tmp0, q_tmp1, q_tmp2
    vlut_16_16 \qd, \q_tmp0, \q_idx_16, \a_lut, \q_tmp1, \q_tmp2
    ee.vunzip.8 \qd, \q_tmp0
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
    .literal .Lrgb5652rgb888_mask_tab_addr\@, rgb5652rgb888_mask_tab
    l32r \a_tmp0, .Lrgb5652rgb888_mask_tab_addr\@
    ee.vldbc.8.ip \qd0, \a_tmp0, 2

    ee.andq \qd0, \qs0, \qd0
    vsr_u8 \qd0, \qd0, \qd1, \a_tmp1, 3

    vsl_8 \qd1, \qs1, \qd2, \a_tmp1, 5
    ee.orq \qd1, \qd0, \qd1

    vsl_8 \qd2, \qs0, \qd0, \a_tmp1, 3

    ee.vldbc.8.ip \qd0, \a_tmp0, 0
    ee.andq \qd0, \qs1, \qd0
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

.macro rgb5652bgr565 qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    .literal .Lrgb5652bgr565_mask_tab_addr\@, rgb5652rgb888_mask_tab
    l32r \a_tmp0, .Lrgb5652bgr565_mask_tab_addr\@
    ee.vldbc.8.ip \qd0, \a_tmp0, 1
    ee.andq \qd0, \qs0, \qd0
    vsl_8 \qs0, \qs0, \qd1, \a_tmp1, 3
    ee.vldbc.8.ip \qd1, \a_tmp0, 0
    ee.andq \qd1, \qs1, \qd1
    ee.orq \qd1, \qd1, \qs0
    vsr_u8 \qs1, \qs1, \qs0, \a_tmp1, 3
    ee.orq \qd0, \qd0, \qs1
.endm

.macro rgb565le2bgr565le qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd0, \qd1, \qs0, \qs1, \a_tmp0, \a_tmp1
.endm

.macro rgb565be2bgr565be qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd1, \qd0, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb565le2bgr565be qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd1, \qd0, \qs0, \qs1, \a_tmp0, \a_tmp1
.endm

.macro rgb565be2bgr565le qd0, qd1, qs0, qs1, a_tmp0, a_tmp1
    rgb5652bgr565 \qd0, \qd1, \qs1, \qs0, \a_tmp0, \a_tmp1
.endm

.macro rgb8882rgb565 qd0, qd1, qs0, qs1, qs2, a_tmp0, a_tmp1
    .literal .Lrgb8882rgb565_mask_tab_addr\@, rgb8882rgb565_mask_tab
    l32r \a_tmp0, .Lrgb8882rgb565_mask_tab_addr\@
    ee.vldbc.8.ip \qd0, \a_tmp0, 1
    ee.andq \qs0, \qs0, \qd0
    vsr_u8 \qs2, \qs2, \qd0, \a_tmp1, 3

    ee.vldbc.8.ip \qd0, \a_tmp0, 1
    ee.andq \qd0, \qs1, \qd0
    vsl_8 \qd0, \qd0, \qd1, \a_tmp1, 3
    ee.orq \qd0, \qd0, \qs2

    ee.vldbc.8.ip \qd1, \a_tmp0, 0
    ee.andq \qd1, \qs1, \qd1
    vsr_u8 \qd1, \qd1, \qs1, \a_tmp1, 5
    ee.orq \qd1, \qd1, \qs0
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
    .literal .Lrgb8882gray_const_tab_addr\@, rgb8882gray_const_tab
    l32r \a_tmp, .Lrgb8882gray_const_tab_addr\@

    ee.vldbc.16.ip \qd, \a_tmp, 2
    ee.vldbc.16.ip \q_tmp0, \a_tmp, 2
    ee.vldbc.16.ip \q_tmp1, \a_tmp, 2
    ee.vldbc.16.ip \q_tmp2, \a_tmp, 0
    movi \a_tmp, 15

    ee.mov.u16.qacc \qd
    ee.zero.q \q_tmp3
    ee.vzip.8 \qs0, \q_tmp3
    ee.vmulas.u16.qacc \q_tmp3, \q_tmp0
    ee.zero.q \q_tmp3
    ee.vzip.8 \qs1, \q_tmp3
    ee.vmulas.u16.qacc \q_tmp3, \q_tmp1
    ee.zero.q \q_tmp3
    ee.vzip.8 \qs2, \q_tmp3
    ee.vmulas.u16.qacc \q_tmp3, \q_tmp2
    ee.srcmb.s16.qacc \q_tmp3, \a_tmp, 0

    ee.mov.u16.qacc \qd
    ee.vmulas.u16.qacc \qs0, \q_tmp0
    ee.vmulas.u16.qacc \qs1, \q_tmp1
    ee.vmulas.u16.qacc \qs2, \q_tmp2
    ee.srcmb.s16.qacc \qd, \a_tmp, 0

    ee.vunzip.8 \qd, \q_tmp3
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

.macro hsv2hsv_mask qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp0, q_tmp1, a_tmp, h_across_zero
    .literal .Lhsv_s8_bias_addr\@, rgb8882hsv_const_tab + 6
    l32r \a_tmp, .Lhsv_s8_bias_addr\@
    ee.vldbc.8 \q_tmp1, \a_tmp

    # Convert unsigned [0, 255] to signed [-128, 127] via XOR 0x80
    ee.xorq \qs0, \qs0, \q_tmp1
    ee.xorq \qs1, \qs1, \q_tmp1
    ee.xorq \qs2, \qs2, \q_tmp1

    # h < h_min
    ee.vldbc.8.ip \q_tmp0, \a_hsv_min_addr, 1
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.lt.s8 \qd, \qs0, \q_tmp0
    # h > h_max
    ee.vldbc.8.ip \q_tmp0, \a_hsv_max_addr, 1
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.gt.s8 \q_tmp0, \qs0, \q_tmp0
    .if \h_across_zero == 0
        ee.orq \qd, \qd, \q_tmp0
    .elseif \h_across_zero == 1
        ee.andq \qd, \qd, \q_tmp0
    .else
        .error "h_across_zero can only be 0/1."
    .endif
    # s < s_min
    ee.vldbc.8.ip \q_tmp0, \a_hsv_min_addr, 1
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.lt.s8 \q_tmp0, \qs1, \q_tmp0
    ee.orq \qd, \qd, \q_tmp0
    # s > s_max
    ee.vldbc.8.ip \q_tmp0, \a_hsv_max_addr, 1
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.gt.s8 \q_tmp0, \qs1, \q_tmp0
    ee.orq \qd, \qd, \q_tmp0
    # v < v_min
    ee.vldbc.8 \q_tmp0, \a_hsv_min_addr
    addi \a_hsv_min_addr, \a_hsv_min_addr, -2
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.lt.s8 \q_tmp0, \qs2, \q_tmp0
    ee.orq \qd, \qd, \q_tmp0
    # v > v_max
    ee.vldbc.8 \q_tmp0, \a_hsv_max_addr
    addi \a_hsv_max_addr, \a_hsv_max_addr, -2
    ee.xorq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vcmp.gt.s8 \q_tmp0, \qs2, \q_tmp0
    ee.orq \qd, \qd, \q_tmp0
    ee.notq \qd, \qd
.endm

.macro hsv2hsv_mask0 qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp0, q_tmp1, a_tmp
    hsv2hsv_mask \qd, \a_hsv_min_addr, \a_hsv_max_addr, \qs0, \qs1, \qs2, \q_tmp0, \q_tmp1, \a_tmp, 0
.endm

.macro hsv2hsv_mask1 qd, a_hsv_min_addr, a_hsv_max_addr, qs0, qs1, qs2, q_tmp0, q_tmp1, a_tmp
    hsv2hsv_mask \qd, \a_hsv_min_addr, \a_hsv_max_addr, \qs0, \qs1, \qs2, \q_tmp0, \q_tmp1, \a_tmp, 1
.endm

.macro mul_u32xu8_round_and_shift qd, q_u32, q_u8, q_round_delta, a_tmp0, a_tmp1, a_tmp2, sel4_q_u8
    ee.movi.32.a \q_u8, \a_tmp2, \sel4_q_u8

    ee.movi.32.a \q_u32, \a_tmp0, 0
    extui \a_tmp1, \a_tmp2, 0, 8
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 0

    ee.movi.32.a \q_u32, \a_tmp0, 1
    extui \a_tmp1, \a_tmp2, 8, 8
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 1

    ee.movi.32.a \q_u32, \a_tmp0, 2
    extui \a_tmp1, \a_tmp2, 16, 8
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 2

    ee.movi.32.a \q_u32, \a_tmp0, 3
    extui \a_tmp1, \a_tmp2, 24, 8
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 3

    ee.vadds.s32 \qd, \qd, \q_round_delta
    ee.vsr.32 \qd, \qd
.endm

.macro cal_s qd, q_vmax, q_diff, q_round_delta, a_sdiv_lut, q_tmp0, q_tmp1, a_tmp0, a_tmp1, a_tmp2
    movi \a_tmp0, 12
    wsr.sar \a_tmp0
    ee.zero.q \q_tmp0
    ee.vzip.8 \q_vmax, \q_tmp0

    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 0, 0
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 1, 1
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 2, 2
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 3, 3
    mul_u32xu8_round_and_shift \qd, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, \a_tmp2, 0

    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 0, 4
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 1, 5
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 2, 6
    ee.ldxq.32 \q_tmp1, \q_vmax, \a_sdiv_lut, 3, 7
    mul_u32xu8_round_and_shift \q_vmax, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, \a_tmp2, 1

    ee.vunzip.16 \qd, \q_vmax

    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 0, 0
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 1, 1
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 2, 2
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 3, 3
    mul_u32xu8_round_and_shift \q_vmax, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, \a_tmp2, 2

    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 0, 4
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 1, 5
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 2, 6
    ee.ldxq.32 \q_tmp1, \q_tmp0, \a_sdiv_lut, 3, 7
    mul_u32xu8_round_and_shift \q_tmp0, \q_tmp1, \q_diff, \q_round_delta, \a_tmp0, \a_tmp1, \a_tmp2, 3

    ee.vunzip.16 \q_vmax, \q_tmp0
    ee.vunzip.8 \qd, \q_vmax
.endm

.macro mul_s32xs32_round_and_shift qd, qs0, qs1, q_round_delta, a_tmp0, a_tmp1
    ee.movi.32.a \qs0, \a_tmp0, 0
    ee.movi.32.a \qs1, \a_tmp1, 0
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 0

    ee.movi.32.a \qs0, \a_tmp0, 1
    ee.movi.32.a \qs1, \a_tmp1, 1
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 1

    ee.movi.32.a \qs0, \a_tmp0, 2
    ee.movi.32.a \qs1, \a_tmp1, 2
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 2

    ee.movi.32.a \qs0, \a_tmp0, 3
    ee.movi.32.a \qs1, \a_tmp1, 3
    mull \a_tmp0, \a_tmp0, \a_tmp1
    ee.movi.32.q \qd, \a_tmp0, 3

    ee.vadds.s32 \qd, \qd, \q_round_delta
    ee.vsr.32 \qd, \qd
.endm

.macro cal_h_helper qd, q_h, q_diff, q_round_delta, a_hdiv_lut, q_tmp0, q_tmp1, a_tmp0, a_tmp1
    ee.zero.q \q_tmp0
    ee.vcmp.lt.s16 \q_tmp0, \q_h, \q_tmp0
    ee.vzip.16 \q_h, \q_tmp0

    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 0, 0
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 1, 1
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 2, 2
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 3, 3
    mul_s32xs32_round_and_shift \qd, \q_h, \q_tmp1, \q_round_delta, \a_tmp0, \a_tmp1

    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 0, 4
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 1, 5
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 2, 6
    ee.ldxq.32 \q_tmp1, \q_diff, \a_hdiv_lut, 3, 7
    mul_s32xs32_round_and_shift \q_h, \q_tmp0, \q_tmp1, \q_round_delta, \a_tmp0, \a_tmp1

    ee.vunzip.16 \qd, \q_h
.endm

.macro rgb8882hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1
    # Stack frame at \a_stack (must have 144 bytes aligned to 16):
    # [0]:   qd0 (vmax)
    # [16]:  qd2 (diff)
    # [32]:  mask1 (vr)
    # [48]:  mask2 (~vr & vg)
    # [64]:  mask3 (~vr & ~vg)
    # [80]:  qs0 (R) / later overwritten by H_low (s16x8)
    # [96]:  qs1 (G)
    # [112]: qs2 (B)
    # [128]: qd1 (S)
    addi \a_tmp0, \a_stack, 0

    .literal .Lrgb8882hsv_const_tab_addr\@, rgb8882hsv_const_tab
    l32r \a_tmp1, .Lrgb8882hsv_const_tab_addr\@
    addi \a_tmp2, \a_tmp1, 6
    ee.vldbc.8 \q_tmp0, \a_tmp2

    # Save original u8 R, G, B to stack [80..127] before XOR 0x80
    addi \a_tmp2, \a_stack, 80
    ee.vst.128.ip \qs0, \a_tmp2, 16
    ee.vst.128.ip \qs1, \a_tmp2, 16
    ee.vst.128.ip \qs2, \a_tmp2, 16

    # Shift u8 [0, 255] to s8 [-128, 127] via XOR 0x80 for unsigned max/min
    ee.xorq \qs0, \qs0, \q_tmp0
    ee.xorq \qs1, \qs1, \q_tmp0
    ee.xorq \qs2, \qs2, \q_tmp0

    ee.vmax.s8 \qd0, \qs0, \qs1
    ee.vmax.s8 \qd0, \qd0, \qs2
    ee.vmin.s8 \qd2, \qs0, \qs1
    ee.vmin.s8 \qd2, \qd2, \qs2

    # Equality masks can be computed on XORed values directly
    ee.vcmp.eq.s8 \q_tmp1, \qd0, \qs0
    # Restore vmax and vmin to u8 [0, 255]
    ee.xorq \qd0, \qd0, \q_tmp0
    ee.xorq \qd2, \qd2, \q_tmp0

    # Save vmax to stack [0]
    ee.vst.128.ip \qd0, \a_tmp0, 16

    # Compute diff = vmax - vmin in u16 (8 lanes x 2) to avoid s8 saturation on [128, 255]
    mv.qr \qs0, \qd0
    ee.zero.q \q_tmp0
    ee.vzip.8 \qs0, \q_tmp0
    ee.zero.q \qs2
    ee.vzip.8 \qd2, \qs2
    ee.vsubs.s16 \qd2, \qs0, \qd2
    ee.vsubs.s16 \q_tmp0, \q_tmp0, \qs2
    ee.vunzip.8 \qd2, \q_tmp0
    # Save diff to stack [16]
    ee.vst.128.ip \qd2, \a_tmp0, 16

    # Save mask1 (vr) to stack [32]
    ee.vst.128.ip \q_tmp1, \a_tmp0, 16
    ee.notq \q_tmp0, \q_tmp1
    # Reload G from stack [96] and compare with vmax
    addi \a_tmp2, \a_stack, 96
    ee.vld.128.ip \qs1, \a_tmp2, 0
    ee.vcmp.eq.s8 \q_tmp1, \qd0, \qs1
    ee.andq \q_tmp1, \q_tmp1, \q_tmp0
    # Save mask2 (~vr & vg) to stack [48]
    ee.vst.128.ip \q_tmp1, \a_tmp0, 16
    ee.notq \q_tmp1, \q_tmp1
    ee.andq \q_tmp1, \q_tmp0, \q_tmp1
    # Save mask3 (~vr & ~vg) to stack [64], advance \a_tmp0 to [128]
    ee.vst.128.ip \q_tmp1, \a_tmp0, 64

    # Broadcast round_delta (2048) from rgb8882hsv_const_tab[0]
    ee.vldbc.32.ip \q_tmp0, \a_tmp1, 0

    cal_s \qd1, \qd0, \qd2, \q_tmp0, \a_sdiv_lut, \q_tmp1, \qs0, \a_tmp1, \a_tmp2, \a_tmp3
    # Save S to stack [128], move \a_tmp0 to [80]
    ee.vst.128.ip \qd1, \a_tmp0, -48

    # First 8 pixels (low half):
    # Load R, G, B from [80], [96], [112] and zero-extend low 8 bytes to u16 in \qs0, \qs1, \qs2
    ee.zero.q \q_tmp0
    ee.vld.128.ip \qs0, \a_tmp0, 16
    ee.vzip.8 \qs0, \q_tmp0
    ee.zero.q \q_tmp0
    ee.vld.128.ip \qs1, \a_tmp0, 16
    ee.vzip.8 \qs1, \q_tmp0
    ee.zero.q \q_tmp0
    ee.vld.128.ip \qs2, \a_tmp0, -96
    ee.vzip.8 \qs2, \q_tmp0

    # diff low 8 bytes from [16] -> u16 in \qd1
    ee.zero.q \q_tmp0
    ee.vld.128.ip \qd1, \a_tmp0, 16
    ee.vzip.8 \qd1, \q_tmp0

    # mask1 low 8 bytes from [32] -> s16 in \q_tmp0
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    mv.qr \q_tmp1, \q_tmp0
    ee.vzip.8 \q_tmp0, \q_tmp1
    ee.vsubs.s16 \q_tmp1, \qs1, \qs2
    ee.andq \qd0, \q_tmp0, \q_tmp1

    # mask2 low 8 bytes from [48] -> s16 in \q_tmp0
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    mv.qr \q_tmp1, \q_tmp0
    ee.vzip.8 \q_tmp0, \q_tmp1
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 1
    ee.vsubs.s16 \q_tmp1, \qs2, \qs0
    ee.vadds.s16 \q_tmp1, \q_tmp1, \qd2
    ee.andq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vadds.s16 \qd0, \qd0, \q_tmp0

    # mask3 low 8 bytes from [64] -> s16 in \q_tmp0, advance \a_tmp0 to [80]
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    mv.qr \q_tmp1, \q_tmp0
    ee.vzip.8 \q_tmp0, \q_tmp1
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 2
    ee.vsubs.s16 \q_tmp1, \qs0, \qs1
    ee.vadds.s16 \q_tmp1, \q_tmp1, \qd2
    ee.andq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vadds.s16 \qd0, \qd0, \q_tmp0

    # Restore \a_tmp1 to rgb8882hsv_const_tab and set SAR = 12
    l32r \a_tmp1, .Lrgb8882hsv_const_tab_addr\@
    movi \a_tmp2, 12
    wsr.sar \a_tmp2
    ee.vldbc.32.ip \q_tmp0, \a_tmp1, 0
    cal_h_helper \qd2, \qd0, \qd1, \q_tmp0, \a_hdiv_lut, \q_tmp1, \qs0, \a_tmp2, \a_tmp3

    # Second 8 pixels (high half):
    # Load R, G, B from [80], [96], [112] and zero-extend high 8 bytes to u16 in \qs0, \qs1, \qs2
    ee.zero.q \qs0
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    ee.vzip.8 \q_tmp0, \qs0
    ee.zero.q \qs1
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    ee.vzip.8 \q_tmp0, \qs1
    ee.zero.q \qs2
    ee.vld.128.ip \q_tmp0, \a_tmp0, -32
    ee.vzip.8 \q_tmp0, \qs2
    # Store H_low (\qd2) to [80], move \a_tmp0 to [16]
    ee.vst.128.ip \qd2, \a_tmp0, -64

    # diff high 8 bytes from [16] -> u16 in \qd1
    ee.zero.q \qd1
    ee.vld.128.ip \q_tmp0, \a_tmp0, 16
    ee.vzip.8 \q_tmp0, \qd1

    # mask1 high 8 bytes from [32] -> s16 in \q_tmp0
    ee.vld.128.ip \q_tmp1, \a_tmp0, 16
    mv.qr \q_tmp0, \q_tmp1
    ee.vzip.8 \q_tmp1, \q_tmp0
    ee.vsubs.s16 \q_tmp1, \qs1, \qs2
    ee.andq \qd0, \q_tmp0, \q_tmp1

    # mask2 high 8 bytes from [48] -> s16 in \q_tmp0
    ee.vld.128.ip \q_tmp1, \a_tmp0, 16
    mv.qr \q_tmp0, \q_tmp1
    ee.vzip.8 \q_tmp1, \q_tmp0
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 1
    ee.vsubs.s16 \q_tmp1, \qs2, \qs0
    ee.vadds.s16 \q_tmp1, \q_tmp1, \qd2
    ee.andq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vadds.s16 \qd0, \qd0, \q_tmp0

    # mask3 high 8 bytes from [64] -> s16 in \q_tmp0, advance \a_tmp0 to [80]
    ee.vld.128.ip \q_tmp1, \a_tmp0, 16
    mv.qr \q_tmp0, \q_tmp1
    ee.vzip.8 \q_tmp1, \q_tmp0
    vsl_16 \qd2, \qd1, \q_tmp1, \a_tmp2, 2
    ee.vsubs.s16 \q_tmp1, \qs0, \qs1
    ee.vadds.s16 \q_tmp1, \q_tmp1, \qd2
    ee.andq \q_tmp0, \q_tmp0, \q_tmp1
    ee.vadds.s16 \qd0, \qd0, \q_tmp0

    # RestoreSAR = 12 (clobbered by vsl_16), load round_delta (2048) and advance \a_tmp1 to +4 (h_range=180)
    movi \a_tmp2, 12
    wsr.sar \a_tmp2
    ee.vldbc.32.ip \q_tmp0, \a_tmp1, 4
    cal_h_helper \qs1, \qd0, \qd1, \q_tmp0, \a_hdiv_lut, \q_tmp1, \qs0, \a_tmp2, \a_tmp3

    # Add 180 if H < 0
    ee.vldbc.16.ip \q_tmp0, \a_tmp1, 0
    ee.vld.128.ip \qs0, \a_tmp0, 48
    ee.zero.q \q_tmp1
    ee.vcmp.lt.s16 \qd0, \qs0, \q_tmp1
    ee.andq \qd0, \qd0, \q_tmp0
    ee.vadds.s16 \qd0, \qd0, \qs0

    ee.vcmp.lt.s16 \qd1, \qs1, \q_tmp1
    ee.andq \qd1, \qd1, \q_tmp0
    ee.vadds.s16 \qd1, \qd1, \qs1
    ee.vunzip.8 \qd0, \qd1
    # Load S from [128] into \qd1, move \a_tmp0 to [0]
    ee.vld.128.ip \qd1, \a_tmp0, -128
    # Load V from [0] into \qd2
    ee.vld.128.ip \qd2, \a_tmp0, 0
.endm

.macro bgr8882hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \qs2, \qs1, \qs0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp0, \q_tmp1
.endm

.macro rgb8882hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask0 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \qs0, \a_tmp0
.endm

.macro rgb8882hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    rgb8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask1 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \qs0, \a_tmp0
.endm

.macro bgr8882hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask0 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \qs0, \a_tmp0
.endm

.macro bgr8882hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, qs2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3
    bgr8882hsv \q_tmp0, \q_tmp1, \q_tmp2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \qs0, \qs1, \qs2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \qd
    hsv2hsv_mask1 \qd, \a_hsv_min_addr, \a_hsv_max_addr, \q_tmp0, \q_tmp1, \q_tmp2, \q_tmp3, \qs0, \a_tmp0
.endm

.macro rgb565le2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro rgb565be2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs1, \qs0, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro bgr565le2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \q_tmp2, \q_tmp1, \q_tmp0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro bgr565be2hsv qd0, qd1, qd2, a_sdiv_lut, a_hdiv_lut, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2
    rgb5652rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs1, \qs0, \a_tmp0, \a_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv_lut, \a_hdiv_lut, \a_stack, \q_tmp2, \q_tmp1, \q_tmp0, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \qs0, \qs1
.endm

.macro rgb565le2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565le2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565be2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro rgb565be2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2rgb888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565le2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565le2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565le2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565be2hsv_mask0 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask0 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm

.macro bgr565be2hsv_mask1 qd, a_sdiv_lut, a_hdiv_lut, a_hsv_min_addr, a_hsv_max_addr, a_stack, qs0, qs1, a_tmp0, a_tmp1, a_tmp2, a_tmp3, q_tmp0, q_tmp1, q_tmp2, q_tmp3, q_tmp4
    rgb565be2bgr888 \q_tmp0, \q_tmp1, \q_tmp2, \qs0, \qs1, \a_tmp0, \a_tmp1
    rgb8882hsv_mask1 \qd, \a_sdiv_lut, \a_hdiv_lut, \a_hsv_min_addr, \a_hsv_max_addr, \a_stack, \q_tmp0, \q_tmp1, \q_tmp2, \a_tmp0, \a_tmp1, \a_tmp2, \a_tmp3, \q_tmp3, \q_tmp4, \qs0, \qs1
.endm


# byte_off is 0 for the pixel address, 1 for the next byte (UYVY Y).
.macro resize_nn_load_1chn_128 qd, base_addr, offsets, a_stack, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, byte_off=0
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \a_stack, 0
    s8i \a_tmp1, \a_stack, 1
    s8i \a_tmp2, \a_stack, 2
    s8i \a_tmp3, \a_stack, 3

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \a_stack, 4
    s8i \a_tmp1, \a_stack, 5
    s8i \a_tmp2, \a_stack, 6
    s8i \a_tmp3, \a_stack, 7

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \a_stack, 8
    s8i \a_tmp1, \a_stack, 9
    s8i \a_tmp2, \a_stack, 10
    s8i \a_tmp3, \a_stack, 11

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \a_stack, 12
    s8i \a_tmp1, \a_stack, 13
    s8i \a_tmp2, \a_stack, 14
    s8i \a_tmp3, \a_stack, 15
    ee.vld.128.ip \qd, \a_stack, 0
.endm

.macro resize_nn_load_store_1chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, byte_off=0
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \dst_addr, 0
    s8i \a_tmp1, \dst_addr, 1
    s8i \a_tmp2, \dst_addr, 2
    s8i \a_tmp3, \dst_addr, 3

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \dst_addr, 4
    s8i \a_tmp1, \dst_addr, 5
    s8i \a_tmp2, \dst_addr, 6
    s8i \a_tmp3, \dst_addr, 7

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \dst_addr, 8
    s8i \a_tmp1, \dst_addr, 9
    s8i \a_tmp2, \dst_addr, 10
    s8i \a_tmp3, \dst_addr, 11

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l8ui \a_tmp0, \a_tmp0, \byte_off
    l8ui \a_tmp1, \a_tmp1, \byte_off
    l8ui \a_tmp2, \a_tmp2, \byte_off
    l8ui \a_tmp3, \a_tmp3, \byte_off
    s8i \a_tmp0, \dst_addr, 12
    s8i \a_tmp1, \dst_addr, 13
    s8i \a_tmp2, \dst_addr, 14
    s8i \a_tmp3, \dst_addr, 15
    addi \dst_addr, \dst_addr, 16
.endm

.macro resize_nn_load_2chn_128 qd0, qd1, base_addr, offsets, a_stack, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \a_stack, 0
    s16i \a_tmp1, \a_stack, 2
    s16i \a_tmp2, \a_stack, 4
    s16i \a_tmp3, \a_stack, 6

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \a_stack, 8
    s16i \a_tmp1, \a_stack, 10
    s16i \a_tmp2, \a_stack, 12
    s16i \a_tmp3, \a_stack, 14
    ee.vld.128.ip \qd0, \a_stack, 0

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \a_stack, 0
    s16i \a_tmp1, \a_stack, 2
    s16i \a_tmp2, \a_stack, 4
    s16i \a_tmp3, \a_stack, 6

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \a_stack, 8
    s16i \a_tmp1, \a_stack, 10
    s16i \a_tmp2, \a_stack, 12
    s16i \a_tmp3, \a_stack, 14
    ee.vld.128.ip \qd1, \a_stack, 0
.endm

.macro resize_nn_load_store_2chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \dst_addr, 0
    s16i \a_tmp1, \dst_addr, 2
    s16i \a_tmp2, \dst_addr, 4
    s16i \a_tmp3, \dst_addr, 6

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \dst_addr, 8
    s16i \a_tmp1, \dst_addr, 10
    s16i \a_tmp2, \dst_addr, 12
    s16i \a_tmp3, \dst_addr, 14

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \dst_addr, 16
    s16i \a_tmp1, \dst_addr, 18
    s16i \a_tmp2, \dst_addr, 20
    s16i \a_tmp3, \dst_addr, 22

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    l16ui \a_tmp0, \a_tmp0, 0
    l16ui \a_tmp1, \a_tmp1, 0
    l16ui \a_tmp2, \a_tmp2, 0
    l16ui \a_tmp3, \a_tmp3, 0
    s16i \a_tmp0, \dst_addr, 24
    s16i \a_tmp1, \dst_addr, 26
    s16i \a_tmp2, \dst_addr, 28
    s16i \a_tmp3, \dst_addr, 30
    addi \dst_addr, \dst_addr, 32
.endm

.macro resize_nn_load_3chn_128 qd0, qd1, qd2, base_addr, offsets, a_stack, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 0
    s8i \a_tmp3, \a_stack, 1
    s8i \a_tmp0, \a_stack, 2
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 3
    s8i \a_tmp3, \a_stack, 4
    s8i \a_tmp1, \a_stack, 5
    ee.movi.32.a \q_tmp1, \a_tmp0, 2
    ee.movi.32.a \q_tmp1, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 6
    s8i \a_tmp3, \a_stack, 7
    s8i \a_tmp0, \a_stack, 8
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 9
    s8i \a_tmp3, \a_stack, 10
    s8i \a_tmp1, \a_stack, 11

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 12
    s8i \a_tmp3, \a_stack, 13
    s8i \a_tmp0, \a_stack, 14
    l8ui \a_tmp2, \a_tmp1, 0
    s8i \a_tmp2, \a_stack, 15
    ee.vld.128.ip \qd0, \a_stack, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp3, \a_stack, 0
    s8i \a_tmp1, \a_stack, 1
    ee.movi.32.a \q_tmp2, \a_tmp0, 2
    ee.movi.32.a \q_tmp2, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 2
    s8i \a_tmp3, \a_stack, 3
    s8i \a_tmp0, \a_stack, 4
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 5
    s8i \a_tmp3, \a_stack, 6
    s8i \a_tmp1, \a_stack, 7

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 8
    s8i \a_tmp3, \a_stack, 9
    s8i \a_tmp0, \a_stack, 10
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 11
    s8i \a_tmp3, \a_stack, 12
    s8i \a_tmp1, \a_stack, 13
    ee.movi.32.a \q_tmp1, \a_tmp0, 2
    ee.movi.32.a \q_tmp1, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    s8i \a_tmp2, \a_stack, 14
    s8i \a_tmp3, \a_stack, 15
    ee.vld.128.ip \qd1, \a_stack, 0
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp0, \a_stack, 0
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 1
    s8i \a_tmp3, \a_stack, 2
    s8i \a_tmp1, \a_stack, 3

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 4
    s8i \a_tmp3, \a_stack, 5
    s8i \a_tmp0, \a_stack, 6
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 7
    s8i \a_tmp3, \a_stack, 8
    s8i \a_tmp1, \a_stack, 9
    ee.movi.32.a \q_tmp2, \a_tmp0, 2
    ee.movi.32.a \q_tmp2, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \a_stack, 10
    s8i \a_tmp3, \a_stack, 11
    s8i \a_tmp0, \a_stack, 12
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \a_stack, 13
    s8i \a_tmp3, \a_stack, 14
    s8i \a_tmp1, \a_stack, 15
    ee.vld.128.ip \qd2, \a_stack, 0
.endm

.macro resize_nn_load_store_3chn_128 dst_addr, base_addr, offsets, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 0
    s8i \a_tmp3, \dst_addr, 1
    s8i \a_tmp0, \dst_addr, 2
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 3
    s8i \a_tmp3, \dst_addr, 4
    s8i \a_tmp1, \dst_addr, 5
    ee.movi.32.a \q_tmp1, \a_tmp0, 2
    ee.movi.32.a \q_tmp1, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 6
    s8i \a_tmp3, \dst_addr, 7
    s8i \a_tmp0, \dst_addr, 8
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 9
    s8i \a_tmp3, \dst_addr, 10
    s8i \a_tmp1, \dst_addr, 11

    ee.vadds.s32.ld.incp \q_tmp1, \offsets, \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 12
    s8i \a_tmp3, \dst_addr, 13
    s8i \a_tmp0, \dst_addr, 14
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 15
    s8i \a_tmp3, \dst_addr, 16
    s8i \a_tmp1, \dst_addr, 17
    ee.movi.32.a \q_tmp2, \a_tmp0, 2
    ee.movi.32.a \q_tmp2, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 18
    s8i \a_tmp3, \dst_addr, 19
    s8i \a_tmp0, \dst_addr, 20
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 21
    s8i \a_tmp3, \dst_addr, 22
    s8i \a_tmp1, \dst_addr, 23

    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 24
    s8i \a_tmp3, \dst_addr, 25
    s8i \a_tmp0, \dst_addr, 26
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 27
    s8i \a_tmp3, \dst_addr, 28
    s8i \a_tmp1, \dst_addr, 29
    ee.movi.32.a \q_tmp1, \a_tmp0, 2
    ee.movi.32.a \q_tmp1, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 30
    s8i \a_tmp3, \dst_addr, 31
    s8i \a_tmp0, \dst_addr, 32
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 33
    s8i \a_tmp3, \dst_addr, 34
    s8i \a_tmp1, \dst_addr, 35

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 36
    s8i \a_tmp3, \dst_addr, 37
    s8i \a_tmp0, \dst_addr, 38
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 39
    s8i \a_tmp3, \dst_addr, 40
    s8i \a_tmp1, \dst_addr, 41
    ee.movi.32.a \q_tmp2, \a_tmp0, 2
    ee.movi.32.a \q_tmp2, \a_tmp1, 3
    l8ui \a_tmp2, \a_tmp0, 0
    l8ui \a_tmp3, \a_tmp0, 1
    l8ui \a_tmp0, \a_tmp0, 2
    s8i \a_tmp2, \dst_addr, 42
    s8i \a_tmp3, \dst_addr, 43
    s8i \a_tmp0, \dst_addr, 44
    l8ui \a_tmp2, \a_tmp1, 0
    l8ui \a_tmp3, \a_tmp1, 1
    l8ui \a_tmp1, \a_tmp1, 2
    s8i \a_tmp2, \dst_addr, 45
    s8i \a_tmp3, \dst_addr, 46
    s8i \a_tmp1, \dst_addr, 47
    addi \dst_addr, \dst_addr, 48
.endm

# One output pixel. a_addr is the pixel byte address. a_base is the row pointer.
# Parity is (a_addr - a_base) & 2. Y is read before a_addr is aligned to the pair.
# l8ui zero-extends into an s16 lane on a_stack (0..15: Y, 16..31: U, 32..47: V).
# YUYV memory is Y0 U Y1 V. Y is at the pixel address.
.macro resize_nn_yuyv_lane a_stack, a_addr, a_scratch, a_base, lane
    l8ui \a_scratch, \a_addr, 0
    s16i \a_scratch, \a_stack, 2 * \lane
    sub \a_scratch, \a_addr, \a_base
    extui \a_scratch, \a_scratch, 1, 1
    slli \a_scratch, \a_scratch, 1
    sub \a_addr, \a_addr, \a_scratch
    l8ui \a_scratch, \a_addr, 1
    s16i \a_scratch, \a_stack, 16 + 2 * \lane
    l8ui \a_scratch, \a_addr, 3
    s16i \a_scratch, \a_stack, 32 + 2 * \lane
.endm

# UYVY memory is U Y0 V Y1. Y is at the pixel address + 1.
.macro resize_nn_uyvy_lane a_stack, a_addr, a_scratch, a_base, lane
    l8ui \a_scratch, \a_addr, 1
    s16i \a_scratch, \a_stack, 2 * \lane
    sub \a_scratch, \a_addr, \a_base
    extui \a_scratch, \a_scratch, 1, 1
    slli \a_scratch, \a_scratch, 1
    sub \a_addr, \a_addr, \a_scratch
    l8ui \a_scratch, \a_addr, 0
    s16i \a_scratch, \a_stack, 16 + 2 * \lane
    l8ui \a_scratch, \a_addr, 2
    s16i \a_scratch, \a_stack, 32 + 2 * \lane
.endm

# 8 pixels. yuv_lane is resize_nn_yuyv_lane or resize_nn_uyvy_lane.
# offsets advances 32 bytes. q_y/q_u/q_v hold 8 zero-extended s16 values.
# a_base receives the row pointer from q_tmp0 and stays live across the lanes.
# a_stack points to a 48-byte 16-byte-aligned stack buffer.
.macro resize_nn_load_yuv_8 q_y, q_u, q_v, base_addr, offsets, a_stack, q_tmp0, q_tmp1, q_tmp2, a_tmp0, a_tmp1, a_tmp2, a_tmp3, a_scratch, a_base, yuv_lane
    ee.vldbc.32.ip \q_tmp0, \base_addr, 0
    ee.movi.32.a \q_tmp0, \a_base, 0
    ee.vld.128.ip \q_tmp1, \offsets, 16
    ee.vadds.s32.ld.incp \q_tmp2, \offsets, \q_tmp1, \q_tmp0, \q_tmp1
    ee.movi.32.a \q_tmp1, \a_tmp0, 0
    ee.movi.32.a \q_tmp1, \a_tmp1, 1
    ee.movi.32.a \q_tmp1, \a_tmp2, 2
    ee.movi.32.a \q_tmp1, \a_tmp3, 3
    \yuv_lane \a_stack, \a_tmp0, \a_scratch, \a_base, 0
    \yuv_lane \a_stack, \a_tmp1, \a_scratch, \a_base, 1
    \yuv_lane \a_stack, \a_tmp2, \a_scratch, \a_base, 2
    \yuv_lane \a_stack, \a_tmp3, \a_scratch, \a_base, 3

    ee.vadds.s32 \q_tmp2, \q_tmp0, \q_tmp2
    ee.movi.32.a \q_tmp2, \a_tmp0, 0
    ee.movi.32.a \q_tmp2, \a_tmp1, 1
    ee.movi.32.a \q_tmp2, \a_tmp2, 2
    ee.movi.32.a \q_tmp2, \a_tmp3, 3
    \yuv_lane \a_stack, \a_tmp0, \a_scratch, \a_base, 4
    \yuv_lane \a_stack, \a_tmp1, \a_scratch, \a_base, 5
    \yuv_lane \a_stack, \a_tmp2, \a_scratch, \a_base, 6
    \yuv_lane \a_stack, \a_tmp3, \a_scratch, \a_base, 7

    addi \a_scratch, \a_stack, 0
    ee.vld.128.ip \q_y, \a_scratch, 16
    ee.vld.128.ip \q_u, \a_scratch, 16
    ee.vld.128.ip \q_v, \a_scratch, 0
.endm

# 8 deltas from one U and one V. q_tmp receives dR, q_v receives dG, q_u receives dB.
# q_cof is a coefficient scratch and is dead on exit. a_shift holds 14.
# a_tmp points at the 255 half in yuv2rgb888_const_tab on exit.
.macro yuv2rgb888_delta q_u, q_v, q_tmp, q_cof, a_tmp, a_shift
    .literal .Lyuv2rgb888_const_tab_addr\@, yuv2rgb888_const_tab
    l32r \a_tmp, .Lyuv2rgb888_const_tab_addr\@
    ee.vldbc.16.ip \q_tmp, \a_tmp, 2
    ee.mov.u16.qacc \q_tmp
    ee.vldbc.16.ip \q_tmp, \a_tmp, 2
    ee.vadds.s16 \q_u, \q_u, \q_tmp
    ee.vadds.s16 \q_v, \q_v, \q_tmp
    ee.vldbc.16.ip \q_tmp, \a_tmp, 2
    ee.vmulas.s16.qacc \q_v, \q_tmp
    ee.srcmb.s16.qacc \q_tmp, \a_shift, 0

    ee.vldbc.16.ip \q_cof, \a_tmp, 2
    ee.mov.u16.qacc \q_cof
    ee.vldbc.16.ip \q_cof, \a_tmp, 2
    ee.vmulas.s16.qacc \q_u, \q_cof
    ee.vldbc.16.ip \q_cof, \a_tmp, 2
    ee.vmulas.s16.qacc \q_v, \q_cof
    ee.srcmb.s16.qacc \q_v, \a_shift, 0

    ee.vldbc.16.ip \q_cof, \a_tmp, 2
    ee.mov.u16.qacc \q_cof
    ee.vldbc.16.ip \q_cof, \a_tmp, 2
    ee.vmulas.s16.qacc \q_u, \q_cof
    ee.srcmb.s16.qacc \q_u, \a_shift, 0
.endm

# Clamp six s16 QRs to 0..255. q_zero and q_255 are free and receive the bounds.
# a_255 already points at the 255 half in yuv2rgb888_const_tab.
.macro yuv2rgb888_clamp6 q0, q1, q2, q3, q4, q5, q_zero, q_255, a_255
    ee.zero.q \q_zero
    ee.vldbc.16.ip \q_255, \a_255, 0
    ee.vmax.s16 \q0, \q0, \q_zero
    ee.vmax.s16 \q1, \q1, \q_zero
    ee.vmax.s16 \q2, \q2, \q_zero
    ee.vmax.s16 \q3, \q3, \q_zero
    ee.vmax.s16 \q4, \q4, \q_zero
    ee.vmax.s16 \q5, \q5, \q_zero
    ee.vmin.s16 \q0, \q0, \q_255
    ee.vmin.s16 \q1, \q1, \q_255
    ee.vmin.s16 \q2, \q2, \q_255
    ee.vmin.s16 \q3, \q3, \q_255
    ee.vmin.s16 \q4, \q4, \q_255
    ee.vmin.s16 \q5, \q5, \q_255
.endm

# 16 output pixels. The eight QRs are distinct.
# q_r/q_g/q_b end as planar packed R/G/B. q_y is both groups' Y and then zero.
# q_r1/q_g1/q_b1 are the high halves. q_255 is the address temp and then 255.
# Group 1 gather aliases its q_tmp1 with q_y.
# The gather borrows a_shift as the row base, then puts 14 back before each delta.
.macro resize_nn_yuv2rgb888_16 q_r, q_g, q_b, base_addr, offsets, a_stack, yuv_lane, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_load_yuv_8 \q_y, \q_b, \q_g, \base_addr, \offsets, \a_stack, \q_r, \q_r1, \q_255, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \a_shift, \yuv_lane
    movi \a_shift, 14
    yuv2rgb888_delta \q_b, \q_g, \q_r, \q_r1, \a_scratch, \a_shift
    ee.vadds.s16 \q_r, \q_y, \q_r
    ee.vadds.s16 \q_g, \q_y, \q_g
    ee.vadds.s16 \q_b, \q_y, \q_b

    resize_nn_load_yuv_8 \q_y, \q_b1, \q_g1, \base_addr, \offsets, \a_stack, \q_r1, \q_y, \q_255, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \a_shift, \yuv_lane
    movi \a_shift, 14
    yuv2rgb888_delta \q_b1, \q_g1, \q_r1, \q_255, \a_scratch, \a_shift
    ee.vadds.s16 \q_r1, \q_y, \q_r1
    ee.vadds.s16 \q_g1, \q_y, \q_g1
    ee.vadds.s16 \q_b1, \q_y, \q_b1

    yuv2rgb888_clamp6 \q_b, \q_g, \q_r, \q_r1, \q_b1, \q_g1, \q_y, \q_255, \a_scratch

    ee.vunzip.8 \q_r, \q_r1
    ee.vunzip.8 \q_g, \q_g1
    ee.vunzip.8 \q_b, \q_b1
.endm

# qd0/qd1/qd2 are planar output-order channels. The eight QRs are distinct.
# On exit only qd0/qd1/qd2 are live. BGR feeds qd2, qd1, qd0 in as R, G, B.
.macro resize_nn_yuyv2rgb888 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, resize_nn_yuyv_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_uyvy2rgb888 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, resize_nn_uyvy_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_yuyv2bgr888 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd2, \qd1, \qd0, \base_addr, \offsets, \a_stack, resize_nn_yuyv_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

.macro resize_nn_uyvy2bgr888 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuv2rgb888_16 \qd2, \qd1, \qd0, \base_addr, \offsets, \a_stack, resize_nn_uyvy_lane, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
.endm

# Quant temps are q_y, q_r1, q_255. They stay off qd0/qd1/qd2.
.macro resize_nn_yuyv2rgb888_qint8 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuyv2rgb888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_uyvy2rgb888_qint8 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_uyvy2rgb888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_yuyv2bgr888_qint8 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_yuyv2bgr888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

.macro resize_nn_uyvy2bgr888_qint8 qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_y, q_r1, q_g1, q_b1, q_255
    resize_nn_uyvy2bgr888 \qd0, \qd1, \qd2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \q_y, \q_r1, \q_g1, \q_b1, \q_255
    norm_quant_3chn_8 \qd0, \qd1, \qd2, \a_lut, \q_y, \q_r1, \q_255
.endm

# qd0/qd1, qd2/qd3, qd4/qd5 are the three channels' low/high halves.
# Planar RGB is written into qd3/qd4/qd5, then consumed as the quant indices.
# qd0..qd5, q_tmp0, q_tmp1 are eight distinct QRs.
.macro resize_nn_yuyv2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_yuyv2rgb888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_uyvy2rgb888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_uyvy2rgb888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_yuyv2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_yuyv2bgr888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

.macro resize_nn_uyvy2bgr888_qint16 qd0, qd1, qd2, qd3, qd4, qd5, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_lut, q_tmp0, q_tmp1
    resize_nn_uyvy2bgr888 \qd3, \qd4, \qd5, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_tmp0, \q_tmp1, \qd2
    norm_quant_3chn_16 \qd0, \qd1, \qd2, \qd3, \qd4, \qd5, \a_lut, \q_tmp0, \q_tmp1
.endm

# qd0/qd1/qd2 end as planar packed R/G/B. qs0 = y1, qs1 = y2, qs2 = u, qs3 = v.
# qd0 is the coefficient scratch, then y1+dR. a_shift already holds 14.
.macro yuyv2rgb888 qd0, qd1, qd2, qs0, qs1, qs2, qs3, a_tmp, a_shift, q_tmp
    yuv2rgb888_delta \qs2, \qs3, \q_tmp, \qd0, \a_tmp, \a_shift
    ee.vadds.s16 \qd0, \qs0, \q_tmp
    ee.vadds.s16 \q_tmp, \qs1, \q_tmp
    ee.vadds.s16 \qd1, \qs0, \qs3
    ee.vadds.s16 \qs3, \qs1, \qs3
    ee.vadds.s16 \qd2, \qs0, \qs2
    ee.vadds.s16 \qs2, \qs1, \qs2

    yuv2rgb888_clamp6 \qd0, \q_tmp, \qd1, \qs3, \qd2, \qs2, \qs0, \qs1, \a_tmp

    ee.vzip.16 \qd0, \q_tmp
    ee.vunzip.8 \qd0, \q_tmp
    ee.vzip.16 \qd1, \qs3
    ee.vunzip.8 \qd1, \qs3
    ee.vzip.16 \qd2, \qs2
    ee.vunzip.8 \qd2, \qs2
.endm

# 16 contiguous pixels (unaligned-safe). q_y1/q_y2/q_u/q_v are YUYV byte positions and are distinct.
.macro cvt_load_yuyv16 q_y1, q_y2, q_u, q_v, a_src
    vld_2chn_unaligned \q_y1, \q_y2, \a_src, \q_u
    ee.vunzip.8 \q_y1, \q_y2
    ee.vunzip.8 \q_y1, \q_y2
    ee.zero.q \q_u
    ee.vzip.8 \q_y1, \q_u
    ee.zero.q \q_v
    ee.vzip.8 \q_y2, \q_v
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

# qd0/qd1 are the 565 halves. qs0/qs1/qs2 stay planar RGB.
# The bgr565 packer swaps R and B. q_t0..q_t2 complete the gather temps.
# a_shift and a_g0 are the packer temps.
.macro resize_nn_yuyv2rgb565le qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2rgb565be qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2bgr565le qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_yuyv2bgr565be qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2rgb565le qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2rgb565be qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    rgb8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2bgr565le qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565le \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

.macro resize_nn_uyvy2bgr565be qd0, qd1, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, qs0, qs1, qs2, q_t0, q_t1, q_t2
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \q_t2
    bgr8882rgb565be \qd0, \qd1, \qs0, \qs1, \qs2, \a_shift, \a_g0
.endm

# qd0 is the yuv scratch and the low 565 half. qd1 is q_v, then the high 565 half.
.macro cvt_yuyv2rgb565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_yuyv2rgb565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_yuyv2bgr565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_yuyv2bgr565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_uyvy2rgb565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_uyvy2rgb565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    rgb8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_uyvy2bgr565le qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565le \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

.macro cvt_uyvy2bgr565be qd0, qd1, a_src, q_y1, q_y2, q_u, qr, qg, qb, a_const, a_shift
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \q_y1, \q_y2, \q_u, \qd1, \a_const, \a_shift, \qd0
    bgr8882rgb565be \qd0, \qd1, \qr, \qg, \qb, \a_const, \a_shift
.endm

# qd0/qd1/qd2 are H/S/V. qs0/qs1/qs2 are planar RGB.
.macro resize_nn_yuyv2hsv qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \qd2
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1
.endm

.macro resize_nn_uyvy2hsv qd0, qd1, qd2, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd0, \qd1, \q_t0, \q_t1, \qd2
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1
.endm

# qd is the mask byte. qs0/qs1/qs2 are planar RGB. q_t0..q_t2 are H/S/V and q_t3 is scratch.
# Loads hsv_min / hsv_max from [a_stack + 192] / [a_stack + 196] after the YUV->RGB888 gather.
.macro resize_nn_yuyv2hsv_mask0 qd, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    l32i \a_shift, \a_stack, 192
    l32i \a_scratch, \a_stack, 196
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_shift, \a_scratch, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_uyvy2hsv_mask0 qd, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    l32i \a_shift, \a_stack, 192
    l32i \a_scratch, \a_stack, 196
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_shift, \a_scratch, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_yuyv2hsv_mask1 qd, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_yuyv2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    l32i \a_shift, \a_stack, 192
    l32i \a_scratch, \a_stack, 196
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_shift, \a_scratch, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

.macro resize_nn_uyvy2hsv_mask1 qd, base_addr, offsets, a_stack, a_shift, a_g0, a_g1, a_g2, a_g3, a_scratch, a_sdiv, a_hdiv, qs0, qs1, qs2, q_t0, q_t1, q_t2, q_t3
    resize_nn_uyvy2rgb888 \qs0, \qs1, \qs2, \base_addr, \offsets, \a_stack, \a_shift, \a_g0, \a_g1, \a_g2, \a_g3, \a_scratch, \qd, \q_t0, \q_t1, \q_t2, \q_t3
    l32i \a_shift, \a_stack, 192
    l32i \a_scratch, \a_stack, 196
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_shift, \a_scratch, \a_stack, \qs0, \qs1, \qs2, \a_g0, \a_g1, \a_g2, \a_g3, \q_t0, \q_t1, \q_t2, \q_t3
.endm

# qd0/qd1/qd2 are H/S/V and also q_y1/q_y2/q_u. q_tmp0 is q_v.
.macro cvt_yuyv2hsv qd0, qd1, qd2, a_src, a_sdiv, a_hdiv, a_stack, a_const, a_shift, a_t0, a_t1, qr, qg, qb, q_tmp0, q_tmp1
    cvt_yuyv2rgb888 \qr, \qg, \qb, \a_src, \qd0, \qd1, \qd2, \q_tmp0, \a_const, \a_shift, \q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \a_stack, \qr, \qg, \qb, \a_const, \a_shift, \a_t0, \a_t1, \q_tmp0, \q_tmp1
.endm

.macro cvt_uyvy2hsv qd0, qd1, qd2, a_src, a_sdiv, a_hdiv, a_stack, a_const, a_shift, a_t0, a_t1, qr, qg, qb, q_tmp0, q_tmp1
    cvt_uyvy2rgb888 \qr, \qg, \qb, \a_src, \qd0, \qd1, \qd2, \q_tmp0, \a_const, \a_shift, \q_tmp1
    rgb8882hsv \qd0, \qd1, \qd2, \a_sdiv, \a_hdiv, \a_stack, \qr, \qg, \qb, \a_const, \a_shift, \a_t0, \a_t1, \q_tmp0, \q_tmp1
.endm

# qd is the mask byte and the yuv scratch. q_h/q_s/q_v/q_scratch are the YUV lanes, then H/S/V/scratch.
.macro cvt_yuyv2hsv_mask0 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_stack, a_const, a_shift, a_t0, a_t1, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_yuyv2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \a_stack, \qs0, \qs1, \qs2, \a_const, \a_shift, \a_t0, \a_t1, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_uyvy2hsv_mask0 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_stack, a_const, a_shift, a_t0, a_t1, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_uyvy2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask0 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \a_stack, \qs0, \qs1, \qs2, \a_const, \a_shift, \a_t0, \a_t1, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_yuyv2hsv_mask1 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_stack, a_const, a_shift, a_t0, a_t1, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_yuyv2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \a_stack, \qs0, \qs1, \qs2, \a_const, \a_shift, \a_t0, \a_t1, \q_h, \q_s, \q_v, \q_scratch
.endm

.macro cvt_uyvy2hsv_mask1 qd, a_src, a_sdiv, a_hdiv, a_min, a_max, a_stack, a_const, a_shift, a_t0, a_t1, qs0, qs1, qs2, q_h, q_s, q_v, q_scratch
    cvt_uyvy2rgb888 \qs0, \qs1, \qs2, \a_src, \q_h, \q_s, \q_v, \q_scratch, \a_const, \a_shift, \qd
    rgb8882hsv_mask1 \qd, \a_sdiv, \a_hdiv, \a_min, \a_max, \a_stack, \qs0, \qs1, \qs2, \a_const, \a_shift, \a_t0, \a_t1, \q_h, \q_s, \q_v, \q_scratch
.endm
