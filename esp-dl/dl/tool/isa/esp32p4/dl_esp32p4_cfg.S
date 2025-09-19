    .text
    .global dl_esp32p4_cfg_round
    .type   dl_esp32p4_cfg_round, @function
    .balign 4
    .option norvc
dl_esp32p4_cfg_round:

    # a0: int8_t round_mode
    # a1:
    # a2:

    # a3:
    # a4:
    # a5:
    # t3:
    # t4:
    # t5:
    # t6:

    # s0:
    # s1:
    # s8:
    # s9:
    # s10:
    # s11:

    esp.movx.r.cfg a1
    li a2, 0xffffff0f
    and a1, a1, a2
    slli a0, a0, 4
    or a1, a1, a0
    esp.movx.w.cfg a1
    ret


    .align 2
    .text
    .global dl_esp32p4_cfg_misalign
    .type   dl_esp32p4_cfg_misalign, @function
dl_esp32p4_cfg_misalign:

    # a0: bool store
    # a1: bool load

    slli a1, a1, 1
    add a0, a0, a1
    esp.movx.r.cfg a1
    li a2, 0xfffffffc
    and a1, a1, a2
    or a1, a1, a0
    esp.movx.w.cfg a1
    ret

    .align 2
    .text
    .global dl_esp32p4_get_cfg
    .type   dl_esp32p4_get_cfg, @function
dl_esp32p4_get_cfg:

    # a0: uint32_t

    esp.movx.r.cfg a0
    ret
