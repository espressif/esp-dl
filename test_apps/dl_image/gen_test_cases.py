from itertools import permutations, product

pix_types = [
    "rgb888",
    "rgb888_qint8",
    "rgb888_qint16",
    "bgr888",
    "bgr888_qint8",
    "bgr888_qint16",
    "gray",
    "gray_qint8",
    "gray_qint16",
    "rgb565le",
    "rgb565be",
    "bgr565le",
    "bgr565be",
    "hsv",
    "hsv_mask",
]

pix_type_pairs = [
    ("rgb888", "rgb888"),
    ("rgb888", "rgb888_qint8"),
    ("rgb888", "rgb888_qint16"),
    ("rgb888", "bgr888"),
    ("rgb888", "bgr888_qint8"),
    ("rgb888", "bgr888_qint16"),
    ("rgb888", "gray"),
    ("rgb888", "gray_qint8"),
    ("rgb888", "gray_qint16"),
    ("rgb888", "rgb565le"),
    ("rgb888", "rgb565be"),
    ("rgb888", "bgr565le"),
    ("rgb888", "bgr565be"),
    ("rgb888", "hsv"),
    ("rgb888", "hsv_mask"),
    ("bgr888", "rgb888"),
    ("bgr888", "rgb888_qint8"),
    ("bgr888", "rgb888_qint16"),
    ("bgr888", "bgr888"),
    ("bgr888", "bgr888_qint8"),
    ("bgr888", "bgr888_qint16"),
    ("bgr888", "gray"),
    ("bgr888", "gray_qint8"),
    ("bgr888", "gray_qint16"),
    ("bgr888", "rgb565le"),
    ("bgr888", "rgb565be"),
    ("bgr888", "bgr565le"),
    ("bgr888", "bgr565be"),
    ("bgr888", "hsv"),
    ("bgr888", "hsv_mask"),
    ("gray", "gray"),
    ("gray", "gray_qint8"),
    ("gray", "gray_qint16"),
    ("rgb565le", "rgb888"),
    ("rgb565le", "rgb888_qint8"),
    ("rgb565le", "rgb888_qint16"),
    ("rgb565le", "bgr888"),
    ("rgb565le", "bgr888_qint8"),
    ("rgb565le", "bgr888_qint16"),
    ("rgb565le", "gray"),
    ("rgb565le", "gray_qint8"),
    ("rgb565le", "gray_qint16"),
    ("rgb565le", "rgb565le"),
    ("rgb565le", "rgb565be"),
    ("rgb565le", "bgr565le"),
    ("rgb565le", "bgr565be"),
    ("rgb565le", "hsv"),
    ("rgb565le", "hsv_mask"),
    ("rgb565be", "rgb888"),
    ("rgb565be", "rgb888_qint8"),
    ("rgb565be", "rgb888_qint16"),
    ("rgb565be", "bgr888"),
    ("rgb565be", "bgr888_qint8"),
    ("rgb565be", "bgr888_qint16"),
    ("rgb565be", "gray"),
    ("rgb565be", "gray_qint8"),
    ("rgb565be", "gray_qint16"),
    ("rgb565be", "rgb565le"),
    ("rgb565be", "rgb565be"),
    ("rgb565be", "bgr565le"),
    ("rgb565be", "bgr565be"),
    ("rgb565be", "hsv"),
    ("rgb565be", "hsv_mask"),
    ("bgr565le", "rgb888"),
    ("bgr565le", "rgb888_qint8"),
    ("bgr565le", "rgb888_qint16"),
    ("bgr565le", "bgr888"),
    ("bgr565le", "bgr888_qint8"),
    ("bgr565le", "bgr888_qint16"),
    ("bgr565le", "gray"),
    ("bgr565le", "gray_qint8"),
    ("bgr565le", "gray_qint16"),
    ("bgr565le", "rgb565le"),
    ("bgr565le", "rgb565be"),
    ("bgr565le", "bgr565le"),
    ("bgr565le", "bgr565be"),
    ("bgr565le", "hsv"),
    ("bgr565le", "hsv_mask"),
    ("bgr565be", "rgb888"),
    ("bgr565be", "rgb888_qint8"),
    ("bgr565be", "rgb888_qint16"),
    ("bgr565be", "bgr888"),
    ("bgr565be", "bgr888_qint8"),
    ("bgr565be", "bgr888_qint16"),
    ("bgr565be", "gray"),
    ("bgr565be", "gray_qint8"),
    ("bgr565be", "gray_qint16"),
    ("bgr565be", "rgb565le"),
    ("bgr565be", "rgb565be"),
    ("bgr565be", "bgr565le"),
    ("bgr565be", "bgr565be"),
    ("bgr565be", "hsv"),
    ("bgr565be", "hsv_mask"),
    ("hsv", "hsv_mask"),
]

if __name__ == "__main__":
    # print(len(list(product(pix_types, pix_types))))
    # print(len(list(permutations(pix_types, 2))))
    # print(len(pix_types))
    # for pair in list(product(pix_types, pix_types)):
    #     print(pair)
    flags = [
        "0b0000",
        "0b0001",
        "0b0010",
        "0b0011",
        "0b0100",
        "0b0101",
        "0b0110",
        "0b0111",
        "0b1000",
        "0b1001",
        "0b1010",
        "0b1011",
        "0b1100",
        "0b1101",
        "0b1110",
        "0b1111",
    ]
    with open("cvt_color_test_cases.txt", "w") as f:
        for (src_pix_type, dst_pix_type), flag in product(pix_type_pairs, flags):
            if "gray" in dst_pix_type and flag.endswith("10"):
                continue
            print(
                f'TEST_CVT_COLOR("{src_pix_type}2{dst_pix_type}", dl::image::DL_IMAGE_PIX_CVT_{src_pix_type.upper()}2{dst_pix_type.upper()}, {flag})',
                file=f,
            )
    flags = [
        "0b000",
        "0b001",
        "0b010",
        "0b011",
        "0b100",
        "0b101",
        "0b110",
        "0b111",
    ]
    # print(len(pix_type_pairs))
    for src_pix_type, dst_pix_type in pix_type_pairs:
        # print(f"DL_IMAGE_PIX_CVT_{src_pix_type.upper()}2{dst_pix_type.upper()}")
        # print(f"DL_IMAGE_PIX_CVT_{src_pix_type.upper()}2{dst_pix_type.upper()} = pix_cvt_id(DL_IMAGE_PIX_TYPE_{src_pix_type.upper()}, DL_IMAGE_PIX_TYPE_{dst_pix_type.upper()}),")
        with open("resize_nn_test_cases.txt", "w") as f:
            for (src_pix_type, dst_pix_type), flag in product(pix_type_pairs, flags):
                if "gray" in dst_pix_type and flag.endswith("10"):
                    continue
                print(
                    f'TEST_RESIZE_NN("{src_pix_type}2{dst_pix_type}", dl::image::DL_IMAGE_PIX_CVT_{src_pix_type.upper()}2{dst_pix_type.upper()}, {flag})',
                    file=f,
                )
