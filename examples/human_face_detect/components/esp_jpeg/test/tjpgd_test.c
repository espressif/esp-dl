/*
 * SPDX-FileCopyrightText: 2021-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "sdkconfig.h"
#include "unity.h"


#include "../include/jpeg_decoder.h"
#include "test_logo_jpg.h"
#include "test_logo_rgb888.h"

#define TESTW 46
#define TESTH 46

TEST_CASE("Test JPEG decompression library", "[esp_jpeg]")
{
    char aapix[] = " .:;+=xX$$";
    unsigned char *decoded, *p, *o;
    int x, y, v;
    int decoded_outsize = TESTW * TESTH * 3;

    decoded = malloc(decoded_outsize);
    for (x = 0; x < decoded_outsize; x += 2) {
        decoded[x] = 0;
        decoded[x + 1] = 0xff;
    }

    /* JPEG decode */
    esp_jpeg_image_cfg_t jpeg_cfg = {
        .indata = (uint8_t *)logo_jpg,
        .indata_size = sizeof(logo_jpg),
        .outbuf = decoded,
        .outbuf_size = decoded_outsize,
        .out_format = JPEG_IMAGE_FORMAT_RGB888,
        .out_scale = JPEG_IMAGE_SCALE_0,
        .flags = {
            .swap_color_bytes = 0,
        }
    };
    esp_jpeg_image_output_t outimg;
    esp_err_t err = esp_jpeg_decode(&jpeg_cfg, &outimg);
    TEST_ASSERT_EQUAL(err, ESP_OK);

    /* Decoded image size */
    TEST_ASSERT_EQUAL(outimg.width, TESTW);
    TEST_ASSERT_EQUAL(outimg.height, TESTH);

    p = decoded;
    o = logo_rgb888;
    for (x = 0; x < outimg.width * outimg.height; x++) {
        /* The color can be +- 2 */
        TEST_ASSERT(p[0] >= (o[0] - 2) && p[0] <= (o[0] + 2));
        TEST_ASSERT(p[1] >= (o[1] - 2) && p[1] <= (o[1] + 2));
        TEST_ASSERT(p[2] >= (o[2] - 2) && p[2] <= (o[2] + 2));

        p += 3;
        o += 3;
    }

    p = decoded + 2;
    for (y = 0; y < outimg.width; y++) {
        for (x = 0; x < outimg.height; x++) {
            v = ((*p) * (sizeof(aapix) - 2) * 2) / 256;
            printf("%c%c", aapix[v / 2], aapix[(v + 1) / 2]);
            p += 3;
        }
        printf("%c%c", ' ', '\n');
    }


    free(decoded);
}
