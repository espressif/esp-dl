#include "dl_image.hpp"
#include "esp_log.h"
#include <assert.h>
#include <stdio.h>

namespace dl {
namespace image {
template <typename T>
void crop_and_resize(T *dst_image,
                     int dst_width,
                     int dst_channel,
                     int dst_y_start,
                     int dst_y_end,
                     int dst_x_start,
                     int dst_x_end,
                     uint16_t *src_image,
                     int src_height,
                     int src_width,
                     int src_channel,
                     int src_y_start,
                     int src_y_end,
                     int src_x_start,
                     int src_x_end,
                     resize_type_t resize_type,
                     bool byte_swap,
                     bool rgb_swap)
{
    assert(src_channel == 3);
    assert(dst_y_start >= 0);
    assert(dst_x_start >= 0);

    float scale_y = (float)(src_y_end - src_y_start) / (dst_y_end - dst_y_start);
    float scale_x = (float)(src_x_end - src_x_start) / (dst_x_end - dst_x_start);
    int temp[13];

    switch (resize_type) {
    case IMAGE_RESIZE_BILINEAR:
        for (size_t y = dst_y_start; y < dst_y_end; y++) {
            float ratio_y[2];
            ratio_y[0] = (float)(((y - dst_y_start) + 0.5) * scale_y - 0.5); // y
            int src_y = (int)ratio_y[0];                                     // y1
            ratio_y[0] -= src_y;                                             // y - y1
            src_y += src_y_start;

            if (src_y < 0) {
                ratio_y[0] = 0;
                src_y = 0;
            } else if (src_y > src_height - 2) {
                ratio_y[0] = 0;
                src_y = src_height - 2;
            }
            ratio_y[1] = 1 - ratio_y[0]; // y2 - y

            int _dst_i = y * dst_width;

            int _src_row_0 = src_y * src_width;
            int _src_row_1 = _src_row_0 + src_width;

            for (size_t x = dst_x_start; x < dst_x_end; x++) {
                float ratio_x[2];
                ratio_x[0] = (float)(((x - dst_x_start) + 0.5) * scale_x - 0.5); // x
                int src_x = (int)ratio_x[0];                                     // x1
                ratio_x[0] -= src_x;                                             // x - x1
                src_x += src_x_start;

                if (src_x < 0) {
                    ratio_x[0] = 0;
                    src_x = 0;
                } else if (src_x > src_width - 2) {
                    ratio_x[0] = 0;
                    src_x = src_width - 2;
                }
                ratio_x[1] = 1 - ratio_x[0]; // x2 - x

                int dst_i = (_dst_i + x) * dst_channel;

                int src_row_0 = _src_row_0 + src_x;
                int src_row_1 = _src_row_1 + src_x;

                convert_pixel_rgb565_to_rgb888(src_image[src_row_0], temp, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_0 + 1], temp + 3, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_1], temp + 6, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_1 + 1], temp + 9, byte_swap);

                if (dst_channel == 3) {
                    if (rgb_swap) {
                        dst_image[dst_i] = (int)(temp[2] * ratio_x[1] * ratio_y[1]    //
                                                 + temp[5] * ratio_x[0] * ratio_y[1]  //
                                                 + temp[8] * ratio_x[1] * ratio_y[0]  //
                                                 + temp[11] * ratio_x[0] * ratio_y[0] //
                                                 + 0.5);                              //

                        dst_image[dst_i + 1] = (int)(temp[1] * ratio_x[1] * ratio_y[1]    //
                                                     + temp[4] * ratio_x[0] * ratio_y[1]  //
                                                     + temp[7] * ratio_x[1] * ratio_y[0]  //
                                                     + temp[10] * ratio_x[0] * ratio_y[0] //
                                                     + 0.5);                              //

                        dst_image[dst_i + 2] = (int)(temp[0] * ratio_x[1] * ratio_y[1]   //
                                                     + temp[3] * ratio_x[0] * ratio_y[1] //
                                                     + temp[6] * ratio_x[1] * ratio_y[0] //
                                                     + temp[9] * ratio_x[0] * ratio_y[0] //
                                                     + 0.5);                             //
                    } else {
                        dst_image[dst_i] = (int)(temp[0] * ratio_x[1] * ratio_y[1]   //
                                                 + temp[3] * ratio_x[0] * ratio_y[1] //
                                                 + temp[6] * ratio_x[1] * ratio_y[0] //
                                                 + temp[9] * ratio_x[0] * ratio_y[0] //
                                                 + 0.5);                             //

                        dst_image[dst_i + 1] = (int)(temp[1] * ratio_x[1] * ratio_y[1]    //
                                                     + temp[4] * ratio_x[0] * ratio_y[1]  //
                                                     + temp[7] * ratio_x[1] * ratio_y[0]  //
                                                     + temp[10] * ratio_x[0] * ratio_y[0] //
                                                     + 0.5);                              //

                        dst_image[dst_i + 2] = (int)(temp[2] * ratio_x[1] * ratio_y[1]    //
                                                     + temp[5] * ratio_x[0] * ratio_y[1]  //
                                                     + temp[8] * ratio_x[1] * ratio_y[0]  //
                                                     + temp[11] * ratio_x[0] * ratio_y[0] //
                                                     + 0.5);                              //
                    }
                } else if (dst_channel == 1) // RGB -> Gray
                {
                    int blue, green, red;
                    if (rgb_swap) {
                        blue = (int)(temp[2] * ratio_x[1] * ratio_y[1]     //
                                     + temp[5] * ratio_x[0] * ratio_y[1]   //
                                     + temp[8] * ratio_x[1] * ratio_y[0]   //
                                     + temp[11] * ratio_x[0] * ratio_y[0]  //
                                     + 0.5);                               //
                        green = (int)(temp[1] * ratio_x[1] * ratio_y[1]    //
                                      + temp[4] * ratio_x[0] * ratio_y[1]  //
                                      + temp[7] * ratio_x[1] * ratio_y[0]  //
                                      + temp[10] * ratio_x[0] * ratio_y[0] //
                                      + 0.5);                              //
                        red = (int)(temp[0] * ratio_x[1] * ratio_y[1]      //
                                    + temp[3] * ratio_x[0] * ratio_y[1]    //
                                    + temp[6] * ratio_x[1] * ratio_y[0]    //
                                    + temp[9] * ratio_x[0] * ratio_y[0]    //
                                    + 0.5);                                //
                    } else {
                        blue = (int)(temp[0] * ratio_x[1] * ratio_y[1]     //
                                     + temp[3] * ratio_x[0] * ratio_y[1]   //
                                     + temp[6] * ratio_x[1] * ratio_y[0]   //
                                     + temp[9] * ratio_x[0] * ratio_y[0]   //
                                     + 0.5);                               //
                        green = (int)(temp[1] * ratio_x[1] * ratio_y[1]    //
                                      + temp[4] * ratio_x[0] * ratio_y[1]  //
                                      + temp[7] * ratio_x[1] * ratio_y[0]  //
                                      + temp[10] * ratio_x[0] * ratio_y[0] //
                                      + 0.5);                              //
                        red = (int)(temp[2] * ratio_x[1] * ratio_y[1]      //
                                    + temp[5] * ratio_x[0] * ratio_y[1]    //
                                    + temp[8] * ratio_x[1] * ratio_y[0]    //
                                    + temp[11] * ratio_x[0] * ratio_y[0]   //
                                    + 0.5);                                //
                    }
                    dst_image[dst_i] = convert_pixel_rgb888_to_gray(red, green, blue);
                } else {
                    printf("Not implement dst_channel = %d\n", dst_channel);
                }
            }
        }
        break;

    case IMAGE_RESIZE_MEAN:
        for (int y = dst_y_start; y < dst_y_end; y++) {
            int _dst_i = y * dst_width;

            float src_y = rintf((y - dst_y_start) * scale_y + src_y_start);
            src_y = DL_CLIP(src_y, 0, src_height - 2);
            float _src_row_0 = src_y * src_width;
            float _src_row_1 = _src_row_0 + src_width;

            for (int x = dst_x_start; x < dst_x_end; x++) {
                int dst_i = (_dst_i + x) * dst_channel;

                float src_x = rintf((x - dst_x_start) * scale_x) + src_x_start;
                src_x = DL_CLIP(src_x, 0, src_width - 2);
                int src_row_0 = _src_row_0 + src_x;
                int src_row_1 = _src_row_1 + src_x;

                convert_pixel_rgb565_to_rgb888(src_image[src_row_0], temp, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_0 + 1], temp + 3, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_1], temp + 6, byte_swap);
                convert_pixel_rgb565_to_rgb888(src_image[src_row_1 + 1], temp + 9, byte_swap);

                if (dst_channel == 3) {
                    if (rgb_swap) {
                        dst_image[dst_i] = (temp[2] + temp[5] + temp[8] + temp[11]) >> 2;     // blue
                        dst_image[dst_i + 1] = (temp[1] + temp[4] + temp[7] + temp[10]) >> 2; // green
                        dst_image[dst_i + 2] = (temp[0] + temp[3] + temp[6] + temp[9]) >> 2;  // red
                    } else {
                        dst_image[dst_i] = (temp[0] + temp[3] + temp[6] + temp[9]) >> 2;      // blue
                        dst_image[dst_i + 1] = (temp[1] + temp[4] + temp[7] + temp[10]) >> 2; // green
                        dst_image[dst_i + 2] = (temp[2] + temp[5] + temp[8] + temp[11]) >> 2; // red
                    }
                } else if (dst_channel == 1) {
                    int blue, green, red;
                    if (rgb_swap) {
                        blue = (temp[2] + temp[5] + temp[8] + temp[11]) >> 2;
                        green = (temp[1] + temp[4] + temp[7] + temp[10]) >> 2;
                        red = (temp[0] + temp[3] + temp[6] + temp[9]) >> 2;
                    } else {
                        blue = (temp[0] + temp[3] + temp[6] + temp[9]) >> 2;
                        green = (temp[1] + temp[4] + temp[7] + temp[10]) >> 2;
                        red = (temp[2] + temp[5] + temp[8] + temp[11]) >> 2;
                    }
                    dst_image[dst_i] = convert_pixel_rgb888_to_gray(red, green, blue);
                } else {
                    printf("Not implement dst_channel = %d\n", dst_channel);
                }
            }
        }
        break;

    case IMAGE_RESIZE_NEAREST:
        for (size_t y = dst_y_start; y < dst_y_end; y++) {
            int _dst_i = y * dst_width;

            float src_y = rintf((y - dst_y_start) * scale_y + src_y_start);
            src_y = DL_CLIP(src_y, 0, src_height - 1);
            float _src_i = src_y * src_width;

            for (size_t x = dst_x_start; x < dst_x_end; x++) {
                int dst_i = (_dst_i + x) * dst_channel;

                float src_x = rintf((x - dst_x_start) * scale_x) + src_x_start;
                src_x = DL_CLIP(src_x, 0, src_width - 1);
                int src_i = _src_i + src_x;

                convert_pixel_rgb565_to_rgb888(src_image[src_i], temp, byte_swap);

                if (dst_channel == 3) {
                    if (rgb_swap) {
                        dst_image[dst_i] = temp[2];
                        dst_image[dst_i + 1] = temp[1];
                        dst_image[dst_i + 2] = temp[0];
                    } else {
                        dst_image[dst_i] = temp[0];
                        dst_image[dst_i + 1] = temp[1];
                        dst_image[dst_i + 2] = temp[2];
                    }
                } else if (dst_channel == 1) // RGB -> Gray
                {
                    int blue, green, red;
                    if (rgb_swap) {
                        blue = temp[2];
                        green = temp[1];
                        red = temp[0];
                    } else {
                        blue = temp[0];
                        green = temp[1];
                        red = temp[2];
                    }
                    dst_image[dst_i] = convert_pixel_rgb888_to_gray(red, green, blue);
                } else {
                    printf("Not implement dst_channel = %d\n", dst_channel);
                }
            }
        }
        break;

    default:
        printf("Not implement image_resize_type = %d\n", resize_type);
        break;
    }
}
template void crop_and_resize(uint8_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint16_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap,
                              bool byte_swap);
template void crop_and_resize(int16_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint16_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap,
                              bool byte_swap);
template void crop_and_resize(int8_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint16_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap,
                              bool byte_swap);

template <typename T>
void crop_and_resize(T *dst_image,
                     int dst_width,
                     int dst_channel,
                     int dst_y_start,
                     int dst_y_end,
                     int dst_x_start,
                     int dst_x_end,
                     uint8_t *src_image,
                     int src_height,
                     int src_width,
                     int src_channel,
                     int src_y_start,
                     int src_y_end,
                     int src_x_start,
                     int src_x_end,
                     resize_type_t resize_type,
                     bool rgb_swap)
{
    assert(dst_y_start >= 0);
    assert(dst_x_start >= 0);

    float scale_y = (float)(src_y_end - src_y_start) / (dst_y_end - dst_y_start);
    float scale_x = (float)(src_x_end - src_x_start) / (dst_x_end - dst_x_start);
    int temp;

    switch (resize_type) {
    case IMAGE_RESIZE_BILINEAR:
        for (size_t y = dst_y_start; y < dst_y_end; y++) {
            float ratio_y[2];
            ratio_y[0] = (float)(((y - dst_y_start) + 0.5) * scale_y - 0.5); // y
            int src_y = (int)ratio_y[0];                                     // y1
            ratio_y[0] -= src_y;                                             // y - y1
            src_y += src_y_start;

            if (src_y < 0) {
                ratio_y[0] = 0;
                src_y = 0;
            }
            if (src_y > src_height - 2) {
                ratio_y[0] = 0;
                src_y = src_height - 2;
            }
            ratio_y[1] = 1 - ratio_y[0]; // y2 - y

            int _dst_i = y * dst_width;

            int _src_row_0 = src_y * src_width;
            int _src_row_1 = _src_row_0 + src_width;

            for (size_t x = dst_x_start; x < dst_x_end; x++) {
                float ratio_x[2];
                ratio_x[0] = (float)(((x - dst_x_start) + 0.5) * scale_x - 0.5); // x
                int src_x = (int)ratio_x[0];                                     // x1
                ratio_x[0] -= src_x;                                             // x - x1
                src_x += src_x_start;

                if (src_x < 0) {
                    ratio_x[0] = 0;
                    src_x = 0;
                }
                if (src_x > src_width - 2) {
                    ratio_x[0] = 0;
                    src_x = src_width - 2;
                }
                ratio_x[1] = 1 - ratio_x[0]; // x2 - x

                int dst_i = (_dst_i + x) * dst_channel;

                int src_row_0 = (_src_row_0 + src_x) * src_channel;
                int src_row_1 = (_src_row_1 + src_x) * src_channel;

                if (src_channel == dst_channel) {
                    for (int c = 0; c < dst_channel; c++) {
                        temp = round(src_image[src_row_0 + c] * ratio_x[1] * ratio_y[1]                   //
                                     + src_image[src_row_0 + src_channel + c] * ratio_x[0] * ratio_y[1]   //
                                     + src_image[src_row_1 + c] * ratio_x[1] * ratio_y[0]                 //
                                     + src_image[src_row_1 + src_channel + c] * ratio_x[0] * ratio_y[0]); //
                        if (rgb_swap)
                            dst_image[dst_i + dst_channel - 1 - c] = temp;
                        else
                            dst_image[dst_i + c] = temp;
                    }
                } else if (src_channel == 3 && dst_channel == 1) // RGB -> Gray
                {
                    int blue = round(src_image[src_row_0] * ratio_x[1] * ratio_y[1]                   //
                                     + src_image[src_row_0 + src_channel] * ratio_x[0] * ratio_y[1]   //
                                     + src_image[src_row_1] * ratio_x[1] * ratio_y[0]                 //
                                     + src_image[src_row_1 + src_channel] * ratio_x[0] * ratio_y[0]); //

                    int green = round(src_image[src_row_0 + 1] * ratio_x[1] * ratio_y[1]                   //
                                      + src_image[src_row_0 + src_channel + 1] * ratio_x[0] * ratio_y[1]   //
                                      + src_image[src_row_1 + 1] * ratio_x[1] * ratio_y[0]                 //
                                      + src_image[src_row_1 + src_channel + 1] * ratio_x[0] * ratio_y[0]); //

                    int red = round(src_image[src_row_0 + 2] * ratio_x[1] * ratio_y[1]                   //
                                    + src_image[src_row_0 + src_channel + 2] * ratio_x[0] * ratio_y[1]   //
                                    + src_image[src_row_1 + 2] * ratio_x[1] * ratio_y[0]                 //
                                    + src_image[src_row_1 + src_channel + 2] * ratio_x[0] * ratio_y[0]); //
                    if (rgb_swap)
                        dst_image[dst_i] = convert_pixel_rgb888_to_gray(blue, green, red);
                    else
                        dst_image[dst_i] = convert_pixel_rgb888_to_gray(red, green, blue);
                } else {
                    printf("Not implement src_channel = %d and dst_channel = %d\n", src_channel, dst_channel);
                }
            }
        }
        break;

    case IMAGE_RESIZE_MEAN:

        for (size_t y = dst_y_start; y < dst_y_end; y++) {
            int _dst_i = y * dst_width;

            float src_y = rintf((y - dst_y_start) * scale_y + src_y_start);
            src_y = DL_CLIP(src_y, 0, src_height - 2);
            float _src_row_0 = src_y * src_width;
            float _src_row_1 = _src_row_0 + src_width;

            for (size_t x = dst_x_start; x < dst_x_end; x++) {
                int dst_i = (_dst_i + x) * dst_channel;

                float src_x = rintf((x - dst_x_start) * scale_x) + src_x_start;
                src_x = DL_CLIP(src_x, 0, src_width - 2);
                int src_row_0 = (_src_row_0 + src_x) * src_channel;
                int src_row_1 = (_src_row_1 + src_x) * src_channel;

                if (src_channel == dst_channel) {
                    for (size_t c = 0; c < dst_channel; c++) {
                        temp = (int)src_image[src_row_0 + c] + (int)src_image[src_row_0 + dst_channel + c] +
                            (int)src_image[src_row_1 + c] + (int)src_image[src_row_1 + dst_channel + c];
                        if (rgb_swap)
                            dst_image[dst_i + dst_channel - 1 - c] = temp >> 2;
                        else
                            dst_image[dst_i + c] = temp >> 2;
                    }
                } else if (src_channel == 3 && dst_channel == 1) // RGB -> Gray
                {
                    int blue = (int)src_image[src_row_0]               //
                        + (int)src_image[src_row_0 + dst_channel]      //
                        + (int)src_image[src_row_1]                    //
                        + (int)src_image[src_row_1 + dst_channel];     //
                    int green = (int)src_image[src_row_0 + 1]          //
                        + (int)src_image[src_row_0 + dst_channel + 1]  //
                        + (int)src_image[src_row_1 + 1]                //
                        + (int)src_image[src_row_1 + dst_channel + 1]; //
                    int red = (int)src_image[src_row_0 + 2]            //
                        + (int)src_image[src_row_0 + dst_channel + 2]  //
                        + (int)src_image[src_row_1 + 2]                //
                        + (int)src_image[src_row_1 + dst_channel + 2]; //
                    if (rgb_swap)
                        dst_image[dst_i] = convert_pixel_rgb888_to_gray(blue, green, red) >> 2;
                    else
                        dst_image[dst_i] = convert_pixel_rgb888_to_gray(red, green, blue) >> 2;
                } else {
                    printf("Not implement src_channel = %d and dst_channel = %d\n", src_channel, dst_channel);
                }
            }
        }
        break;

    case IMAGE_RESIZE_NEAREST:
        for (size_t y = dst_y_start; y < dst_y_end; y++) {
            int _dst_i = y * dst_width;

            float src_y = rintf((y - dst_y_start) * scale_y + src_y_start);
            src_y = DL_CLIP(src_y, 0, src_height - 1);
            float _src_i = src_y * src_width;

            for (size_t x = dst_x_start; x < dst_x_end; x++) {
                int dst_i = (_dst_i + x) * dst_channel;

                float src_x = rintf((x - dst_x_start) * scale_x) + src_x_start;
                src_x = DL_CLIP(src_x, 0, src_width - 1);
                int src_i = (_src_i + src_x) * dst_channel;

                if (src_channel == dst_channel) {
                    for (size_t c = 0; c < dst_channel; c++) {
                        if (rgb_swap)
                            dst_image[dst_i + dst_channel - 1 - c] = (T)src_image[src_i + c];
                        else
                            dst_image[dst_i + c] = (T)src_image[src_i + c];
                    }
                } else if (src_channel == 3 && dst_channel == 1) // RGB -> Gray
                {
                    if (rgb_swap)
                        dst_image[dst_i] =
                            convert_pixel_rgb888_to_gray(src_image[src_i], src_image[src_i + 1], src_image[src_i + 2]);
                    else
                        dst_image[dst_i] =
                            convert_pixel_rgb888_to_gray(src_image[src_i + 2], src_image[src_i + 1], src_image[src_i]);
                } else {
                    printf("Not implement src_channel = %d and dst_channel = %d\n", src_channel, dst_channel);
                }
            }
        }
        break;

    default:
        printf("Not implement image_resize_type = %d\n", resize_type);
        break;
    }
}
template void crop_and_resize(uint8_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint8_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap);
template void crop_and_resize(int16_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint8_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap);
template void crop_and_resize(int8_t *dst_image,
                              int dst_width,
                              int dst_channel,
                              int dst_y_start,
                              int dst_y_end,
                              int dst_x_start,
                              int dst_x_end,
                              uint8_t *src_image,
                              int src_height,
                              int src_width,
                              int src_channel,
                              int src_y_start,
                              int src_y_end,
                              int src_x_start,
                              int src_x_end,
                              resize_type_t resize_type,
                              bool rgb_swap);

void draw_filled_rectangle(uint8_t *image,
                           const uint32_t image_height,
                           const uint32_t image_width,
                           uint32_t x1,
                           uint32_t y1,
                           uint32_t x2,
                           uint32_t y2,
                           const uint32_t color)
{
    assert(x2 >= x1);
    assert(y2 >= y1);

    x1 = DL_MIN(x1, image_width - 1);
    y1 = DL_MIN(y1, image_height - 1);
    x2 = DL_MIN(x2, image_width - 1);
    y2 = DL_MIN(y2, image_height - 1);

    uint8_t c0 = color >> 16;
    uint8_t c1 = color >> 8;
    uint8_t c2 = color;

    uint8_t *ptr = image + (y1 * image_width + x1) * 3;
    uint32_t offset = image_width * 3;
    for (int y = y1; y <= y2; y++) {
        uint8_t *row = ptr;
        for (int x = x1; x <= x2; x++) {
            row[0] = c0;
            row[1] = c1;
            row[2] = c2;
            row += 3;
        }
        ptr += offset;
    }
}

void draw_filled_rectangle(uint16_t *image,
                           const uint32_t image_height,
                           const uint32_t image_width,
                           uint32_t x1,
                           uint32_t y1,
                           uint32_t x2,
                           uint32_t y2,
                           const uint16_t color)
{
    assert(x2 >= x1);
    assert(y2 >= y1);

    x1 = DL_MIN(x1, image_width - 1);
    y1 = DL_MIN(y1, image_height - 1);
    x2 = DL_MIN(x2, image_width - 1);
    y2 = DL_MIN(y2, image_height - 1);

    uint16_t *ptr = image + y1 * image_width + x1;
    for (int y = y1; y <= y2; y++) {
        uint16_t *row = ptr;

        for (int x = x1; x <= x2; x++) {
            *row = color;
            row++;
        }
        ptr += image_width;
    }
}

void draw_point(uint8_t *image,
                const uint32_t image_height,
                const uint32_t image_width,
                const uint32_t x,
                const uint32_t y,
                const uint32_t size,
                const uint32_t color)
{
    int half_size = size >> 1;
    draw_filled_rectangle(image,
                          image_height,
                          image_width,
                          DL_MAX((int)x - half_size, 0),
                          DL_MAX((int)y - half_size, 0),
                          x + half_size,
                          y + half_size,
                          color);
}

void draw_point(uint16_t *image,
                const uint32_t image_height,
                const uint32_t image_width,
                const uint32_t x,
                const uint32_t y,
                const uint32_t size,
                uint16_t color)
{
    int half_size = size >> 1;
    draw_filled_rectangle(image,
                          image_height,
                          image_width,
                          DL_MAX((int)x - half_size, 0),
                          DL_MAX((int)y - half_size, 0),
                          x + half_size,
                          y + half_size,
                          color);
}

void draw_hollow_rectangle(uint8_t *image,
                           const uint32_t image_height,
                           const uint32_t image_width,
                           uint32_t x1,
                           uint32_t y1,
                           uint32_t x2,
                           uint32_t y2,
                           uint32_t color)
{
    assert(x2 >= x1);
    assert(y2 >= y1);

    x1 = DL_MIN(x1, image_width - 1);
    y1 = DL_MIN(y1, image_height - 1);
    x2 = DL_MIN(x2, image_width - 1);
    y2 = DL_MIN(y2, image_height - 1);

    uint8_t c0 = color >> 16;
    uint8_t c1 = color >> 8;
    uint8_t c2 = color;

    // draw horizon
    uint8_t *row_up = image + (y1 * image_width + x1) * 3;
    uint8_t *row_down = image + (y2 * image_width + x1) * 3;
    for (int x = x1; x <= x2; x++) {
        row_up[0] = c0;
        row_up[1] = c1;
        row_up[2] = c2;
        row_up += 3;

        row_down[0] = c0;
        row_down[1] = c1;
        row_down[2] = c2;
        row_down += 3;
    }

    // draw vertical
    uint8_t *colum_left = image + (y1 * image_width + x1) * 3;
    uint8_t *colum_right = image + (y1 * image_width + x2) * 3;
    uint32_t offset = image_width * 3;
    for (int y = y1; y <= y2; y++) {
        colum_left[0] = c0;
        colum_left[1] = c1;
        colum_left[2] = c2;
        colum_left += offset;

        colum_right[0] = c0;
        colum_right[1] = c1;
        colum_right[2] = c2;
        colum_right += offset;
    }
}

void draw_hollow_rectangle(uint16_t *image,
                           const uint32_t image_height,
                           const uint32_t image_width,
                           uint32_t x1,
                           uint32_t y1,
                           uint32_t x2,
                           uint32_t y2,
                           const uint16_t color)
{
    assert(x2 >= x1);
    assert(y2 >= y1);

    x1 = DL_MIN(x1, image_width - 1);
    y1 = DL_MIN(y1, image_height - 1);
    x2 = DL_MIN(x2, image_width - 1);
    y2 = DL_MIN(y2, image_height - 1);

    // draw horizon
    uint16_t *row_up = image + y1 * image_width + x1;
    uint16_t *row_down = image + y2 * image_width + x1;
    for (int x = x1; x <= x2; x++) {
        *row_up = color;
        row_up++;

        *row_down = color;
        row_down++;
    }

    // draw vertical
    uint16_t *colum_left = image + y1 * image_width + x1;
    uint16_t *colum_right = image + y1 * image_width + x2;
    for (int y = y1; y <= y2; y++) {
        *colum_left = color;
        colum_left += image_width;

        *colum_right = color;
        colum_right += image_width;
    }
}

uint32_t get_moving_point_number(uint16_t *f1,
                                 uint16_t *f2,
                                 const uint32_t height,
                                 const uint32_t width,
                                 const uint32_t stride,
                                 const uint32_t threshold)
{
    uint32_t stride_y_offset = width * stride;
    uint32_t count = 0;
    for (uint32_t y = 0; y < height; y += stride) {
        uint16_t *f1_row = f1;
        uint16_t *f2_row = f2;
        for (uint32_t x = 0; x < width; x += stride) {
            int f1_gray = convert_pixel_rgb565_to_gray(*f1_row);
            int f2_gray = convert_pixel_rgb565_to_gray(*f2_row);

            if (DL_ABS(f1_gray - f2_gray) > threshold)
                count++;

            f1_row += stride;
            f2_row += stride;
        }
        f1 += stride_y_offset;
        f2 += stride_y_offset;
    }
    return count;
}

uint32_t get_moving_point_number(uint8_t *f1,
                                 uint8_t *f2,
                                 const uint32_t height,
                                 const uint32_t width,
                                 const uint32_t stride,
                                 const uint32_t threshold)
{
    uint32_t stride_y_offset = width * stride * 3;
    uint32_t stride_x_offset = stride * 3;
    uint32_t count = 0;
    for (uint32_t y = 0; y < height; y += stride) {
        uint8_t *f1_row = f1;
        uint8_t *f2_row = f2;
        for (uint32_t x = 0; x < width; x += stride) {
            int f1_gray = convert_pixel_rgb888_to_gray(f1_row[2], f1_row[1], f1_row[0]);
            int f2_gray = convert_pixel_rgb888_to_gray(f2_row[2], f2_row[1], f2_row[0]);

            if (DL_ABS(f1_gray - f2_gray) > threshold)
                count++;

            f1_row += stride_x_offset;
            f2_row += stride_x_offset;
        }
        f1 += stride_y_offset;
        f2 += stride_y_offset;
    }
    return count;
}

template <typename T>
void warp_affine(uint8_t *input,
                 const std::vector<int> &input_shape,
                 T *output,
                 const std::vector<int> &output_shape,
                 dl::math::Matrix<float> *M_inv,
                 bool byte_swap)
{
    int input_stride = input_shape[1] * input_shape[2]; // stride = w * c
    int c = input_shape[2];
    int output_h = output_shape[0];
    int output_w = output_shape[1];

    float x_src = 0.0;
    float y_src = 0.0;
    int x1 = 0;
    int x2 = 0;
    int y1 = 0;
    int y2 = 0;

    for (int i = 0; i < output_h; i++) {
        for (int j = 0; j < output_w; j++) {
            x_src = (M_inv->array[0][0] * j + M_inv->array[0][1] * i + M_inv->array[0][2]) /
                (M_inv->array[2][0] * j + M_inv->array[2][1] * i + M_inv->array[2][2]);
            y_src = (M_inv->array[1][0] * j + M_inv->array[1][1] * i + M_inv->array[1][2]) /
                (M_inv->array[2][0] * j + M_inv->array[2][1] * i + M_inv->array[2][2]);
            if ((x_src < 0) || (y_src < 0) || (x_src >= (input_shape[1] - 1)) || (y_src >= (input_shape[0] - 1))) {
                for (int k = 0; k < c; k++) {
                    *output++ = 0;
                }
            } else {
                x1 = floor(x_src);
                x2 = x1 + 1;
                y1 = floor(y_src);
                y2 = y1 + 1;
                for (int k = 0; k < c; k++) {
                    *output++ = (T)rintf(((input[y1 * input_stride + x1 * c + k]) * (x2 - x_src) * (y2 - y_src)) +
                                         ((input[y1 * input_stride + x2 * c + k]) * (x_src - x1) * (y2 - y_src)) +
                                         ((input[y2 * input_stride + x1 * c + k]) * (x2 - x_src) * (y_src - y1)) +
                                         ((input[y2 * input_stride + x2 * c + k]) * (x_src - x1) * (y_src - y1)));
                }
            }
        }
    }
}
template void warp_affine(uint8_t *input,
                          const std::vector<int> &input_shape,
                          uint8_t *output,
                          const std::vector<int> &output_shape,
                          dl::math::Matrix<float> *M_inv,
                          bool byte_swap);
template void warp_affine(uint8_t *input,
                          const std::vector<int> &input_shape,
                          int16_t *output,
                          const std::vector<int> &output_shape,
                          dl::math::Matrix<float> *M_inv,
                          bool byte_swap);

template <typename T>
void warp_affine(uint16_t *input,
                 const std::vector<int> &input_shape,
                 T *output,
                 const std::vector<int> &output_shape,
                 dl::math::Matrix<float> *M_inv,
                 bool byte_swap)
{
    int input_stride = input_shape[1]; // stride = w
    int c = input_shape[2];
    assert(c == 3);
    int output_h = output_shape[0];
    int output_w = output_shape[1];

    float x_src = 0.0;
    float y_src = 0.0;
    int x1 = 0;
    int x2 = 0;
    int y1 = 0;
    int y2 = 0;

    uint8_t src_x1y1[3] = {0};
    uint8_t src_x1y2[3] = {0};
    uint8_t src_x2y1[3] = {0};
    uint8_t src_x2y2[3] = {0};

    for (int i = 0; i < output_h; i++) {
        for (int j = 0; j < output_w; j++) {
            x_src = (M_inv->array[0][0] * j + M_inv->array[0][1] * i + M_inv->array[0][2]) /
                (M_inv->array[2][0] * j + M_inv->array[2][1] * i + M_inv->array[2][2]);
            y_src = (M_inv->array[1][0] * j + M_inv->array[1][1] * i + M_inv->array[1][2]) /
                (M_inv->array[2][0] * j + M_inv->array[2][1] * i + M_inv->array[2][2]);
            if ((x_src < 0) || (y_src < 0) || (x_src >= (input_shape[1] - 1)) || (y_src >= (input_shape[0] - 1))) {
                for (int k = 0; k < c; k++) {
                    *output++ = 0;
                }
            } else {
                x1 = floor(x_src);
                x2 = x1 + 1;
                y1 = floor(y_src);
                y2 = y1 + 1;

                dl::image::convert_pixel_rgb565_to_rgb888(input[y1 * input_stride + x1], src_x1y1);
                dl::image::convert_pixel_rgb565_to_rgb888(input[y2 * input_stride + x1], src_x1y2);
                dl::image::convert_pixel_rgb565_to_rgb888(input[y1 * input_stride + x2], src_x2y1);
                dl::image::convert_pixel_rgb565_to_rgb888(input[y2 * input_stride + x2], src_x2y2);

                *output++ =
                    (T)rintf((src_x1y1[0] * (x2 - x_src) * (y2 - y_src)) + (src_x2y1[0] * (x_src - x1) * (y2 - y_src)) +
                             (src_x1y2[0] * (x2 - x_src) * (y_src - y1)) + (src_x2y2[0] * (x_src - x1) * (y_src - y1)));
                *output++ =
                    (T)rintf((src_x1y1[1] * (x2 - x_src) * (y2 - y_src)) + (src_x2y1[1] * (x_src - x1) * (y2 - y_src)) +
                             (src_x1y2[1] * (x2 - x_src) * (y_src - y1)) + (src_x2y2[1] * (x_src - x1) * (y_src - y1)));
                *output++ =
                    (T)rintf((src_x1y1[2] * (x2 - x_src) * (y2 - y_src)) + (src_x2y1[2] * (x_src - x1) * (y2 - y_src)) +
                             (src_x1y2[2] * (x2 - x_src) * (y_src - y1)) + (src_x2y2[2] * (x_src - x1) * (y_src - y1)));
            }
        }
    }
}
template void warp_affine(uint16_t *input,
                          const std::vector<int> &input_shape,
                          uint8_t *output,
                          const std::vector<int> &output_shape,
                          dl::math::Matrix<float> *M_inv,
                          bool byte_swap);
template void warp_affine(uint16_t *input,
                          const std::vector<int> &input_shape,
                          int16_t *output,
                          const std::vector<int> &output_shape,
                          dl::math::Matrix<float> *M_inv,
                          bool byte_swap);

uint8_t get_otsu_thresh(Tensor<uint8_t> &image)
{
    if (image.shape.size() == 3) {
        assert(image.shape[2] == 1);
    } else {
        assert(image.shape.size() == 2);
    }
    int numPixels = image.get_size();

    const int HISTOGRAM_SIZE = 256;
    unsigned int histogram[HISTOGRAM_SIZE];
    memset(histogram, 0, (HISTOGRAM_SIZE) * sizeof(unsigned int));
    uint8_t *ptr = image.element;
    int length = numPixels;
    while (length--) {
        uint8_t value = *ptr++;
        histogram[value]++;
    }

    int sum = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
        sum += i * histogram[i];
    }

    int sumB = 0;
    int q1 = 0;
    double max = 0;
    uint8_t threshold = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
        q1 += histogram[i];
        if (q1 == 0)
            continue;

        const int q2 = numPixels - q1;
        if (q2 == 0)
            break;

        sumB += i * histogram[i];
        const double m1 = (double)sumB / q1;
        const double m2 = ((double)sum - sumB) / q2;
        const double m1m2 = m1 - m2;
        const double variance = m1m2 * m1m2 * q1 * q2;
        if (variance > max) {
            threshold = i;
            max = variance;
        }
    }

    return threshold;
}

Tensor<uint8_t> *rgb2gray(Tensor<uint8_t> &image, bool bgr)
{
    assert(image.shape.size() == 3);
    assert(image.shape[2] == 3);

    Tensor<uint8_t> *gray = new Tensor<uint8_t>;
    gray->set_shape({image.shape[0], image.shape[1], 1}).malloc_element();
    int count = gray->get_size();
    uint8_t *r = NULL;
    uint8_t *g = NULL;
    uint8_t *b = NULL;
    if (bgr) {
        b = image.element;
        g = b + 1;
        r = b + 2;
    } else {
        r = image.element;
        g = r + 1;
        b = r + 2;
    }

    uint8_t *pgray = gray->element;
    int x = 0;
    for (int i = 0; i < count; ++i) {
        // TODO: use tie instructions.
        x = (19595 * (*r) + 38469 * (*g) + 7472 * (*b)) >> 16; // fast algorithm
        // Gray = R*0.299 + G*0.587 + B*0.114
        // Gray = (R*30 + G*59 + B*11 + 50) / 100
        // Gray = (R*38 + G*75 + B*15) >> 7

        *(pgray++) = (uint8_t)x;

        r += 3;
        g += 3;
        b += 3;
    }
    return gray;
}

Tensor<uint8_t> *rgb2lab(Tensor<uint8_t> &image, bool bgr, bool fast)
{
    assert(image.shape.size() == 3);
    assert(image.shape[2] == 3);

    Tensor<uint8_t> *lab = new Tensor<uint8_t>;
    lab->set_shape(image.shape).malloc_element();
    int count = image.shape[0] * image.shape[1];
    uint8_t *r = NULL;
    uint8_t *g = NULL;
    uint8_t *b = NULL;
    if (bgr) {
        b = image.element;
        g = b + 1;
        r = b + 2;
    } else {
        r = image.element;
        g = r + 1;
        b = r + 2;
    }

    if (fast) {
        int x, y, z;
        uint8_t *plab = lab->element;
        for (int i = 0; i < count; ++i) {
            // TODO: use tie
            x = (13933 * (*r) + 46871 * (*g) + 4732 * (*b)) >> 16;
            y = (((5467631 * (*r) - 8376186 * (*g) + 2908178 * (*b))) >> 24) + 128;
            z = (((2043680 * (*r) + 6351200 * (*g) - 8394880 * (*b))) >> 24) + 128;

            *(plab++) = (uint8_t)x;
            *(plab++) = (uint8_t)y;
            *(plab++) = (uint8_t)z;

            r += 3;
            g += 3;
            b += 3;
        }
        return lab;
    } else {
        float x, y, z;
        uint8_t *plab = lab->element;
        for (int i = 0; i < count; ++i) {
            x = (0.433953 * (*r) + 0.376219 * (*g) + 0.189828 * (*b)) / 255;
            y = (0.212671 * (*r) + 0.715160 * (*g) + 0.072169 * (*b)) / 255;
            z = (0.017758 * (*r) + 0.109476 * (*g) + 0.872766 * (*b)) / 255;

            x = (x > 0.008856) ? pow(x, 1.0 / 3) : (7.787037 * x + 0.137931);
            y = (y > 0.008856) ? pow(y, 1.0 / 3) : (7.787037 * y + 0.137931);
            z = (z > 0.008856) ? pow(z, 1.0 / 3) : (7.787037 * z + 0.137931);

            *(plab++) = (uint8_t)(116 * y - 16);
            *(plab++) = (uint8_t)(500 * (x - y) + 128);
            *(plab++) = (uint8_t)(200 * (y - z) + 128);

            r += 3;
            g += 3;
            b += 3;
        }
        return lab;
    }
}

Tensor<uint8_t> *rgb2hsv(Tensor<uint8_t> &image, bool bgr, bool fast)
{
    assert(image.shape.size() == 3);
    assert(image.shape[2] == 3);

    Tensor<uint8_t> *hsv = new Tensor<uint8_t>;
    hsv->set_shape(image.shape).malloc_element();
    int count = image.shape[0] * image.shape[1];
    uint8_t *r = NULL;
    uint8_t *g = NULL;
    uint8_t *b = NULL;
    if (bgr) {
        b = image.element;
        g = b + 1;
        r = b + 2;
    } else {
        r = image.element;
        g = r + 1;
        b = r + 2;
    }
    if (fast) {
        int h, s, v, min_rgb, delta;
        uint8_t *phsv = hsv->element;
        for (int i = 0; i < count; ++i) {
            v = DL_MAX(DL_MAX(*r, *g), *b);
            min_rgb = DL_MIN(DL_MIN(*r, *g), *b);
            if (v == min_rgb) {
                *(phsv++) = 0;
                *(phsv++) = 0;
                *(phsv++) = (uint8_t)(v);
            } else {
                delta = v - min_rgb;
                s = (delta * 255) / v;
                if (v == (*r)) {
                    h = (60 * ((*g) - (*b)) / (delta)) >> 1;
                    h = (h < 0) ? (h + 180) : h;
                } else if (v == (*g)) {
                    h = (120 + 60 * ((*b) - (*r)) / delta) >> 1;
                } else {
                    h = (240 + 60 * ((*r) - (*g)) / delta) >> 1;
                }
                *(phsv++) = (uint8_t)(h);
                *(phsv++) = (uint8_t)(s);
                *(phsv++) = (uint8_t)(v);
            }

            r += 3;
            g += 3;
            b += 3;
        }

        return hsv;
    } else {
        float h, s, v, min_rgb;
        uint8_t *phsv = hsv->element;
        float h_scale = 180.0 / 360.0;
        for (int i = 0; i < count; ++i) {
            v = DL_MAX(DL_MAX(*r, *g), *b);
            min_rgb = DL_MIN(DL_MIN(*r, *g), *b);
            if (v == min_rgb) {
                *(phsv++) = 0;
                *(phsv++) = 0;
                *(phsv++) = (uint8_t)(v);
            } else {
                s = (v - min_rgb) * 255.0 / v;
                if (v == (*r)) {
                    h = h_scale * 60.0 * ((*g) - (*b)) / (v - min_rgb);
                    h = (h < 0) ? (h + 180) : h;
                } else if (v == (*g)) {
                    h = h_scale * (120.0 + 60.0 * ((*b) - (*r)) / (v - min_rgb));
                } else {
                    h = h_scale * (240.0 + 60.0 * ((*r) - (*g)) / (v - min_rgb));
                }
                *(phsv++) = (uint8_t)(h);
                *(phsv++) = (uint8_t)(s);
                *(phsv++) = (uint8_t)(v);
            }

            r += 3;
            g += 3;
            b += 3;
        }

        return hsv;
    }
}

Tensor<uint8_t> *convert_image_rgb565_to_rgb888(uint16_t *image, std::vector<int> &image_shape)
{
    Tensor<uint8_t> *rgb = new Tensor<uint8_t>;
    rgb->set_shape({image_shape[0], image_shape[1], 3}).malloc_element();
    int count = image_shape[0] * image_shape[1];
    uint8_t *element_ptr = rgb->element;
    for (int i = 0; i < count; ++i) {
        convert_pixel_rgb565_to_rgb888(image[i], element_ptr);
        element_ptr += 3;
    }
    return rgb;
}

Tensor<uint8_t> *gen_binary_img(Tensor<uint8_t> &image, std::vector<int> thresh)
{
    assert(image.shape.size() == 3);
    assert(image.shape[2] == 3);
    assert(thresh.size() == 6);
    Tensor<uint8_t> *bin = new Tensor<uint8_t>;
    bin->set_shape({image.shape[0], image.shape[1], 1}).malloc_element();
    uint8_t *c1 = image.element;
    uint8_t *c2 = c1 + 1;
    uint8_t *c3 = c1 + 2;
    uint8_t *pbin = bin->element;
    int count = bin->get_size();
    // int num = 0;
    for (int i = 0; i < count; i++) {
        if (((*c1) >= thresh[0]) && ((*c1) <= thresh[1]) && ((*c2) >= thresh[2]) && ((*c2) <= thresh[3]) &&
            ((*c3) >= thresh[4]) && ((*c3) <= thresh[5])) {
            *(pbin++) = 255;
            // num++;
        } else {
            *(pbin++) = 0;
        }
        c1 += 3;
        c2 += 3;
        c3 += 3;
    }

    return bin;
}

Tensor<uint8_t> *resize_image(Tensor<uint8_t> &image, std::vector<int> target_shape, resize_type_t resize_type)
{
    assert(image.shape.size() == 3);
    assert(target_shape.size() == 3);
    Tensor<uint8_t> *resized_image = new Tensor<uint8_t>;
    resized_image->set_shape({target_shape[0], target_shape[1], image.shape[2]});
    float h_ratio = (float)(image.shape[0]) / target_shape[0];
    float w_ratio = (float)(image.shape[1]) / target_shape[1];

    if (image.shape.back() == 3) {
        resized_image->malloc_element();
        uint8_t *resized_ptr = resized_image->element;
        float h_origin = 0;
        float w_origin = 0;
        if (resize_type == IMAGE_RESIZE_BILINEAR) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                float h1_weight = (float)h2 - h_origin;
                float h2_weight = h_origin - (float)h1;
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    float w1_weight = (float)w2 - w_origin;
                    float w2_weight = w_origin - (float)w1;
                    resized_ptr[0] = (uint8_t)(image.get_element_value({h1, w1, 0}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 0}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 0}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 0}) * h2_weight * w2_weight);
                    resized_ptr[1] = (uint8_t)(image.get_element_value({h1, w1, 1}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 1}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 1}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 1}) * h2_weight * w2_weight);
                    resized_ptr[2] = (uint8_t)(image.get_element_value({h1, w1, 2}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 2}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 2}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 2}) * h2_weight * w2_weight);
                    resized_ptr += 3;
                }
            }
            return resized_image;
        } else if (resize_type == IMAGE_RESIZE_MEAN) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    resized_ptr[0] = (uint8_t)(((int)image.get_element_value({h1, w1, 0}) +
                                                (int)image.get_element_value({h1, w2, 0}) +
                                                (int)image.get_element_value({h2, w1, 0}) +
                                                (int)image.get_element_value({h2, w2, 0})) >>
                                               2);
                    resized_ptr[1] = (uint8_t)(((int)image.get_element_value({h1, w1, 1}) +
                                                (int)image.get_element_value({h1, w2, 1}) +
                                                (int)image.get_element_value({h2, w1, 1}) +
                                                (int)image.get_element_value({h2, w2, 1})) >>
                                               2);
                    resized_ptr[2] = (uint8_t)(((int)image.get_element_value({h1, w1, 2}) +
                                                (int)image.get_element_value({h1, w2, 2}) +
                                                (int)image.get_element_value({h2, w1, 2}) +
                                                (int)image.get_element_value({h2, w2, 2})) >>
                                               2);
                    resized_ptr += 3;
                }
            }
            return resized_image;
        } else if (resize_type == IMAGE_RESIZE_NEAREST) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h_real = (int)(round(h_origin));
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w_real = (int)(round(w_origin));
                    resized_ptr[0] = image.get_element_value({h_real, w_real, 0});
                    resized_ptr[1] = image.get_element_value({h_real, w_real, 1});
                    resized_ptr[2] = image.get_element_value({h_real, w_real, 2});
                    resized_ptr += 3;
                }
            }
            return resized_image;
        } else {
            delete resized_image;
            ESP_LOGE("resize image", "resize type is not supported!");
            return NULL;
        }
    } else if (image.shape.back() == 1) {
        resized_image->malloc_element();
        uint8_t *resized_ptr = resized_image->element;
        float h_origin = 0;
        float w_origin = 0;
        if (resize_type == IMAGE_RESIZE_BILINEAR) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                float h1_weight = (float)h2 - h_origin;
                float h2_weight = h_origin - (float)h1;
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    float w1_weight = (float)w2 - w_origin;
                    float w2_weight = w_origin - (float)w1;
                    resized_ptr[0] = (uint8_t)(image.get_element_value({h1, w1, 0}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 0}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 0}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 0}) * h2_weight * w2_weight);
                    ++resized_ptr;
                }
            }
            return resized_image;
        } else if (resize_type == IMAGE_RESIZE_MEAN) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    resized_ptr[0] = (uint8_t)(((int)image.get_element_value({h1, w1, 0}) +
                                                (int)image.get_element_value({h1, w2, 0}) +
                                                (int)image.get_element_value({h2, w1, 0}) +
                                                (int)image.get_element_value({h2, w2, 0})) >>
                                               2);
                    ++resized_ptr;
                }
            }
            return resized_image;
        } else if (resize_type == IMAGE_RESIZE_NEAREST) {
            for (int h = 0; h < target_shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h_real = (int)(round(h_origin));
                for (int w = 0; w < target_shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w_real = (int)(round(w_origin));
                    resized_ptr[0] = image.get_element_value({h_real, w_real, 0});
                    ++resized_ptr;
                }
            }
            return resized_image;
        } else {
            delete resized_image;
            ESP_LOGE("resize image", "resize type is not supported!");
            return NULL;
        }
    } else {
        delete resized_image;
        ESP_LOGE("resize image", "the image shape is invaild!");
        return NULL;
    }
}

void resize_image(Tensor<uint8_t> &image, Tensor<uint8_t> &resized_image, resize_type_t resize_type)
{
    assert(image.shape.size() == 3);
    assert(resized_image.shape.size() == 3);
    float h_ratio = (float)(image.shape[0]) / resized_image.shape[0];
    float w_ratio = (float)(image.shape[1]) / resized_image.shape[1];

    if (image.shape.back() == 3) {
        resized_image.malloc_element();
        uint8_t *resized_ptr = resized_image.element;
        float h_origin = 0;
        float w_origin = 0;
        if (resize_type == IMAGE_RESIZE_BILINEAR) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                float h1_weight = (float)h2 - h_origin;
                float h2_weight = h_origin - (float)h1;
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    float w1_weight = (float)w2 - w_origin;
                    float w2_weight = w_origin - (float)w1;
                    resized_ptr[0] = (uint8_t)(image.get_element_value({h1, w1, 0}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 0}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 0}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 0}) * h2_weight * w2_weight);
                    resized_ptr[1] = (uint8_t)(image.get_element_value({h1, w1, 1}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 1}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 1}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 1}) * h2_weight * w2_weight);
                    resized_ptr[2] = (uint8_t)(image.get_element_value({h1, w1, 2}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 2}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 2}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 2}) * h2_weight * w2_weight);
                    resized_ptr += 3;
                }
            }
            return;
        } else if (resize_type == IMAGE_RESIZE_MEAN) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    resized_ptr[0] = (uint8_t)(((int)image.get_element_value({h1, w1, 0}) +
                                                (int)image.get_element_value({h1, w2, 0}) +
                                                (int)image.get_element_value({h2, w1, 0}) +
                                                (int)image.get_element_value({h2, w2, 0})) >>
                                               2);
                    resized_ptr[1] = (uint8_t)(((int)image.get_element_value({h1, w1, 1}) +
                                                (int)image.get_element_value({h1, w2, 1}) +
                                                (int)image.get_element_value({h2, w1, 1}) +
                                                (int)image.get_element_value({h2, w2, 1})) >>
                                               2);
                    resized_ptr[2] = (uint8_t)(((int)image.get_element_value({h1, w1, 2}) +
                                                (int)image.get_element_value({h1, w2, 2}) +
                                                (int)image.get_element_value({h2, w1, 2}) +
                                                (int)image.get_element_value({h2, w2, 2})) >>
                                               2);
                    resized_ptr += 3;
                }
            }
            return;
        } else if (resize_type == IMAGE_RESIZE_NEAREST) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h_real = (int)(round(h_origin));
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w_real = (int)(round(w_origin));
                    resized_ptr[0] = image.get_element_value({h_real, w_real, 0});
                    resized_ptr[1] = image.get_element_value({h_real, w_real, 1});
                    resized_ptr[2] = image.get_element_value({h_real, w_real, 2});
                    resized_ptr += 3;
                }
            }
            return;
        } else {
            ESP_LOGE("resize image", "resize type is not supported!");
            return;
        }
    } else if (image.shape.back() == 1) {
        resized_image.malloc_element();
        uint8_t *resized_ptr = resized_image.element;
        float h_origin = 0;
        float w_origin = 0;
        if (resize_type == IMAGE_RESIZE_BILINEAR) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                float h1_weight = (float)h2 - h_origin;
                float h2_weight = h_origin - (float)h1;
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    float w1_weight = (float)w2 - w_origin;
                    float w2_weight = w_origin - (float)w1;
                    resized_ptr[0] = (uint8_t)(image.get_element_value({h1, w1, 0}) * h1_weight * w1_weight +
                                               image.get_element_value({h1, w2, 0}) * h1_weight * w2_weight +
                                               image.get_element_value({h2, w1, 0}) * h2_weight * w1_weight +
                                               image.get_element_value({h2, w2, 0}) * h2_weight * w2_weight);
                    ++resized_ptr;
                }
            }
            return;
        } else if (resize_type == IMAGE_RESIZE_MEAN) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h1 = (int)h_origin;
                int h2 = h1 + 1;
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w1 = (int)w_origin;
                    int w2 = w1 + 1;
                    resized_ptr[0] = (uint8_t)(((int)image.get_element_value({h1, w1, 0}) +
                                                (int)image.get_element_value({h1, w2, 0}) +
                                                (int)image.get_element_value({h2, w1, 0}) +
                                                (int)image.get_element_value({h2, w2, 0})) >>
                                               2);
                    ++resized_ptr;
                }
            }
            return;
        } else if (resize_type == IMAGE_RESIZE_NEAREST) {
            for (int h = 0; h < resized_image.shape[0]; ++h) {
                h_origin = h * h_ratio;
                int h_real = (int)(round(h_origin));
                for (int w = 0; w < resized_image.shape[1]; ++w) {
                    w_origin = w * w_ratio;
                    int w_real = (int)(round(w_origin));
                    resized_ptr[0] = image.get_element_value({h_real, w_real, 0});
                    ++resized_ptr;
                }
            }
            return;
        } else {
            ESP_LOGE("resize image", "resize type is not supported!");
            return;
        }
    } else {
        ESP_LOGE("resize image", "the image shape is invaild!");
        return;
    }
}

template <typename T>
T *resize_image_nearest(T *image, std::vector<int> input_shape, std::vector<int> target_shape)
{
    assert(input_shape.size() == 3);
    assert(target_shape.size() == 3);
    T *resized_image = (T *)dl::tool::malloc_aligned(
        target_shape[0] * target_shape[1] * target_shape[2], sizeof(T), 16, MALLOC_CAP_8BIT);
    float h_ratio = (float)(input_shape[0]) / target_shape[0];
    float w_ratio = (float)(input_shape[1]) / target_shape[1];

    if (input_shape.back() == 3) {
        T *resized_ptr = resized_image;
        float h_origin = 0;
        float w_origin = 0;

        for (int h = 0; h < target_shape[0]; ++h) {
            h_origin = h * h_ratio;
            int h_real = (int)(round(h_origin));
            for (int w = 0; w < target_shape[1]; ++w) {
                w_origin = w * w_ratio;
                int w_real = (int)(round(w_origin));
                T *origin_ptr = image + (h_real * input_shape[1] + w_real * 3);

                resized_ptr[0] = origin_ptr[0];
                resized_ptr[1] = origin_ptr[1];
                resized_ptr[2] = origin_ptr[2];
                resized_ptr += 3;
            }
        }
        return resized_image;
    } else if (input_shape.back() == 1) {
        T *resized_ptr = resized_image;
        float h_origin = 0;
        float w_origin = 0;

        for (int h = 0; h < target_shape[0]; ++h) {
            h_origin = h * h_ratio;
            int h_real = (int)(round(h_origin));
            for (int w = 0; w < target_shape[1]; ++w) {
                w_origin = w * w_ratio;
                int w_real = (int)(round(w_origin));
                resized_ptr[0] = *(image + (h_real * input_shape[1] + w_real));
                ++resized_ptr;
            }
        }
        return resized_image;
    } else {
        dl::tool::free_aligned(resized_image);
        ESP_LOGE("resize image", "the image shape is invaild!");
        return NULL;
    }
}
template int32_t *resize_image_nearest(int32_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template uint32_t *resize_image_nearest(uint32_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template int16_t *resize_image_nearest(int16_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template uint16_t *resize_image_nearest(uint16_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template int8_t *resize_image_nearest(int8_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template uint8_t *resize_image_nearest(uint8_t *image, std::vector<int> input_shape, std::vector<int> target_shape);
template float *resize_image_nearest(float *image, std::vector<int> input_shape, std::vector<int> target_shape);

template <typename T>
void resize_image_nearest(T *image, std::vector<int> input_shape, T *resized_image, std::vector<int> target_shape)
{
    assert(input_shape.size() == 3);
    assert(target_shape.size() == 3);
    float h_ratio = (float)(input_shape[0]) / target_shape[0];
    float w_ratio = (float)(input_shape[1]) / target_shape[1];

    if (input_shape.back() == 3) {
        T *resized_ptr = resized_image;
        float h_origin = 0;
        float w_origin = 0;

        for (int h = 0; h < target_shape[0]; ++h) {
            h_origin = h * h_ratio;
            int h_real = (int)(round(h_origin));
            for (int w = 0; w < target_shape[1]; ++w) {
                w_origin = w * w_ratio;
                int w_real = (int)(round(w_origin));
                T *origin_ptr = image + (h_real * input_shape[1] + w_real * 3);

                resized_ptr[0] = origin_ptr[0];
                resized_ptr[1] = origin_ptr[1];
                resized_ptr[2] = origin_ptr[2];
                resized_ptr += 3;
            }
        }
        return;
    } else if (input_shape.back() == 1) {
        T *resized_ptr = resized_image;
        float h_origin = 0;
        float w_origin = 0;

        for (int h = 0; h < target_shape[0]; ++h) {
            h_origin = h * h_ratio;
            int h_real = (int)(round(h_origin));
            for (int w = 0; w < target_shape[1]; ++w) {
                w_origin = w * w_ratio;
                int w_real = (int)(round(w_origin));
                resized_ptr[0] = *(image + (h_real * input_shape[1] + w_real));
                ++resized_ptr;
            }
        }
        return;
    } else {
        ESP_LOGE("resize image", "the image shape is invaild!");
        return;
    }
}
template void resize_image_nearest(int32_t *image,
                                   std::vector<int> input_shape,
                                   int32_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(uint32_t *image,
                                   std::vector<int> input_shape,
                                   uint32_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(int16_t *image,
                                   std::vector<int> input_shape,
                                   int16_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(uint16_t *image,
                                   std::vector<int> input_shape,
                                   uint16_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(int8_t *image,
                                   std::vector<int> input_shape,
                                   int8_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(uint8_t *image,
                                   std::vector<int> input_shape,
                                   uint8_t *resized_image,
                                   std::vector<int> target_shape);
template void resize_image_nearest(float *image,
                                   std::vector<int> input_shape,
                                   float *resized_image,
                                   std::vector<int> target_shape);

} // namespace image
} // namespace dl
