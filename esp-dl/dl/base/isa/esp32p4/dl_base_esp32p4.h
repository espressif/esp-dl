#pragma once

#include <stdint.h>

extern "C" {

/* W8A16 API: int8 filter, int16 activations. */
void dl_esp32p4_w8a16_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_conv2d_per_channel_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu(int16_t *output_ptr,
                                                                  int16_t *input_ptr,
                                                                  void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_conv2d_per_channel_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu(int16_t *output_ptr,
                                                                  int16_t *input_ptr,
                                                                  void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_conv2d_per_channel_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_conv2d_per_channel_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu(int16_t *output_ptr,
                                                                  int16_t *input_ptr,
                                                                  void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

/* Int16 API */
void dl_esp32p4_s16_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_conv2d_per_channel_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_bias_relu(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_conv2d_per_channel_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_per_channel_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_33cn_bias_relu(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_conv2d_per_channel_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_conv2d_per_channel_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_conv2d_per_channel_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_hwcn_bias_relu(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_conv2d_per_channel_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_depthwise_conv2d_33c1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_33c1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_33c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_33c1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_depthwise_conv2d_hwc1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_hwc1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_hwc1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_depthwise_conv2d_hwc1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_hwc1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_hwc1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_depthwise_conv2d_per_channel_33c1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_33c1_bias_relu(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_33c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_33c1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_33c1_bias(int16_t *output_ptr,
                                                                     int16_t *input_ptr,
                                                                     void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_33c1_bias_relu(int16_t *output_ptr,
                                                                          int16_t *input_ptr,
                                                                          void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_33c1(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_33c1_relu(int16_t *output_ptr,
                                                                     int16_t *input_ptr,
                                                                     void *args_ptr);

void dl_esp32p4_s16_depthwise_conv2d_per_channel_hwc1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_hwc1_bias_relu(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_depthwise_conv2d_per_channel_hwc1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_hwc1_bias(int16_t *output_ptr,
                                                                     int16_t *input_ptr,
                                                                     void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_hwc1_bias_relu(int16_t *output_ptr,
                                                                          int16_t *input_ptr,
                                                                          void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_hwc1(int16_t *output_ptr,
                                                                int16_t *input_ptr,
                                                                void *args_ptr);
void dl_esp32p4_s16_unaligned_depthwise_conv2d_per_channel_hwc1_relu(int16_t *output_ptr,
                                                                     int16_t *input_ptr,
                                                                     void *args_ptr);

void dl_esp32p4_s16_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_add2d_11c_relu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_add2d_11c_prelu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_rescale_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_rescale_add2d_11c_relu(int16_t *output_ptr,
                                           int16_t *input0_ptr,
                                           int16_t *input1_ptr,
                                           void *args_ptr);
void dl_esp32p4_s16_rescale_add2d_11c_prelu(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_unaligned_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_add2d_11c_relu(int16_t *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_unaligned_add2d_11c_prelu(int16_t *output_ptr,
                                              int16_t *input0_ptr,
                                              int16_t *input1_ptr,
                                              void *args_ptr);

void dl_esp32p4_s16_mul2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mul2d_11c_relu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mul2d_11c_prelu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_mul2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_unaligned_mul2d_11c_relu(int16_t *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_unaligned_mul2d_11c_prelu(int16_t *output_ptr,
                                              int16_t *input0_ptr,
                                              int16_t *input1_ptr,
                                              void *args_ptr);

/* Int8 API */
void dl_esp32p4_s8_conv2d_11cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_11cn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_11cn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_11cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_11cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_11cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_11cn_bias_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_11cn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_11cn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_11cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_conv2d_33cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_33cn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_33cn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_33cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_33cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_33cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_33cn_bias_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_33cn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_33cn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_33cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_conv2d_hwcn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_hwcn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_hwcn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_hwcn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_hwcn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_hwcn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_hwcn_bias_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_hwcn_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_hwcn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_hwcn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_depthwise_conv2d_33c1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_33c1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_33c1_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_33c1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_33c1_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_depthwise_conv2d_33c1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_33c1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_33c1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_depthwise_conv2d_hwc1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_hwc1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_hwc1_bias_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_hwc1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_hwc1_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_depthwise_conv2d_hwc1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_hwc1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_hwc1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_conv2d_per_channel_11cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_11cn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_11cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_per_channel_11cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_11cn_bias_leakyrelu(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_11cn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_conv2d_per_channel_33cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_33cn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_33cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_per_channel_33cn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_33cn_bias_leakyrelu(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_33cn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_conv2d_per_channel_hwcn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_hwcn_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_conv2d_per_channel_hwcn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_conv2d_per_channel_hwcn_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_hwcn_bias_leakyrelu(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_conv2d_per_channel_hwcn_leakyrelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_depthwise_conv2d_per_channel_33c1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_33c1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_33c1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_33c1_bias(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_33c1_bias_relu(int8_t *output_ptr,
                                                                         int8_t *input_ptr,
                                                                         void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_33c1_relu(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);

void dl_esp32p4_s8_depthwise_conv2d_per_channel_hwc1_bias(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_hwc1_bias_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_depthwise_conv2d_per_channel_hwc1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_hwc1_bias(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_hwc1_bias_relu(int8_t *output_ptr,
                                                                         int8_t *input_ptr,
                                                                         void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_depthwise_conv2d_per_channel_hwc1_relu(int8_t *output_ptr,
                                                                    int8_t *input_ptr,
                                                                    void *args_ptr);

void dl_esp32p4_s8_mul2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_mul2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_mul2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_mul2d_11c_prelu(int8_t *output_ptr,
                                             int8_t *input0_ptr,
                                             int8_t *input1_ptr,
                                             void *args_ptr);

void dl_esp32p4_s8_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_rescale_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_rescale_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_rescale_add2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_add2d_11c_prelu(int8_t *output_ptr,
                                             int8_t *input0_ptr,
                                             int8_t *input1_ptr,
                                             void *args_ptr);

void dl_esp32p4_s8_max_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_max_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_max_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_max_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_avg_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_avg_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_avg_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_avg_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_resize_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_resize_nearest_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_resize_nearest_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_resize_linear_taps(int8_t *output_ptr, void *taps_ptr);

void dl_esp32p4_s8_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_add_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add_w1_16_w2_16_unaligned(int8_t *output_ptr,
                                             int8_t *input0_ptr,
                                             int8_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s8_add_w1_16_w2_1_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_add_w1_1_w2_16_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_add_w1_8_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_add_w1_8_w2_1(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_add_w1_1_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_add_w1_8_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_add_w1_8_w2_1_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_add_w1_1_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);

void dl_esp32p4_s8_sub_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_sub_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_sub_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_sub_w1_16_w2_16_unaligned(int8_t *output_ptr,
                                             int8_t *input0_ptr,
                                             int8_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s8_sub_w1_16_w2_1_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_sub_w1_1_w2_16_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_sub_w1_8_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_sub_w1_8_w2_1(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_sub_w1_1_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_sub_w1_8_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_sub_w1_8_w2_1_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_sub_w1_1_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);

void dl_esp32p4_s8_mul_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul_w1_16_w2_16_unaligned(int8_t *output_ptr,
                                             int8_t *input0_ptr,
                                             int8_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s8_mul_w1_16_w2_1_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mul_w1_1_w2_16_unaligned(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_mul_w1_8_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mul_w1_8_w2_1(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mul_w1_1_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mul_w1_8_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_mul_w1_8_w2_1_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_esp32p4_s16_mul_w1_1_w2_8_unaligned(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);

void dl_esp32p4_s8_mod_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mod_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_mod_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_mod_w1_8_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mod_w1_8_w2_1(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_mod_w1_1_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s8_equal_w1_16_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_equal_w1_16_w2_1(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_equal_w1_1_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_equal_w1_16_w2_16_unaligned(bool *output_ptr,
                                               int8_t *input0_ptr,
                                               int8_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s8_equal_w1_16_w2_1_unaligned(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_equal_w1_1_w2_16_unaligned(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_equal_w1_8_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_equal_w1_8_w2_1(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_equal_w1_1_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_equal_w1_8_w2_8_unaligned(bool *output_ptr,
                                              int16_t *input0_ptr,
                                              int16_t *input1_ptr,
                                              void *args_ptr);
void dl_esp32p4_s16_equal_w1_8_w2_1_unaligned(bool *output_ptr,
                                              int16_t *input0_ptr,
                                              int16_t *input1_ptr,
                                              void *args_ptr);
void dl_esp32p4_s16_equal_w1_1_w2_8_unaligned(bool *output_ptr,
                                              int16_t *input0_ptr,
                                              int16_t *input1_ptr,
                                              void *args_ptr);

void dl_esp32p4_s8_greater_w1_16_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_greater_w1_16_w2_1(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_greater_w1_1_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s8_greater_w1_16_w2_16_unaligned(bool *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_greater_w1_16_w2_1_unaligned(bool *output_ptr,
                                                int8_t *input0_ptr,
                                                int8_t *input1_ptr,
                                                void *args_ptr);
void dl_esp32p4_s8_greater_w1_1_w2_16_unaligned(bool *output_ptr,
                                                int8_t *input0_ptr,
                                                int8_t *input1_ptr,
                                                void *args_ptr);

void dl_esp32p4_s16_greater_w1_8_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_greater_w1_8_w2_1(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_greater_w1_1_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_greater_w1_8_w2_8_unaligned(bool *output_ptr,
                                                int16_t *input0_ptr,
                                                int16_t *input1_ptr,
                                                void *args_ptr);
void dl_esp32p4_s16_greater_w1_8_w2_1_unaligned(bool *output_ptr,
                                                int16_t *input0_ptr,
                                                int16_t *input1_ptr,
                                                void *args_ptr);
void dl_esp32p4_s16_greater_w1_1_w2_8_unaligned(bool *output_ptr,
                                                int16_t *input0_ptr,
                                                int16_t *input1_ptr,
                                                void *args_ptr);

void dl_esp32p4_s8_less_w1_16_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_less_w1_16_w2_1(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_less_w1_1_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_less_w1_16_w2_16_unaligned(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_less_w1_16_w2_1_unaligned(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_less_w1_1_w2_16_unaligned(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_esp32p4_s16_less_w1_8_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_less_w1_8_w2_1(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_less_w1_1_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_less_w1_8_w2_8_unaligned(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_less_w1_8_w2_1_unaligned(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_less_w1_1_w2_8_unaligned(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);

void dl_esp32p4_s8_lessorequal_w1_16_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_lessorequal_w1_16_w2_1(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_lessorequal_w1_1_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_lessorequal_w1_16_w2_16_unaligned(bool *output_ptr,
                                                     int8_t *input0_ptr,
                                                     int8_t *input1_ptr,
                                                     void *args_ptr);
void dl_esp32p4_s8_lessorequal_w1_16_w2_1_unaligned(bool *output_ptr,
                                                    int8_t *input0_ptr,
                                                    int8_t *input1_ptr,
                                                    void *args_ptr);
void dl_esp32p4_s8_lessorequal_w1_1_w2_16_unaligned(bool *output_ptr,
                                                    int8_t *input0_ptr,
                                                    int8_t *input1_ptr,
                                                    void *args_ptr);

void dl_esp32p4_s16_lessorequal_w1_8_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_lessorequal_w1_8_w2_1(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_lessorequal_w1_1_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s16_lessorequal_w1_8_w2_8_unaligned(bool *output_ptr,
                                                    int16_t *input0_ptr,
                                                    int16_t *input1_ptr,
                                                    void *args_ptr);
void dl_esp32p4_s16_lessorequal_w1_8_w2_1_unaligned(bool *output_ptr,
                                                    int16_t *input0_ptr,
                                                    int16_t *input1_ptr,
                                                    void *args_ptr);
void dl_esp32p4_s16_lessorequal_w1_1_w2_8_unaligned(bool *output_ptr,
                                                    int16_t *input0_ptr,
                                                    int16_t *input1_ptr,
                                                    void *args_ptr);

void dl_esp32p4_s8_greaterorequal_w1_16_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_greaterorequal_w1_16_w2_1(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_greaterorequal_w1_1_w2_16(bool *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_esp32p4_s8_greaterorequal_w1_16_w2_16_unaligned(bool *output_ptr,
                                                        int8_t *input0_ptr,
                                                        int8_t *input1_ptr,
                                                        void *args_ptr);
void dl_esp32p4_s8_greaterorequal_w1_16_w2_1_unaligned(bool *output_ptr,
                                                       int8_t *input0_ptr,
                                                       int8_t *input1_ptr,
                                                       void *args_ptr);
void dl_esp32p4_s8_greaterorequal_w1_1_w2_16_unaligned(bool *output_ptr,
                                                       int8_t *input0_ptr,
                                                       int8_t *input1_ptr,
                                                       void *args_ptr);

void dl_esp32p4_s16_greaterorequal_w1_8_w2_8(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_greaterorequal_w1_8_w2_1(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_greaterorequal_w1_1_w2_8(bool *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);
void dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned(bool *output_ptr,
                                                       int16_t *input0_ptr,
                                                       int16_t *input1_ptr,
                                                       void *args_ptr);
void dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned(bool *output_ptr,
                                                       int16_t *input0_ptr,
                                                       int16_t *input1_ptr,
                                                       void *args_ptr);
void dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned(bool *output_ptr,
                                                       int16_t *input0_ptr,
                                                       int16_t *input1_ptr,
                                                       void *args_ptr);

void dl_esp32p4_s8_and4d_bchw_w1_16_w2_16_simdand(int8_t *output_ptr,
                                                  int8_t *input0_ptr,
                                                  int8_t *input1_ptr,
                                                  void *args_ptr);
void dl_esp32p4_s8_and4d_bchw_w1_16_w2_1_simdand(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_and4d_bchw_w1_1_w2_16_simdand(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_8_w2_8_simdand(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_8_w2_1_simdand(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_1_w2_8_simdand(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_and4d_bchw_w1_16_w2_16_simdand_unaligned(int8_t *output_ptr,
                                                            int8_t *input0_ptr,
                                                            int8_t *input1_ptr,
                                                            void *args_ptr);
void dl_esp32p4_s8_and4d_bchw_w1_16_w2_1_simdand_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s8_and4d_bchw_w1_1_w2_16_simdand_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_8_w2_8_simdand_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_8_w2_1_simdand_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_and4d_bchw_w1_1_w2_8_simdand_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);

void dl_esp32p4_s8_xor4d_bchw_w1_16_w2_16_simdxor(int8_t *output_ptr,
                                                  int8_t *input0_ptr,
                                                  int8_t *input1_ptr,
                                                  void *args_ptr);
void dl_esp32p4_s8_xor4d_bchw_w1_16_w2_1_simdxor(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_xor4d_bchw_w1_1_w2_16_simdxor(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_8_w2_8_simdxor(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_8_w2_1_simdxor(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_1_w2_8_simdxor(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_xor4d_bchw_w1_16_w2_16_simdxor_unaligned(int8_t *output_ptr,
                                                            int8_t *input0_ptr,
                                                            int8_t *input1_ptr,
                                                            void *args_ptr);
void dl_esp32p4_s8_xor4d_bchw_w1_16_w2_1_simdxor_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s8_xor4d_bchw_w1_1_w2_16_simdxor_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_8_w2_8_simdxor_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_8_w2_1_simdxor_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_xor4d_bchw_w1_1_w2_8_simdxor_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);

void dl_esp32p4_s8_or4d_bchw_w1_16_w2_16_simdor(int8_t *output_ptr,
                                                int8_t *input0_ptr,
                                                int8_t *input1_ptr,
                                                void *args_ptr);
void dl_esp32p4_s8_or4d_bchw_w1_16_w2_1_simdor(int8_t *output_ptr,
                                               int8_t *input0_ptr,
                                               int8_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s8_or4d_bchw_w1_1_w2_16_simdor(int8_t *output_ptr,
                                               int8_t *input0_ptr,
                                               int8_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_8_w2_8_simdor(int16_t *output_ptr,
                                               int16_t *input0_ptr,
                                               int16_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_8_w2_1_simdor(int16_t *output_ptr,
                                               int16_t *input0_ptr,
                                               int16_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_1_w2_8_simdor(int16_t *output_ptr,
                                               int16_t *input0_ptr,
                                               int16_t *input1_ptr,
                                               void *args_ptr);
void dl_esp32p4_s8_or4d_bchw_w1_16_w2_16_simdor_unaligned(int8_t *output_ptr,
                                                          int8_t *input0_ptr,
                                                          int8_t *input1_ptr,
                                                          void *args_ptr);
void dl_esp32p4_s8_or4d_bchw_w1_16_w2_1_simdor_unaligned(int8_t *output_ptr,
                                                         int8_t *input0_ptr,
                                                         int8_t *input1_ptr,
                                                         void *args_ptr);
void dl_esp32p4_s8_or4d_bchw_w1_1_w2_16_simdor_unaligned(int8_t *output_ptr,
                                                         int8_t *input0_ptr,
                                                         int8_t *input1_ptr,
                                                         void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_8_w2_8_simdor_unaligned(int16_t *output_ptr,
                                                         int16_t *input0_ptr,
                                                         int16_t *input1_ptr,
                                                         void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_8_w2_1_simdor_unaligned(int16_t *output_ptr,
                                                         int16_t *input0_ptr,
                                                         int16_t *input1_ptr,
                                                         void *args_ptr);
void dl_esp32p4_s16_or4d_bchw_w1_1_w2_8_simdor_unaligned(int16_t *output_ptr,
                                                         int16_t *input0_ptr,
                                                         int16_t *input1_ptr,
                                                         void *args_ptr);

void dl_esp32p4_s8_min4d_bchw_w1_16_w2_16_simdmin(int8_t *output_ptr,
                                                  int8_t *input0_ptr,
                                                  int8_t *input1_ptr,
                                                  void *args_ptr);
void dl_esp32p4_s8_min4d_bchw_w1_16_w2_1_simdmin(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_min4d_bchw_w1_1_w2_16_simdmin(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_8_w2_8_simdmin(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_8_w2_1_simdmin(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_1_w2_8_simdmin(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_min4d_bchw_w1_16_w2_16_simdmin_unaligned(int8_t *output_ptr,
                                                            int8_t *input0_ptr,
                                                            int8_t *input1_ptr,
                                                            void *args_ptr);
void dl_esp32p4_s8_min4d_bchw_w1_16_w2_1_simdmin_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s8_min4d_bchw_w1_1_w2_16_simdmin_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_8_w2_8_simdmin_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_8_w2_1_simdmin_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_min4d_bchw_w1_1_w2_8_simdmin_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);

void dl_esp32p4_s8_max4d_bchw_w1_16_w2_16_simdmax(int8_t *output_ptr,
                                                  int8_t *input0_ptr,
                                                  int8_t *input1_ptr,
                                                  void *args_ptr);
void dl_esp32p4_s8_max4d_bchw_w1_16_w2_1_simdmax(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_max4d_bchw_w1_1_w2_16_simdmax(int8_t *output_ptr,
                                                 int8_t *input0_ptr,
                                                 int8_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_8_w2_8_simdmax(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_8_w2_1_simdmax(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_1_w2_8_simdmax(int16_t *output_ptr,
                                                 int16_t *input0_ptr,
                                                 int16_t *input1_ptr,
                                                 void *args_ptr);
void dl_esp32p4_s8_max4d_bchw_w1_16_w2_16_simdmax_unaligned(int8_t *output_ptr,
                                                            int8_t *input0_ptr,
                                                            int8_t *input1_ptr,
                                                            void *args_ptr);
void dl_esp32p4_s8_max4d_bchw_w1_16_w2_1_simdmax_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s8_max4d_bchw_w1_1_w2_16_simdmax_unaligned(int8_t *output_ptr,
                                                           int8_t *input0_ptr,
                                                           int8_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_8_w2_8_simdmax_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_8_w2_1_simdmax_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);
void dl_esp32p4_s16_max4d_bchw_w1_1_w2_8_simdmax_unaligned(int16_t *output_ptr,
                                                           int16_t *input0_ptr,
                                                           int16_t *input1_ptr,
                                                           void *args_ptr);

void dl_esp32p4_s8_s8_requantize_linear(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_s16_requantize_linear(int8_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_s16_requantize_linear(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_esp32p4_s16_s8_requantize_linear(int16_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_dotprod_i16k16o16(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int shift, int n);
void dl_esp32p4_dotprod_i16k16o32(int32_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int shift, int n);
void dl_esp32p4_dotprod_i16k8o16(int16_t *output_ptr, int8_t *input0_ptr, int16_t *input1_ptr, int shift, int n);
void dl_esp32p4_dotprod_i8k8o16(int16_t *out2_int8, int8_t *in1_int8, int8_t *in2_int8, int shift, int n);
void dl_esp32p4_mat_vecs_i8k8o16(
    int16_t *output_ptr, int8_t *matrix, int8_t *vectors, int shift, int cols, int rows, int vectors_num);
void dl_esp32p4_dotprod_f32(float *output_ptr, float *input0_ptr, float *input1_ptr, int length);

/* Attention batched kernels: one call covers nq×tile_n (QK) or nq×D (P·V).
 * This specialized QK kernel keeps a complete D=64 Q row in PIE registers. */
void dl_esp32p4_s8_attention_qk_qacc_d64(int32_t *dots,
                                         const int8_t *q,
                                         const int8_t *k_blk,
                                         int nq,
                                         int tile_n,
                                         int score_stride,
                                         int32_t *row_max,
                                         int32_t *tmp16);
/* Outer-product Q·K for D=32: 16 t-lanes in QACC, d is the reduction.
 * k_blk must be t-block-major (see build_k_head_tblk_s8); tmp16 is scratch. */
void dl_esp32p4_s8_attention_qk_qacc_d32(int32_t *dots,
                                         const int8_t *q,
                                         const int8_t *k_blk,
                                         int nq,
                                         int tile_n,
                                         int score_stride,
                                         int32_t *row_max,
                                         int32_t *tmp16);
/* Generic D_pad (multiple of 16, <= 128) with the same K layout and contract. */
void dl_esp32p4_s8_attention_qk_qacc_generic(int32_t *dots,
                                             const int8_t *q,
                                             const int8_t *k_blk,
                                             int nq,
                                             int tile_n,
                                             int score_stride,
                                             int32_t *row_max,
                                             int32_t *tmp16,
                                             int D_pad);
/* Outer-product P·V: 16 d-lanes live in QACC, reduction runs along t, so
 * accumulator setup/readout is amortized over 16 outputs instead of 1.
 * v_blk must be d-block-major (see load_v_tile_dblk_s8). Walks D/16 d-groups,
 * so callers pass D_pad and rely on the packer zero-filling the tail group;
 * acc must have room for D_pad lanes. tmp16 is 16 int32 of 16-byte-aligned
 * scratch. */
void dl_esp32p4_s8_attention_pv_qacc(
    float *acc, const int8_t *e, const int8_t *v_blk, const float *deq, int nq, int D, int acc_stride, int32_t *tmp16);
void dl_esp32p4_s8_attention_pv_qacc_i32(
    int32_t *acc, const int8_t *e, const int8_t *v_blk, int nq, int D, int acc_stride, int32_t *tmp16);
/* e[t] = table[min(m_new - dots[t], sat) >> shift]. The index is a shift
 * rather than a multiply, so the clamp/index chain runs 4 lanes wide (PIE has
 * no 32-bit vector multiply). ``table`` must be the 512-entry shift table built for the same scale, with
 * each int8 entry in its own int32 slot: ESP.LDXQ.32 scales its index by 4,
 * so a word-wide table is gathered with idx directly (an int8 table would
 * need idx>>2 plus a byte select). ``e`` is a 16-byte-aligned 64-byte row,
 * ``dots`` a 64-int32 row; e[n..63] is zero-filled and the return value is
 * sum(e[0:64]). ``scratch`` is 272 bytes, 16-byte aligned. */
int32_t dl_esp32p4_s8_attention_exp_lut_shift(int8_t *e,
                                              const int32_t *dots,
                                              int n,
                                              int32_t m_new,
                                              int32_t sat,
                                              int32_t shift,
                                              const int32_t *table,
                                              int32_t *scratch);

/* int16 QACC batched QK for D=32.
 * Layout of k_blk matches build_k_head_tblk_s16 (8-token t-block groups).
 * tmp16 must be a 16 B-aligned scratch of >= 16 int32 (64 B): the kernel
 * dumps all four st.qacc halves (8 sign-extended s64) into it.
 * scores[t] = float(round_half_even((Q.K)[t] / 2^INT16_QACC_QK_D32_SHIFT)) *
 * dot_scale: the integer step equals the per-row dot under CFG HALF_EVEN, and
 * is needed because a D=32 int16 dot takes up to 36 bits. Pass
 * dot_scale = logit_scale * 2^INT16_QACC_QK_D32_SHIFT. Each scores row must
 * hold tile_n rounded up to 8. */
void dl_esp32p4_s16_attention_qk_qacc_d32(float *scores,
                                          const int16_t *q,
                                          const int16_t *k_blk,
                                          int nq,
                                          int tile_n,
                                          int score_stride,
                                          float dot_scale,
                                          int32_t *tmp16);
/* Generic logical D (1..128) form of the kernel above. Q and K rows are
 * physically padded to D_pad=round_up(D, 8); rounding still uses
 * int16_dot_shift(D), so pass dot_scale = logit_scale * 2^int16_dot_shift(D). */
void dl_esp32p4_s16_attention_qk_qacc_generic(float *scores,
                                              const int16_t *q,
                                              const int16_t *k_blk,
                                              int nq,
                                              int tile_n,
                                              int score_stride,
                                              int32_t *tmp16,
                                              int D_real,
                                              float dot_scale);
/* int16 QACC P·V on split weights (see load_v_tile_dblk_s16 / softmax_exp_row_split_s16):
 * acc[iq][d] += scale * sum_t (e_hi[t] * 2^15 + e_lo[t]) V[t][d].
 * n_tok and D multiples of 8; e rows are [e_hi[n_tok], e_lo[n_tok]];
 * tmp32 is 16 B aligned with >= 32 int32. */
void dl_esp32p4_s16_attention_pv_qacc(float *acc,
                                      const int16_t *e,
                                      const int16_t *v_dblk,
                                      int nq,
                                      int n_tok,
                                      int D,
                                      int acc_stride,
                                      int32_t *tmp32,
                                      float scale);

/* Native row-major A[M,K] x B[K,N] MatMul kernels. */
void dl_esp32p4_s8_matmul_kn(const void *args);
void dl_esp32p4_s16_matmul_kn(const void *args);

int32_t dl_esp32p4_reduce_l2_s8_aligned(int8_t *input, int32_t size);
int64_t dl_esp32p4_reduce_l2_s16_aligned(int16_t *input, int32_t size);
int8_t dl_esp32p4_reduce_max_s8_aligned(int8_t *input, int32_t size);
int16_t dl_esp32p4_reduce_max_s16_aligned(int16_t *input, int32_t size);
int8_t dl_esp32p4_reduce_min_s8_aligned(int8_t *input, int32_t size);
int16_t dl_esp32p4_reduce_min_s16_aligned(int16_t *input, int32_t size);
int8_t dl_esp32p4_reduce_abs_max_s8_aligned(int8_t *input, int32_t size);
int16_t dl_esp32p4_reduce_abs_max_s16_aligned(int16_t *input, int32_t size);
int32_t dl_esp32p4_reduce_sum_s8_aligned(int8_t *input, int32_t size);
int64_t dl_esp32p4_reduce_sum_s16_aligned(int16_t *input, int32_t size);
int32_t dl_esp32p4_reduce_l1_s8_aligned(int8_t *input, int32_t size);
int64_t dl_esp32p4_reduce_l1_s16_aligned(int16_t *input, int32_t size);

void dl_esp32p4_s8_avg_pool2d_hwc_sum(int32_t *buffer_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_rmsnorm_s8(int8_t *output, int8_t *input, float *scale, float *rms, int n);
void dl_esp32p4_rmsnorm_s16(int16_t *output, int16_t *input, float *scale, float *rms, int n);

/* LUT API */
void dl_esp32p4_s8_lut(int8_t *output, int8_t *input, int32_t n_8, int8_t *table);
void dl_esp32p4_s16_lut_nearest_neighbor(int16_t *output, int16_t *input, int32_t n_8, int16_t *table, int32_t shift);

/* INT8 transpose kernels.
 *   dl_esp32p4_s8_transpose: N×M byte matrix, N%8==0, M%16==0.
 *   dl_esp32p4_block_transpose: N×M grid of K-byte blocks, any K>=1.
 */
void dl_esp32p4_s8_transpose(int8_t *output, const int8_t *input, int N, int M);
void dl_esp32p4_block_transpose(int8_t *output, const int8_t *input, int N, int M, int K);

/* INT16 transpose kernels.
 *   dl_esp32p4_s16_transpose: N×M int16 matrix, N%8==0, M%8==0.
 *   dl_esp32p4_s16_block_transpose: N×M grid of K-element blocks, any K>=1.
 */
void dl_esp32p4_s16_transpose(int16_t *output, const int16_t *input, int N, int M);
void dl_esp32p4_s16_block_transpose(int16_t *output, const int16_t *input, int N, int M, int K);

/* Non-interleaved RotaryEmbedding (RoPE) kernels. */
void dl_esp32p4_s8_rope_neox_simd(int8_t *y_first,
                                  int8_t *y_second,
                                  const int8_t *x_first,
                                  const int8_t *x_second,
                                  const int8_t *cos_row,
                                  const int8_t *sin_row,
                                  int iters,
                                  int mac_shift);
void dl_esp32p4_s16_rope_neox_simd(int16_t *y_first,
                                   int16_t *y_second,
                                   const int16_t *x_first,
                                   const int16_t *x_second,
                                   const int16_t *cos_row,
                                   const int16_t *sin_row,
                                   int iters,
                                   int mac_shift);
void dl_esp32p4_s8_s16c_rope_neox_simd(int8_t *y_first,
                                       int8_t *y_second,
                                       const int8_t *x_first,
                                       const int8_t *x_second,
                                       const int16_t *cos_row,
                                       const int16_t *sin_row,
                                       int iters,
                                       int mac_shift);
void dl_esp32p4_s8_s16c8_rope_neox_simd(int8_t *y_first,
                                        int8_t *y_second,
                                        const int8_t *x_first,
                                        const int8_t *x_second,
                                        const int16_t *cos_row,
                                        const int16_t *sin_row,
                                        int iters,
                                        int mac_shift);
}
