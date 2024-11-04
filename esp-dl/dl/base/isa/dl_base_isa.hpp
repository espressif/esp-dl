#pragma once

extern "C" {
#if CONFIG_XTENSA_BOOST
void dl_xtensa_s16_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_xtensa_s16_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_xtensa_s16_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_xtensa_s16_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
#endif

#if CONFIG_TIE728_BOOST
void dl_tie728_s16_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_11cn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_11cn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_unaligned_conv2d_11cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_bias_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_11cn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_33cn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_33cn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_unaligned_conv2d_33cn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_bias_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_33cn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_hwcn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_conv2d_hwcn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_unaligned_conv2d_hwcn(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_bias_leakyrelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_conv2d_hwcn_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_depthwise_conv2d_33c1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_33c1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_33c1_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_33c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_33c1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_33c1_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_unaligned_depthwise_conv2d_33c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_33c1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_33c1_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_33c1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_33c1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_33c1_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_depthwise_conv2d_hwc1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_hwc1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_hwc1_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_hwc1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_depthwise_conv2d_hwc1_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1_bias(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1_bias_relu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_depthwise_conv2d_hwc1_bias_prelu(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_max_pool2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_max_pool2d_22c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_max_pool2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_max_pool2d_22c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_avg_pool2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_avg_pool2d_22c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_avg_pool2d_hwc1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_avg_pool2d_22c1(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_add2d_11c_relu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_add2d_11c_prelu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_rescale_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_rescale_add2d_11c_relu(int16_t *output_ptr,
                                          int16_t *input0_ptr,
                                          int16_t *input1_ptr,
                                          void *args_ptr);
void dl_tie728_s16_rescale_add2d_11c_prelu(int16_t *output_ptr,
                                           int16_t *input0_ptr,
                                           int16_t *input1_ptr,
                                           void *args_ptr);
void dl_tie728_s16_unaligned_add2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_add2d_11c_relu(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_tie728_s16_unaligned_add2d_11c_prelu(int16_t *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);

void dl_tie728_s16_sub2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_sub2d_11c_relu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_sub2d_11c_prelu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_rescale_sub2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_rescale_sub2d_11c_relu(int16_t *output_ptr,
                                          int16_t *input0_ptr,
                                          int16_t *input1_ptr,
                                          void *args_ptr);
void dl_tie728_s16_rescale_sub2d_11c_prelu(int16_t *output_ptr,
                                           int16_t *input0_ptr,
                                           int16_t *input1_ptr,
                                           void *args_ptr);
void dl_tie728_s16_unaligned_sub2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_sub2d_11c_relu(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_tie728_s16_unaligned_sub2d_11c_prelu(int16_t *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);

void dl_tie728_s16_mul2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_mul2d_11c_relu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_mul2d_11c_prelu(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_mul2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_mul2d_11c_relu(int16_t *output_ptr,
                                            int16_t *input0_ptr,
                                            int16_t *input1_ptr,
                                            void *args_ptr);
void dl_tie728_s16_unaligned_mul2d_11c_prelu(int16_t *output_ptr,
                                             int16_t *input0_ptr,
                                             int16_t *input1_ptr,
                                             void *args_ptr);

void dl_tie728_s16_max2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_max2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);

void dl_tie728_s16_min2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_min2d_11c(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, void *args_ptr);

void dl_tie728_s16_relu_11c(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_relu_11c(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

void dl_tie728_s16_prelu_11c(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);
void dl_tie728_s16_unaligned_prelu_11c(int16_t *output_ptr, int16_t *input_ptr, void *args_ptr);

/* Int8 API */
void dl_tie728_s8_conv2d_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_11cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_11cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_conv2d_11cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_conv2d_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_33cn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_33cn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_conv2d_33cn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_conv2d_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_hwcn_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_conv2d_hwcn_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_conv2d_hwcn(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_depthwise_conv2d_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_depthwise_conv2d_33c1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_depthwise_conv2d_33c1_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_depthwise_conv2d_33c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_depthwise_conv2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_depthwise_conv2d_hwc1_relu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_depthwise_conv2d_hwc1_prelu(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_depthwise_conv2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_max_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_max_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_max_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_max_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_avg_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_avg_pool2d_22c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_avg_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_avg_pool2d_hwc1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_add2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_add2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_add2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_add2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_add2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_tie728_s8_sub2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_sub2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_sub2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_sub2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_sub2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_rescale_sub2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_sub2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_sub2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_sub2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_tie728_s8_mul2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_mul2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_mul2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_mul2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_mul2d_11c_relu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_mul2d_11c_prelu(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_tie728_s8_max2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_max2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_tie728_s8_min2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_min2d_11c(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, void *args_ptr);

void dl_tie728_s8_relu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_relu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_tie728_s8_resize2d_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_tie728_s8_unaligned_resize2d_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
#endif

#if CONFIG_IDF_TARGET_ESP32P4
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

void dl_esp32p4_s8_resize2d_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_resize2d_nearest_2x2_c1(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

void dl_esp32p4_s8_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);
void dl_esp32p4_s8_unaligned_prelu_11c(int8_t *output_ptr, int8_t *input_ptr, void *args_ptr);

#endif
}
