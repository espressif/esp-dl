#pragma once

extern "C" {
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
}
