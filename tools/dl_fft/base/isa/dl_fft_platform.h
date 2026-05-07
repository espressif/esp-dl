#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_IDF_TARGET_ESP32
void dl_fft2r_fc32_ae32_(float *data, int N, float *table);
void dl_ifft2r_fc32_ae32_(float *data, int N, float *table);
void dl_fft4r_fc32_ae32_(float *data, int N, float *table, int table_size);
void dl_ifft4r_fc32_ae32_(float *data, int N, float *table, int table_size);

#elif CONFIG_IDF_TARGET_ESP32S3
void dl_fft2r_fc32_aes3_(float *data, int N, float *table);
void dl_ifft2r_fc32_aes3_(float *data, int N, float *table);
void dl_fft4r_fc32_aes3_(float *data, int N, float *table, int table_size);
void dl_ifft4r_fc32_aes3_(float *data, int N, float *table, int table_size);
int16_t dl_fft_reduce_max_aes3_(const int16_t *inputs, int len);
int16_t dl_reduce_abs_max_aes3_(const int16_t *inputs, int len);
void dl_bitrev2r_sc16_aes3_(int16_t *data, int cpx_points, int log2N);
esp_err_t dl_rfft_post_proc_sc16_aes3_(int16_t *data, int N, int16_t *table);
esp_err_t dl_rfft_pre_proc_sc16_aes3_(int16_t *data, int N, int16_t *table);
void test_fftr_s16(int16_t *, int16_t *, int16_t);
void test_ffti_s16(int16_t *, int16_t *, int16_t);
void dl_fft2r_sc16_dif_aes3_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
void dl_ifft2r_sc16_dif_aes3_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
void dl_fft2r_sc16_dif_hp_aes3_(int16_t *data, int16_t *table, int num_stages, int cpx_points, int *out_shift);
void dl_ifft2r_sc16_dif_hp_aes3_(int16_t *data, int16_t *table, int num_stages, int cpx_points, int *out_shift);
int test_radix2_fft_bf_s16_hp(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
#elif CONFIG_IDF_TARGET_ESP32P4
void dl_fft2r_fc32_arp4_(float *data, int N, float *table);
void dl_ifft2r_fc32_arp4_(float *data, int N, float *table);
void dl_fft4r_fc32_arp4_(float *data, int N, float *table, int table_size);
void dl_ifft4r_fc32_arp4_(float *data, int N, float *table, int table_size);
int16_t dl_fft_reduce_max_arp4_(const int16_t *inputs, int len);
int16_t dl_reduce_abs_max_arp4_(const int16_t *inputs, int len);
void dl_bitrev2r_sc16_arp4_(int16_t *data, int cpx_points, int log2N);
void dl_fft2r_sc16_dif_arp4_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
void dl_ifft2r_sc16_dif_arp4_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
void dl_fft2r_sc16_dif_hp_arp4_(int16_t *data, int16_t *table, int num_stages, int cpx_points, int *out_shift);
void dl_ifft2r_sc16_dif_hp_arp4_(int16_t *data, int16_t *table, int num_stages, int cpx_points, int *out_shift);
#endif

#ifdef __cplusplus
}
#endif
