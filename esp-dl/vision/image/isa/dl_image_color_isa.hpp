#pragma once
#include <cstdint>

extern "C" {
void cvt_color_simd_helper_rgb5652rgb565(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565le2rgb565be(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565le2bgr565le(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565be2bgr565be(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565le2bgr565be(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565be2bgr565le(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb565le2rgb888(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565be2rgb888(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565le2bgr888(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565be2bgr888(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb565le2gray(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb565be2gray(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_bgr565le2gray(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_bgr565be2gray(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb8882rgb888(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb8882bgr888(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb8882rgb565le(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_bgr8882rgb565le(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_rgb8882rgb565be(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_bgr8882rgb565be(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb8882gray(uint8_t *src, uint8_t *dst, int n);
void cvt_color_simd_helper_bgr8882gray(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_gray2gray(uint8_t *src, uint8_t *dst, int n);

void cvt_color_simd_helper_rgb8882hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void cvt_color_simd_helper_bgr8882hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);

void cvt_color_simd_helper_rgb565le2hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void cvt_color_simd_helper_rgb565be2hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void cvt_color_simd_helper_bgr565le2hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void cvt_color_simd_helper_bgr565be2hsv(uint8_t *src, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);

void cvt_color_simd_helper_rgb565le2rgb888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2rgb888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565le2bgr888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2bgr888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565le2rgb888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2rgb888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565le2bgr888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2bgr888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);

void cvt_color_simd_helper_rgb565le2gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr565le2gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr565be2gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565le2gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb565be2gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr565le2gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr565be2gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);

void cvt_color_simd_helper_rgb8882rgb888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb8882bgr888_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb8882rgb888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb8882bgr888_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);

void cvt_color_simd_helper_rgb8882gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr8882gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_rgb8882gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_bgr8882gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);

void cvt_color_simd_helper_gray2gray_qint8(uint8_t *src, uint8_t *dst, int n, int *lut);
void cvt_color_simd_helper_gray2gray_qint16(uint8_t *src, uint8_t *dst, int n, int *lut);

void resize_nn_simd_helper_rgb5652rgb565(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565le2rgb565be(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565le2bgr565le(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565be2bgr565be(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565le2bgr565be(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565be2bgr565le(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb565le2rgb888(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565be2rgb888(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565le2bgr888(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565be2bgr888(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb565le2gray(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb565be2gray(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_bgr565le2gray(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_bgr565be2gray(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb8882rgb888(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb8882bgr888(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb8882rgb565le(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_bgr8882rgb565le(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_rgb8882rgb565be(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_bgr8882rgb565be(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb8882gray(uint8_t **src, int *offsets, uint8_t *dst, int n);
void resize_nn_simd_helper_bgr8882gray(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_gray2gray(uint8_t **src, int *offsets, uint8_t *dst, int n);

void resize_nn_simd_helper_rgb8882hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void resize_nn_simd_helper_bgr8882hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);

void resize_nn_simd_helper_rgb565le2hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void resize_nn_simd_helper_rgb565be2hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void resize_nn_simd_helper_bgr565le2hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);
void resize_nn_simd_helper_bgr565be2hsv(uint8_t **src, int *offsets, uint8_t *dst, int n, int *sdiv_lut, int *hdiv_lut);

void resize_nn_simd_helper_rgb565le2rgb888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2rgb888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565le2bgr888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2bgr888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565le2rgb888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2rgb888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565le2bgr888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2bgr888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);

void resize_nn_simd_helper_rgb565le2gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr565le2gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr565be2gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565le2gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb565be2gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr565le2gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr565be2gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);

void resize_nn_simd_helper_rgb8882rgb888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb8882bgr888_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb8882rgb888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb8882bgr888_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);

void resize_nn_simd_helper_rgb8882gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr8882gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_rgb8882gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_bgr8882gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);

void resize_nn_simd_helper_gray2gray_qint8(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
void resize_nn_simd_helper_gray2gray_qint16(uint8_t **src, int *offsets, uint8_t *dst, int n, int *lut);
}
