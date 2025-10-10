#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {
/**
 * @brief Computes the dot product of two int8_t arrays and stores the result in the output int16_t array.
 *
 * @param input0_ptr Pointer to the first input array.
 * @param input1_ptr Pointer to the second input array.
 * @param output_ptr Pointer to the output array to store the computed result.
 * @param length Length of the input arrays (number of elements).
 * @param shift Number of bits to right-shift the result for precision or range adjustment.
 *
 * @note This function assumes the input arrays have the same length and the output array has sufficient allocated
 * space.
 */
void dotprod(int8_t *input0_ptr, int8_t *input1_ptr, int16_t *output_ptr, int length, int shift = 0);
void dotprod(int8_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift = 0);
void dotprod(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift = 0);
void dotprod(float *input0_ptr, float *input1_ptr, float *output_ptr, int length, int shift = 0);

/**
 * @brief Performs matrix-vector dot product operation.
 *
 * @tparam T The data type of the elements in the matrix and vector (e.g., float, double, int).
 * @param matrix Pointer to the input matrix stored in row-major order.
 * @param vector Pointer to the input vector.
 * @param result Pointer to the output vector where the result will be stored.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @param shift Optional parameter to apply a shift to the result (default is 0).
 */

template <typename TM, typename TV, typename TO>
void mat_vec_dotprod(TM *matrix, TV *vector, TO *result, int rows, int cols, int shift = 0)
{
    for (int i = 0; i < rows; i++) {
        TM *matrix_row = matrix + i * cols;
        dotprod(matrix_row, vector, result + i, cols, shift);
    }
}
template void mat_vec_dotprod(int8_t *matrix, int8_t *vector, int16_t *result, int rows, int cols, int shift);
template void mat_vec_dotprod(int8_t *matrix, int16_t *vector, int16_t *result, int rows, int cols, int shift);
template void mat_vec_dotprod(int16_t *matrix, int16_t *vector, int16_t *result, int rows, int cols, int shift);
template void mat_vec_dotprod(float *matrix, float *vector, float *result, int rows, int cols, int shift);
} // namespace base
} // namespace dl
