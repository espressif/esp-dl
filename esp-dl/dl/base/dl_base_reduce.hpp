#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

/**
 * @brief Compute the sum of squares (L2 norm without the final sqrt) of int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int32_t to avoid overflow.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of squares (int32_t).
 */
int32_t reduce_l2(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the sum of squares (L2 norm without the final sqrt) of int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int64_t to avoid overflow
 * (each int16 square can reach ~2^30).
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of squares (int64_t).
 */
int64_t reduce_l2(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the sum of squares (L2 norm without the final sqrt) of float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. No SIMD acceleration is available for the float path.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a 4-way unrolled scalar
 *               path is taken.
 * @return Sum of squares (float).
 */
float reduce_l2(float *input, int32_t size, int32_t stride);

/**
 * @brief Find the maximum value among int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to INT8_MIN.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Maximum element value (int8_t).
 */
int8_t reduce_max(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the maximum value among int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to INT16_MIN.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Maximum element value (int16_t).
 */
int16_t reduce_max(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the minimum value among int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running minimum to INT8_MAX.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Minimum element value (int8_t).
 */
int8_t reduce_min(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the minimum value among int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running minimum to INT16_MAX.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Minimum element value (int16_t).
 */
int16_t reduce_min(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the maximum value among float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to -INFINITY.
 * No SIMD acceleration is available for the float path.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a 4-way unrolled scalar
 *               path is taken.
 * @return Maximum element value (float).
 */
float reduce_max(float *input, int32_t size, int32_t stride);

/**
 * @brief Find the minimum value among float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running minimum to INFINITY.
 * No SIMD acceleration is available for the float path.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a 4-way unrolled scalar
 *               path is taken.
 * @return Minimum element value (float).
 */
float reduce_min(float *input, int32_t size, int32_t stride);

/**
 * @brief Find the element with the maximum absolute value among int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to 0.
 * Handles INT8_MIN specially: since |INT8_MIN| = 128 does not fit in a positive
 * int8_t, it is clamped to INT8_MAX (127). For all other negative values the
 * standard negation is used.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Maximum absolute element value (int8_t, non-negative).
 */
int8_t reduce_abs_max(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the element with the maximum absolute value among int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to 0.
 * Handles INT16_MIN specially: since |INT16_MIN| does not fit in a positive
 * int16_t, it is clamped to INT16_MAX. For all other negative values the
 * standard negation is used.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Maximum absolute element value (int16_t, non-negative).
 */
int16_t reduce_abs_max(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Find the element with the maximum absolute value among float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. Initialises the running maximum to 0.0f. Uses fabsf()
 * to compute the absolute value of each element.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a 4-way unrolled scalar
 *               path is taken.
 * @return Maximum absolute element value (float, non-negative).
 */
float reduce_abs_max(float *input, int32_t size, int32_t stride);

/**
 * @brief Compute the sum of int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int32_t to avoid overflow.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of elements (int32_t).
 */
int32_t reduce_sum(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the sum of int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int64_t to avoid overflow.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of elements (int64_t).
 */
int64_t reduce_sum(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the sum of float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. No SIMD acceleration is available for the float path.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a 4-way unrolled scalar
 *               path is taken.
 * @return Sum of elements (float).
 */
float reduce_sum(float *input, int32_t size, int32_t stride);

/**
 * @brief Compute the L1 norm (sum of absolute values) of int8 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int32_t to avoid overflow.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of absolute values (int32_t).
 */
int32_t reduce_l1(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the L1 norm (sum of absolute values) of int16 elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. The result is accumulated as int64_t to avoid overflow.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a fast SIMD path may be taken.
 * @return Sum of absolute values (int64_t).
 */
int64_t reduce_l1(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the L1 norm (sum of absolute values) of float elements.
 *
 * Scans @p size elements starting from @p input, advancing by @p stride elements
 * between each sample. No SIMD acceleration is available for the float path.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 *               When stride == 1 the elements are contiguous and a plain loop is taken.
 * @return Sum of absolute values (float).
 */
float reduce_l1(float *input, int32_t size, int32_t stride);

/**
 * @brief Compute the mean of int8 elements.
 *
 * Internally calls reduce_sum() to get the int32_t sum, then rounds to the nearest
 * int8_t by adding half the size before division. This is a convenience wrapper
 * around reduce_sum(); the module-level ReduceMean provides full quantization-aware
 * mean computation.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 * @return Mean value rounded to nearest (int8_t).
 */
int8_t reduce_mean(int8_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the mean of int16 elements.
 *
 * Internally calls reduce_sum() to get the int64_t sum, then rounds to the nearest
 * int16_t by adding half the size before division. This is a convenience wrapper
 * around reduce_sum(); the module-level ReduceMean provides full quantization-aware
 * mean computation.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 * @return Mean value rounded to nearest (int16_t).
 */
int16_t reduce_mean(int16_t *input, int32_t size, int32_t stride);

/**
 * @brief Compute the mean of float elements.
 *
 * Internally calls reduce_sum() and divides by @p size. This is a convenience
 * wrapper around reduce_sum(); the module-level ReduceMean provides full
 * quantization-aware mean computation for quantized types.
 *
 * @param input Pointer to the first element.
 * @param size  Number of elements to reduce.
 * @param stride Spacing between consecutive elements in memory.
 * @return Mean value (float).
 */
float reduce_mean(float *input, int32_t size, int32_t stride);

} // namespace base
} // namespace dl
