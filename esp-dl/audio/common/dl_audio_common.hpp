#pragma once
#include "dl_fft.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace dl {
namespace audio {

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define M_PI 3.14159265358979323846

/**
 * @brief Window function type enumeration.
 */
enum class WinType {
    HANNING,     /*!< Hanning window */
    SINE,        /*!< Sine window */
    HAMMING,     /*!< Hamming window */
    POVEY,       /*!< Povey window, used in Kaldi */
    RECTANGULAR, /*!< Rectangular window */
    WN9,         /*!< Special predefined window */
    UNKNOWN      /*!< Unknown type */
};

/**
 * @brief Mel filter structure
 */
typedef struct {
    float *coeff;  /*!< Filter coefficients */
    int *bank_pos; /*!< Filter bank positions */
    int nfilter;   /*!< Number of filters */
} mel_filter_t;

/**
 * @brief Initialize window function data of the specified type.
 *
 * @param win_type Window type, use WinType enum.
 * @param win_len  Window length.
 * @return float*  Pointer to the allocated window data. Caller must free it. Returns nullptr on failure or invalid
 * arguments.
 */
float *win_func_init(WinType win_type, int win_len);

/**
 * @brief Convert a string to the corresponding window type enum.
 *
 * @param str Window type string (e.g., "hanning", "sine").
 * @return WinType Corresponding window type enum. Returns WinType::UNKNOWN if not recognized.
 */
WinType win_type_from_string(const char *str);

/**
 * @brief Initialize mel filter data.
 *
 * @param nfft          FFT size.
 * @param nfilter       Number of mel filterbanks.
 * @param low_freq      Lowest frequency.
 * @param high_freq     Highest frequency.
 * @param sample_rate   Sample rate.
 * @param caps          Memory allocation capabilities.
 *
 * @return mel_filter_t* Pointer to the allocated mel filter data. Caller must free it. Returns nullptr on failure or
 * invalid arguments.
 */
mel_filter_t *mel_filter_init(int nfft, int nfilter, int low_freq, int high_freq, int sample_rate, uint32_t caps);

/**
 * @brief Deinitialize mel filter data.
 *
 * @param mel_filter Pointer to mel filter data to deinitialize.
 */
void mel_filter_deinit(mel_filter_t *mel_filter);

/**
 * @brief Apply window function and preemphasis to input signal.
 *
 * @param input             Input signal.
 * @param win_len           Window length.
 * @param output            Output signal.
 * @param preemphasis_coeff Preemphasis coefficient.
 * @param win_func          Window function.
 * @param prev              Previous sample for preemphasis.
 * @return float*           Pointer to output.
 */
float *apply_window_and_preemphasis(
    const float *input, int win_len, float *output, float preemphasis_coeff, float *win_func, float prev = 0);

/**
 * @brief Apply mel filterbank to spectrum.
 *
 * @param x         Input spectrum.
 * @param mel_filter Mel filter data.
 * @param output    Output mel spectrum.
 * @return float*   Pointer to output.
 */
float *mel_dotprod(float *x, mel_filter_t *mel_filter, float *output);

/**
 * @brief Convert Hz to Mel scale.
 *
 * @param x Frequency in Hz.
 * @return float Frequency in Mel.
 */
float hz2mel(float x);

/**
 * @brief Convert Mel to Hz scale.
 *
 * @param x Frequency in Mel.
 * @return float Frequency in Hz.
 */
float mel2hz(float x);

/**
 * @brief Compute spectrum from time domain signal.
 *
 * @param fft_handle FFT handle.
 * @param x         Input signal.
 * @param win_len   Window length.
 * @param use_power If true, compute power spectrum, else magnitude.
 * @return float*   Pointer to output spectrum.
 */
float *compute_spectrum(FFT *fft_handle, float *x, int win_len, bool use_power);

/**
 * @brief Dot product of two float arrays.
 *
 * @param x1 First array.
 * @param x2 Second array.
 * @param y  Output (accumulated result).
 * @param len Length of arrays.
 */
void dotprod_f32(float *x1, float *x2, float *y, int len);
} // namespace audio
} // namespace dl
