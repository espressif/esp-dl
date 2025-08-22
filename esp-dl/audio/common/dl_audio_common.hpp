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
#define M_2PI 6.283185307179586476925286766559005

/**
 * @brief Window function type enumeration.
 */
enum class WinType {
    HANNING,     /*!< Hanning window */
    HANN,        /*!< Hanning window, see https://pytorch.org/docs/stable/generated/torch.hann_window.html */
    SINE,        /*!< Sine window */
    HAMMING,     /*!< Hamming window */
    POVEY,       /*!< Povey window, used in Kaldi */
    BLACKMAN,    /*!< Blackman window */
    RECTANGULAR, /*!< Rectangular window */
    NONE,        /*!< No window function */
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
 * This function allocates and initializes the window function data based on the specified type and length.
 * The caller is responsible for freeing the allocated memory.
 *
 * @param win_type Window type, specified using the WinType enum.
 * @param win_len  Length of the window function.
 * @param caps     Memory allocation capabilities (default is MALLOC_CAP_DEFAULT).
 * @param blackman_coeff Coefficient for Blackman window (default is 0.42f, ignored for other window types).
 * @return float*  Pointer to the allocated window data. Returns nullptr on failure or invalid arguments.
 */
float *win_func_init(WinType win_type, int win_len, uint32_t caps = MALLOC_CAP_DEFAULT, float blackman_coeff = 0.42f);

/**
 * @brief Convert a string to the corresponding window type enum.
 *
 * @param str Window type string (e.g., "hanning", "sine").
 * @return WinType Corresponding window type enum. Returns WinType::NONE if not recognized.
 */
WinType win_type_from_string(const char *str);

/**
 * @brief Convert a window type enum to its corresponding string representation.
 *
 * @param win_type Window type enum value.
 * @return const char* String representation of the window type. Returns "none" if not recognized.
 */
const char *win_type_to_string(WinType win_type);

/**
 * @brief Calculate the number of frames for audio processing.
 *
 * @param input_len Length of input audio data.
 * @param win_len   Window length.
 * @param win_step  Window step size.
 * @return int      Number of frames.
 */
int get_frame_num(int input_len, int win_len, int win_step);

/**
 * @brief Calculate the next power of 2 for a given number.
 *
 * @param x Input number.
 * @return uint32_t Next power of 2.
 */
uint32_t next_power_of_2(uint32_t x);

/**
 * @brief Remove DC offset from audio signal.
 *
 * @param x Input/output signal.
 * @param n Length of signal.
 */
void remove_dc_offset(float *x, int n);

/**
 * @brief Compute energy of audio signal and return its logarithm.
 *
 * @param x       Input signal.
 * @param len     Length of signal.
 * @param epsilon Minimum value for logarithm to avoid log(0).
 * @return float  Logarithm of energy.
 */
float compute_energy(float *x, int len, float epsilon);

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
 * @param x             Input signal.
 * @param win_len           Window length.
 * @param preemphasis_coeff Preemphasis coefficient.
 * @param prev              Previous sample for preemphasis.
 * @return float*           Pointer to output.
 */
float *apply_preemphasis(float *x, int win_len, float preemphasis_coeff, float prev);

/**
 * @brief Apply window function and preemphasis to input signal.
 *
 * @param x             Input signal.
 * @param win_len           Window length.
 * @param win_func          Window function.
 * @return float*           Pointer to output.
 */
float *apply_window(float *x, int win_len, float *win_func);
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
 * @brief Compute spectrum.
 *
 * @param x         Input signal.
 * @param win_len   Window length.
 * @param use_power If true, compute power spectrum, else magnitude.
 * @return float*   Pointer to output spectrum.
 */
float *compute_spectrum(float *x, int win_len, bool use_power);

/**
 * @brief Dot product of two float arrays.
 *
 * @param x1 First array.
 * @param x2 Second array.
 * @param len Length of arrays.
 * @return float Dot product result.
 */
float dotprod_f32(float *x1, float *x2, int len);

} // namespace audio
} // namespace dl
