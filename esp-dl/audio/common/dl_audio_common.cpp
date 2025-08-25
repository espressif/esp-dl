#include "dl_audio_common.hpp"

namespace dl {
namespace audio {

float *win_func_init(WinType win_type, int win_len, uint32_t caps, float blackman_coeff)
{
    if (win_len <= 0)
        return nullptr;

    float *window_data = (float *)heap_caps_malloc(sizeof(float) * win_len, caps);
    if (!window_data)
        return nullptr;

    double a = 2.0f * M_PI / (win_len - 1);

    if (win_type == WinType::HANN) {
        a = 2.0f * M_PI / win_len;
    }

    for (int i = 0; i < win_len; i++) {
        double i_fl = static_cast<double>(i);
        switch (win_type) {
        case WinType::HANNING:
            window_data[i] = 0.5f - 0.5f * cosf(a * i_fl);
            break;
        case WinType::SINE:
            window_data[i] = sinf(0.5f * a * i_fl);
            break;
        case WinType::HAMMING:
            window_data[i] = 0.54f - 0.46f * cosf(a * i_fl);
            break;
        case WinType::HANN:
            window_data[i] = 0.5f - 0.5f * cosf(a * i_fl);
            break;
        case WinType::POVEY:
            window_data[i] = powf(0.5f - 0.5f * cosf(a * i_fl), 0.85f);
            break;
        case WinType::RECTANGULAR:
            window_data[i] = 1.0f;
            break;
        case WinType::BLACKMAN:
            window_data[i] = blackman_coeff - 0.5f * cosf(a * i_fl) + (0.5 - blackman_coeff) * cosf(2.0f * a * i_fl);
            break;
        default:
            free(window_data);
            return nullptr;
        }
    }

    return window_data;
}

// 可选：字符串转枚举辅助函数
WinType win_type_from_string(const char *str)
{
    if (!str)
        return WinType::NONE;
    if (strcmp(str, "hanning") == 0)
        return WinType::HANNING;
    if (strcmp(str, "sine") == 0)
        return WinType::SINE;
    if (strcmp(str, "hamming") == 0)
        return WinType::HAMMING;
    if (strcmp(str, "povey") == 0)
        return WinType::POVEY;
    if (strcmp(str, "rectangular") == 0)
        return WinType::RECTANGULAR;
    if (strcmp(str, "NONE") == 0)
        return WinType::NONE;
    return WinType::NONE;
}

const char *win_type_to_string(WinType win_type)
{
    switch (win_type) {
    case WinType::HANNING:
        return "hanning";
    case WinType::SINE:
        return "sine";
    case WinType::HAMMING:
        return "hamming";
    case WinType::POVEY:
        return "povey";
    case WinType::RECTANGULAR:
        return "rectangular";
    default:
        return "none";
    }
}

int get_frame_num(int input_len, int win_len, int win_step)
{
    int num_frames = 0;
    if (input_len >= win_len) {
        num_frames = 1 + (input_len - win_len) / win_step;
    }
    return num_frames;
}

void remove_dc_offset(float *x, int n)
{
    if (n <= 0) {
        return; // Avoid division by zero
    }

    float sum = 0;
    for (int i = 0; i != n; ++i) {
        sum += x[i];
    }

    float mean = sum / n;

    for (int i = 0; i != n; ++i) {
        x[i] -= mean;
    }
}

float hz2mel(float x)
{
    // return 2595.0 * log10f(1.0 + (x) / 700.0);
    return 1127.0 * logf(1.0 + (x) / 700.0);
}

float mel2hz(float x)
{
    // return 700.0 * (powf(10.0, (x) / 2595.0) - 1.0);
    return 700.0 * (exp(x / 1127.0) - 1.0);
}

uint32_t next_power_of_2(uint32_t x)
{
    if (x == 0) {
        return 1;
    }

    uint32_t power = 1;
    x--;

// Count leading zeros and calculate bit length
#if defined(__GNUC__) || defined(__clang__)
    // Use compiler built-in for efficiency if available
    uint32_t leading_zeros = __builtin_clz(x);
    power = 32 - leading_zeros;
#else
    // Manual bit length calculation
    while (x >>= 1) {
        power++;
    }
#endif

    // Return 2^power
    return 1U << power;
}

float compute_energy(float *x, int len, float epsilon)
{
    float sum_squares = 0.0f;

    // Compute sum of squares
    for (int i = 0; i < len; i++) {
        sum_squares += x[i] * x[i];
    }

    // Return log of the energy
    return logf(MAX(sum_squares, epsilon));
}

mel_filter_t *mel_filter_init(int nfft, int nfilter, int low_freq, int high_freq, int sample_rate, uint32_t caps)
{
    int feat_width = nfft / 2 + 1;
    int bands_to_zero = 1;
    float lowmel = hz2mel(low_freq);
    if (high_freq <= low_freq)
        high_freq = sample_rate / 2;
    float highmel = hz2mel(high_freq);
    float nyquist_hertz = sample_rate / 2.0;
    float *bin = (float *)malloc(sizeof(float) * (nfilter + 2));
    float *bin_mel = (float *)malloc(sizeof(float) * (feat_width));

    // init mel_filter_t　
    mel_filter_t *mel_filter = (mel_filter_t *)heap_caps_malloc(sizeof(mel_filter_t), caps);
    mel_filter->coeff = (float *)heap_caps_malloc(sizeof(float) * feat_width * 2, caps);
    mel_filter->bank_pos = (int *)heap_caps_malloc(sizeof(int) * nfilter * 2, caps);
    mel_filter->nfilter = nfilter;

    for (int i = 0; i < nfilter + 2; i++) {
        float melpoint = ((highmel - lowmel) / (float)(nfilter + 1) * i) + lowmel;
        bin[i] = melpoint;
    }

    for (int i = 0; i < feat_width; i++) bin_mel[i] = hz2mel(i * nyquist_hertz / (feat_width - 1));

    for (int i = 0, idx = 0; i < nfilter; i++) {
        int start = -1, stop = -1;
        for (int j = bands_to_zero; j < feat_width; j++) {
            float lower_slope = (bin_mel[j] - bin[i]) / (bin[i + 1] - bin[i]);
            float upper_slope = (bin[i + 2] - bin_mel[j]) / (bin[i + 2] - bin[i + 1]);
            float temp = MIN(lower_slope, upper_slope);
            if (lower_slope > 0 && start == -1)
                start = j;
            if (upper_slope <= 0 && stop == -1)
                stop = j - 1;

            if (temp > 0.0)
                mel_filter->coeff[idx++] = temp;
        }
        mel_filter->bank_pos[i * 2] = start;
        mel_filter->bank_pos[i * 2 + 1] = stop;
    }
    free(bin);
    free(bin_mel);

    return mel_filter;
}

void mel_filter_deinit(mel_filter_t *mel_filter)
{
    if (mel_filter) {
        free(mel_filter->bank_pos);
        free(mel_filter->coeff);
        free(mel_filter);
        mel_filter = NULL;
    }
}

float *apply_preemphasis(float *x, int win_len, float preemphasis_coeff, float prev)
{
    // Preemphasis
    if (preemphasis_coeff > 1e-7) {
        for (int i = win_len - 1; i >= 1; i--) {
            x[i] = x[i] - x[i - 1] * preemphasis_coeff;
        }
        x[0] = x[0] - prev * preemphasis_coeff;
    }
    return x;
}

float *apply_window(float *x, int win_len, float *win_func)
{
    if (win_func != NULL) {
        for (int i = 0; i < win_len; i++) x[i] = x[i] * win_func[i];
    }

    return x;
}

float *compute_spectrum(float *x, int win_len, bool use_power)
{
    int spect_len = win_len / 2 + 1;

    if (use_power) {
        // power
        x[0] = x[0] * x[0];
        float temp = x[1] * x[1];
        for (int i = 1; i < spect_len - 1; i++) {
            int idx = i * 2;
            x[i] = x[idx] * x[idx] + x[idx + 1] * x[idx + 1];
        }
        x[spect_len - 1] = temp;
    } else {
        // magnitude
        x[0] = fabsf(x[0]);
        float temp = fabsf(x[1]);
        for (int i = 1; i < spect_len - 1; i++) {
            int idx = i * 2;
            x[i] = sqrtf(x[idx] * x[idx] + x[idx + 1] * x[idx + 1]);
        }
        x[spect_len - 1] = temp;
    }

    return x; // x: [nfft//2+1], power or magnitude
}

float dotprod_f32(float *x1, float *x2, int len)
{
    float sum = 0;
    for (int i = 0; i < len; i++) sum += x1[i] * x2[i];
    return sum;
}

float *mel_dotprod(float *x, mel_filter_t *mel_filter, float *output)
{
    float *coeff = mel_filter->coeff;
    int *bank_pos = mel_filter->bank_pos;
    int nfilter = mel_filter->nfilter;
    int coeff_shift = 0;
    int x_shift = 0;
    int len = 0;
    for (int j = 0; j < nfilter; j++) {
        len = bank_pos[j * 2 + 1] - bank_pos[j * 2] + 1;
        x_shift = bank_pos[j * 2];
        output[j] = dotprod_f32(coeff + coeff_shift, x + x_shift, len);
        coeff_shift += len;
    }

    return output;
}

} // namespace audio
} // namespace dl
