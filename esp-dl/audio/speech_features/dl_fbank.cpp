#include "dl_fbank.hpp"
#include "dl_audio_fixed_point.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

namespace dl {
namespace audio {

static inline int bit_len32(uint32_t v)
{
    return v ? 32 - __builtin_clz(v) : 0;
}

static inline int bit_len64(uint64_t v)
{
    return v ? 64 - __builtin_clzll(v) : 0;
}

static inline int32_t mulh32(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 32);
}

static inline int16_t sat_s16(int64_t v)
{
    return v > INT16_MAX ? INT16_MAX : (v < INT16_MIN ? INT16_MIN : (int16_t)v);
}

// ln(v * 2^e) in Q36, v > 0
static inline int64_t ln_q36(uint64_t v, int e)
{
    return ln_u64_q36(v) + (int64_t)e * LN2_Q36;
}

// round(l * 2^-36 * 2^-exponent), saturated to int16
static int16_t q36_to_s16(int64_t l, int exponent)
{
    int k = 36 + exponent;
    if (k <= 0) {
        if (k <= -16) {
            return l > 0 ? INT16_MAX : (l < 0 ? INT16_MIN : 0);
        }
        return sat_s16(l * (1LL << -k));
    }
    if (k >= 62) {
        return 0;
    }
    return sat_s16((l + (1LL << (k - 1))) >> k);
}

// round(v * 2^e * 2^-exponent), saturated to int16
static int16_t linear_to_s16(uint64_t v, int e, int exponent)
{
    int k = e - exponent;
    if (v == 0) {
        return 0;
    }
    if (k >= 0) {
        return (k >= 15 || v > (uint64_t)(INT16_MAX >> k)) ? INT16_MAX : (int16_t)(v << k);
    }
    if (k <= -64) {
        return 0;
    }
    uint64_t q = ((v >> (-k - 1)) + 1) >> 1;
    return q > INT16_MAX ? INT16_MAX : (int16_t)q;
}

// ln(v * 2^e + m * 2^me) in Q36, m in [2^61, 2^62)
static int64_t ln_add_q36(uint64_t v, int e, uint64_t m, int me)
{
    if (v) {
        int s = 62 - bit_len64(v);
        if (s >= 0) {
            v <<= s;
        } else {
            v >>= -s;
        }
        e -= s;
        if (e < me) {
            std::swap(v, m);
            std::swap(e, me);
        }
        int d = e - me;
        v += d < 63 ? m >> d : 0;
    } else {
        v = m;
        e = me;
    }
    return ln_q36(v, e);
}

// DC removal, pre-emphasis and window in Q(frac_bits). Returns the OR of |y| over the frame.
template <bool kWindow>
static uint32_t preprocess_s32(const int16_t *input,
                               int len,
                               int32_t mean,
                               int32_t frac,
                               int frac_bits,
                               int32_t preemph_q31,
                               const int32_t *win_q31,
                               int32_t *y)
{
    uint32_t mag = 0;
    int32_t prev = ((input[0] - mean) << frac_bits) - frac;
    for (int i = 0; i < len; i++) {
        int32_t x = ((input[i] - mean) << frac_bits) - frac;
        int32_t v = x - mulh32(prev << 1, preemph_q31);
        prev = x;
        if (kWindow) {
            v = mulh32(v << 1, win_q31[i]);
        }
        y[i] = v;
        mag |= (uint32_t)(v ^ (v >> 31));
    }
    return mag;
}

} // namespace audio
} // namespace dl

namespace dl {
namespace audio {

esp_err_t Fbank::process_frame(const float *input, int win_len, float *output, float prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    if (m_cache != input) {
        memcpy(m_cache, input, sizeof(float) * win_len);
    }

    if (m_config.remove_dc_offset) {
        remove_dc_offset(m_cache, win_len);
    }

    if (m_config.raw_energy && m_config.use_energy) {
        output[0] = compute_energy(m_cache, win_len, m_config.log_epsilon);
        output += 1;
    }

    // Use m_cache[0] as prev to match Kaldi's replicate padding: y[0] = x[0] - α·x[0] = (1-α)·x[0]
    apply_preemphasis(m_cache, win_len, m_config.preemphasis, m_cache[0]);

    apply_window(m_cache, win_len, m_win_func);

    if (!m_config.raw_energy && m_config.use_energy) {
        output[0] = compute_energy(m_cache, win_len, m_config.log_epsilon);
        output += 1;
    }

    if (win_len < m_fft_size) {
        memset(m_cache + win_len, 0, sizeof(float) * (m_fft_size - win_len));
    }

    dl_rfft_f32_run(m_fft_config, m_cache);

    compute_spectrum(m_cache, m_fft_size, m_config.use_power);

    mel_dotprod(m_cache, m_mel_filter, output);

    if (m_config.use_log_fbank == 1) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(DL_MAX(output[j], epsilon));
    } else if (m_config.use_log_fbank == 2) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(output[j] + epsilon);
    }

    return ESP_OK;
}

esp_err_t Fbank::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < win_len; i++) {
        m_cache[i] = input[i] / 32768.0f;
    }

    return process_frame(m_cache, win_len, output, prev / 32768.0f);
}

FbankS32::FbankS32(const SpeechFeatureConfig config, uint32_t caps) :
    SpeechFeatureBase(config, caps), m_mel_filter(nullptr), m_win_q31(nullptr), m_buf(nullptr), m_fft_s32(nullptr)
{
    m_feature_dim = config.use_energy ? config.num_mel_bins + 1 : config.num_mel_bins;
    m_buf = (int32_t *)heap_caps_aligned_alloc(
        16, sizeof(int32_t) * (m_fft_size + 2), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    m_fft_s32 = dl_rfft_s32_init(m_fft_size, caps);
    m_mel_filter = mel_filter_q30_init(
        m_fft_size, config.num_mel_bins, config.low_freq, config.high_freq, config.sample_rate, m_caps);
    m_power_bits = m_mel_filter ? DL_MIN(32, 63 - m_mel_filter->sum_bits) : 32;

    float *win = win_func_init(config.window_type, m_win_len);
    if (win) {
        bool all_ones = true;
        for (int i = 0; i < m_win_len; i++) {
            all_ones = all_ones && win[i] == 1.0f;
        }
        if (!all_ones) {
            m_win_q31 = (int32_t *)heap_caps_malloc(sizeof(int32_t) * m_win_len, caps);
            if (m_win_q31) {
                for (int i = 0; i < m_win_len; i++) {
                    double q = round(win[i] * 2147483648.0);
                    m_win_q31[i] = (int32_t)DL_MIN(q, 2147483647.0);
                }
            } else {
                // process_frame_int16() reports ESP_ERR_NO_MEM.
                heap_caps_free(m_buf);
                m_buf = nullptr;
            }
        }
        free(win);
    }

    m_preemph_q31 = m_config.preemphasis > 1e-7f ? (int32_t)lrint(m_config.preemphasis * 2147483648.0) : 0;

    double eps = m_config.log_epsilon;
    if (eps > 0) {
        int e2;
        double m = frexp(eps, &e2); // eps = m * 2^e2, m in [0.5, 1)
        m_log_eps_q36 = llround(log(eps) * 68719476736.0);
        m_eps_mant = (uint64_t)llround(m * 4611686018427387904.0); // m * 2^62
        m_eps_exp = e2 - 62;
        if (m_eps_mant >> 62) {
            m_eps_mant >>= 1;
            m_eps_exp++;
        }
    } else {
        m_log_eps_q36 = INT64_MIN / 2;
        m_eps_mant = 1ULL << 61;
        m_eps_exp = -1100;
    }
}

FbankS32::~FbankS32()
{
    if (m_fft_s32) {
        dl_rfft_s32_deinit(m_fft_s32);
    }
    heap_caps_free(m_buf);
    heap_caps_free(m_win_q31);
    if (m_mel_filter) {
        mel_filter_q30_deinit(m_mel_filter);
    }
}

esp_err_t FbankS32::process_frame(const float *input, int win_len, float *output, float prev)
{
    (void)input;
    (void)win_len;
    (void)output;
    (void)prev;
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t FbankS32::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
{
    (void)input;
    (void)win_len;
    (void)output;
    (void)prev;
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t FbankS32::process_frame_int16(
    const int16_t *input, int win_len, int16_t *output, int16_t prev, int output_exponent)
{
    (void)prev;

    if (input == nullptr || output == nullptr || win_len <= 0 || win_len > m_fft_size ||
        (m_win_q31 && win_len > m_win_len)) {
        return ESP_ERR_INVALID_ARG;
    }
    if (m_fft_s32 == nullptr) {
        return ESP_ERR_NO_MEM;
    }
    if (m_buf == nullptr || m_mel_filter == nullptr) {
        return ESP_ERR_NO_MEM;
    }

    const bool remove_dc = m_config.remove_dc_offset;
    const int64_t log_eps = m_log_eps_q36;

    int32_t sum = 0, vmin = input[0], vmax = input[0];
    for (int i = 0; i < win_len; i++) {
        int32_t v = input[i];
        sum += v;
        vmin = DL_MIN(vmin, v);
        vmax = DL_MAX(vmax, v);
    }
    // x = input - (mean + frac / 2^frac_bits) with |x| < bound
    int32_t mean = 0;
    uint32_t rem = 0, bound;
    if (remove_dc) {
        mean = sum / win_len;
        if (mean * win_len > sum) {
            mean--;
        }
        rem = (uint32_t)(sum - mean * win_len);
        bound = (uint32_t)(vmax - vmin + 1);
    } else {
        bound = (uint32_t)DL_MAX(vmax, -vmin);
    }

    if (m_config.use_energy && m_config.raw_energy) {
        uint64_t sumsq = 0;
        for (int i = 0; i < win_len; i++) {
            sumsq += (uint32_t)((int32_t)input[i] * input[i]);
        }
        int64_t l = log_eps;
        if (remove_dc) {
            // sum((x - mean)^2) = (n * sum(x^2) - sum(x)^2) / n
            uint64_t ne = (uint64_t)win_len * sumsq - (uint64_t)((int64_t)sum * sum);
            if (ne) {
                l = ln_q36(ne, -30) - ln_u64_q36(win_len);
            }
        } else if (sumsq) {
            l = ln_q36(sumsq, -30);
        }
        *output++ = q36_to_s16(DL_MAX(l, log_eps), output_exponent);
    }

    // |x| < 2^29 in Q(frac_bits), so pre-emphasis and window stay below 2^30.
    int frac_bits = 29 - bit_len32(bound);
    int32_t frac = remove_dc ? (int32_t)((((uint64_t)rem << frac_bits) + win_len / 2) / win_len) : 0;
    int32_t *x = m_buf;
    uint32_t mag = m_win_q31 ? preprocess_s32<true>(input, win_len, mean, frac, frac_bits, m_preemph_q31, m_win_q31, x)
                             : preprocess_s32<false>(input, win_len, mean, frac, frac_bits, m_preemph_q31, nullptr, x);
    int mag_bits = bit_len32(mag);

    uint64_t *power = (uint64_t *)m_buf;
    int cpx = m_fft_size / 2;
    int power_exp; // float power = power[k] * 2^power_exp
    uint64_t energy = 0;
    int energy_exp = 0;
    const bool win_energy = m_config.use_energy && !m_config.raw_energy;
    // Scale to [2^28, 2^29) for the int32 FFT.
    int shift = 29 - mag_bits;
    int scale = frac_bits + shift; // x_s32 = x_pcm * 2^scale
    if (shift > 0) {
        for (int i = 0; i < win_len; i++) {
            x[i] <<= shift;
        }
    } else if (shift < 0) {
        int r = -shift;
        int32_t half = 1 << (r - 1);
        for (int i = 0; i < win_len; i++) {
            x[i] = (x[i] + half) >> r;
        }
    }
    if (win_energy) {
        for (int i = 0; i < win_len; i++) {
            int32_t v = x[i] >> 14;
            energy += (uint32_t)(v * v);
        }
        energy_exp = 28 - 2 * scale - 30;
    }
    if (win_len < m_fft_size) {
        memset(x + win_len, 0, sizeof(int32_t) * (m_fft_size - win_len));
    }

    int fft_exp = 0;
    esp_err_t ret = dl_rfft_s32_run(m_fft_s32, x, -scale - 15, &fft_exp);
    if (ret != ESP_OK) {
        return ret;
    }
    // |X| < 2^30, so every power fits in 61 bits.
    int32_t dc = x[0], nyquist = x[1];
    for (int k = 1; k < cpx; k++) {
        int32_t re = x[2 * k], im = x[2 * k + 1];
        power[k] = (uint64_t)((int64_t)re * re) + (uint64_t)((int64_t)im * im);
    }
    power[0] = (uint64_t)((int64_t)dc * dc);
    power[cpx] = (uint64_t)((int64_t)nyquist * nyquist);
    power_exp = 2 * fft_exp;

    if (win_energy) {
        int64_t l = energy ? ln_q36(energy, energy_exp) : log_eps;
        *output++ = q36_to_s16(DL_MAX(l, log_eps), output_exponent);
    }

    if (!m_config.use_power) {
        for (int k = 0; k <= cpx; k++) {
            power[k] = sqrt_u64_q8(power[k]);
        }
        power_exp = power_exp / 2 - 8;
    }

    const uint32_t *coeff = m_mel_filter->coeff;
    const int *bank_pos = m_mel_filter->bank_pos;
    for (int j = 0; j < m_mel_filter->nfilter; j++) {
        int start = bank_pos[j * 2];
        int len = bank_pos[j * 2 + 1] - start + 1;
        const uint64_t *p = power + start;

        // Per-band shift keeps the accumulator in 64 bits without losing weak bands.
        uint64_t any = 0;
        for (int k = 0; k < len; k++) {
            any |= p[k];
        }
        int sh = DL_MAX(0, bit_len64(any) - m_power_bits);
        uint64_t acc = 0;
        if (sh == 0) {
            for (int k = 0; k < len; k++) {
                acc += (uint64_t)coeff[k] * (uint32_t)p[k];
            }
        } else if (sh >= 32) {
            int s = sh - 32;
            for (int k = 0; k < len; k++) {
                acc += (uint64_t)coeff[k] * ((uint32_t)(p[k] >> 32) >> s);
            }
        } else {
            for (int k = 0; k < len; k++) {
                uint32_t lo = (uint32_t)p[k], hi = (uint32_t)(p[k] >> 32);
                acc += (uint64_t)coeff[k] * ((lo >> sh) | (hi << (32 - sh)));
            }
        }
        coeff += len;

        int e = sh + power_exp - 30; // mel = acc * 2^e
        if (m_config.use_log_fbank == 1) {
            int64_t l = acc ? ln_q36(acc, e) : log_eps;
            output[j] = q36_to_s16(DL_MAX(l, log_eps), output_exponent);
        } else if (m_config.use_log_fbank == 2) {
            output[j] = q36_to_s16(ln_add_q36(acc, e, m_eps_mant, m_eps_exp), output_exponent);
        } else {
            output[j] = linear_to_s16(acc, e, output_exponent);
        }
    }

    return ESP_OK;
}

} // namespace audio
} // namespace dl
