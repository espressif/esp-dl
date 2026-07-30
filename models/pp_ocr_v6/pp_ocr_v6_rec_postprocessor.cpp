#include "pp_ocr_v6_rec_postprocessor.hpp"
#include "dl_define.hpp"
#include "esp_log.h"
#include "pp_ocr_v6_dict.hpp"
#include <cmath>

static const char *TAG = "pp_ocr_v6";

namespace pp_ocr_v6 {
static constexpr int kCtcBlank = 0;

template <typename T>
static int argmax(const T *row, int class_count, T *best_q)
{
    int best = 0;
    T q = row[0];
    for (int c = 1; c < class_count; ++c) {
        if (row[c] > q) {
            q = row[c];
            best = c;
        }
    }
    *best_q = q;
    return best;
}

template <typename T>
static float softmax_argmax_prob(const T *row, int class_count, T best_q, float scale)
{
    float best_logit = dl::dequantize(best_q, scale);
    float exp_sum = 0.0f;
    for (int c = 0; c < class_count; ++c) {
        exp_sum += std::exp(dl::dequantize(row[c], scale) - best_logit);
    }
    return exp_sum > 0.0f ? 1.0f / exp_sum : 0.0f;
}

static const char *class_to_char(int class_id)
{
    if (class_id > 0 && class_id < static_cast<int>(dict::charset_size)) {
        return dict::charset[class_id];
    }
    return "?";
}

// CTCLabelDecode greedy; softmax only if score != nullptr.
template <typename T>
static std::string ctc_greedy_decode(const T *logits, float scale, int time_steps, int class_count, float *score)
{
    const bool need_score = score != nullptr;
    int prev = kCtcBlank;
    int emitted = 0;
    float score_sum = 0.0f;
    std::string text;
    text.reserve(static_cast<size_t>(time_steps));

    for (int t = 0; t < time_steps; ++t) {
        const T *step = logits + static_cast<size_t>(t) * class_count;
        T best_q;
        int best = argmax(step, class_count, &best_q);

        if (best == kCtcBlank || best == prev) {
            prev = best;
            continue;
        }

        text += class_to_char(best);
        if (need_score) {
            score_sum += softmax_argmax_prob(step, class_count, best_q, scale);
            ++emitted;
        }
        prev = best;
    }

    if (need_score) {
        *score = emitted > 0 ? score_sum / emitted : 0.0f;
    }
    return text;
}

RecCTCPostprocessor::RecCTCPostprocessor(dl::Model *model) : m_model_output(nullptr)
{
    auto outputs = model->get_outputs();
    if (!outputs.empty()) {
        m_model_output = outputs.begin()->second;
    }
}

std::string RecCTCPostprocessor::postprocess(float *score)
{
    auto fail = [score]() -> std::string {
        if (score) {
            *score = 0.0f;
        }
        return {};
    };

    if (!m_model_output || !m_model_output->data || m_model_output->shape.empty()) {
        ESP_LOGW(TAG, "Recognition model has no valid output.");
        return fail();
    }

    const int class_count = m_model_output->shape.back();
    if (class_count <= 0) {
        return fail();
    }

    int element_count = 1;
    for (int d : m_model_output->shape) {
        element_count *= d;
    }
    const int time_steps = element_count / class_count;
    const float scale = DL_SCALE(m_model_output->exponent);

    if (m_model_output->dtype == dl::DATA_TYPE_INT8) {
        return ctc_greedy_decode(
            static_cast<const int8_t *>(m_model_output->data), scale, time_steps, class_count, score);
    }
    if (m_model_output->dtype == dl::DATA_TYPE_INT16) {
        return ctc_greedy_decode(
            static_cast<const int16_t *>(m_model_output->data), scale, time_steps, class_count, score);
    }

    ESP_LOGW(TAG, "Unsupported recognition dtype: %s", dl::dtype_to_string(m_model_output->dtype));
    return fail();
}

} // namespace pp_ocr_v6
