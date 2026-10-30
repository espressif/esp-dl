#pragma once

#include "dl_base_reduce.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief Normalize X over shape[axis:] and apply Scale and optional Bias.
 *
 * The quantized kernel gathers sum and sum-of-squares with the reduce_base
 * SIMD kernels (reduce_sum, reduce_l2), which keep the wide accumulation out
 * of the inner scalar loop. The variance is recovered as
 * (n*sum(x^2) - sum(x)^2) / n^2 so the cancellation happens exactly in
 * integers, and the folded affine scales are cached.
 * Rows may be split across two cores unless SINGLE_CORE is requested.
 *
 * Reference: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
 */
class LayerNormalization : public Module {
private:
    int m_axis; /*!< axis to normalize, default is -1 */

    // inv_std is 1 / sqrt_newton(var * input_scale^2 + epsilon), and
    // dl::math::sqrt_newton stops on an absolute step of 1e-5, so it cannot
    // resolve a root below ~7.6e-6. Epsilon floors that argument: 1e-7 still
    // holds 6e-5 relative error, 1e-8 reaches 0.3%, and a BERT-style 1e-12
    // leaves inv_std 7.7x too small. The ONNX default 1e-5 is safe; going far
    // below it needs a relative stop condition in sqrt_newton first.
    float m_epsilon; /*!< small value to avoid division by zero, default is 1e-5 */

    // Accepted and ignored: stash_type asks for a wider accumulation dtype,
    // and the moments are already gathered in fp32 (float kernel) or exact
    // integers (quantized kernel).
    int m_stash_type; /*!< whether to stash intermediate values in higher precision (0 or 1) */

    // Cached quantized affine parameters. Fp32 uses the model tensors directly.
    int m_affine_dim = 0;       /*!< cached inner_dim, 0 means uninitialized */
    float *m_scale_f = nullptr; /*!< dequantized scale, length inner_dim */
    float *m_bias_f = nullptr;  /*!< dequantized bias, length inner_dim */
    float *m_norm_a = nullptr;  /*!< scale multiplied by output scale, length inner_dim */
    float *m_norm_b = nullptr;  /*!< bias multiplied by output scale, length inner_dim */
    float m_input_scale = 1.f;
    float m_mean_output_scale = 1.f;
    float m_inv_std_output_scale = 1.f;
    static constexpr size_t AUTO_DUAL_CORE_MIN_ELEMENTS = 768; /*!< conservative P4 crossover threshold */

    // Arguments shared by the single-core and dual-core kernels.
    struct LNTask {
        const void *x = nullptr;
        void *y = nullptr;
        void *mean = nullptr;
        void *inv_std = nullptr;
        const float *scale_f = nullptr; /*!< quantized: dequantized scale; fp32: model scale */
        const float *bias_f = nullptr;  /*!< quantized: dequantized bias; fp32: model bias */
        const float *norm_a = nullptr;  /*!< quantized path only */
        const float *norm_b = nullptr;  /*!< quantized path only */
        int outer_begin = 0;
        int outer_end = 0;
        int inner_dim = 0;
        float input_scale = 1.f;
        float mean_output_scale = 1.f;
        float inv_std_output_scale = 1.f;
        float epsilon = 1e-5f;
        bool is_float = false;
    };

public:
    /**
     * @brief Construct a new LayerNormalization object.
     *
     * @param axis           The axis to normalize (default -1)
     * @param epsilon        Small value to avoid division by zero (default 1e-5)
     * @param stash_type     Whether to stash intermediate values in higher precision (default 0)
     * @param name           Name of module
     * @param inplace        Inplace type
     * @param quant_type     Quantization type
     */
    LayerNormalization(int axis = -1,
                       float epsilon = 1e-5f,
                       int stash_type = 0,
                       const char *name = NULL,
                       module_inplace_t inplace = MODULE_NON_INPLACE,
                       quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_axis(axis), m_epsilon(epsilon), m_stash_type(stash_type)
    {
    }

    /**
     * @brief Destroy the LayerNormalization object.
     */
    ~LayerNormalization()
    {
        heap_caps_free(m_scale_f);
        heap_caps_free(m_bias_f);
        heap_caps_free(m_norm_a);
        heap_caps_free(m_norm_b);
    }

    /**
     * @brief Calculate output shape based on input shapes.
     *
     * @param input_shapes   Input shapes [X_shape, (optional) Scale_shape, (optional) B_shape]
     * @return std::vector<std::vector<int>> Output shapes [Y_shape, Mean_shape, InvStdDev_shape]
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        assert(input_shapes.size() >= 1);
        std::vector<int> y_shape = input_shapes[0];
        int valid_axis = get_valid_axis(y_shape);
        std::vector<int> reduced_shape = calculate_normalized_shape(y_shape, valid_axis);
        return {y_shape, reduced_shape, reduced_shape};
    }

    /**
     * @brief Forward pass for LayerNormalization.
     *
     * @param context   Model context
     * @param mode      Runtime mode
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    /**
     * @brief Low-level forward interface.
     *
     * @param args   Pointer to LNTask
     */
    void forward_args(void *args) override
    {
        const LNTask *task = static_cast<const LNTask *>(args);
        if (task->is_float) {
            ln_float_range(*task);
        } else if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            ln_quant_range<int8_t>(*task);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            ln_quant_range<int16_t>(*task);
        }
    }

    /**
     * @brief Template forward implementation.
     *
     * @tparam T        Data type (int8_t, int16_t, float)
     * @param context   Model context
     * @param mode      Runtime mode
     */
    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *scale = (m_inputs_index.size() > 1) ? context->get_tensor(m_inputs_index[1]) : nullptr;
        TensorBase *bias = (m_inputs_index.size() > 2) ? context->get_tensor(m_inputs_index[2]) : nullptr;

        TensorBase *y = context->get_tensor(m_outputs_index[0]);
        TensorBase *mean = (m_outputs_index.size() > 1) ? context->get_tensor(m_outputs_index[1]) : nullptr;
        TensorBase *inv_std_dev = (m_outputs_index.size() > 2) ? context->get_tensor(m_outputs_index[2]) : nullptr;

        const std::vector<int> &x_shape = x->shape;
        int valid_axis = get_valid_axis(x_shape);

        int outer_dim = 1;
        for (int i = 0; i < valid_axis; i++) {
            outer_dim *= x_shape[i];
        }
        int inner_dim = 1;
        for (size_t i = valid_axis; i < x_shape.size(); i++) {
            inner_dim *= x_shape[i];
        }

        LNTask base;
        base.x = x->get_element_ptr();
        base.y = y->get_element_ptr();
        base.mean = mean ? mean->get_element_ptr() : nullptr;
        base.inv_std = inv_std_dev ? inv_std_dev->get_element_ptr() : nullptr;
        base.inner_dim = inner_dim;
        base.epsilon = m_epsilon;
        base.is_float = std::is_same_v<T, float>;

        if constexpr (std::is_same_v<T, float>) {
            base.scale_f = scale ? static_cast<const float *>(scale->get_element_ptr()) : nullptr;
            base.bias_f = bias ? static_cast<const float *>(bias->get_element_ptr()) : nullptr;
        } else {
            if (!ensure_quant_affine_cache<T>(inner_dim, x, y, scale, bias, mean, inv_std_dev)) {
                return; // OOM already logged
            }
            base.scale_f = m_scale_f;
            base.bias_f = m_bias_f;
            base.input_scale = m_input_scale;
            base.mean_output_scale = m_mean_output_scale;
            base.inv_std_output_scale = m_inv_std_output_scale;
            base.norm_a = m_norm_a;
            base.norm_b = m_norm_b;
        }

        // Split independent rows when the chip has two cores.
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
        const bool dual = outer_dim >= 2 &&
            (mode == RUNTIME_MODE_MULTI_CORE ||
             (mode == RUNTIME_MODE_AUTO &&
              static_cast<size_t>(outer_dim) * static_cast<size_t>(inner_dim) >= AUTO_DUAL_CORE_MIN_ELEMENTS));
#else
        (void)mode;
        const bool dual = false;
#endif
        if (dual) {
            LNTask t0 = base;
            LNTask t1 = base;
            t0.outer_begin = 0;
            t0.outer_end = outer_dim / 2;
            t1.outer_begin = t0.outer_end;
            t1.outer_end = outer_dim;
            module_forward_dual_core(this, &t0, &t1);
        } else {
            base.outer_begin = 0;
            base.outer_end = outer_dim;
            forward_args(&base);
        }
    }

private:
    /**
     * @brief Materialize the quantized-path affine cache on first use.
     *
     * @return true on success; false on allocation failure
     */
    template <typename T>
    bool ensure_quant_affine_cache(int inner_dim,
                                   TensorBase *x,
                                   TensorBase *y,
                                   TensorBase *scale,
                                   TensorBase *bias,
                                   TensorBase *mean,
                                   TensorBase *inv_std_dev)
    {
        static_assert(!std::is_same_v<T, float>, "ensure_quant_affine_cache is for the quantized path");

        if (m_affine_dim == inner_dim) {
            return true;
        }
        assert(m_affine_dim == 0 && "LayerNormalization inner_dim changed after first forward");

        const size_t bytes = static_cast<size_t>(inner_dim) * sizeof(float);
        m_scale_f = static_cast<float *>(tool::malloc_aligned(bytes, MALLOC_CAP_DEFAULT));
        m_bias_f = static_cast<float *>(tool::malloc_aligned(bytes, MALLOC_CAP_DEFAULT));
        m_norm_a = static_cast<float *>(tool::malloc_aligned(bytes, MALLOC_CAP_DEFAULT));
        m_norm_b = static_cast<float *>(tool::malloc_aligned(bytes, MALLOC_CAP_DEFAULT));
        if (!m_scale_f || !m_bias_f || !m_norm_a || !m_norm_b) {
            ESP_LOGE("LayerNormalization",
                     "OOM allocating %u B affine cache (inner_dim=%d)",
                     static_cast<unsigned>(bytes * 4),
                     inner_dim);
            heap_caps_free(m_scale_f);
            m_scale_f = nullptr;
            heap_caps_free(m_bias_f);
            m_bias_f = nullptr;
            heap_caps_free(m_norm_a);
            m_norm_a = nullptr;
            heap_caps_free(m_norm_b);
            m_norm_b = nullptr;
            return false;
        }

        const float scale_scale = scale ? DL_SCALE(scale->exponent) : 1.0f;
        const float bias_scale = bias ? DL_SCALE(bias->exponent) : 0.0f;
        const T *sp = scale ? static_cast<const T *>(scale->get_element_ptr()) : nullptr;
        const T *bp = bias ? static_cast<const T *>(bias->get_element_ptr()) : nullptr;
        for (int j = 0; j < inner_dim; j++) {
            m_scale_f[j] = sp ? (static_cast<float>(sp[j]) * scale_scale) : 1.f;
            m_bias_f[j] = bp ? (static_cast<float>(bp[j]) * bias_scale) : 0.f;
        }

        m_input_scale = DL_SCALE(x->exponent);
        m_mean_output_scale = mean ? DL_RESCALE(mean->exponent) : 1.0f;
        m_inv_std_output_scale = inv_std_dev ? DL_RESCALE(inv_std_dev->exponent) : 1.0f;

        const float output_scale = DL_RESCALE(y->exponent);
        for (int j = 0; j < inner_dim; j++) {
            m_norm_a[j] = m_scale_f[j] * output_scale;
            m_norm_b[j] = m_bias_f[j] * output_scale;
        }

        m_affine_dim = inner_dim;
        return true;
    }

public:
    /**
     * @brief Calculate the broadcast shape of Mean and InvStdDev.
     *
     * @param shape   Input shape
     * @param axis    Normalization axis
     * @return std::vector<int> Normalized shape
     */
    static std::vector<int> calculate_normalized_shape(const std::vector<int> &shape, int axis)
    {
        if (axis < 0) {
            axis = shape.size() + axis;
        }
        std::vector<int> normalized_shape(shape.size(), 1);
        for (int i = 0; i < axis; i++) {
            normalized_shape[i] = shape[i];
        }
        return normalized_shape;
    }

    /**
     * @brief Deserialize LayerNormalization module instance.
     *
     * @param fbs_model  Flatbuffer model
     * @param node_name  Node name
     * @return Module*   Deserialized module
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        int axis = -1;
        float epsilon = 1e-5f;
        int stash_type = 0;
        quant_type_t quant_type;

        fbs_model->get_operation_attribute(node_name, "axis", axis);
        fbs_model->get_operation_attribute(node_name, "epsilon", epsilon);
        fbs_model->get_operation_attribute(node_name, "stash_type", stash_type);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        return new LayerNormalization(
            axis, epsilon, stash_type, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
    }

    /**
     * @brief Print module information.
     */
    void print() override
    {
        ESP_LOGI("LayerNormalization",
                 "axis: %d, epsilon: %f, stash_type: %d, quant_type: %s.",
                 m_axis,
                 m_epsilon,
                 m_stash_type,
                 quant_type_to_string(quant_type));
    }

private:
    /**
     * @brief Float32 LayerNormalization kernel.
     */
    static void ln_float_range(const LNTask &t)
    {
        const float *x_ptr = static_cast<const float *>(t.x);
        float *y_ptr = static_cast<float *>(t.y);
        float *mean_ptr = static_cast<float *>(t.mean);
        float *inv_ptr = static_cast<float *>(t.inv_std);
        const float *sp = t.scale_f;
        const float *bp = t.bias_f;
        for (int i = t.outer_begin; i < t.outer_end; i++) {
            const float *x_slice = x_ptr + i * t.inner_dim;
            float *y_slice = y_ptr + i * t.inner_dim;
            float mean_val = 0.0f;
            for (int j = 0; j < t.inner_dim; j++) {
                mean_val += x_slice[j];
            }
            mean_val /= static_cast<float>(t.inner_dim);
            if (mean_ptr) {
                mean_ptr[i] = mean_val;
            }
            float variance = 0.0f;
            for (int j = 0; j < t.inner_dim; j++) {
                const float diff = x_slice[j] - mean_val;
                variance += diff * diff;
            }
            variance /= static_cast<float>(t.inner_dim);
            const float inv_std = 1.0f / dl::math::sqrt_newton(variance + t.epsilon);
            if (inv_ptr) {
                inv_ptr[i] = inv_std;
            }
            for (int j = 0; j < t.inner_dim; j++) {
                const float normalized = (x_slice[j] - mean_val) * inv_std;
                const float scaled = sp ? normalized * sp[j] : normalized;
                y_slice[j] = bp ? scaled + bp[j] : scaled;
            }
        }
    }

    /**
     * @brief Quantized LayerNormalization kernel.
     */
    template <typename T>
    static void ln_quant_range(const LNTask &t)
    {
        T *x_ptr = static_cast<T *>(const_cast<void *>(t.x));
        T *y_ptr = static_cast<T *>(t.y);
        T *mean_ptr = static_cast<T *>(t.mean);
        T *inv_ptr = static_cast<T *>(t.inv_std);
        const int n = t.inner_dim;
        const float inv_n = 1.0f / static_cast<float>(n);
        const float inv_n2 = inv_n * inv_n;
        const float is2 = t.input_scale * t.input_scale;

        for (int i = t.outer_begin; i < t.outer_end; i++) {
            T *x_slice = x_ptr + i * n;
            T *y_slice = y_ptr + i * n;

            // Delegate both reductions to reduce_base. The SIMD kernels batch
            // 8/16 elements per iteration and only fold into a wide scalar at
            // the end of each aligned chunk, so the per-element int64 add that
            // dominated the previous scalar loop is gone.
            const int64_t sum_val = dl::base::reduce_sum(x_slice, n, 1);
            const int64_t sumsq_val = dl::base::reduce_l2(x_slice, n, 1);

            const float mean_q = static_cast<float>(sum_val) * inv_n;
            if (mean_ptr) {
                tool::truncate(mean_ptr[i], tool::round(mean_q * t.mean_output_scale * t.input_scale));
            }

            // Subtract the two moments in the integer domain. Rows with a
            // large mean and a small spread (int16 values crowded near 32767)
            // make n*sum(x^2) and sum(x)^2 both reach ~2^42 while their
            // difference is only ~2^9, so rounding either one to fp32 first
            // wipes out the variance. n*sum(x^2) stays inside int64 as long as
            // n < 2^16 for int16 and n < 2^24 for int8, far above any
            // deployable normalized length.
            const int64_t var_num = static_cast<int64_t>(n) * sumsq_val - sum_val * sum_val;
            // Cauchy-Schwarz keeps var_num >= 0, so the clamp only guards the
            // int64 overflow that an out-of-range n would cause.
            float var_q = static_cast<float>(var_num) * inv_n2;
            if (var_q < 0.0f) {
                var_q = 0.0f;
            }
            const float inv_std = 1.0f / dl::math::sqrt_newton(var_q * is2 + t.epsilon);
            if (inv_ptr) {
                tool::truncate(inv_ptr[i], tool::round(inv_std * t.inv_std_output_scale));
            }

            const float g = inv_std * t.input_scale;
            for (int j = 0; j < n; j++) {
                const float centered = (static_cast<float>(x_slice[j]) - mean_q) * g;
                tool::truncate(y_slice[j], tool::round(centered * t.norm_a[j] + t.norm_b[j]));
            }
        }
    }

    /**
     * @brief Convert a negative axis to its non-negative equivalent.
     *
     * @param shape   Input shape
     * @return int    Valid axis value
     */
    int get_valid_axis(const std::vector<int> &shape)
    {
        if (m_axis < 0) {
            return shape.size() + m_axis;
        }
        return m_axis;
    }
};

} // namespace module
} // namespace dl
