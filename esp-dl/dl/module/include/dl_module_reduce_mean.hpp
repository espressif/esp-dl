#pragma once

#include "dl_module_reduce_base.hpp"
#include <climits>
#include <type_traits>
#if CONFIG_IDF_TARGET_ESP32P4 && CONFIG_PIE_V2_BOOST
extern "C" void dl_esp32p4_reduce_inner_s16(const int16_t *input, int stride_bytes, int rows, int32_t *output);
#endif

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
class ReduceMean : public ReduceBase {
public:
    using ReduceBase::ReduceBase;
    /**
     * @brief Destroy the ReduceMean object.
     */
    ~ReduceMean() {}

    template <typename V_T, typename T>
    static T reduce(std::string &op_type,
                    int input_exponent,
                    int output_exponent,
                    V_T v0,
                    const T *ptr,
                    int size0,
                    int stride0,
                    int size1,
                    int stride1,
                    void *arg)
    {
        // For float types, skip quantization operations
        if constexpr (std::is_same<T, float>::value && std::is_same<V_T, float>::value) {
            // Directly return the mean for float32
            V_T sum = ReduceBase::reduce_sum<V_T>(ptr, size0, stride0, size1, stride1);
            size_t count = size0 * size1;
            return sum / static_cast<float>(count);
        } else {
            // For quantized types, perform scaling and truncation
            T ret = 0;
            V_T tmp = ReduceBase::reduce_sum<V_T>(ptr, size0, stride0, size1, stride1);
            size_t count = size0 * size1;
            float input_scale = DL_SCALE(input_exponent);
            float output_scale = DL_RESCALE(output_exponent);
            tool::truncate(ret, tool::round(tmp * input_scale / count * output_scale));

            return ret;
        }
    }

#if CONFIG_IDF_TARGET_ESP32P4 && CONFIG_PIE_V2_BOOST
    template <typename T>
    bool try_inner(TensorBase *input, TensorBase *output)
    {
        // Read the input as [outer, reduce, inner].
        const auto &shape = input->shape;
        if (shape.size() != m_axes_reduce_flag.size())
            return false;
        int64_t sizes[3] = {1, 1, 1};
        int phase = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] <= 0)
                return false;
            if (m_axes_reduce_flag[d]) {
                if (phase == 2)
                    return false;
                phase = 1;
            } else if (phase == 1)
                phase = 2;
            sizes[phase] *= shape[d];
            if (sizes[phase] > INT_MAX)
                return false;
        }
        const int outer = sizes[0], count = sizes[1], inner = sizes[2];
        // The count limit keeps all partial sums in int32, including -32768 * 65536.
        if (phase != 2 || count < 4 || count > 65536 || inner < 8 || int64_t(outer) * inner != output->get_size() ||
            int64_t(outer) * count * inner != input->get_size())
            return false;
        const auto *src = input->get_element_ptr<T>();
        auto *dst = output->get_element_ptr<T>();
        const uintptr_t a = reinterpret_cast<uintptr_t>(src), b = reinterpret_cast<uintptr_t>(dst);
        // Equal input and output addresses are safe. Write before the next unread row.
        if (a != b && uint64_t(a) < uint64_t(b) + output->get_bytes() && uint64_t(b) < uint64_t(a) + input->get_bytes())
            return false;
        const float input_scale = DL_SCALE(input->get_exponent());
        const float output_scale = DL_RESCALE(output->get_exponent());
        for (int o = 0; o < outer; ++o) {
            for (int c = 0; c < inner; c += 8) {
                alignas(16) int32_t sums[8] = {};
                const int n = std::min(8, inner - c);
                const T *row = src + o * count * inner + c;
                bool native = false;
                if constexpr (sizeof(T) == 2) {
                    if (n == 8 && !(inner & 7) && !(reinterpret_cast<uintptr_t>(row) & 15)) {
                        dl_esp32p4_reduce_inner_s16(row, inner * sizeof(T), count, sums);
                        native = true;
                    }
                }
                if (!native) {
                    if (n == 8) {
                        for (int r = 0; r < count; ++r) {
#pragma GCC unroll 8
                            for (int k = 0; k < 8; ++k) sums[k] += row[k];
                            if (r + 1 < count)
                                row += inner;
                        }
                    } else {
                        for (int r = 0; r < count; ++r) {
                            for (int k = 0; k < n; ++k) sums[k] += row[k];
                            if (r + 1 < count)
                                row += inner;
                        }
                    }
                }
                for (int k = 0; k < n; ++k) {
                    // Keep the original scale, division, rounding and limits.
                    const int64_t sum = sums[k];
                    tool::truncate(dst[o * inner + c + k],
                                   tool::round(sum * input_scale / size_t(count) * output_scale));
                }
            }
        }
        return true;
    }

#endif

    void forward(ModelContext *context, runtime_mode_t mode)
    {
#if CONFIG_IDF_TARGET_ESP32P4 && CONFIG_PIE_V2_BOOST
        auto *input = context->get_tensor(m_inputs_index[0]);
        auto *output = context->get_tensor(m_outputs_index[0]);
        if ((quant_type == QUANT_TYPE_SYMM_8BIT && try_inner<int8_t>(input, output)) ||
            (quant_type == QUANT_TYPE_SYMM_16BIT && try_inner<int16_t>(input, output))) {
            return;
        }
#endif
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            int32_t v0 = 0;
            forward_template<int32_t, int8_t>(context, mode, v0, reduce<int32_t, int8_t>, nullptr);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            int64_t v0 = 0;
            forward_template<int64_t, int16_t>(context, mode, v0, reduce<int64_t, int16_t>, nullptr);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            float v0 = 0.0f;
            forward_template<float, float>(context, mode, v0, reduce<float, float>, nullptr);
        }
    }

    /**
     * @brief deserialize ReduceMean module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int keepdims = 1;
        std::vector<bool> axes_reduce_flag;
        get_attributes(fbs_model, node_name, keepdims, axes_reduce_flag, quant_type);

        // Create module
        op = new ReduceMean(
            keepdims, axes_reduce_flag, "ReduceMean", node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ReduceBase::print("ReduceMean"); }
};
} // namespace module
} // namespace dl
