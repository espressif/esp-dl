#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * Softmax. int8 and int16 each keep an exp table indexed by (max - x) and fall
 * back to dequantize + expf when the table cannot be allocated. float32 calls
 * the out-of-place forward_float directly; the one-buffer overload remains.
 */
class Softmax : public Module {
private:
    int axis;
    float *exp_table;

    // int16 exp lookup, split by range so two small tables replace one
    // impossible 65536-entry one:
    //
    //     exp(-n * s) = exp(-(n >> 8) * 256 * s) * exp(-(n & 255) * s)
    //
    // with n = max - x >= 0. m_exp16_lo covers the low byte, m_exp16_hi the
    // high byte until a term is negligible next to a sum of at least 1, so
    // together they stay under ~2 KB. Cost is two loads and a multiply against
    // the ~60-100 cycles of a libc expf call per element, which is why the
    // int8 path (which has always used a LUT) measured 1.8 ms on shapes where
    // this one measured 20.8 ms.
    //
    // Accuracy: one extra fp32 multiply rounding versus calling expf, i.e.
    // ~1e-7 relative. The int8 table samples expf on the same integer code
    // points; its error comes from the 8-bit input, not from the table.
    static constexpr int EXP16_LO_BITS = 8;
    static constexpr int EXP16_LO_SIZE = 1 << EXP16_LO_BITS;
    // Past exp(-88) (~6e-39) a term is negligible against a sum >= 1, so the
    // high table stops there. expf itself only returns 0 near -103.97.
    static constexpr float EXP16_UNDERFLOW = 88.0f;

    float *m_exp16_lo = nullptr;
    float *m_exp16_hi = nullptr;
    int m_exp16_hi_len = 0;
    int m_exp16_exponent = INT32_MIN;
    int m_exp8_exponent = INT32_MIN;

    void build_exp16_lut(int exponent)
    {
        // Both tables have to be live. A rebuild that allocates the low table and
        // then fails the high one leaves m_exp16_exponent at the previous value;
        // checking only m_exp16_lo would treat that as a hit and never retry.
        if (m_exp16_lo != nullptr && m_exp16_hi != nullptr && m_exp16_exponent == exponent) {
            return;
        }
        if (m_exp16_lo) {
            free(m_exp16_lo);
            m_exp16_lo = nullptr;
        }
        if (m_exp16_hi) {
            free(m_exp16_hi);
            m_exp16_hi = nullptr;
        }

        const float scale = DL_SCALE(exponent);
        const float hi_step = scale * EXP16_LO_SIZE;

        // Largest high index still worth storing, capped by the widest possible
        // int16 difference (65535 >> 8).
        int hi_len = (65535 >> EXP16_LO_BITS) + 2;
        if (hi_step > 0.f) {
            const int needed = static_cast<int>(EXP16_UNDERFLOW / hi_step) + 2;
            if (needed < hi_len) {
                hi_len = needed;
            }
        }

        m_exp16_lo = (float *)tool::malloc_aligned(EXP16_LO_SIZE * sizeof(float), MALLOC_CAP_DEFAULT);
        m_exp16_hi = (float *)tool::malloc_aligned(hi_len * sizeof(float), MALLOC_CAP_DEFAULT);
        if (m_exp16_lo == nullptr || m_exp16_hi == nullptr) {
            return;
        }
        for (int i = 0; i < EXP16_LO_SIZE; i++) {
            m_exp16_lo[i] = expf(-scale * i);
        }
        for (int i = 0; i < hi_len; i++) {
            m_exp16_hi[i] = expf(-hi_step * i);
        }
        m_exp16_hi_len = hi_len;
        m_exp16_exponent = exponent;
    }

    // exp((x - max) * scale) for the tabulated exponent. n = max - x >= 0.
    inline float exp16(int n) const
    {
        const int hi = n >> EXP16_LO_BITS;
        // After max-sub the mass is usually in the low byte: hi[0] == 1, so
        // skip the multiply. Deep tails still take the two-load path.
        if (hi == 0) {
            return m_exp16_lo[n];
        }
        if (hi >= m_exp16_hi_len) {
            return 0.f;
        }
        return m_exp16_hi[hi] * m_exp16_lo[n & (EXP16_LO_SIZE - 1)];
    }

    void build_exp8_lut(int exponent)
    {
        if (this->exp_table != nullptr && m_exp8_exponent == exponent) {
            return;
        }
        if (this->exp_table == nullptr) {
            this->exp_table = (float *)tool::malloc_aligned(256 * sizeof(float), MALLOC_CAP_DEFAULT);
            if (this->exp_table == nullptr) {
                return;
            }
        }
        // Store exp(-n * scale), n = max - x in [0, 255]. Indexing by the
        // integer gap keeps the LUT finite when the raw activation overflows.
        const float scale = DL_SCALE(exponent);
        for (int i = 0; i < 256; i++) {
            this->exp_table[i] = expf(-scale * i);
        }
        m_exp8_exponent = exponent;
    }

public:
    /**
     * @brief Construct a new Softmax object.
     *
     * @param name            name of module
     * @param axis            the axis of the inputs. Accepted range is [-r, r-1] where r = rank(input).
     * @param inplace         inplace type.
     */
    Softmax(const char *name = NULL,
            int axis = -1,
            module_inplace_t inplace = MODULE_NON_INPLACE,
            quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), axis(axis)
    {
        this->exp_table = nullptr;
    }

    /**
     * @brief Destroy the Softmax object.
     */
    ~Softmax()
    {
        if (this->exp_table != nullptr) {
            free(this->exp_table);
        }
        if (m_exp16_lo != nullptr) {
            free(m_exp16_lo);
        }
        if (m_exp16_hi != nullptr) {
            free(m_exp16_hi);
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        // The output values with the same shape as the input tensor.
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_lut(input, output);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_lut_i16(input, output);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            float *input_element = (float *)input->get_element_ptr();
            float *output_element = (float *)output->get_element_ptr();
            forward_float(input_element, output_element, output->get_size(), output->get_shape(), this->axis);
        }
    }

    // In-place form kept for callers that still pass one buffer. The out-of-place
    // overload reads each element before writing it, so the same pointer is safe.
    void forward_float(float *output_element, int size, std::vector<int> shape, int axis)
    {
        forward_float(output_element, output_element, size, shape, axis);
    }

    void forward_float(const float *input_element, float *output_element, int size, std::vector<int> shape, int axis)
    {
        int dims = shape.size();
        if (dims == 0) {
            output_element[0] = 1.0f;
            return;
        }
        int positive_axis = axis < 0 ? dims + axis : axis;
        int len = shape[positive_axis]; // the size of positive_axis

        if (positive_axis == dims - 1) { // positive_axis == dims - 1
            int outer_loop = 1;
            if (dims > 1) {
                outer_loop = size / len;
            }

            for (int o = 0; o < outer_loop; o++) {
                float max = input_element[0];
                for (int i = 1; i < len; i++) {
                    max = DL_MAX(max, input_element[i]);
                }

                float sum = 0.f;
                for (int i = 0; i < len; i++) {
                    output_element[i] = expf(input_element[i] - max);
                    sum += output_element[i];
                }

                const float inv_sum = 1.f / sum;
                for (int i = 0; i < len; i++) {
                    output_element[i] *= inv_sum;
                }
                input_element += len;
                output_element += len;
            }
        } else {
            // convert input tensor to [outer_loop, len, inner_loop]
            int outer_loop = 1;
            int inner_loop = 1;
            for (int i = 0; i < dims; i++) {
                if (i < positive_axis) {
                    outer_loop *= shape[i];
                } else if (i > positive_axis) {
                    inner_loop *= shape[i];
                }
            }

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float max = input_element[0];
                    for (int i = 1; i < len; i++) {
                        max = DL_MAX(max, input_element[i * inner_loop]);
                    }

                    float sum = 0.f;
                    for (int i = 0; i < len; i++) {
                        const int idx = i * inner_loop;
                        output_element[idx] = expf(input_element[idx] - max);
                        sum += output_element[idx];
                    }

                    const float inv_sum = 1.f / sum;
                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] *= inv_sum;
                    }
                    input_element += 1;
                    output_element += 1;
                }
                input_element += inner_loop * (len - 1);
                output_element += inner_loop * (len - 1);
            }
        }
    }

    // int16 counterpart of forward_lut. Reads the quantized input directly --
    // the old path materialized the whole tensor as fp32 first -- and keeps the
    // max subtraction in the integer domain so exp16() only ever sees n >= 0.
    void forward_lut_i16(TensorBase *input, TensorBase *output)
    {
        assert(output->get_dtype() == DATA_TYPE_FLOAT);
        build_exp16_lut(input->exponent);
        if (m_exp16_lo == nullptr || m_exp16_hi == nullptr) {
            // Same fallback as the int8 path: dequantize, then the generic softmax.
            int16_t *input_element = (int16_t *)input->get_element_ptr();
            float *output_element = (float *)output->get_element_ptr();
            float scale = DL_SCALE(input->exponent);
            for (int i = 0; i < input->get_size(); i++) {
                output_element[i] = scale * input_element[i];
            }
            forward_float(output_element, output_element, output->get_size(), output->get_shape(), this->axis);
            return;
        }

        float *output_element = (float *)output->get_element_ptr();
        int16_t *input_element = (int16_t *)input->get_element_ptr();

        int dims = input->get_shape().size();
        if (dims == 0) {
            output_element[0] = 1.0f;
            return;
        }
        int positive_axis = axis < 0 ? dims + axis : axis;
        int len = input->get_shape()[positive_axis];

        if (positive_axis == dims - 1) {
            int outer_loop = 1;
            if (dims > 1) {
                outer_loop = input->get_size() / len;
            }

            for (int o = 0; o < outer_loop; o++) {
                int max = input_element[0];
                for (int i = 1; i < len; i++) {
                    max = DL_MAX(max, (int)input_element[i]);
                }

                float sum = 0.f;
                for (int i = 0; i < len; i++) {
                    output_element[i] = exp16(max - input_element[i]);
                    sum += output_element[i];
                }

                const float inv_sum = 1.f / sum;
                for (int i = 0; i < len; i++) {
                    output_element[i] *= inv_sum;
                }
                input_element += len;
                output_element += len;
            }
        } else {
            // convert input tensor to [outer_loop, len, inner_loop]
            int outer_loop = 1;
            int inner_loop = 1;
            for (int i = 0; i < dims; i++) {
                if (i < positive_axis) {
                    outer_loop *= input->get_shape()[i];
                } else if (i > positive_axis) {
                    inner_loop *= input->get_shape()[i];
                }
            }

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    int max = input_element[0];
                    for (int i = 1; i < len; i++) {
                        max = DL_MAX(max, (int)input_element[i * inner_loop]);
                    }

                    float sum = 0.f;
                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] = exp16(max - input_element[i * inner_loop]);
                        sum += output_element[i * inner_loop];
                    }

                    const float inv_sum = 1.f / sum;
                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] *= inv_sum;
                    }
                    input_element += 1;
                    output_element += 1;
                }
                input_element += inner_loop * (len - 1);
                output_element += inner_loop * (len - 1);
            }
        }
    }

    void forward_lut(TensorBase *input, TensorBase *output)
    {
        assert(output->get_dtype() == DATA_TYPE_FLOAT);
        build_exp8_lut(input->exponent);
        if (this->exp_table == nullptr) {
            // Same fallback as the int16 path: dequantize, then the generic softmax.
            int8_t *input_element = (int8_t *)input->get_element_ptr();
            float *output_element = (float *)output->get_element_ptr();
            float scale = DL_SCALE(input->exponent);
            for (int i = 0; i < input->get_size(); i++) {
                output_element[i] = scale * input_element[i];
            }
            forward_float(output_element, output_element, output->get_size(), output->get_shape(), this->axis);
            return;
        }

        float *output_element = (float *)output->get_element_ptr();

        int dims = input->get_shape().size();
        if (dims == 0) {
            output_element[0] = 1.0f;
            return;
        }
        int positive_axis = axis < 0 ? dims + axis : axis;
        int len = input->get_shape()[positive_axis]; // the size of positive_axis
        int8_t *input_element = (int8_t *)input->get_element_ptr();

        if (positive_axis == dims - 1) { // positive_axis == dims - 1
            int outer_loop = 1;
            if (dims > 1) {
                outer_loop = input->get_size() / len;
            }

            for (int o = 0; o < outer_loop; o++) {
                int max = input_element[0];
                for (int i = 1; i < len; i++) {
                    max = DL_MAX(max, (int)input_element[i]);
                }

                float sum = 0.f;
                for (int i = 0; i < len; i++) {
                    output_element[i] = this->exp_table[max - input_element[i]];
                    sum += output_element[i];
                }

                const float inv_sum = 1.f / sum;
                for (int i = 0; i < len; i++) {
                    output_element[i] *= inv_sum;
                }
                input_element += len;
                output_element += len;
            }
        } else {
            // convert input tensor to [outer_loop, len, inner_loop]
            int outer_loop = 1;
            int inner_loop = 1;
            for (int i = 0; i < dims; i++) {
                if (i < positive_axis) {
                    outer_loop *= input->get_shape()[i];
                } else if (i > positive_axis) {
                    inner_loop *= input->get_shape()[i];
                }
            }

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    int max = input_element[0];
                    for (int i = 1; i < len; i++) {
                        max = DL_MAX(max, (int)input_element[i * inner_loop]);
                    }

                    float sum = 0.f;
                    for (int i = 0; i < len; i++) {
                        const int idx = i * inner_loop;
                        output_element[idx] = this->exp_table[max - input_element[idx]];
                        sum += output_element[idx];
                    }

                    const float inv_sum = 1.f / sum;
                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] *= inv_sum;
                    }
                    input_element += 1;
                    output_element += 1;
                }
                input_element += inner_loop * (len - 1);
                output_element += inner_loop * (len - 1);
            }
        }
    }

    /**
     * @brief deserialize Softmax module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int axis = -1;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "axis", axis);

        // Create module
        op = new Softmax(node_name.c_str(), axis, MODULE_NON_INPLACE, quant_type);
        return op;
    }

    void print() { ESP_LOGI("Softmax", "quant_type: %s. axis:%d", quant_type_to_string(quant_type), axis); }
};
} // namespace module
} // namespace dl
