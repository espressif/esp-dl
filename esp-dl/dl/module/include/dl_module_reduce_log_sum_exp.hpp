#pragma once

#include "dl_module_reduce_base.hpp"
#include <cmath>

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html
class ReduceLogSumExp : public ReduceBase {
public:
    float *m_exp_table = nullptr;

    using ReduceBase::ReduceBase;
    /**
     * @brief Destroy the ReduceLogSumExp object.
     */
    ~ReduceLogSumExp()
    {
        if (m_exp_table) {
            free(m_exp_table);
        }
    }

    // 16bit
    template <typename V_T, typename T>
    struct reduce_op_exp_add {
        V_T operator()(const V_T &x, const T &y, void *arg) const { return x + expf(y); }
    };

    // 8bit
    template <typename V_T>
    struct reduce_op_exp_add<V_T, int8_t> {
        V_T operator()(const V_T &x, const int8_t &y, void *arg) const
        {
            ReduceLogSumExp *obj = static_cast<ReduceLogSumExp *>(arg);
            if (obj && obj->m_exp_table) {
                return x + obj->m_exp_table[y + 128];
            } else {
                return x + expf(y);
            }
        }
    };

    // 16bit
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
        T ret = 0;
        V_T tmp = ReduceBase::reduce<reduce_op_exp_add<V_T, float>>(
            input_exponent, v0, ptr, size0, stride0, size1, stride1, arg);
        float output_tmp = std::log(static_cast<float>(tmp));
        float output_scale = DL_RESCALE(output_exponent);
        tool::truncate(ret, tool::round(output_tmp * output_scale));

        return ret;
    }

    // 8bit
    template <typename V_T>
    static int8_t reduce(std::string &op_type,
                         int input_exponent,
                         int output_exponent,
                         V_T v0,
                         const int8_t *ptr,
                         int size0,
                         int stride0,
                         int size1,
                         int stride1,
                         void *arg)
    {
        int8_t ret = 0;
        V_T tmp = ReduceBase::reduce<reduce_op_exp_add<V_T, int8_t>>(v0, ptr, size0, stride0, size1, stride1, arg);
        float output_tmp = std::log(static_cast<float>(tmp));
        float output_scale = DL_RESCALE(output_exponent);
        tool::truncate(ret, tool::round(output_tmp * output_scale));

        return ret;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            float v0 = 0.f;
            if (m_exp_table == nullptr) {
#if CONFIG_SPIRAM
                m_exp_table = (float *)heap_caps_malloc(256 * sizeof(float), MALLOC_CAP_SPIRAM);
#else
                m_exp_table = (float *)heap_caps_malloc(256 * sizeof(float), MALLOC_CAP_DEFAULT);
#endif
                tool::gen_lut_8bit(m_exp_table, context->get_tensor(m_inputs_index[0])->get_exponent(), expf);
            }
            forward_template<float, int8_t>(context, mode, v0, reduce<float>, this);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            float v0 = 0.f;
            forward_template<float, int16_t>(context, mode, v0, reduce<float, int16_t>, nullptr);
        }
    }

    /**
     * @brief deserialize ReduceLogSumExp module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int keepdims = 1;
        std::vector<bool> axes_reduce_flag;
        get_attributes(fbs_model, node_name, keepdims, axes_reduce_flag, quant_type);

        // Create module
        op = new ReduceLogSumExp(keepdims,
                                 axes_reduce_flag,
                                 "ReduceLogSumExp",
                                 node_name.c_str(),
                                 MODULE_INPLACE_CHANGED_BUFFER,
                                 quant_type);
        return op;
    }

    void print() { ReduceBase::print("ReduceLogSumExp"); }
};
} // namespace module
} // namespace dl
