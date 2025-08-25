#pragma once

#include "dl_module_reduce_base.hpp"

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__ReduceProd.html
class ReduceProd : public ReduceBase {
public:
    using ReduceBase::ReduceBase;
    /**
     * @brief Destroy the ReduceProd object.
     */
    ~ReduceProd() {}

    template <typename V_T, typename T>
    struct reduce_op_mul {
        V_T operator()(const V_T &x, const T &y, void *arg) const { return x * y; }
    };

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
        float output_scale = DL_RESCALE(output_exponent);
        V_T tmp =
            ReduceBase::reduce<reduce_op_mul<V_T, float>>(input_exponent, v0, ptr, size0, stride0, size1, stride1, arg);
        tool::truncate(ret, tool::round(tmp * output_scale));

        return ret;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            float v0 = 1.f;
            forward_template<float, int8_t>(context, mode, v0, reduce<float, int8_t>, nullptr);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            double v0 = 1.0;
            forward_template<double, int16_t>(context, mode, v0, reduce<double, int16_t>, nullptr);
        }
    }

    /**
     * @brief deserialize ReduceProd module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int keepdims = 1;
        std::vector<bool> axes_reduce_flag;
        get_attributes(fbs_model, node_name, keepdims, axes_reduce_flag, quant_type);

        // Create module
        op = new ReduceProd(
            keepdims, axes_reduce_flag, "ReduceProd", node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ReduceBase::print("ReduceProd"); }
};
} // namespace module
} // namespace dl
