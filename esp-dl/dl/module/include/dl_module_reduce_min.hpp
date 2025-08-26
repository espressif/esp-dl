#pragma once

#include "dl_module_reduce_base.hpp"
#include <climits>

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__ReduceMin.html
class ReduceMin : public ReduceBase {
public:
    using ReduceBase::ReduceBase;
    /**
     * @brief Destroy the ReduceMin object.
     */
    ~ReduceMin() {}

    template <typename V_T, typename T>
    struct reduce_op_min {
        V_T operator()(const V_T &x, const T &y, void *arg) const { return std::min(x, y); }
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
        T ret = ReduceBase::reduce<reduce_op_min<T, T>>(v0, ptr, size0, stride0, size1, stride1, arg);
        return ret;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            int8_t v0 = INT8_MAX;
            forward_template<int8_t, int8_t>(context, mode, v0, reduce<int8_t, int8_t>, nullptr);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            int16_t v0 = INT16_MAX;
            forward_template<int16_t, int16_t>(context, mode, v0, reduce<int16_t, int16_t>, nullptr);
        }
    }

    /**
     * @brief deserialize ReduceMin module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int keepdims = 1;
        std::vector<bool> axes_reduce_flag;
        get_attributes(fbs_model, node_name, keepdims, axes_reduce_flag, quant_type);

        // Create module
        op = new ReduceMin(
            keepdims, axes_reduce_flag, "ReduceMin", node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ReduceBase::print("ReduceMin"); }
};
} // namespace module
} // namespace dl
