#pragma once

#include "dl_module_reduce_base.hpp"
#include <cmath>

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html
class ReduceLogSum : public ReduceBase {
public:
    using ReduceBase::ReduceBase;
    /**
     * @brief Destroy the ReduceLogSum object.
     */
    ~ReduceLogSum() {}

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
        V_T tmp = ReduceBase::reduce<reduce_op_add<V_T, T>>(v0, ptr, size0, stride0, size1, stride1, arg);
        float input_scale = DL_SCALE(input_exponent);
        float output_scale = DL_RESCALE(output_exponent);
        float output_tmp = std::log(static_cast<float>(tmp * input_scale));
        tool::truncate(ret, tool::round(output_tmp * output_scale));

        return ret;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            int32_t v0 = 0;
            forward_template<int32_t, int8_t>(context, mode, v0, reduce<int32_t, int8_t>, nullptr);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            int64_t v0 = 0;
            forward_template<int64_t, int16_t>(context, mode, v0, reduce<int64_t, int16_t>, nullptr);
        }
    }

    /**
     * @brief deserialize ReduceLogSum module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int keepdims = 1;
        std::vector<bool> axes_reduce_flag;
        get_attributes(fbs_model, node_name, keepdims, axes_reduce_flag, quant_type);

        // Create module
        op = new ReduceLogSum(
            keepdims, axes_reduce_flag, "ReduceLogSum", node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ReduceBase::print("ReduceLogSum"); }
};
} // namespace module
} // namespace dl
