#pragma once

#include "dl_module_base.hpp"
#include <cmath>

namespace dl {
namespace module {

class ReduceBase : public Module {
protected:
    int m_keepdims;                       /*!< Keep the reduced dimension or not. */
    std::vector<bool> m_axes_reduce_flag; /*!< A bool list with the same dims as input0, indicating whether to perform
                                             reduction on the axes. */
    std::string m_op_type;                /*!< Reduce operation type. */

public:
    /**
     * @brief Construct a new ReduceBase object.
     *
     * @param axes            a list of integers, along which to reduce.
     * @param name            name of module.
     * @param inplace         inplace type.
     * @param quant_type      quant type.
     */
    ReduceBase(int keepdims,
               std::vector<bool> axes_reduce_flag,
               std::string op_type,
               const char *name = NULL,
               module_inplace_t inplace = MODULE_NON_INPLACE,
               quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type),
        m_keepdims(keepdims),
        m_axes_reduce_flag(axes_reduce_flag),
        m_op_type(op_type)
    {
    }

    /**
     * @brief Destroy the ReduceBase object.
     */
    ~ReduceBase() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape;
        if (m_keepdims) {
            for (int i = 0; i < input_shape.size(); i++) {
                if (m_axes_reduce_flag[i]) {
                    output_shape.push_back(1);
                } else {
                    output_shape.push_back(input_shape[i]);
                }
            }
        } else {
            uint32_t reduce_dims_count = 0;
            for (int i = 0; i < input_shape.size(); i++) {
                if (m_axes_reduce_flag[i]) {
                    reduce_dims_count++;
                    continue;
                } else {
                    output_shape.push_back(input_shape[i]);
                }
            }
            if (reduce_dims_count == input_shape.size()) {
                output_shape = {1};
            }
        }
        return {output_shape};
    }

    template <typename V_T, typename T>
    struct reduce_op_add {
        V_T operator()(const V_T &x, const T &y, void *arg) const { return x + y; }
    };

    template <typename V_T, typename T>
    struct reduce_op_square_add {
        V_T operator()(const V_T &x, const T &y, void *arg) const { return x + y * y; }
    };

    template <typename Op, typename V_T, typename T>
    static V_T reduce(V_T v0, const T *ptr, int size0, int stride0, int size1, int stride1, void *arg)
    {
        Op op;
        V_T sum = v0;

        for (int i = 0; i < size1; i++) {
            const T *ptr0 = ptr;
            for (int j = 0; j < size0; j++) {
                sum = op(sum, *ptr0, arg);
                ptr0 += stride0;
            }
            ptr += stride1;
        }

        return sum;
    }

    template <typename Op, typename V_T, typename T>
    static V_T reduce(
        int input_exponent, V_T v0, const T *ptr, int size0, int stride0, int size1, int stride1, void *arg)
    {
        Op op;
        V_T sum = v0;
        float input_scale = DL_SCALE(input_exponent);
        for (int i = 0; i < size1; i++) {
            const T *ptr0 = ptr;
            for (int j = 0; j < size0; j++) {
                float tmp = (*ptr0) * input_scale;
                sum = op(sum, tmp, arg);
                ptr0 += stride0;
            }
            ptr += stride1;
        }

        return sum;
    }

    template <typename V_T, typename T, typename ReduceFn>
    void forward_template(ModelContext *context, runtime_mode_t mode, V_T v0, ReduceFn &&reduce_fn, void *arg)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        int merged_dims = input->get_shape().size();
        int i_exp = input->get_exponent();
        int o_exp = output->get_exponent();
        std::vector<int> new_input_shape = input->get_shape(); // NCHW
        std::vector<bool> new_reduce_flag = m_axes_reduce_flag;
        T *input_ptr = input->get_element_ptr<T>();
        T *output_ptr = output->get_element_ptr<T>();
        int stride0 = 1;
        int size1 = 1;
        int stride1 = 0;

        // Merge input shape and reduce flags.
        if (new_reduce_flag.size() > 1) {
            for (int i = 0; i < new_reduce_flag.size() - 1; ++i) {
                if (new_reduce_flag[i] == new_reduce_flag[i + 1]) {
                    new_input_shape[i] *= new_input_shape[i + 1];
                    new_input_shape.erase(new_input_shape.begin() + i + 1);
                    new_reduce_flag.erase(new_reduce_flag.begin() + i + 1);
                    // Since an element was removed, we need to step back one position and continue checking.
                    --i;
                }
            }
            merged_dims = new_input_shape.size();
        }
        assert(new_input_shape.size() == new_reduce_flag.size());

        if (merged_dims == 1) {
            output_ptr[0] =
                reduce_fn(m_op_type, i_exp, o_exp, v0, input_ptr, input->get_size(), stride0, size1, stride1, arg);
        } else if (merged_dims == 2) {
            if (!new_reduce_flag[0] && new_reduce_flag[1]) {
                T *input_ptr_tmp = input_ptr;
                for (int i = 0; i < new_input_shape[0]; i++) {
                    output_ptr[i] = reduce_fn(
                        m_op_type, i_exp, o_exp, v0, input_ptr_tmp, new_input_shape[1], stride0, size1, stride1, arg);
                    input_ptr_tmp += new_input_shape[1];
                }
            } else if (new_reduce_flag[0] && !new_reduce_flag[1]) {
                for (int i = 0; i < new_input_shape[1]; i++) {
                    output_ptr[i] = reduce_fn(m_op_type,
                                              i_exp,
                                              o_exp,
                                              v0,
                                              input_ptr + i,
                                              new_input_shape[0],
                                              new_input_shape[1],
                                              size1,
                                              stride1,
                                              arg);
                }
            }
        } else if (merged_dims == 3) {
            if (new_reduce_flag[0] && !new_reduce_flag[1] && new_reduce_flag[2]) {
                T *input_ptr_tmp = input_ptr;
                int stride = new_input_shape[1] * new_input_shape[2];
                for (int i = 0; i < new_input_shape[1]; i++) {
                    output_ptr[i] = reduce_fn(m_op_type,
                                              i_exp,
                                              o_exp,
                                              v0,
                                              input_ptr_tmp,
                                              new_input_shape[2],
                                              1,
                                              new_input_shape[0],
                                              stride,
                                              arg);
                    input_ptr_tmp += new_input_shape[2];
                }
            } else if (!new_reduce_flag[0] && new_reduce_flag[1] && !new_reduce_flag[2]) {
                int offset = new_input_shape[1] * new_input_shape[2];
                T *input_ptr_tmp = input_ptr;
                T *output_ptr_tmp = output_ptr;
                for (int i = 0; i < new_input_shape[0]; i++) {
                    for (int j = 0; j < new_input_shape[2]; j++) {
                        output_ptr_tmp[j] = reduce_fn(m_op_type,
                                                      i_exp,
                                                      o_exp,
                                                      v0,
                                                      input_ptr_tmp + j,
                                                      new_input_shape[1],
                                                      new_input_shape[2],
                                                      size1,
                                                      stride1,
                                                      arg);
                    }
                    input_ptr_tmp += offset;
                    output_ptr_tmp += new_input_shape[2];
                }
            }
        } else if (merged_dims == 4) {
            if (!new_reduce_flag[0] && new_reduce_flag[1] && !new_reduce_flag[2] && new_reduce_flag[3]) {
                int offset0 = new_input_shape[1] * new_input_shape[2] * new_input_shape[3];
                int offset1 = new_input_shape[3];
                int stride = new_input_shape[2] * new_input_shape[3];
                T *input_ptr_tmp0 = input_ptr;
                T *output_ptr_tmp = output_ptr;
                for (int i = 0; i < new_input_shape[0]; i++) {
                    T *input_ptr_tmp1 = input_ptr_tmp0;
                    for (int j = 0; j < new_input_shape[2]; j++) {
                        output_ptr_tmp[j] = reduce_fn(m_op_type,
                                                      i_exp,
                                                      o_exp,
                                                      v0,
                                                      input_ptr_tmp1,
                                                      new_input_shape[3],
                                                      1,
                                                      new_input_shape[1],
                                                      stride,
                                                      arg);
                        input_ptr_tmp1 += offset1;
                    }
                    input_ptr_tmp0 += offset0;
                    output_ptr_tmp += new_input_shape[2];
                }
            } else if (new_reduce_flag[0] && !new_reduce_flag[1] && new_reduce_flag[2] && !new_reduce_flag[3]) {
                int offset = new_input_shape[2] * new_input_shape[3];
                int stride = new_input_shape[1] * new_input_shape[2] * new_input_shape[3];
                T *input_ptr_tmp0 = input_ptr;
                T *output_ptr_tmp = output_ptr;
                for (int i = 0; i < new_input_shape[1]; i++) {
                    T *input_ptr_tmp1 = input_ptr_tmp0;
                    for (int j = 0; j < new_input_shape[3]; j++) {
                        output_ptr_tmp[j] = reduce_fn(m_op_type,
                                                      i_exp,
                                                      o_exp,
                                                      v0,
                                                      input_ptr_tmp1 + j,
                                                      new_input_shape[2],
                                                      new_input_shape[3],
                                                      new_input_shape[0],
                                                      stride,
                                                      arg);
                    }
                    input_ptr_tmp0 += offset;
                    output_ptr_tmp += new_input_shape[3];
                }
            }
        }
    }

    static void get_attributes(fbs::FbsModel *fbs_model,
                               std::string node_name,
                               int &keepdims,
                               std::vector<bool> &axes_reduce_flag,
                               quant_type_t &quant_type)
    {
        int noop_with_empty_axes = 0;
        std::vector<int> input0_shape;

        TensorBase *axes = fbs_model->get_operation_parameter(node_name, 1);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "keepdims", keepdims);
        fbs_model->get_operation_attribute(node_name, "noop_with_empty_axes", noop_with_empty_axes);
        fbs_model->get_operation_input_shape(node_name, 0, input0_shape);

        std::vector<bool> axes_reduce_flag_tmp(input0_shape.size(), false);
        if (axes && axes->get_size() > 0) {
            for (int i = 0; i < axes->get_size(); i++) {
                int axis = static_cast<int>(axes->get_element<int64_t>(i));
                if (axis < 0) {
                    axis += input0_shape.size();
                }
                axes_reduce_flag_tmp[axis] = true;
            }
        } else {
            if (!noop_with_empty_axes) {
                for (int i = 0; i < axes_reduce_flag_tmp.size(); i++) {
                    axes_reduce_flag_tmp[i] = true;
                }
            }
        }
        delete axes;
        axes_reduce_flag = axes_reduce_flag_tmp;
    }

    void print(std::string tag)
    {
        ESP_LOGI(tag.c_str(),
                 "quant_type: %s, op_type: %s, keepdims: %d, axes_reduce_flag: %s.",
                 quant_type_to_string(quant_type),
                 m_op_type.c_str(),
                 m_keepdims,
                 vector_to_string(m_axes_reduce_flag).c_str());
    }

    virtual void print() { print("ReduceBase"); }
};
} // namespace module
} // namespace dl
