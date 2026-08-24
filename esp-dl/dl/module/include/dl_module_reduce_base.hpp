#pragma once

#include "dl_base_reduce.hpp"
#include "dl_module_base.hpp"
#include <array>
#include <cmath>
#include <limits>

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

    template <typename V_T, typename T>
    static V_T reduce_sum(const T *ptr, int size0, int stride0, int size1, int stride1)
    {
        V_T sum = 0;
        for (int i = 0; i < size1; i++) {
            sum += dl::base::reduce_sum(const_cast<T *>(ptr), size0, stride0);
            ptr += stride1;
        }
        return sum;
    }

    template <typename V_T, typename T>
    static V_T reduce_l1(const T *ptr, int size0, int stride0, int size1, int stride1)
    {
        V_T sum = 0;
        for (int i = 0; i < size1; i++) {
            sum += dl::base::reduce_l1(const_cast<T *>(ptr), size0, stride0);
            ptr += stride1;
        }
        return sum;
    }

    template <typename V_T, typename T>
    static V_T reduce_l2(const T *ptr, int size0, int stride0, int size1, int stride1)
    {
        V_T sum = 0;
        for (int i = 0; i < size1; i++) {
            sum += dl::base::reduce_l2(const_cast<T *>(ptr), size0, stride0);
            ptr += stride1;
        }
        return sum;
    }

    template <typename T>
    static T reduce_max(const T *ptr, int size0, int stride0, int size1, int stride1)
    {
        T value = std::numeric_limits<T>::lowest();
        for (int i = 0; i < size1; i++) {
            value = std::max(value, dl::base::reduce_max(const_cast<T *>(ptr), size0, stride0));
            ptr += stride1;
        }
        return value;
    }

    template <typename T>
    static T reduce_min(const T *ptr, int size0, int stride0, int size1, int stride1)
    {
        T value = std::numeric_limits<T>::max();
        for (int i = 0; i < size1; i++) {
            value = std::min(value, dl::base::reduce_min(const_cast<T *>(ptr), size0, stride0));
            ptr += stride1;
        }
        return value;
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
        int i_exp = input->get_exponent();
        int o_exp = output->get_exponent();
        T *input_ptr = input->get_element_ptr<T>();
        T *output_ptr = output->get_element_ptr<T>();
        const std::vector<int> input_shape = input->get_shape();

        assert(input_shape.size() == m_axes_reduce_flag.size());
        assert(input_shape.size() <= 4);

        bool has_reduce_axis = false;
        for (bool reduce_axis : m_axes_reduce_flag) {
            has_reduce_axis |= reduce_axis;
        }
        if (!has_reduce_axis) {
            assert(input->get_size() == output->get_size());
            if (input_ptr != output_ptr) {
                output->assign(input);
            }
            return;
        }

        // Merge adjacent axes with the same reduction state without allocating or erasing vectors.
        std::array<int, 4> shape = {1, 1, 1, 1};
        std::array<bool, 4> reduce_axis = {false, false, false, false};
        int merged_dims = 0;
        for (int i = 0; i < input_shape.size(); i++) {
            if (merged_dims > 0 && reduce_axis[merged_dims - 1] == m_axes_reduce_flag[i]) {
                shape[merged_dims - 1] *= input_shape[i];
            } else {
                shape[merged_dims] = input_shape[i];
                reduce_axis[merged_dims] = m_axes_reduce_flag[i];
                merged_dims++;
            }
        }

        auto run_reduce = [&](T *ptr, int size0, int stride0, int size1 = 1, int stride1 = 0) {
            return reduce_fn(m_op_type, i_exp, o_exp, v0, ptr, size0, stride0, size1, stride1, arg);
        };

        // Merged axes alternate between output and reduction runs. Keep the innermost
        // reduction run as size0 so contiguous reductions use the longest SIMD kernel.
        if (merged_dims == 1) {
            output_ptr[0] = run_reduce(input_ptr, shape[0], 1);
        } else if (merged_dims == 2) {
            if (reduce_axis[1]) { // [output, reduce]
                for (int i = 0; i < shape[0]; i++) {
                    output_ptr[i] = run_reduce(input_ptr + i * shape[1], shape[1], 1);
                }
            } else { // [reduce, output]
                for (int i = 0; i < shape[1]; i++) {
                    output_ptr[i] = run_reduce(input_ptr + i, shape[0], shape[1]);
                }
            }
        } else if (merged_dims == 3) {
            if (reduce_axis[0]) { // [reduce, output, reduce]
                int outer_stride = shape[1] * shape[2];
                for (int i = 0; i < shape[1]; i++) {
                    output_ptr[i] = run_reduce(input_ptr + i * shape[2], shape[2], 1, shape[0], outer_stride);
                }
            } else { // [output, reduce, output]
                int input_block_size = shape[1] * shape[2];
                for (int i = 0; i < shape[0]; i++) {
                    T *input_block = input_ptr + i * input_block_size;
                    T *output_block = output_ptr + i * shape[2];
                    for (int j = 0; j < shape[2]; j++) {
                        output_block[j] = run_reduce(input_block + j, shape[1], shape[2]);
                    }
                }
            }
        } else {
            assert(merged_dims == 4);
            if (reduce_axis[1]) { // [output, reduce, output, reduce]
                int input_block_size = shape[1] * shape[2] * shape[3];
                int outer_stride = shape[2] * shape[3];
                for (int i = 0; i < shape[0]; i++) {
                    T *input_block = input_ptr + i * input_block_size;
                    T *output_block = output_ptr + i * shape[2];
                    for (int j = 0; j < shape[2]; j++) {
                        output_block[j] = run_reduce(input_block + j * shape[3], shape[3], 1, shape[1], outer_stride);
                    }
                }
            } else { // [reduce, output, reduce, output]
                int middle_block_size = shape[2] * shape[3];
                int outer_stride = shape[1] * middle_block_size;
                for (int i = 0; i < shape[1]; i++) {
                    T *input_block = input_ptr + i * middle_block_size;
                    T *output_block = output_ptr + i * shape[3];
                    for (int j = 0; j < shape[3]; j++) {
                        output_block[j] = run_reduce(input_block + j, shape[2], shape[3], shape[0], outer_stride);
                    }
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
