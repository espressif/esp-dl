#pragma once

#include "dl_base_dotprod.hpp"
#include "dl_base_shape.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include <cmath>

namespace dl {
namespace module {

class GRU : public Module {
public:
    int m_hidden_size;          ///< Number of neurons in the hidden layer
    std::string m_direction;    ///< Specify if the RNN is forward, reverse, or bidirectional
    int m_layout;               ///< The shape format of inputs X, initial_h and outputs Y, Y_h
    bool m_first_step;          ///< First step of the RNN
    TensorBase *m_input_cache;  ///< Input cache tensor
    TensorBase *m_hidden_cache; ///< Hidden cache tensor
    TensorBase *m_h_prev;       ///< Previous hidden state tensor

    GRU(const char *name = NULL,
        int hidden_size = 0,
        const std::string &direction = "forward",
        int layout = 0,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_hidden_size(hidden_size), m_direction(direction), m_layout(layout)
    {
        m_first_step = true;
        m_input_cache = nullptr;
        m_hidden_cache = nullptr;
        m_h_prev = nullptr;
    }

    ~GRU()
    {
        if (m_input_cache != nullptr) {
            delete m_input_cache;
        }
        if (m_hidden_cache != nullptr) {
            delete m_hidden_cache;
        }
        if (m_h_prev != nullptr) {
            delete m_h_prev;
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        std::vector<std::vector<int>> output_shapes;

        if (input_shapes.size() < 3) {
            ESP_LOGE("GRU", "GRU requires at least 3 inputs: X, W, R");
            return output_shapes;
        }

        std::vector<int> x_shape = input_shapes[0];
        int seq_length = (m_layout == 0) ? x_shape[0] : x_shape[1];
        int batch_size = (m_layout == 0) ? x_shape[1] : x_shape[0];

        int num_directions = (m_direction == "bidirectional") ? 2 : 1;

        std::vector<int> y_shape;
        if (m_layout == 0) {
            y_shape = {seq_length, num_directions, batch_size, m_hidden_size};
        } else {
            y_shape = {batch_size, seq_length, num_directions, m_hidden_size};
        }
        output_shapes.push_back(y_shape);

        std::vector<int> y_h_shape;
        if (m_layout == 0) {
            y_h_shape = {num_directions, batch_size, m_hidden_size};
        } else {
            y_h_shape = {batch_size, num_directions, m_hidden_size};
        }
        output_shapes.push_back(y_h_shape);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode) override
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            // forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            // forward_template<int16_t>(context, mode);
        } else {
            forward_float(context, mode);
        }
    }

    void forward_float(ModelContext *context, runtime_mode_t mode)
    {
        // Get input tensors
        TensorBase *input_x = context->get_tensor(m_inputs_index[0]); // X
        TensorBase *input_w = context->get_tensor(m_inputs_index[1]); // W
        TensorBase *input_r = context->get_tensor(m_inputs_index[2]); // R
        float *bias_items = nullptr;                                  // Bias
        if (m_inputs_index.size() > 3) {
            TensorBase *input_bias = context->get_tensor(m_inputs_index[3]);
            if (input_bias)
                bias_items = input_bias->get_element_ptr<float>();
        }

        // Get output tensors
        TensorBase *output_y = context->get_tensor(m_outputs_index[0]); // Y
        TensorBase *output_h = context->get_tensor(m_outputs_index[1]); // Y_h

        // Extract dimensions
        std::vector<int> x_shape = input_x->get_shape();
        int seq_length = (m_layout == 0) ? x_shape[0] : x_shape[1];
        int input_size = x_shape[2];

        if (m_first_step) {
            m_first_step = false;
            m_input_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_hidden_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_h_prev = new TensorBase({m_hidden_size}, nullptr);

            if (m_inputs_index.size() > 5) {
                TensorBase *initial_h = context->get_tensor(m_inputs_index[5]);
                // Copy initial hidden state to h_prev
                const float *initial_h_data = initial_h->get_element_ptr<float>();
                memcpy(m_h_prev->get_element_ptr(), initial_h_data, m_hidden_size * sizeof(float));
            } else {
                // Initialize h_prev to zeros
                memset(m_h_prev->get_element_ptr(), 0, m_hidden_size * sizeof(float));
            }

            // Validate input dimensions
            std::vector<int> w_shape = input_w->get_shape();
            std::vector<int> r_shape = input_r->get_shape();
            int num_directions = (m_direction == "bidirectional") ? 2 : 1;

            if (w_shape.size() != 3 || w_shape[0] != num_directions || w_shape[1] != 3 * m_hidden_size ||
                w_shape[2] != input_size) {
                ESP_LOGE("GRU", "Invalid W tensor shape");
                return;
            }

            if (r_shape.size() != 3 || r_shape[0] != num_directions || r_shape[1] != 3 * m_hidden_size ||
                r_shape[2] != m_hidden_size) {
                ESP_LOGE("GRU", "Invalid R tensor shape");
                return;
            }
        }

        // Process sequence, currently only support batch_size = 1 and forward direction
        float *x_items = input_x->get_element_ptr<float>();
        float *w_items = input_w->get_element_ptr<float>();
        float *r_items = input_r->get_element_ptr<float>();

        float *y_items = output_y->get_element_ptr<float>();
        float *h_items = output_h->get_element_ptr<float>();

        float *input_cache_items = m_input_cache->get_element_ptr<float>();
        float *hidden_cache_items = m_hidden_cache->get_element_ptr<float>();
        float *h_prev_items = m_h_prev->get_element_ptr<float>();

        float *z_gate = input_cache_items;
        float *r_gate = input_cache_items + m_hidden_size;
        float *n_gate = input_cache_items + 2 * m_hidden_size;

        for (int seq = 0; seq < seq_length; seq++) {
            // W * x_t
            base::mat_vec_dotprod(w_items, x_items, input_cache_items, 3 * m_hidden_size, input_size, bias_items, 0);

            // R * h_prev
            base::mat_vec_dotprod(r_items,
                                  h_prev_items,
                                  hidden_cache_items,
                                  3 * m_hidden_size,
                                  m_hidden_size,
                                  bias_items + 3 * m_hidden_size,
                                  0);

            // Compute gates
            for (int i = 0; i < m_hidden_size * 2; i++) {
                input_cache_items[i] = math::sigmoid(input_cache_items[i] + hidden_cache_items[i]);
            }

            int offset = 2 * m_hidden_size;
            for (int i = 0; i < m_hidden_size; i++) {
                n_gate[i] = math::tanh(n_gate[i] + r_gate[i] * hidden_cache_items[i + offset]);
            }

            for (int i = 0; i < m_hidden_size; i++) {
                h_prev_items[i] = (1 - z_gate[i]) * n_gate[i] + z_gate[i] * h_prev_items[i];
                y_items[i] = h_prev_items[i];
            }

            // Get input and output pointer for next time step
            x_items += input_size;
            y_items += m_hidden_size;
        }
        memcpy(h_items, h_prev_items, m_hidden_size * sizeof(float));
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        int hidden_size = 0;
        std::string direction = "forward";
        int layout = 0;
        int linear_before_reset = 0;

        fbs_model->get_operation_attribute(node_name, "hidden_size", hidden_size);
        fbs_model->get_operation_attribute(node_name, "direction", direction);
        fbs_model->get_operation_attribute(node_name, "layout", layout);
        fbs_model->get_operation_attribute(node_name, "linear_before_reset", linear_before_reset);

        if (linear_before_reset == 0) {
            ESP_LOGE("GRU", "linear_before_reset must be 1. The GRU implementation is allgned with PyTorch.");
            return op;
        }

        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT ||
            quant_type == QUANT_TYPE_FLOAT32) {
            op = new GRU(node_name.c_str(), hidden_size, direction, layout, MODULE_NON_INPLACE, quant_type);
        }
        return op;
    }

    void print() override
    {
        ESP_LOGI("GRU",
                 "hidden_size: %d, direction: %s, layout: %d, quant_type: %s",
                 m_hidden_size,
                 m_direction.c_str(),
                 m_layout,
                 quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
