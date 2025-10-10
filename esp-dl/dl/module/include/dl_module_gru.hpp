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
    int m_direction_num;        ///< Specify if the RNN is forward, reverse, or bidirectional
    int m_layout;               ///< The shape format of inputs X, initial_h and outputs Y, Y_h
    bool m_first_step;          ///< First step of the RNN
    int m_gate_exponent;        ///< Exponent for gate calculations
    TensorBase *m_input_cache;  ///< Input cache tensor
    TensorBase *m_hidden_cache; ///< Hidden cache tensor
    TensorBase *m_h_prev;       ///< Previous hidden state tensor
    float *m_gates;
    math::SigmoidLUT *m_sigmoid_lut; ///< Sigmoid lookup table for quantized operations
    math::TanhLUT *m_tanh_lut;       ///< Tanh lookup table for quantized operations

    GRU(const char *name = NULL,
        int hidden_size = 0,
        int direction_num = 1,
        int layout = 0,
        int gate_exponent = -8,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type),
        m_hidden_size(hidden_size),
        m_direction_num(direction_num),
        m_layout(layout),
        m_gate_exponent(gate_exponent)
    {
        m_first_step = true;
        m_input_cache = nullptr;
        m_hidden_cache = nullptr;
        m_h_prev = nullptr;
        m_gates = nullptr;
        m_sigmoid_lut = nullptr;
        m_tanh_lut = nullptr;

        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            uint32_t caps = MALLOC_CAP_SPIRAM;
            if (!DL_SPIRAM_SUPPORT) {
                caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
            }

            m_sigmoid_lut = new math::SigmoidLUT(m_gate_exponent, -6.3, 6.3, caps); // +-6.3 is the boundary for sigmoid
            m_tanh_lut = new math::TanhLUT(m_gate_exponent, -3.5, 3.5, caps);       // +-3.5 is the boundary for tanh
            m_gates = (float *)heap_caps_malloc(3 * hidden_size * sizeof(float), caps);
        }
    }

    ~GRU()
    {
        if (m_input_cache != nullptr) {
            delete m_input_cache;
            m_input_cache = nullptr;
        }
        if (m_hidden_cache != nullptr) {
            delete m_hidden_cache;
            m_hidden_cache = nullptr;
        }
        if (m_h_prev != nullptr) {
            delete m_h_prev;
            m_h_prev = nullptr;
        }
        if (m_sigmoid_lut != nullptr) {
            delete m_sigmoid_lut;
        }
        if (m_tanh_lut != nullptr) {
            delete m_tanh_lut;
        }
        if (m_gates) {
            free(m_gates);
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

        std::vector<int> y_shape;
        y_shape = {seq_length, m_direction_num, batch_size, m_hidden_size};
        output_shapes.push_back(y_shape);

        std::vector<int> y_h_shape;
        y_h_shape = {m_direction_num, batch_size, m_hidden_size};
        output_shapes.push_back(y_h_shape);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode) override
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else {
            forward_float(context, mode);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        // Get input tensors
        TensorBase *input_x = context->get_tensor(m_inputs_index[0]); // X
        TensorBase *input_w = context->get_tensor(m_inputs_index[1]); // W
        TensorBase *input_r = context->get_tensor(m_inputs_index[2]); // R
        TensorBase *input_b = nullptr;                                // B
        TensorBase *initial_h = nullptr;                              // initial_h
        if (m_inputs_index.size() > 3) {
            input_b = context->get_tensor(m_inputs_index[3]);
        }
        if (m_inputs_index.size() > 5) {
            initial_h = context->get_tensor(m_inputs_index[5]);
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
            m_input_cache = new TensorBase({3 * m_hidden_size}, nullptr, m_gate_exponent, DATA_TYPE_INT16);
            m_hidden_cache = new TensorBase({3 * m_hidden_size}, nullptr, m_gate_exponent, DATA_TYPE_INT16);
            m_h_prev = new TensorBase({m_hidden_size}, nullptr, output_h->get_exponent(), output_h->get_dtype());

            // Validate input dimensions
            std::vector<int> w_shape = input_w->get_shape();
            std::vector<int> r_shape = input_r->get_shape();

            if (w_shape.size() != 3 || w_shape[0] != m_direction_num || w_shape[1] != 3 * m_hidden_size ||
                w_shape[2] != input_size) {
                ESP_LOGE("GRU", "Invalid W tensor shape");
                return;
            }

            if (r_shape.size() != 3 || r_shape[0] != m_direction_num || r_shape[1] != 3 * m_hidden_size ||
                r_shape[2] != m_hidden_size) {
                ESP_LOGE("GRU", "Invalid R tensor shape");
                return;
            }
        }

        // Process sequence, currently only support batch_size = 1 and forward direction
        T *x_items = input_x->get_element_ptr<T>();
        T *w_items = input_w->get_element_ptr<T>();
        T *r_items = input_r->get_element_ptr<T>();
        T *h_prev_items = m_h_prev->get_element_ptr<T>();
        T *initial_h_items = nullptr;
        int16_t *wb_items = nullptr;
        int16_t *rb_items = nullptr;
        if (input_b) {
            wb_items = input_b->get_element_ptr<int16_t>();
            rb_items = wb_items + 3 * m_hidden_size;
        }
        if (initial_h) {
            initial_h_items = initial_h->get_element_ptr<T>();
            memcpy(h_prev_items, initial_h_items, m_hidden_size * sizeof(T));
        } else {
            memset(h_prev_items, 0, m_hidden_size * sizeof(T));
        }

        T *y_items = output_y->get_element_ptr<T>();
        T *h_items = output_h->get_element_ptr<T>();
        int16_t *input_cache_items = m_input_cache->get_element_ptr<int16_t>();
        int16_t *hidden_cache_items = m_hidden_cache->get_element_ptr<int16_t>();

        float *z_gate = m_gates;
        float *r_gate = m_gates + m_hidden_size;
        float *n_gate = m_gates + 2 * m_hidden_size;
        float rescale = DL_RESCALE(output_y->get_exponent());
        int w_shift = m_gate_exponent - (input_x->exponent + input_w->exponent);
        int r_shift = m_gate_exponent - (m_h_prev->exponent + input_r->exponent);

        for (int di = 0; di < m_direction_num; di++) {
            if (di == 1) {
                x_items = x_items - input_size;
                y_items = y_items - m_hidden_size;
                w_items += 3 * m_hidden_size * input_size;
                r_items += 3 * m_hidden_size * m_hidden_size;
                if (input_b) {
                    wb_items = wb_items + 6 * m_hidden_size;
                    rb_items = wb_items + 3 * m_hidden_size;
                }
                if (initial_h_items) {
                    memcpy(h_prev_items, initial_h_items + m_hidden_size, m_hidden_size * sizeof(T));
                } else {
                    memset(h_prev_items, 0, m_hidden_size * sizeof(T));
                }
            }

            for (int seq = 0; seq < seq_length; seq++) {
                // W * x_t
                base::mat_vec_dotprod(w_items, x_items, input_cache_items, 3 * m_hidden_size, input_size, w_shift);
                // R * h_prev
                base::mat_vec_dotprod(
                    r_items, h_prev_items, hidden_cache_items, 3 * m_hidden_size, m_hidden_size, r_shift);

                // add bias
                if (input_b) {
                    for (int i = 0; i < 3 * m_hidden_size; i++) {
                        input_cache_items[i] += wb_items[i];
                        hidden_cache_items[i] += rb_items[i];
                    }
                }

                // Compute gates
                for (int i = 0; i < m_hidden_size * 2; i++) {
                    int index = input_cache_items[i] + hidden_cache_items[i];
                    m_gates[i] = m_sigmoid_lut->get(index);
                }

                int offset = 2 * m_hidden_size;
                for (int i = 0; i < m_hidden_size; i++) {
                    int index = tool::round(input_cache_items[i + offset] + r_gate[i] * hidden_cache_items[i + offset]);
                    n_gate[i] = m_tanh_lut->get(index);
                }

                for (int i = 0; i < m_hidden_size; i++) {
                    float temp = (1 - z_gate[i]) * n_gate[i] * rescale;
                    temp += z_gate[i] * h_prev_items[i];
                    tool::truncate(h_prev_items[i], tool::round(temp));
                    y_items[i] = h_prev_items[i];
                }

                // Get input and output pointer for next time step
                if (di == 0) {
                    x_items += input_size;
                    y_items += m_hidden_size * m_direction_num;
                } else {
                    x_items -= input_size;
                    y_items -= m_hidden_size * m_direction_num;
                }
            }
            memcpy(h_items, h_prev_items, m_hidden_size * sizeof(T));
            h_items += m_hidden_size;
        }
    }

    void forward_float(ModelContext *context, runtime_mode_t mode)
    {
        // Get input tensors
        TensorBase *input_x = context->get_tensor(m_inputs_index[0]); // X
        TensorBase *input_w = context->get_tensor(m_inputs_index[1]); // W
        TensorBase *input_r = context->get_tensor(m_inputs_index[2]); // R
        TensorBase *input_b = nullptr;                                // B
        TensorBase *initial_h = nullptr;                              // initial_h
        if (m_inputs_index.size() > 3) {
            input_b = context->get_tensor(m_inputs_index[3]);
        }
        if (m_inputs_index.size() > 5) {
            initial_h = context->get_tensor(m_inputs_index[5]);
        }

        // Get output tensors
        TensorBase *output_y = context->get_tensor(m_outputs_index[0]); // Y
        TensorBase *output_h = context->get_tensor(m_outputs_index[1]); // Y_h

        // Extract dimensions and New variables
        std::vector<int> x_shape = input_x->get_shape();
        int seq_length = (m_layout == 0) ? x_shape[0] : x_shape[1];
        int input_size = x_shape[2];
        if (m_first_step) {
            m_first_step = false;
            m_input_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_hidden_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_h_prev = new TensorBase({m_hidden_size}, nullptr);

            // Validate input dimensions
            std::vector<int> w_shape = input_w->get_shape();
            std::vector<int> r_shape = input_r->get_shape();

            if (w_shape.size() != 3 || w_shape[0] != m_direction_num || w_shape[1] != 3 * m_hidden_size ||
                w_shape[2] != input_size) {
                ESP_LOGE("GRU", "Invalid W tensor shape");
                return;
            }

            if (r_shape.size() != 3 || r_shape[0] != m_direction_num || r_shape[1] != 3 * m_hidden_size ||
                r_shape[2] != m_hidden_size) {
                ESP_LOGE("GRU", "Invalid R tensor shape");
                return;
            }
        }

        // Process sequence, currently only support batch_size = 1
        float *x_items = input_x->get_element_ptr<float>();
        float *w_items = input_w->get_element_ptr<float>();
        float *r_items = input_r->get_element_ptr<float>();
        float *wb_items = nullptr;
        float *rb_items = nullptr;
        float *initial_h_items = nullptr;
        float *h_prev_items = m_h_prev->get_element_ptr<float>();
        if (input_b) {
            wb_items = input_b->get_element_ptr<float>();
            rb_items = wb_items + 3 * m_hidden_size;
        }
        if (initial_h) {
            initial_h_items = initial_h->get_element_ptr<float>();
            memcpy(h_prev_items, initial_h_items, m_hidden_size * sizeof(float));
        } else {
            memset(h_prev_items, 0, m_hidden_size * sizeof(float));
        }

        float *y_items = output_y->get_element_ptr<float>();
        float *h_items = output_h->get_element_ptr<float>();

        float *input_cache_items = m_input_cache->get_element_ptr<float>();
        float *hidden_cache_items = m_hidden_cache->get_element_ptr<float>();

        float *z_gate = input_cache_items;
        float *r_gate = input_cache_items + m_hidden_size;
        float *n_gate = input_cache_items + 2 * m_hidden_size;

        for (int di = 0; di < m_direction_num; di++) {
            if (di == 1) {
                x_items = x_items - input_size;
                y_items = y_items - m_hidden_size;
                w_items += 3 * m_hidden_size * input_size;
                r_items += 3 * m_hidden_size * m_hidden_size;
                if (input_b) {
                    wb_items = input_b->get_element_ptr<float>() + 6 * m_hidden_size;
                    rb_items = wb_items + 3 * m_hidden_size;
                }
                if (initial_h_items) {
                    memcpy(h_prev_items, initial_h_items + m_hidden_size, m_hidden_size * sizeof(float));
                } else {
                    memset(h_prev_items, 0, m_hidden_size * sizeof(float));
                }
            }
            for (int seq = 0; seq < seq_length; seq++) {
                // W * x_t + bias
                // R * h_t + bias
                base::mat_vec_dotprod(w_items, x_items, input_cache_items, 3 * m_hidden_size, input_size, 0);
                base::mat_vec_dotprod(r_items, h_prev_items, hidden_cache_items, 3 * m_hidden_size, m_hidden_size, 0);

                if (input_b) {
                    for (int i = 0; i < 3 * m_hidden_size; i++) {
                        input_cache_items[i] += wb_items[i];
                        hidden_cache_items[i] += rb_items[i];
                    }
                }

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
                if (di == 0) {
                    x_items += input_size;
                    y_items += m_hidden_size * m_direction_num;
                } else {
                    x_items -= input_size;
                    y_items -= m_hidden_size * m_direction_num;
                }
            }
            memcpy(h_items, h_prev_items, m_hidden_size * sizeof(float));
            h_items += m_hidden_size;
        }
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
        int gate_exponent = -8;

        fbs_model->get_operation_attribute(node_name, "hidden_size", hidden_size);
        fbs_model->get_operation_attribute(node_name, "direction", direction);
        fbs_model->get_operation_attribute(node_name, "layout", layout);
        fbs_model->get_operation_attribute(node_name, "linear_before_reset", linear_before_reset);
        fbs_model->get_operation_attribute(node_name, "gate_exponent", gate_exponent);

        if (linear_before_reset == 0) {
            ESP_LOGE("GRU", "linear_before_reset must be 1. The GRU implementation is allgned with PyTorch.");
            return op;
        }
        int direction_num = (direction == "bidirectional") ? 2 : 1;

        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT ||
            quant_type == QUANT_TYPE_FLOAT32) {
            op = new GRU(
                node_name.c_str(), hidden_size, direction_num, layout, gate_exponent, MODULE_NON_INPLACE, quant_type);
        }
        return op;
    }

    void print() override
    {
        ESP_LOGI("GRU",
                 "hidden_size: %d, direction: %s, layout: %d, gate exponent:%d, quant_type: %s",
                 m_hidden_size,
                 (m_direction_num == 2) ? "bidirectional" : "forward",
                 m_layout,
                 m_gate_exponent,
                 quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
