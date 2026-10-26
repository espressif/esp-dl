#pragma once

#include "dl_base_dotprod.hpp"
#include "dl_base_shape.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>

namespace dl {
namespace module {

class GRU : public Module {
public:
    int m_hidden_size;          ///< Number of neurons in the hidden layer
    int m_direction_num;        ///< Number of directions: forward (1) or bidirectional (2)
    int m_layout;               ///< The shape format of inputs X, initial_h and outputs Y, Y_h
    bool m_initialized;         ///< Internal buffers have been allocated
    bool m_reset_state;         ///< Hidden state must be seeded on the next forward call
    int m_gate_exponent;        ///< Exponent for gate calculations
    TensorBase *m_input_cache;  ///< Input cache tensor
    TensorBase *m_hidden_cache; ///< Hidden cache tensor
    TensorBase *m_h_prev;       ///< Previous hidden state tensor
    int16_t *m_wx_all;          ///< Precomputed W * x for all steps in the current call, [seq][3 * hidden]
    int m_wx_all_steps;         ///< Number of sequence steps allocated in m_wx_all
    float *m_gates;
    float m_sigmoid_lut_min;
    float m_sigmoid_lut_max;
    float m_tanh_lut_min;
    float m_tanh_lut_max;
    std::shared_ptr<math::SigmoidLUT> m_sigmoid_lut; ///< Sigmoid lookup table for quantized operations
    std::shared_ptr<math::TanhLUT> m_tanh_lut;       ///< Tanh lookup table for quantized operations

    static constexpr float DEFAULT_SIGMOID_LUT_MIN = -6.3f;
    static constexpr float DEFAULT_SIGMOID_LUT_MAX = 6.3f;
    static constexpr float DEFAULT_TANH_LUT_MIN = -3.5f;
    static constexpr float DEFAULT_TANH_LUT_MAX = 3.5f;

    // Share tables only when both their exponent and range match.
    template <typename LUT>
    static std::shared_ptr<LUT> shared_lut(int exponent, float lo, float hi)
    {
        using key_t = std::tuple<int, float, float>;
        static std::map<key_t, std::weak_ptr<LUT>> cache;
        std::weak_ptr<LUT> &slot = cache[key_t(exponent, lo, hi)];
        std::shared_ptr<LUT> lut = slot.lock();
        if (!lut) {
            lut = std::make_shared<LUT>(exponent, lo, hi, MALLOC_CAP_DEFAULT);
            slot = lut;
        }
        return lut;
    }

    bool initialize_hidden_state(TensorBase *initial_h, int direction)
    {
        if (!initial_h) {
            m_h_prev->memset(0);
            return true;
        }
        if (direction < 0 || direction >= m_direction_num || m_hidden_size <= 0) {
            ESP_LOGE("GRU", "Invalid direction %d or hidden size %d", direction, m_hidden_size);
            return false;
        }
        const size_t offset = (size_t)direction * (size_t)m_hidden_size;
        if ((size_t)initial_h->get_size() < offset + (size_t)m_hidden_size) {
            ESP_LOGE("GRU",
                     "Invalid initial_h size %d for direction %d and hidden size %d",
                     initial_h->get_size(),
                     direction,
                     m_hidden_size);
            return false;
        }
        const uint8_t *src = (const uint8_t *)initial_h->data + offset * initial_h->get_dtype_bytes();
        TensorBase state({m_hidden_size}, src, initial_h->get_exponent(), initial_h->get_dtype(), false);
        if (!m_h_prev->assign(&state)) {
            ESP_LOGE("GRU", "Failed to convert initial_h for direction %d", direction);
            return false;
        }
        return true;
    }

    GRU(const char *name = NULL,
        int hidden_size = 0,
        int direction_num = 1,
        int layout = 0,
        int gate_exponent = -8,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE,
        float sigmoid_lut_min = DEFAULT_SIGMOID_LUT_MIN,
        float sigmoid_lut_max = DEFAULT_SIGMOID_LUT_MAX,
        float tanh_lut_min = DEFAULT_TANH_LUT_MIN,
        float tanh_lut_max = DEFAULT_TANH_LUT_MAX) :
        Module(name, inplace, quant_type),
        m_hidden_size(hidden_size),
        m_direction_num(direction_num),
        m_layout(layout),
        m_gate_exponent(gate_exponent),
        m_sigmoid_lut_min(sigmoid_lut_min),
        m_sigmoid_lut_max(sigmoid_lut_max),
        m_tanh_lut_min(tanh_lut_min),
        m_tanh_lut_max(tanh_lut_max)
    {
        m_initialized = false;
        m_reset_state = true;
        m_input_cache = nullptr;
        m_hidden_cache = nullptr;
        m_h_prev = nullptr;
        m_wx_all = nullptr;
        m_wx_all_steps = 0;
        m_gates = nullptr;

        if ((quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) && hidden_size > 0) {
            uint32_t caps = MALLOC_CAP_DEFAULT;
            m_sigmoid_lut = shared_lut<math::SigmoidLUT>(m_gate_exponent, m_sigmoid_lut_min, m_sigmoid_lut_max);
            m_tanh_lut = shared_lut<math::TanhLUT>(m_gate_exponent, m_tanh_lut_min, m_tanh_lut_max);
            m_gates = (float *)tool::malloc_aligned(3 * hidden_size * sizeof(float), caps);
        } else if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            ESP_LOGE("GRU", "hidden_size must be positive, got %d", hidden_size);
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
        if (m_gates) {
            free(m_gates);
        }
        if (m_wx_all) {
            free(m_wx_all);
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
        if (x_shape.size() != 3) {
            ESP_LOGE("GRU", "X must have rank 3");
            return output_shapes;
        }
        if (m_layout != 0 || (m_direction_num != 1 && m_direction_num != 2)) {
            ESP_LOGE("GRU", "Only layout=0 with forward or bidirectional GRU is supported");
            return output_shapes;
        }
        int seq_length = x_shape[0];
        int batch_size = x_shape[1];
        if (batch_size != 1) {
            ESP_LOGE("GRU", "Only batch_size=1 is supported, got %d", batch_size);
            return output_shapes;
        }

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
        if (x_shape.size() != 3 || m_layout != 0 || (m_direction_num != 1 && m_direction_num != 2)) {
            ESP_LOGE("GRU", "Only rank-3 X and layout=0 with forward or bidirectional GRU are supported");
            return;
        }
        if (x_shape[1] != 1) {
            ESP_LOGE("GRU", "Only batch_size=1 is supported, got %d", x_shape[1]);
            return;
        }
        if (!m_gates || !m_sigmoid_lut || !m_tanh_lut) {
            ESP_LOGE("GRU", "Quantized GRU activation buffers are not initialized");
            return;
        }
        int seq_length = x_shape[0];
        int input_size = x_shape[2];

        if (!m_initialized) {
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

            m_input_cache = new TensorBase({3 * m_hidden_size}, nullptr, m_gate_exponent, DATA_TYPE_INT16);
            m_hidden_cache = new TensorBase({3 * m_hidden_size}, nullptr, m_gate_exponent, DATA_TYPE_INT16);
            m_h_prev = new TensorBase({m_hidden_size}, nullptr, output_h->get_exponent(), output_h->get_dtype());
            m_initialized = true;
        }

        // Bidirectional GRU processes the full sequence offline. Reset its
        // initial state on every call; never carry reverse state across calls.
        bool reset_hidden_state = m_reset_state || m_direction_num == 2;

        // Process the sequence. batch_size=1 and layout=0 were validated above.
        T *x_items = input_x->get_element_ptr<T>();
        T *w_items = input_w->get_element_ptr<T>();
        T *r_items = input_r->get_element_ptr<T>();
        T *h_prev_items = m_h_prev->get_element_ptr<T>();
        int16_t *wb_items = nullptr;
        int16_t *rb_items = nullptr;
        if (input_b) {
            wb_items = input_b->get_element_ptr<int16_t>();
            rb_items = wb_items + 3 * m_hidden_size;
        }
        // Preserve m_h_prev across chunks. After reset(), seed it from the
        // initial_h supplied to this call, or from zero when initial_h is absent.
        if (reset_hidden_state) {
            if (!initialize_hidden_state(initial_h, 0)) {
                return;
            }
        }

        T *y_items = output_y->get_element_ptr<T>();
        T *h_items = output_h->get_element_ptr<T>();
        int16_t *input_cache_items = m_input_cache->get_element_ptr<int16_t>();
        int16_t *hidden_cache_items = m_hidden_cache->get_element_ptr<int16_t>();
        T *x_base = x_items;
        if (seq_length > m_wx_all_steps) {
            if (seq_length < 0 || m_hidden_size <= 0) {
                ESP_LOGE("GRU", "Invalid GRU cache shape: seq=%d hidden=%d", seq_length, m_hidden_size);
                return;
            }
            const size_t row_elements = 3u * static_cast<size_t>(m_hidden_size);
            const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(int16_t);
            if (static_cast<size_t>(seq_length) > max_elements / row_elements) {
                ESP_LOGE("GRU", "GRU W*x cache size overflow: seq=%d hidden=%d", seq_length, m_hidden_size);
                return;
            }
            const size_t bytes = static_cast<size_t>(seq_length) * row_elements * sizeof(int16_t);
            int16_t *new_wx_all = (int16_t *)tool::malloc_aligned(bytes, MALLOC_CAP_DEFAULT);
            if (!new_wx_all) {
                ESP_LOGE("GRU", "Failed to allocate %zu bytes for W*x cache", bytes);
                return;
            }
            free(m_wx_all);
            m_wx_all = new_wx_all;
            m_wx_all_steps = seq_length;
        }
        m_reset_state = false;

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
                if (reset_hidden_state) {
                    if (!initialize_hidden_state(initial_h, 1)) {
                        return;
                    }
                }
            }

            // W*x is independent of the recurrent state, so batch every time
            // step for both directions. The reverse direction reads this buffer
            // in reverse temporal order below.
            if constexpr (std::is_same_v<T, int8_t>) {
                base::mat_vecs_dotprod(w_items, x_base, m_wx_all, 3 * m_hidden_size, input_size, seq_length, w_shift);
            } else {
                base::mat_vecs_dotprod(w_items,
                                       x_base,
                                       m_wx_all,
                                       3 * m_hidden_size,
                                       input_size,
                                       seq_length,
                                       input_size,
                                       3 * m_hidden_size,
                                       w_shift);
            }

            for (int seq = 0; seq < seq_length; seq++) {
                int t = (di == 0) ? seq : seq_length - 1 - seq;
                input_cache_items = m_wx_all + t * 3 * m_hidden_size;
                // R * h_prev
                if constexpr (std::is_same_v<T, int8_t>) {
                    base::mat_vecs_dotprod(
                        r_items, h_prev_items, hidden_cache_items, 3 * m_hidden_size, m_hidden_size, 1, r_shift);
                } else {
                    base::mat_vec_dotprod(
                        r_items, h_prev_items, hidden_cache_items, 3 * m_hidden_size, m_hidden_size, r_shift);
                }

                // add bias
                if (input_b) {
                    // The dot products above SATURATE at int16, so a plain int16 += wraps once
                    // the cache is at a rail, flipping a saturated gate to the opposite side.
                    // PPQ adds the bias in float and never wraps: saturate here to match it.
                    for (int i = 0; i < 3 * m_hidden_size; i++) {
                        tool::truncate(input_cache_items[i], (int)input_cache_items[i] + (int)wb_items[i]);
                        tool::truncate(hidden_cache_items[i], (int)hidden_cache_items[i] + (int)rb_items[i]);
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
        if (x_shape.size() != 3 || m_layout != 0 || (m_direction_num != 1 && m_direction_num != 2)) {
            ESP_LOGE("GRU", "Only rank-3 X and layout=0 with forward or bidirectional GRU are supported");
            return;
        }
        if (x_shape[1] != 1) {
            ESP_LOGE("GRU", "Only batch_size=1 is supported, got %d", x_shape[1]);
            return;
        }
        int seq_length = x_shape[0];
        int input_size = x_shape[2];
        if (!m_initialized) {
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

            m_input_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_hidden_cache = new TensorBase({3 * m_hidden_size}, nullptr);
            m_h_prev = new TensorBase({m_hidden_size}, nullptr);
            m_initialized = true;
        }

        // Bidirectional GRU processes the full sequence offline. Reset its
        // initial state on every call; never carry reverse state across calls.
        bool reset_hidden_state = m_reset_state || m_direction_num == 2;

        // Process the sequence. batch_size=1 and layout=0 were validated above.
        float *x_items = input_x->get_element_ptr<float>();
        float *w_items = input_w->get_element_ptr<float>();
        float *r_items = input_r->get_element_ptr<float>();
        float *wb_items = nullptr;
        float *rb_items = nullptr;
        float *h_prev_items = m_h_prev->get_element_ptr<float>();
        if (input_b) {
            wb_items = input_b->get_element_ptr<float>();
            rb_items = wb_items + 3 * m_hidden_size;
        }
        if (reset_hidden_state) {
            if (!initialize_hidden_state(initial_h, 0)) {
                return;
            }
        }
        m_reset_state = false;

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
                if (reset_hidden_state) {
                    if (!initialize_hidden_state(initial_h, 1)) {
                        return;
                    }
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
        float sigmoid_lut_min = DEFAULT_SIGMOID_LUT_MIN;
        float sigmoid_lut_max = DEFAULT_SIGMOID_LUT_MAX;
        float tanh_lut_min = DEFAULT_TANH_LUT_MIN;
        float tanh_lut_max = DEFAULT_TANH_LUT_MAX;

        fbs_model->get_operation_attribute(node_name, "hidden_size", hidden_size);
        fbs_model->get_operation_attribute(node_name, "direction", direction);
        fbs_model->get_operation_attribute(node_name, "layout", layout);
        fbs_model->get_operation_attribute(node_name, "linear_before_reset", linear_before_reset);
        fbs_model->get_operation_attribute(node_name, "gate_exponent", gate_exponent);
        fbs_model->get_operation_attribute(node_name, "sigmoid_lut_min", sigmoid_lut_min);
        fbs_model->get_operation_attribute(node_name, "sigmoid_lut_max", sigmoid_lut_max);
        fbs_model->get_operation_attribute(node_name, "tanh_lut_min", tanh_lut_min);
        fbs_model->get_operation_attribute(node_name, "tanh_lut_max", tanh_lut_max);

        if (linear_before_reset == 0) {
            ESP_LOGE("GRU", "linear_before_reset must be 1. The GRU implementation is allgned with PyTorch.");
            return op;
        }
        if (direction != "forward" && direction != "bidirectional") {
            ESP_LOGE("GRU", "Only direction=forward or bidirectional is supported, got %s", direction.c_str());
            return op;
        }
        if (layout != 0) {
            ESP_LOGE("GRU", "Only layout=0 is supported, got %d", layout);
            return op;
        }

        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT ||
            quant_type == QUANT_TYPE_FLOAT32) {
            int direction_num = direction == "bidirectional" ? 2 : 1;
            op = new GRU(node_name.c_str(),
                         hidden_size,
                         direction_num,
                         layout,
                         gate_exponent,
                         MODULE_NON_INPLACE,
                         quant_type,
                         sigmoid_lut_min,
                         sigmoid_lut_max,
                         tanh_lut_min,
                         tanh_lut_max);
        }
        return op;
    }

    void reset() override
    {
        if (m_input_cache != nullptr) {
            m_input_cache->memset(0);
        }
        if (m_hidden_cache != nullptr) {
            m_hidden_cache->memset(0);
        }
        // Buffers stay allocated. The next forward call reseeds m_h_prev from
        // that call's initial_h (or zero), while calls between resets preserve it.
        m_reset_state = true;
    }

    void print() override
    {
        ESP_LOGI("GRU",
                 "hidden_size: %d, direction: %s, layout: %d, gate exponent:%d, quant_type: %s, "
                 "sigmoid_lut:[%.3f, %.3f], tanh_lut:[%.3f, %.3f]",
                 m_hidden_size,
                 (m_direction_num == 2) ? "bidirectional" : "forward",
                 m_layout,
                 m_gate_exponent,
                 quant_type_to_string(quant_type),
                 m_sigmoid_lut_min,
                 m_sigmoid_lut_max,
                 m_tanh_lut_min,
                 m_tanh_lut_max);
    }
};

} // namespace module
} // namespace dl
