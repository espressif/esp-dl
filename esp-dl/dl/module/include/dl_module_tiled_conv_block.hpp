#pragma once

#include "dl_base_conv2d.hpp"
#include "dl_module_base.hpp"
#include "dl_base_esp32p4.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" void dl_esp32p4_s16_lut_pie8(
    int16_t *output, int16_t *input, int32_t n_8,
    int16_t *table, int16_t *ones_buf, int16_t *xor_buf,
    int32_t shift);

namespace dl {
namespace module {

/**
 * @brief TiledConvBlock — Spatial H-tiling with fused activation.
 *
 * Tiles the output height into chunks of @c tile_h rows.
 * For each tile the pipeline is: Conv → Act (in-place on output).
 * Conv writes directly to the output tensor (PSRAM); the tile region
 * stays hot in L2 cache and Act runs in-place on the same cached data.
 *
 * Supports optional channel tiling (@c c_tile) for large C_out.
 *
 * Inputs:
 *   [0] activation input
 *   [1] Conv filter  (HWIO, packed as (N/16)HWC16)
 *   [2] Conv bias    (optional)
 */
class TiledConvBlock : public Module {
private:
    std::vector<int> m_pads;       /*!< padding [top, bottom, left, right] */
    std::vector<int> m_strides;    /*!< stride along each spatial axis */
    std::vector<int> m_dilations;  /*!< dilation along each spatial axis */
    int m_group;                   /*!< convolution group (must be 1) */
    bool m_bias_reseted;           /*!< bias layout reset flag */

    int m_tile_h;                  /*!< output rows per tile */
    int m_c_tile;                  /*!< output channels per tile (0 = all) */

    std::string m_act_type;        /*!< "" | "HardSiluPie8" | "Swish" | "LUT" */
    TensorBase *m_act_table;       /*!< LUT table for Swish/LUT activations */
    int m_act1_input_exponent;     /*!< conv output exponent (= act input exponent) */
    int m_scale_int;               /*!< HardSiluPie8 learned scale (256 = unity) */
    int16_t m_scale_buf[8] __attribute__((aligned(16))); /*!< broadcast scale for PIE SIMD */

    /**
     * @brief Reset bias layout on first forward pass.
     */
    void reset_bias(ModelContext *context)
    {
        if (!m_bias_reseted) {
            if (m_inputs_index.size() >= 3) {
                TensorBase *bias = context->get_tensor(m_inputs_index[2]);
                if (bias) {
                    bias->reset_bias_layout(quant_type, false);
                }
            }
            m_bias_reseted = true;
        }
    }

    /**
     * @brief Apply fused activation in-place.
     *
     * @param dst              destination buffer (may alias @p src)
     * @param src              source buffer
     * @param n_elements       number of elements to process
     * @param act_input_tensor tensor view with input exponent
     * @param act_output_tensor tensor view with output exponent
     */
    void apply_act(void *dst, void *src, size_t n_elements,
                   TensorBase *act_input_tensor,
                   TensorBase *act_output_tensor)
    {
        if (m_act_type.empty()) {
            if (dst != src) {
                int elem_size = (quant_type == QUANT_TYPE_SYMM_16BIT) ? 2 : 1;
                memcpy(dst, src, n_elements * elem_size);
            }
            return;
        }

        if (m_act_type == "HardSiluPie8") {
            assert(quant_type == QUANT_TYPE_SYMM_8BIT);
            assert((n_elements % 16) == 0);

            int e1 = act_input_tensor->exponent;
            int e2 = act_output_tensor->exponent;
            int half_val  = 1 << (-e1 + 2);
            int clamp_hi  = 1 << (-e1 + 3);
            int sar_total = -2 * e1 + 3 + e2 + 8;

            int16_t half_buf[8] __attribute__((aligned(16)));
            for (int i = 0; i < 8; i++) {
                half_buf[i] = (int16_t)half_val;
            }

            dl_esp32p4_s8_hard_silu_pie8(
                (int8_t *)dst, (int8_t *)src,
                n_elements / 16, half_buf, clamp_hi, sar_total, m_scale_buf);

        } else if (m_act_type == "Swish" || m_act_type == "LUT") {
            if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                int8_t *d = (int8_t *)dst;
                int8_t *s = (int8_t *)src;
                if (m_act_table) {
                    int8_t *t = (int8_t *)m_act_table->get_element_ptr();
                    for (size_t i = 0; i < n_elements; i++) {
                        d[i] = t[s[i] + 128];
                    }
                } else {
                    int e1 = act_input_tensor->exponent;
                    int e2 = act_output_tensor->exponent;
                    float in_scale  = DL_SCALE(e1);
                    float out_scale = DL_RESCALE(e2);
                    for (size_t i = 0; i < n_elements; i++) {
                        float v = s[i] * in_scale;
                        v = dl::math::sigmoid(v) * v;
                        tool::truncate(d[i], tool::round(v * out_scale));
                    }
                }
            } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                int16_t *d = (int16_t *)dst;
                int16_t *s = (int16_t *)src;
                if (m_act_table) {
                    int16_t *t = (int16_t *)m_act_table->get_element_ptr();
                    int step = 65536 / (m_act_table->get_size() - 1);

                    if (step > 1
                        && (step & (step - 1)) == 0
                        && (n_elements % 8) == 0
                        && ((uintptr_t)s % 16) == 0
                        && ((uintptr_t)d % 16) == 0)
                    {
                        static __attribute__((aligned(16))) int16_t ones[8] =
                            {1, 1, 1, 1, 1, 1, 1, 1};
                        static __attribute__((aligned(16))) int16_t xor_buf[8] = {
                            (int16_t)0x8000, (int16_t)0x8000,
                            (int16_t)0x8000, (int16_t)0x8000,
                            (int16_t)0x8000, (int16_t)0x8000,
                            (int16_t)0x8000, (int16_t)0x8000};
                        dl_esp32p4_s16_lut_pie8(d, s, n_elements / 8,
                                                 t, ones, xor_buf,
                                                 __builtin_ctz(step));
                    } else if (step == 1) {
                        for (size_t i = 0; i < n_elements; i++) {
                            d[i] = t[s[i] + 32768];
                        }
                    } else {
                        for (size_t i = 0; i < n_elements; i++) {
                            int idx = s[i] + 32768;
                            int len = idx % step;
                            idx = idx / step;
                            int x = t[idx];
                            int y = t[idx + 1];
                            d[i] = x + len * (y - x) / step;
                        }
                    }
                } else {
                    int e1 = act_input_tensor->exponent;
                    int e2 = act_output_tensor->exponent;
                    float in_scale  = DL_SCALE(e1);
                    float out_scale = DL_RESCALE(e2);
                    for (size_t i = 0; i < n_elements; i++) {
                        float v = s[i] * in_scale;
                        v = dl::math::sigmoid(v) * v;
                        tool::truncate(d[i], tool::round(v * out_scale));
                    }
                }
            }
        } else {
            ESP_LOGE("TiledConvBlock", "Unsupported act: %s", m_act_type.c_str());
        }
    }

public:
    /**
     * @brief Construct a new TiledConvBlock object.
     *
     * @param pads                padding [top, bottom, left, right]
     * @param strides             stride along each spatial axis
     * @param dilations           dilation along each spatial axis
     * @param tile_h              output rows per tile
     * @param act_type            fused activation type
     * @param act_table           LUT table (owned, may be NULL)
     * @param act1_input_exponent conv output exponent
     * @param scale_int           HardSiluPie8 learned scale
     * @param c_tile              output channel tile size (0 = no tiling)
     * @param name                module name
     * @param group               convolution group
     * @param quant_type          quantization type
     */
    TiledConvBlock(std::vector<int> pads,
                   std::vector<int> strides,
                   std::vector<int> dilations,
                   int tile_h,
                   const std::string &act_type,
                   TensorBase *act_table,
                   int act1_input_exponent,
                   int scale_int = 256,
                   int c_tile = 0,
                   const char *name = NULL,
                   int group = 1,
                   quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type),
        m_pads(pads), m_strides(strides), m_dilations(dilations),
        m_group(group), m_bias_reseted(false),
        m_tile_h(tile_h), m_c_tile(c_tile),
        m_act_type(act_type), m_act_table(act_table),
        m_act1_input_exponent(act1_input_exponent),
        m_scale_int(scale_int)
    {
        for (int i = 0; i < 8; i++) {
            m_scale_buf[i] = (int16_t)m_scale_int;
        }
    }

    /**
     * @brief Destroy the TiledConvBlock object.
     */
    ~TiledConvBlock()
    {
        if (m_act_table) {
            delete m_act_table;
        }
    }

    /**
     * @brief Calculate the output shape.
     *
     * @param input_shapes shapes of all inputs
     * @return output shapes
     */
    std::vector<std::vector<int>> get_output_shape(
        std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes.size() >= 2);
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> filter_shape = input_shapes[1];
        std::vector<int> output_shape = input_shape;

        output_shape[1] =
            (input_shape[1] + m_pads[0] + m_pads[1]
             - m_dilations[0] * (filter_shape[0] - 1) - 1) / m_strides[0] + 1;
        if (input_shape.size() == 4) {
            output_shape[2] =
                (input_shape[2] + m_pads[2] + m_pads[3]
                 - m_dilations[1] * (filter_shape[1] - 1) - 1) / m_strides[1] + 1;
            output_shape[3] = filter_shape[3];
        }
        return {output_shape};
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::conv2d<int8_t, int32_t, int32_t>(args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::conv2d<int16_t, int32_t, int64_t>(args);
        }
    }

    /**
     * @brief Run tiled convolution with fused activation.
     *
     * For each H-tile:
     *   1. Conv(input_tile) → output  (PSRAM, L2-cached)
     *   2. Act(output)      → output  (in-place, L2-cached hot)
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        reset_bias(context);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input  = context->get_tensor(m_inputs_index[0]);
        TensorBase *filter = context->get_tensor(m_inputs_index[1]);
        TensorBase *bias   = nullptr;
        if (m_inputs_index.size() >= 3) {
            bias = context->get_tensor(m_inputs_index[2]);
        }
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        const int H_in  = input->shape[1];
        const int W_in  = input->shape[2];
        const int C_in  = input->shape[3];
        const int H_out = output->shape[1];
        const int W_out = output->shape[2];
        const int C_out = output->shape[3];
        const int kH    = filter->shape[0];
        const int kW    = filter->shape[1];
        const int eff_kH = m_dilations[0] * (kH - 1) + 1;
        const int c_tile = (m_c_tile > 0 && m_c_tile < C_out) ? m_c_tile : C_out;

        // Pre-allocate reusable vectors (avoid heap alloc in hot loop)
        std::vector<int> tile_pads(4);
        std::vector<int> tile_input_shape(4);
        std::vector<int> filter_slice_shape(4);
        std::vector<int> tile_output_shape(4);
        std::vector<int> full_tile_shape(4);
        tile_pads[2] = m_pads[2];
        tile_pads[3] = m_pads[3];
        tile_input_shape[0] = 1;
        tile_input_shape[2] = W_in;
        tile_input_shape[3] = C_in;
        filter_slice_shape[0] = kH;
        filter_slice_shape[1] = kW;
        filter_slice_shape[2] = C_in;
        tile_output_shape[0] = 1;
        tile_output_shape[2] = W_out;
        full_tile_shape[0] = 1;
        full_tile_shape[2] = W_out;
        full_tile_shape[3] = C_out;

        for (int h_out = 0; h_out < H_out; h_out += m_tile_h) {
            int tile_h_actual = std::min(m_tile_h, H_out - h_out);
            int tile_n_elements = tile_h_actual * W_out * C_out;

            T *output_dst = (T *)output->get_element_ptr()
                            + h_out * W_out * C_out;

            // Compute input region for this output tile
            int h_in_start   = h_out * m_strides[0] - m_pads[0];
            int h_in_end     = h_in_start + (tile_h_actual - 1) * m_strides[0]
                               + eff_kH - 1;
            int pad_top      = std::max(0, -h_in_start);
            int pad_bottom   = std::max(0, h_in_end - (H_in - 1));
            int h_in_clamped = std::max(0, h_in_start);
            int actual_h_in  = (h_in_end - h_in_start + 1) - pad_top - pad_bottom;

            tile_pads[0] = pad_top;
            tile_pads[1] = pad_bottom;

            T *tile_input_ptr = (T *)input->get_element_ptr()
                                + h_in_clamped * W_in * C_in;
            tile_input_shape[1] = actual_h_in;
            TensorBase tile_input(tile_input_shape, tile_input_ptr,
                                  input->exponent, input->dtype, false);

            // Channel tile loop
            for (int c_start = 0; c_start < C_out; c_start += c_tile) {
                int c_actual = std::min(c_tile, C_out - c_start);

                T *filter_slice_ptr = (T *)filter->get_element_ptr()
                                      + c_start * kH * kW * C_in;
                filter_slice_shape[3] = c_actual;
                TensorBase filter_slice(filter_slice_shape, filter_slice_ptr,
                                        filter->exponent, filter->dtype, false);

                tile_output_shape[1] = tile_h_actual;
                tile_output_shape[3] = c_actual;
                TensorBase tile_output(tile_output_shape, output_dst,
                                       m_act1_input_exponent, output->dtype, false);

                std::vector<base::ArgsType<T>> m_args =
                    base::get_conv_operation_args<T>(
                        &tile_output, &tile_input, tile_pads, &filter_slice,
                        m_strides, m_dilations, m_group, bias,
                        Linear, nullptr, mode);

                if (c_tile < C_out) {
                    for (auto &a : m_args) {
                        a.output_element = output_dst + c_start;
                        a.output_x_offset = C_out;
                        a.output_y_offset = W_out * C_out;
                        if (bias) {
                            if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                                a.bias_element = (const void *)(
                                    (const int64_t *)bias->get_element_ptr() + c_start);
                            } else {
                                a.bias_element = (const void *)(
                                    (const int32_t *)bias->get_element_ptr() + c_start);
                            }
                        }
                        if (filter->exponent.is_per_channel()
                            && a.tie_filter_channel_factor) {
                            for (int i = 0; i < c_actual; i++) {
                                a.tie_filter_channel_factor[i] =
                                    (T)(m_act1_input_exponent
                                        - filter->exponent.get(c_start + i)
                                        - input->exponent);
                            }
                        }
                    }
                }

                int task_size = m_args.size();
                if (task_size == 1) {
                    forward_args((void *)&m_args[0]);
                } else if (task_size == 2) {
                    module_forward_dual_core(this,
                        (void *)&m_args[0], (void *)&m_args[1]);
                }
            }

            // Fused activation in-place on L2-cached output
            full_tile_shape[1] = tile_h_actual;
            TensorBase full_tile_output(full_tile_shape, output_dst,
                                        m_act1_input_exponent, output->dtype, false);

            apply_act(output_dst, output_dst, tile_n_elements,
                      &full_tile_output, output);
        }
    }

    /**
     * @brief Deserialize TiledConvBlock module from flatbuffer model.
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        std::vector<int> pads, strides, dilations;
        int group = 1;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "pads", pads);
        fbs_model->get_operation_attribute(node_name, "strides", strides);
        fbs_model->get_operation_attribute(node_name, "dilations", dilations);
        fbs_model->get_operation_attribute(node_name, "group", group);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        if (pads.size() == 4) {
            pads = {pads[0], pads[2], pads[1], pads[3]};
        } else if (pads.size() == 0) {
            pads = {0, 0, 0, 0};
        }

        int tile_h = 0;
        fbs_model->get_operation_attribute(node_name, "tile_h", tile_h);
        if (tile_h <= 0) {
            ESP_LOGE("TiledConvBlock", "tile_h must be > 0, got %d", tile_h);
            return nullptr;
        }

        int act1_input_exponent = 0;
        fbs_model->get_operation_attribute(node_name, "act1_input_exponent", act1_input_exponent);

        int scale_int = 256;
        fbs_model->get_operation_attribute(node_name, "scale_int", scale_int);

        std::string act_type = "";
        fbs_model->get_operation_attribute(node_name, "act_type", act_type);

        TensorBase *act_table = nullptr;
        if (act_type == "Swish" || act_type == "LUT") {
            act_table = fbs_model->get_operation_lut(node_name);
            if (!act_table) {
                ESP_LOGW("TiledConvBlock",
                    "No LUT table for act_type=%s, will use scalar fallback",
                    act_type.c_str());
            }
        }

        int c_tile = 0;
        fbs_model->get_operation_attribute(node_name, "c_tile", c_tile);

        return new TiledConvBlock(
            pads, strides, dilations, tile_h,
            act_type, act_table, act1_input_exponent, scale_int, c_tile,
            node_name.c_str(), group, quant_type);
    }

    void print()
    {
        ESP_LOGI("TiledConvBlock",
                 "tile_h: %d, c_tile: %d, act: %s, pads: %s, strides: %s, "
                 "dilations: %s, quant_type: %s.",
                 m_tile_h, m_c_tile,
                 m_act_type.empty() ? "none" : m_act_type.c_str(),
                 vector_to_string(m_pads).c_str(),
                 vector_to_string(m_strides).c_str(),
                 vector_to_string(m_dilations).c_str(),
                 quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
