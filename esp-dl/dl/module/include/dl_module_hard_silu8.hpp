// ═══════════════════════════════════════════════════════════════════
// Author: Boumedine Billal (https://github.com/BoumedineBillal)
//
// Contribution:
//   - Designed the HardSiluPie8 ESP-DL module: runtime C++ wrapper
//     for the HardSiLU8 PIE SIMD kernel with per-layer learned scale.
//   - Derives all quantization constants (half_buf, clamp_hi, sar_total)
//     from the input/output exponents at construction time.
//   - Integrates with the ESP-DL FlatBuffers deserialization pipeline
//     (scale_int attribute stored in the .espdl model).
// ═══════════════════════════════════════════════════════════════════

#pragma once

#include "dl_module_base.hpp"
#include "dl_base_esp32p4.h"

namespace dl {
namespace module {

/**
 * @brief HardSiluPie8 — ESP32-P4 PIE SIMD activation with learned scale
 *
 * Float:   y = x × clamp(x/8 + 0.5, 0, 1) × (scale_int / 256)
 * Integer: y_q = x_q × clamp(x_q + 2^(-e1+2), 0, 2^(-e1+3)) × 2^(2*e1-3-e2)
 *          then: y_q = y_q × scale_int >> 8
 *          e1 = input exponent, e2 = output exponent
 *          the × 2^(...) is the SAR shift inside esp.vmul.s16
 *
 * INT8-only, 12 PIE instructions per 16 elements. Uses /8 (2^3) for shift-friendly quantization.
 * esp.vmul.s16 uses banker's rounding (HALF_EVEN) via global PIE CFG register.
 *
 * Constraints (enforced by assert):
 *   - quant_type must be QUANT_TYPE_SYMM_8BIT (no INT16, no F32, no fallback)
 *   - input->size must be a multiple of 16 (no remainder handling)
 *
 * Constants (precomputed from input exponent e1, output exponent e2):
 *   half_offset = 2^(-e1+2)
 *   clamp_hi    = 2^(-e1+3)
 *   SAR         = -2*e1 + 3 + e2
 *
 * Scale (from neural morphing, learned per-layer):
 *   scale_int   = round(sigmoid(a_param) * 256), range [0, 255]
 *   scale_int=256 means no scaling (identity)
 */
class HardSiluPie8 : public Module {
private:
    int m_scale_int;               // learned scale: sigmoid(a) * 256, default 256 (no-op)
    int16_t m_scale_buf[8] __attribute__((aligned(16))); // broadcast for PIE SIMD

public:
    HardSiluPie8(const char *name = NULL,
                 module_inplace_t inplace = MODULE_NON_INPLACE,
                 quant_type_t quant_type = QUANT_TYPE_NONE,
                 int scale_int = 256) :
        Module(name, inplace, quant_type),
        m_scale_int(scale_int)
    {
        for (int i = 0; i < 8; i++) {
            m_scale_buf[i] = (int16_t)m_scale_int;
        }
    }

    ~HardSiluPie8() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        return {input_shapes[0]};
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        assert(quant_type == QUANT_TYPE_SYMM_8BIT && "HardSiluPie8: only INT8 supported");
        assert((input->size % 16) == 0 && "HardSiluPie8: tensor size must be multiple of 16");

        int8_t *input_ptr = (int8_t *)input->get_element_ptr();
        int8_t *output_ptr = (int8_t *)output->get_element_ptr();

        int e1 = input->exponent;
        int e2 = output->exponent;

        // Precompute constants from exponents
        int half_val   = 1 << (-e1 + 2);
        int clamp_hi   = 1 << (-e1 + 3);
        int sar_total  = -2 * e1 + 3 + e2 + 8;  // +8 absorbs scale /256
        int n_16 = input->size / 16;

        // Broadcast half_offset into aligned buffer for q3 load
        int16_t half_buf[8] __attribute__((aligned(16)));
        for (int i = 0; i < 8; i++) {
            half_buf[i] = (int16_t)half_val;
        }

        dl_esp32p4_s8_hard_silu8(output_ptr, input_ptr, n_16,
                                      half_buf, clamp_hi, sar_total, m_scale_buf);
    }

    void forward_args(void *args) {}

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        int scale_int = 256;
        fbs_model->get_operation_attribute(node_name, "scale_int", scale_int);

        return new HardSiluPie8(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER,
                                quant_type, scale_int);
    }

    void print()
    {
        ESP_LOGI("HardSiluPie8", "quant_type: %s, scale_int: %d.",
                 quant_type_to_string(quant_type), m_scale_int);
    }
};

} // namespace module
} // namespace dl
