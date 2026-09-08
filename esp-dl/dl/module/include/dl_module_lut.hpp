#pragma once

#include "dl_base_lut.hpp"
#include "dl_module_base.hpp"
#include <cstdint>

namespace dl {
namespace module {
/**
 * @brief Apply an int8 or int16 lookup table.
 *
 * Int16 uses nearest-neighbor indexing when the table step is greater than one.
 */
class LUT : public Module {
public:
    /**
     * @brief Construct a new LUT object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    LUT(const char *name = NULL,
        module_inplace_t inplace = MODULE_INPLACE_CHANGED_BUFFER,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    bool is_lut_module() const override { return true; }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        assert(m_inputs_index.size() >= 2);
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *table = context->get_tensor(m_inputs_index.back());
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        assert(table != nullptr);
        assert(output->exponent == table->exponent);

        LutTask base;
        base.output = output->get_element_ptr();
        base.input = input->get_element_ptr();
        base.table = table->get_element_ptr();
        base.size = static_cast<int32_t>(input->size);
        base.bits = (quant_type == QUANT_TYPE_SYMM_16BIT) ? 16 : 8;
        if (base.bits == 16) {
            assert(table->get_size() > 1);
            base.step = 65536 / (table->get_size() - 1);
        }

        // Keep each tile 16-byte aligned so it can use the PIE LUT kernel.
        const int32_t align = (base.bits == 16) ? 8 : 16;
        int32_t half = base.size / 2;
        half -= half % align;
        const bool can_split = half > 0 && half < base.size;
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
#if CONFIG_IDF_TARGET_ESP32P4
        const int32_t auto_min_size = (base.bits == 8)
            ? AUTO_DUAL_CORE_MIN_SIZE_S8
            : (base.step == 1 ? AUTO_DUAL_CORE_MIN_SIZE_S16_STEP1 : AUTO_DUAL_CORE_MIN_SIZE_S16_REDUCED);
        const bool auto_dual = mode == RUNTIME_MODE_AUTO && base.size >= auto_min_size;
#else
        const bool auto_dual = false;
#endif
        const bool dual = can_split && (mode == RUNTIME_MODE_MULTI_CORE || auto_dual);
#else
        (void)mode;
        const bool dual = false;
#endif
        if (dual) {
            LutTask t0 = base;
            LutTask t1 = base;
            t0.size = half;
            t1.size = base.size - half;
            const size_t elem = (base.bits == 16) ? sizeof(int16_t) : sizeof(int8_t);
            t1.output = static_cast<uint8_t *>(base.output) + static_cast<size_t>(half) * elem;
            t1.input = static_cast<const uint8_t *>(base.input) + static_cast<size_t>(half) * elem;
            module_forward_dual_core(this, &t0, &t1);
        } else {
            forward_args(&base);
        }
    }

    void forward_args(void *args)
    {
        const LutTask *t = static_cast<const LutTask *>(args);
        if (t->bits == 8) {
            base::lut_s8(static_cast<int8_t *>(t->output),
                         static_cast<const int8_t *>(t->input),
                         t->size,
                         static_cast<const int8_t *>(t->table));
        } else {
            assert((t->step & (t->step - 1)) == 0);
            base::lut_s16_nearest_neighbor(static_cast<int16_t *>(t->output),
                                           static_cast<const int16_t *>(t->input),
                                           t->size,
                                           static_cast<const int16_t *>(t->table),
                                           t->step);
        }
    }

    static bool has_lut(fbs::FbsModel *fbs_model, const std::string &node_name)
    {
        std::string lut_name;
        return fbs_model->get_operation_lut_name(node_name, lut_name) == ESP_OK;
    }

    /**
     * @brief deserialize LUT module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        if (!has_lut(fbs_model, node_name)) {
            ESP_LOGE("LUT", "Table is null!");
            return nullptr;
        }

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new LUT(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        } else {
            ESP_LOGE("LUT", "Only support QUANT_TYPE_SYMM_8BIT or QUANT_TYPE_SYMM_16BIT!");
        }
        return op;
    }

    void print() { ESP_LOGI("LUT", "quant_type: %s.", quant_type_to_string(quant_type)); }

private:
    // Conservative crossover points measured on ESP32-P4 at 400 MHz.
    static constexpr int32_t AUTO_DUAL_CORE_MIN_SIZE_S8 = 8192;
    static constexpr int32_t AUTO_DUAL_CORE_MIN_SIZE_S16_REDUCED = 8192;
    static constexpr int32_t AUTO_DUAL_CORE_MIN_SIZE_S16_STEP1 = 2048;

    struct LutTask {
        void *output = nullptr;
        const void *input = nullptr;
        const void *table = nullptr;
        int32_t size = 0;
        int step = 1;
        int bits = 8;
    };
};
} // namespace module
} // namespace dl
