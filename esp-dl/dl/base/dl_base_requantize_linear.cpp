#include "dl_base_requantize_linear.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace base {

std::vector<requantizeArgsType> get_requantize_operation_args(TensorBase *output,
                                                              TensorBase *input,
                                                              const runtime_mode_t runtime_mode)
{
    assert(input->get_size() == output->get_size());

    requantizeArgsType args;
    args.input_element = input->get_element_ptr();
    args.output_element = output->get_element_ptr(); // output

    args.output_scale = -1;
    args.output_shift = output->exponent - input->exponent;
    if (args.output_shift < 0) {
        args.output_scale = 1 << (-args.output_shift);
        args.output_shift = 0;
    }

    // for ISA
    int in_align = 16 / input->get_dtype_bytes();
    int out_align = 16 / output->get_dtype_bytes();
    args.size_div_x = output->get_size() / out_align;
    args.in_size_remainder = (input->get_size() % in_align) * input->get_dtype_bytes();
    args.out_size_remainder = (output->get_size() % out_align) * output->get_dtype_bytes();

    // slice
    std::vector<requantizeArgsType> m_args(1, args);
    if (runtime_mode == RUNTIME_MODE_MULTI_CORE) {
        // TODO:
    }

    return m_args;
}

void load_requantize_linear_func(ImplFunc_t<int8_t, int8_t> &impl_func, const requantizeArgsType &args)
{
    if (!(reinterpret_cast<uintptr_t>(args.input_element) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 0xf)) {
#if CONFIG_ESP32P4_BOOST
        impl_func = dl_esp32p4_s8_s8_requantize_linear;
#elif CONFIG_TIE728_BOOST
        impl_func = dl_tie728_s8_s8_requantize_linear;
#endif
    } else {
        // TODO: unaligned
    }
}

void load_requantize_linear_func(ImplFunc_t<int8_t, int16_t> &impl_func, const requantizeArgsType &args)
{
    if (!(reinterpret_cast<uintptr_t>(args.input_element) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 0xf)) {
#if CONFIG_ESP32P4_BOOST
        impl_func = dl_esp32p4_s8_s16_requantize_linear;
#elif CONFIG_TIE728_BOOST
        impl_func = dl_tie728_s8_s16_requantize_linear;
#endif
    } else {
        // TODO: unaligned
    }
}

void load_requantize_linear_func(ImplFunc_t<int16_t, int16_t> &impl_func, const requantizeArgsType &args)
{
    if (!(reinterpret_cast<uintptr_t>(args.input_element) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 0xf)) {
#if CONFIG_ESP32P4_BOOST
        impl_func = dl_esp32p4_s16_s16_requantize_linear;
#elif CONFIG_TIE728_BOOST
        impl_func = dl_tie728_s16_s16_requantize_linear;
#endif
    } else {
        // TODO: unaligned
    }
}

void load_requantize_linear_func(ImplFunc_t<int16_t, int8_t> &impl_func, const requantizeArgsType &args)
{
    if (!(reinterpret_cast<uintptr_t>(args.input_element) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(args.output_element) & 0xf)) {
#if CONFIG_ESP32P4_BOOST
        impl_func = dl_esp32p4_s16_s8_requantize_linear;
#elif CONFIG_TIE728_BOOST
        impl_func = dl_tie728_s16_s8_requantize_linear;
#endif
    } else {
        // TODO: unaligned
    }
}

template <typename out_feature_t, typename in_feature_t>
void requantize_linear(void *const args_ptr)
{
    requantizeArgsType &args = *((requantizeArgsType *)args_ptr);

    ImplFunc_t<out_feature_t, in_feature_t> impl_func;

#if CONFIG_ESP32P4_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif

    load_requantize_linear_func(impl_func, args);
    if (!impl_func) {
        ESP_LOGE("requantize_linear", "impl_func is empty.");
    } else {
        impl_func(static_cast<out_feature_t *>(args.output_element),
                  static_cast<in_feature_t *>(args.input_element),
                  args_ptr);
    }
}

template void requantize_linear<int8_t, int8_t>(void *const args_ptr);
template void requantize_linear<int8_t, int16_t>(void *const args_ptr);
template void requantize_linear<int16_t, int16_t>(void *const args_ptr);
template void requantize_linear<int16_t, int8_t>(void *const args_ptr);

} // namespace base
} // namespace dl
