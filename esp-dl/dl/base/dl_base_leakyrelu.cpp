#include "dl_base_leakyrelu.hpp"

#include "dl_base_isa.hpp"

namespace dl
{
    namespace base
    {
        template <typename feature_t, typename buffer_t>
        void leakyrelu_11c(feature_t *output_ptr, feature_t *input_ptr, const ArgsType<feature_t> &args)
        {
            buffer_t temp;
            for (size_t output_c = 0; output_c < args.output_channel; output_c++)
            {
                output_ptr[output_c] = input_ptr[output_c];
                if (output_ptr[output_c] < 0)
                {
                    temp = DL_RIGHT_SHIFT((buffer_t)output_ptr[output_c] * (buffer_t)args.activation_alpha, args.activation_shift);
                    tool::truncate(output_ptr[output_c], temp);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // specialize leakyrelu<int16_t>
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        inline void load_leakyrelu_11cn_s16(i_impl_acti_s16_t &i_impl_func,
                                            c_impl_acti_s16_t &c_impl_func,
                                            const ArgsType<int16_t> &args)
        {
#if CONFIG_TIE728_BOOST
            if (args.input_stride_x_offset % 8 == 0 && args.output_x_offset % 8 == 0 && !((unsigned)&args.input_element[0] & 15) && !((unsigned)&args.output_element[0] & 15))
            {
                i_impl_func = dl_tie728_s16_relu_11c;
            }
            else
            {
                i_impl_func = dl_tie728_s16_unaligned_relu_11c;
            }

#else
            c_impl_func = leakyrelu_11c<int16_t, int32_t>;
#endif // CONFIG_TIE_BOOST
        }

        template <>
        void leakyrelu<int16_t>(void *const args_ptr)
        {
            const ArgsType<int16_t> &args = *((ArgsType<int16_t> *)args_ptr);

            i_impl_acti_s16_t i_impl_func = NULL;
            c_impl_acti_s16_t c_impl_func = NULL;

            load_leakyrelu_11cn_s16(i_impl_func, c_impl_func, args);

            activation_shell<int16_t>(args, i_impl_func, c_impl_func);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // specialize leakyrelu<int8_t>
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        inline void load_leakyrelu_11cn_s8(i_impl_acti_s8_t &i_impl_func, c_impl_acti_s8_t &c_impl_func, const ArgsType<int8_t> &args)
        {

#if CONFIG_TIE728_BOOST
            if (args.input_stride_x_offset % 16 == 0 && args.output_x_offset % 16 == 0 && !((unsigned)&args.input_element[0] & 15) && !((unsigned)&args.output_element[0] & 15))
            {
                i_impl_func = dl_tie728_s8_relu_11c;
            }
            else
            {
                i_impl_func = dl_tie728_s8_unaligned_relu_11c;
            }
#else
            c_impl_func = leakyrelu_11c<int8_t, int16_t>;
#endif // CONFIG_TIE_BOOST
        }

        template <>
        void leakyrelu<int8_t>(void *const args_ptr)
        {
            const ArgsType<int8_t> &args = *((ArgsType<int8_t> *)args_ptr);

            i_impl_acti_s8_t i_impl_func = NULL;
            c_impl_acti_s8_t c_impl_func = NULL;

            load_leakyrelu_11cn_s8(i_impl_func, c_impl_func, args);

            activation_shell<int8_t>(args, i_impl_func, c_impl_func);
        }
    } // namespace c
} // namespace dl