#include "dl_base_add4d.hpp"

#include "dl_base_activate_output.hpp"
#include "dl_base_isa.hpp"

namespace dl {
namespace base {
// template <typename feature_t, typename buffer_t>
// inline void add4d_11c(feature_t *output_ptr,
//                       feature_t *input0_ptr,
//                       feature_t *input1_ptr,
//                       const arithArgsType<feature_t> &args)
// {
//     buffer_t buffer;
//     for (size_t output_c = 0; output_c < args.channel; output_c++) // C
//     {
//         buffer = (buffer_t)input0_ptr[output_c] + (buffer_t)input1_ptr[output_c];
//         tool::truncate(output_ptr[output_c], buffer);
//     }
// }

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_16_w2_16 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s8_add4d_bchw_w1_16_w2_16_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s8_add4d_bchw_w1_16_w2_16_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif
/*
                for (int lLoop = 0; lLoop < args.output_w; lLoop++) 
                {
                    buffer_t buffer = (buffer_t)input0_ptr_base[lLoop * args.input0_w_same] + (buffer_t)input1_ptr_base[lLoop * args.input1_w_same];
                    tool::truncate(*(output_ptr++), buffer);
                }
*/
            }
        }
    }
}

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_16_w2_1 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s8_add4d_bchw_w1_16_w2_1_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s8_add4d_bchw_w1_16_w2_1_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif
            }
        }
    }
}

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_1_w2_16 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s8_add4d_bchw_w1_1_w2_16_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s8_add4d_bchw_w1_1_w2_16_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif

            }
        }
    }
}

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_8_w2_8 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s16_add4d_bchw_w1_8_w2_8_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s16_add4d_bchw_w1_8_w2_8_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif
            }
        }
    }
}

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_8_w2_1 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s16_add4d_bchw_w1_8_w2_1_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s16_add4d_bchw_w1_8_w2_1_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif
            }
        }
    }
}

template <typename feature_t, typename buffer_t>
inline void add4d_bchw_w1_1_w2_8 (feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9    
    for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
    {
        int iLoop_input0_offset = iLoop * input0_chw;
        int iLoop_input1_offset = iLoop * input1_chw;
        int iLoop_output_offset = iLoop * output_chw;

        for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
        {
            int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
            int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;
            int jLoop_output_offset = iLoop_output_offset + jLoop * output_hw;

            for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
            {
                int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;
                int kLoop_output_offset = jLoop_output_offset + kLoop * args.output_w;

                feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;
                feature_t *output_ptr_base = output_ptr + kLoop_output_offset;
#ifdef CONFIG_IDF_TARGET_ESP32P4
                dl_esp32p4_s16_add4d_bchw_w1_1_w2_8_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#elif defined CONFIG_IDF_TARGET_ESP32S3
                dl_esp32s3_s16_add4d_bchw_w1_1_w2_8_simdadd(output_ptr_base, input0_ptr_base, input1_ptr_base, args.output_w);
#endif
            }
        }
    }
}


template <typename feature_t, typename buffer_t>
inline void add4d_bchw_rescale(feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    int index = 0;
    int index0 = 0;
    int index1 = 0;
    int output_chw = args.output_c * args.output_h * args.output_w;//s10
    int output_hw = args.output_h * args.output_w;//s11
    int input0_chw = args.input0_c * args.input0_h * args.input0_w * args.input0_b_same;//a4
    int input0_hw = args.input0_h * args.input0_w * args.input0_c_same;//a5
    int input0_w = args.input0_w * args.input0_h_same;//a6
    int input1_chw = args.input1_c * args.input1_h * args.input1_w * args.input1_b_same;//a7
    int input1_hw = args.input1_h * args.input1_w * args.input1_c_same;//s8
    int input1_w = args.input1_w * args.input1_h_same;//s9

    // printf("input0_chw=%d, input0_hw=%d, input0_w=%d\n", input0_chw, input0_hw, input0_w);
    // printf("input1_chw=%d, input1_hw=%d, input1_w=%d\n", input1_chw, input1_hw, input1_w);
    // printf("output_chw=%d, output_hw=%d\n", output_chw, output_hw);
    // printf("output_b=%d, output_c=%d, output_h=%d, output_w=%d\n", args.output_b, args.output_c, args.output_h, args.output_w);
    // printf("input0_b=%d, input0_c=%d, input0_h=%d, input0_w=%d\n", args.input0_b, args.input0_c, args.input0_h, args.input0_w);
    // printf("input1_b=%d, input1_c=%d, input1_h=%d, input1_w=%d\n", args.input1_b, args.input1_c, args.input1_h, args.input1_w);
    // printf("input0_b_same=%d, input0_c_same=%d, input0_h_same=%d, input0_w_same=%d\n", args.input0_b_same, args.input0_c_same, args.input0_h_same, args.input0_w_same);
    // printf("input1_b_same=%d, input1_c_same=%d, input1_h_same=%d, input1_w_same=%d\n", args.input1_b_same, args.input1_c_same, args.input1_h_same, args.input1_w_same);

/*
    index = 0;
    for (int iLoop = 0; iLoop < args.input0_b; iLoop++)//t0 s5
    {
        for (int jLoop = 0; jLoop < args.input0_c; jLoop++)//t1 s2
        {
            for (int kLoop = 0; kLoop < args.input0_h; kLoop++)//t2 s3
            {
                for (int lLoop = 0; lLoop < args.input0_w; lLoop++)//t3 s4
                {
                    printf("input0[%d]= %d\n", index, input0_ptr[index]);
                    index++;
                }
            }        
        }
    }

    index = 0;
    for (int iLoop = 0; iLoop < args.input1_b; iLoop++)//t0 s5
    {
        for (int jLoop = 0; jLoop < args.input1_c; jLoop++)//t1 s2
        {
            for (int kLoop = 0; kLoop < args.input1_h; kLoop++)//t2 s3
            {
                for (int lLoop = 0; lLoop < args.input1_w; lLoop++)//t3 s4
                {
                    printf("input1[%d]= %d\n", index, input1_ptr[index]);
                    index++;
                }
            }        
        }
    }
*/

    //printf("test add4d by xzs29\n");
    index = 0;
    if (args.output_max_dims == 4)
    {
        for (int iLoop = 0; iLoop < args.output_b; iLoop++) 
        {
            int iLoop_input0_offset = iLoop * input0_chw;
            int iLoop_input1_offset = iLoop * input1_chw;

            for (int jLoop = 0; jLoop < args.output_c; jLoop++) 
            {
                int jLoop_input0_offset = iLoop_input0_offset + jLoop * input0_hw;
                int jLoop_input1_offset = iLoop_input1_offset + jLoop * input1_hw;

                for (int kLoop = 0; kLoop < args.output_h; kLoop++) 
                {
                    int kLoop_input0_offset = jLoop_input0_offset + kLoop * input0_w;
                    int kLoop_input1_offset = jLoop_input1_offset + kLoop * input1_w;

                    feature_t *input0_ptr_base = input0_ptr + kLoop_input0_offset;
                    feature_t *input1_ptr_base = input1_ptr + kLoop_input1_offset;

                    for (int lLoop = 0; lLoop < args.output_w; lLoop++) 
                    {
                        buffer_t buffer = (buffer_t)input0_ptr_base[lLoop * args.input0_w_same] + (buffer_t)input1_ptr_base[lLoop * args.input1_w_same];
                        tool::truncate(*(output_ptr++), buffer);
                    }

                }
            }
        }
    }
  



    else if (args.output_max_dims == 3)
    {
/*        
        for (int jLoop = 0; jLoop < args.output_c; jLoop++)
        {
            for (int kLoop = 0; kLoop < args.output_h; kLoop++)
            {
                for (int lLoop = 0; lLoop < args.output_w; lLoop++)
                {
                    //index = jLoop * output_hw + kLoop * args.output_w + lLoop;
                    index0 = jLoop * input0_hw + kLoop * input0_w + lLoop * args.input0_w_same;
                    index1 = jLoop * input1_hw + kLoop * input1_w + lLoop * args.input1_w_same;
                    buffer = (buffer_t)input0_ptr[index0] + (buffer_t)input1_ptr[index1];
                    //tool::truncate(output_ptr[index], buffer);
                    tool::truncate(*(output_ptr++), buffer);
                }
            }
        }           
*/
        for (int jLoop = 0; jLoop < args.output_c; jLoop++) {
            int base_index0_j = jLoop * input0_hw;
            int base_index1_j = jLoop * input1_hw;
            for (int kLoop = 0; kLoop < args.output_h; kLoop++) {
                int base_index0_k = base_index0_j + kLoop * input0_w;
                int base_index1_k = base_index1_j + kLoop * input1_w;
                for (int lLoop = 0; lLoop < args.output_w; lLoop++) {
                    // Increment the index directly based on the previous calculation
                    int index0 = base_index0_k + lLoop * args.input0_w_same;
                    int index1 = base_index1_k + lLoop * args.input1_w_same;
                    buffer = (buffer_t)input0_ptr[index0] + (buffer_t)input1_ptr[index1];
                    tool::truncate(*(output_ptr++), buffer);
                }
            }
        }
    }
    else if (args.output_max_dims == 2)
    {
/*        
        for (int jLoop = 0; jLoop < args.output_h; jLoop++)
        {
            for (int kLoop = 0; kLoop < args.output_w; kLoop++)
            {
                //index = jLoop * args.output_w + kLoop;
                index0 = jLoop * args.input0_w + kLoop * args.input0_w_same;
                index1 = jLoop * args.input1_w + kLoop * args.input1_w_same;
                buffer = (buffer_t)input0_ptr[index0] + (buffer_t)input1_ptr[index1];
                //tool::truncate(output_ptr[index], buffer);
                tool::truncate(*(output_ptr++), buffer);
            }
        }
*/
        for (int jLoop = 0; jLoop < args.output_h; jLoop++) {
            int base_index0 = jLoop * input0_w;
            int base_index1 = jLoop * input1_w;

            for (int kLoop = 0; kLoop < args.output_w; kLoop++) {
                // Incremental index calculation
                int index0 = base_index0 + kLoop * args.input0_w_same;
                int index1 = base_index1 + kLoop * args.input1_w_same;

                // Fetch values and compute buffer
                buffer = (buffer_t)input0_ptr[index0] + (buffer_t)input1_ptr[index1];

                // Truncate the buffer and store in output
                tool::truncate(*(output_ptr++), buffer);
            }
        }    
    }
    else if (args.output_max_dims == 1)
    {
        for (int jLoop = 0; jLoop < args.output_w; jLoop++)
        {
            //index = jLoop;
            index0 = jLoop * args.input0_w_same;
            index1 = jLoop * args.input1_w_same;
            buffer = (buffer_t)input0_ptr[index0] + (buffer_t)input1_ptr[index1];
            //tool::truncate(output_ptr[index], buffer);
            tool::truncate(*(output_ptr++), buffer);
        }
    }    
}

template <typename feature_t, typename buffer_t>
inline void add4d_11c_rescale(feature_t *output_ptr,
                              feature_t *input0_ptr,
                              feature_t *input1_ptr,
                              const arithArgsType<feature_t> &args)
{
    buffer_t buffer;
    for (size_t output_c = 0; output_c < args.channel; output_c++) // C
    {
        buffer = (buffer_t)input0_ptr[output_c] + (buffer_t)(DL_RIGHT_SHIFT(input1_ptr[output_c], args.input_shift));
        buffer = DL_RIGHT_SHIFT(buffer * args.output_scale, args.output_shift);
        tool::truncate(output_ptr[output_c], buffer);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// specialize add4d<int16_t>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void load_add4d_11c_s16(arith_i_impl_func_s16_t &i_impl_func,
                               arith_c_impl_func_s16_t &c_impl_func,
                               arith_n_wise_tail_s16_t &n_wise_tail,
                               const arithArgsType<int16_t> &args)
{
#if defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S3)
    if (args.input0_w % 8 == 0 && args.input1_w % 8 == 0 && args.input0_w_same == 1 && args.input1_w_same == 1)
    {
        c_impl_func = add4d_bchw_w1_8_w2_8 <int16_t, int32_t>;
    }
    else if (args.input0_w % 8 == 0 && args.input1_w == 1)
    {
        c_impl_func = add4d_bchw_w1_8_w2_1 <int16_t, int32_t>;
    }
    else if (args.input0_w == 1 && args.input1_w % 8 == 0)
    {
        c_impl_func = add4d_bchw_w1_1_w2_8 <int16_t, int32_t>;
    }
    else 
    {
        c_impl_func = add4d_bchw_rescale<int16_t, int32_t>;
    }
#else
    c_impl_func = add4d_bchw_rescale<int16_t, int32_t>;
#endif

}

template <>
void add4d<int16_t>(void *const args_ptr)
{
    const arithArgsType<int16_t> &args = *((arithArgsType<int16_t> *)args_ptr);

    arith_i_impl_func_s16_t i_impl_func = NULL;
    arith_c_impl_func_s16_t c_impl_func = NULL;
    arith_n_wise_tail_s16_t n_wise_tail = NULL;

#if CONFIG_ESP32P4_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif

    load_add4d_11c_s16(i_impl_func, c_impl_func, n_wise_tail, args);

    arith_operation_shell_<int16_t>(args, i_impl_func, c_impl_func, n_wise_tail);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// specialize add4d<int8_t>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void load_add4d_11c_s8(arith_i_impl_func_s8_t &i_impl_func,
                              arith_c_impl_func_s8_t &c_impl_func,
                              arith_n_wise_tail_s8_t &n_wise_tail,
                              const arithArgsType<int8_t> &args)
{
#if defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S3)
    if (args.input0_w % 16 == 0 && args.input1_w % 16 == 0 && args.input0_w_same == 1 && args.input1_w_same == 1)
    {
        c_impl_func = add4d_bchw_w1_16_w2_16 <int8_t, int16_t>;
    }
    else if (args.input0_w % 16 == 0 && args.input1_w == 1)
    {
        c_impl_func = add4d_bchw_w1_16_w2_1 <int8_t, int16_t>;
    }
    else if (args.input0_w == 1 && args.input1_w % 16 == 0)
    {
        c_impl_func = add4d_bchw_w1_1_w2_16 <int8_t, int16_t>;
    }
    else 
    {
        c_impl_func = add4d_bchw_rescale<int8_t, int16_t>;
    }
#else
    c_impl_func = add4d_bchw_rescale<int8_t, int16_t>;
#endif
}

template <>
void add4d<int8_t>(void *const args_ptr)
{
    const arithArgsType<int8_t> &args = *((arithArgsType<int8_t> *)args_ptr);

    arith_i_impl_func_s8_t i_impl_func = NULL;
    arith_c_impl_func_s8_t c_impl_func = NULL;
    arith_n_wise_tail_s8_t n_wise_tail = NULL;

#if CONFIG_ESP32P4_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif
    load_add4d_11c_s8(i_impl_func, c_impl_func, n_wise_tail, args);

    arith_operation_shell_<int8_t>(args, i_impl_func, c_impl_func, n_wise_tail);
}
} // namespace base
} // namespace dl

