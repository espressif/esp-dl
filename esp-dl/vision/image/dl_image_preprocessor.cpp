#include "dl_image_preprocessor.hpp"

#define ALIGN_UP(num, align)    (((num) + ((align) - 1)) & ~((align) - 1))

namespace dl
{
    namespace image
    {
        template <typename feature_t>
        ImagePreprocessor<feature_t>::ImagePreprocessor(TensorBase *model_input,
                                                        const std::vector<float> &mean,
                                                        const std::vector<float> &std,
                                                        bool rgb_swap,
                                                        bool byte_swap,
                                                        bool use_ppa) : model_input(model_input), 
                                                                          mean(mean), 
                                                                          std(std),
                                                                          rgb_swap(rgb_swap),
                                                                          byte_swap(byte_swap),
                                                                          use_ppa(use_ppa)

        {
            this->norm_lut =
                (feature_t *)heap_caps_aligned_alloc(16, mean.size() * 256 * sizeof(feature_t), MALLOC_CAP_SPIRAM);
            this->create_norm_lut();
#if CONFIG_IDF_TARGET_ESP32P4
            memset(&this->ppa_client_config, 0, sizeof(ppa_client_config_t));
            this->ppa_client_config.oper_type = PPA_OPERATION_SRM;
            ESP_ERROR_CHECK(ppa_register_client(&this->ppa_client_config, &this->ppa_client_srm_handle));
            memset(&this->srm_oper_config, 0, sizeof(ppa_srm_oper_config_t));
            size_t cache_line_size;
            ESP_ERROR_CHECK(esp_cache_get_alignment(MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA, &cache_line_size));
            this->ppa_buffer_size = ALIGN_UP(model_input->shape[0] * model_input->shape[1] * model_input->shape[2], cache_line_size);
            this->ppa_buffer = heap_caps_aligned_calloc(cache_line_size, this->ppa_buffer_size, sizeof(uint8_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
#endif
        }

        template <typename feature_t>
        ImagePreprocessor<feature_t>::~ImagePreprocessor()
        {
            heap_caps_free(this->norm_lut);
#if CONFIG_IDF_TARGET_ESP32P4
            heap_caps_free(this->ppa_buffer);
            ESP_ERROR_CHECK(ppa_unregister_client(this->ppa_client_srm_handle));
#endif
        }

        template <typename feature_t>
        void ImagePreprocessor<feature_t>::create_norm_lut()
        {
            for (int i = 0; i < this->mean.size(); i++)
            {
                if (std::is_same<feature_t, int8_t>::value)
                {
                    for (int j = 0; j < 256; j++)
                    {
                        this->norm_lut[i * 256 + j] = (feature_t)DL_CLIP(dl_esp32p4_round_half_even(((float)j - this->mean[i]) / this->std[i] / DL_SCALE(this->model_input->exponent)), -128, 127);
                    }
                }
                else
                {
                    for (int j = 0; j < 65536; j++)
                    {
                        this->norm_lut[i * 65536 + j] = (feature_t)DL_CLIP(dl_esp32p4_round_half_even(((float)j - this->mean[i]) / this->std[i] / DL_SCALE(this->model_input->exponent)), -32768, 32767);
                    }       
                }
            }
        }

        template <typename feature_t>
        void ImagePreprocessor<feature_t>::preprocess(uint8_t *input_element, const std::vector<int> &input_shape, const std::vector<int> &crop_area)
        {
            dl::tool::Latency latency;
            // step1. crop & resize
            if (!crop_area.empty() || input_shape[0] != this->model_input->shape[0] || input_shape[1] != this->model_input->shape[1])
            {
                latency.start();
                if (!crop_area.empty())
                {
                    assert(crop_area.size() == 4);
                    input_area_x_start = crop_area[0];
                    input_area_y_start = crop_area[1];
                    input_area_x_end = crop_area[2];
                    input_area_y_end = crop_area[3];
                }
                else
                {
                    input_area_x_start = 0;
                    input_area_y_start = 0;
                    input_area_x_end = input_shape[1];
                    input_area_y_end = input_shape[0];
                }
                this->resize_scale_y = (float)this->model_input->shape[0] / (input_area_y_end - input_area_y_start);
                this->resize_scale_x = (float)this->model_input->shape[1] / ((input_area_x_end - input_area_x_start));

                // hardware resize with ppa
                // only esp32p4 has ppa, 
                // ppa use 8 bit to store int part of scale, 4 bit to store frac part of scale.
                bool ppa_available = false;
                if (CONFIG_IDF_TARGET_ESP32P4
                    && this->resize_scale_y < 256 && this->resize_scale_x < 256
                    && this->resize_scale_y >= 0.0625 && this->resize_scale_x >= 0.0625
                    && this->use_ppa)
                {
                    int resize_scale_y_int = floor(this->resize_scale_y);
                    int resize_scale_x_int = floor(this->resize_scale_x);
                    float resize_scale_y_frac = this->resize_scale_y - resize_scale_y_int;
                    float resize_scale_x_frac = this->resize_scale_x - resize_scale_x_int;
                    resize_scale_y_frac = floor(resize_scale_y_frac / 0.0625) * 0.0625;
                    resize_scale_x_frac = floor(resize_scale_x_frac / 0.0625) * 0.0625;
                    float new_resize_scale_y = resize_scale_y_int + resize_scale_y_frac;
                    float new_resize_scale_x = resize_scale_x_int + resize_scale_x_frac;
                    float error_percentage_y = (this->model_input->shape[0] - new_resize_scale_y * (input_area_y_end - input_area_y_start)) / this->model_input->shape[0];
                    float error_percentage_x = (this->model_input->shape[1] - new_resize_scale_x * (input_area_x_end - input_area_x_start)) / this->model_input->shape[1];
                    
                    if (error_percentage_x < 0.3 && error_percentage_y < 0.3)
                    {
                        ppa_available = true;
                        this->resize_scale_y = new_resize_scale_y;
                        this->resize_scale_x = new_resize_scale_x;
                    }
                }

                    // printf("%d, %d\n", input_area_x_end - input_area_x_start, input_area_y_end - input_area_y_start);
                    // printf("%f, %f\n", this->resize_scale_x, this->resize_scale_y);
                    // printf("%d, %d\n", (int)(this->resize_scale_x * (input_area_x_end - input_area_x_start)), (int)(this->resize_scale_y * (input_area_y_end - input_area_y_start)));
                    // printf("%d, %d\n", model_input->shape[1], model_input->shape[0]);
                if (ppa_available)
                {

                    srm_oper_config.in.buffer = (const void *)input_element;
                    srm_oper_config.in.pic_h = input_shape[0];
                    srm_oper_config.in.pic_w = input_shape[1];
                    srm_oper_config.in.block_h = input_area_y_end - input_area_y_start;
                    srm_oper_config.in.block_w = input_area_x_end - input_area_x_start;
                    srm_oper_config.in.block_offset_y = input_area_y_start;
                    srm_oper_config.in.block_offset_x = input_area_x_start;
                    srm_oper_config.in.srm_cm = PPA_SRM_COLOR_MODE_RGB888;

                    srm_oper_config.out.buffer = this->ppa_buffer;
                    srm_oper_config.out.buffer_size = this->ppa_buffer_size;
                    srm_oper_config.out.pic_h = model_input->shape[0];
                    srm_oper_config.out.pic_w = model_input->shape[1];
                    srm_oper_config.out.block_offset_x = 0;
                    srm_oper_config.out.block_offset_y = 0;
                    srm_oper_config.out.srm_cm = PPA_SRM_COLOR_MODE_RGB888;

                    srm_oper_config.rotation_angle = PPA_SRM_ROTATION_ANGLE_0;
                    srm_oper_config.scale_x = this->resize_scale_x;
                    srm_oper_config.scale_y = this->resize_scale_y;
                    srm_oper_config.mirror_x = false;
                    srm_oper_config.mirror_y = false;
                    srm_oper_config.rgb_swap = this->rgb_swap;
                    
                    srm_oper_config.mode = PPA_TRANS_MODE_BLOCKING;
                    ESP_ERROR_CHECK(ppa_do_scale_rotate_mirror(this->ppa_client_srm_handle, &srm_oper_config));
                    // TODO change input mem alignment to avoid data copy
                    tool::copy_memory(this->model_input->data, this->ppa_buffer, this->model_input->get_bytes());
                }
                else
                {
                    // software resize
                    crop_and_resize((uint8_t *)this->model_input->get_element_ptr(),
                                    this->model_input->shape[1],
                                    this->model_input->shape[2],
                                    0, this->model_input->shape[0],
                                    0, this->model_input->shape[1],
                                    input_element,
                                    input_shape[0],
                                    input_shape[1],
                                    input_shape[2],
                                    input_area_y_start, input_area_y_end,
                                    input_area_x_start, input_area_x_end,
                                    IMAGE_RESIZE_NEAREST,
                                    this->rgb_swap);
                }
                latency.end();
                if (this->print_info)
                    latency.print("image_preprocess", "resize");
            }
            else
            {
                latency.start();
                this->resize_scale_y = 1;
                this->resize_scale_x = 1;
                tool::copy_memory(this->model_input->data, input_element, this->model_input->get_bytes());
                latency.end();
                if (this->print_info)
                    latency.print("image_preprocess", "copy");
            }

            // step2. normalize quantize
            uint8_t *input = (uint8_t *)this->model_input->get_element_ptr();
            feature_t *norm_quant_input = (feature_t *)this->model_input->get_element_ptr();
            latency.start();
            for (int i = 0; i < this->model_input->get_size(); i++)
            {
                norm_quant_input[i] = this->norm_lut[i % 3 * 256 + input[i]];
            }
            latency.end();
            if (this->print_info)
                latency.print("image_preprocess", "normalize");
        }

        template <typename feature_t>
        void ImagePreprocessor<feature_t>::preprocess(uint16_t *input_element, const std::vector<int> &input_shape, const std::vector<int> &crop_area)
        {
            dl::tool::Latency latency;
            // step1. crop & resize
            if (!crop_area.empty() || input_shape[0] != this->model_input->shape[0] || input_shape[1] != this->model_input->shape[1])
            {
                latency.start();
                if (!crop_area.empty())
                {
                    assert(crop_area.size() == 4);
                    input_area_x_start = crop_area[0];
                    input_area_y_start = crop_area[1];
                    input_area_x_end = crop_area[2];
                    input_area_y_end = crop_area[3];
                }
                else
                {
                    input_area_x_start = 0;
                    input_area_y_start = 0;
                    input_area_x_end = input_shape[1];
                    input_area_y_end = input_shape[0];
                }
                this->resize_scale_y = (float)this->model_input->shape[0] / (input_area_y_end - input_area_y_start);
                this->resize_scale_x = (float)this->model_input->shape[1] / (input_area_x_end - input_area_x_start);

                // hardware resize with ppa
                // only esp32p4 has ppa, 
                // ppa use 8 bit to store int part of scale, 4 bit to store frac part of scale.
                bool ppa_available = false;
                if (CONFIG_IDF_TARGET_ESP32P4
                    && this->resize_scale_y < 256 && this->resize_scale_x < 256
                    && this->resize_scale_y >= 0.0625 && this->resize_scale_x >= 0.0625
                    && this->use_ppa)
                {
                    int resize_scale_y_int = floor(this->resize_scale_y);
                    int resize_scale_x_int = floor(this->resize_scale_x);
                    float resize_scale_y_frac = this->resize_scale_y - resize_scale_y_int;
                    float resize_scale_x_frac = this->resize_scale_x - resize_scale_x_int;
                    resize_scale_y_frac = floor(resize_scale_y_frac / 0.0625) * 0.0625;
                    resize_scale_x_frac = floor(resize_scale_x_frac / 0.0625) * 0.0625;
                    float new_resize_scale_y = resize_scale_y_int + resize_scale_y_frac;
                    float new_resize_scale_x = resize_scale_x_int + resize_scale_x_frac;
                    float error_percentage_y = (this->model_input->shape[0] - new_resize_scale_y * (input_area_y_end - input_area_y_start)) / this->model_input->shape[0];
                    float error_percentage_x = (this->model_input->shape[1] - new_resize_scale_x * (input_area_x_end - input_area_x_start)) / this->model_input->shape[1];
                    
                    if (error_percentage_x < 0.3 && error_percentage_y < 0.3)
                    {
                        ppa_available = true;
                        this->resize_scale_y = new_resize_scale_y;
                        this->resize_scale_x = new_resize_scale_x;
                    }
                }

                    // printf("%d, %d\n", input_area_x_end - input_area_x_start, input_area_y_end - input_area_y_start);
                    // printf("%f, %f\n", this->resize_scale_x, this->resize_scale_y);
                    // printf("%d, %d\n", (int)(this->resize_scale_x * (input_area_x_end - input_area_x_start)), (int)(this->resize_scale_y * (input_area_y_end - input_area_y_start)));
                    // printf("%d, %d\n", model_input->shape[1], model_input->shape[0]);
                    
                if (ppa_available)
                {
                    srm_oper_config.in.buffer = (const void *)input_element;
                    srm_oper_config.in.pic_h = input_shape[0];
                    srm_oper_config.in.pic_w = input_shape[1];
                    srm_oper_config.in.block_h = input_area_y_end - input_area_y_start;
                    srm_oper_config.in.block_w = input_area_x_end - input_area_x_start;
                    srm_oper_config.in.block_offset_y = input_area_y_start;
                    srm_oper_config.in.block_offset_x = input_area_x_start;
                    srm_oper_config.in.srm_cm = PPA_SRM_COLOR_MODE_RGB565;

                    srm_oper_config.out.buffer = this->ppa_buffer;
                    srm_oper_config.out.buffer_size = this->ppa_buffer_size;
                    srm_oper_config.out.pic_h = model_input->shape[0];
                    srm_oper_config.out.pic_w = model_input->shape[1];
                    srm_oper_config.out.block_offset_x = 0;
                    srm_oper_config.out.block_offset_y = 0;
                    srm_oper_config.out.srm_cm = PPA_SRM_COLOR_MODE_RGB888;

                    srm_oper_config.rotation_angle = PPA_SRM_ROTATION_ANGLE_0;
                    srm_oper_config.scale_x = this->resize_scale_x;
                    srm_oper_config.scale_y = this->resize_scale_y;
                    srm_oper_config.mirror_x = false;
                    srm_oper_config.mirror_y = false;
                    srm_oper_config.rgb_swap = this->rgb_swap;
                    srm_oper_config.byte_swap = this->byte_swap;

                    srm_oper_config.mode = PPA_TRANS_MODE_BLOCKING;

                    ESP_ERROR_CHECK(ppa_do_scale_rotate_mirror(this->ppa_client_srm_handle, &srm_oper_config));
                    // TODO change input mem alignment to avoid data copy
                    tool::copy_memory(this->model_input->data, this->ppa_buffer, this->model_input->get_bytes());
                }
                else
                {
                    // software resize
                    crop_and_resize((uint8_t *)this->model_input->get_element_ptr(),
                                    this->model_input->shape[1],
                                    this->model_input->shape[2],
                                    0, this->model_input->shape[0],
                                    0, this->model_input->shape[1],
                                    input_element,
                                    input_shape[0],
                                    input_shape[1],
                                    input_shape[2],
                                    input_area_y_start, input_area_y_end,
                                    input_area_x_start, input_area_x_end,
                                    IMAGE_RESIZE_NEAREST,
                                    this->rgb_swap,
                                    this->byte_swap);
                }
                latency.end();
                if (this->print_info)
                    latency.print("image_preprocess", "resize");
            }
            else
            {
                latency.start();
                this->resize_scale_y = 1;
                this->resize_scale_x = 1;
                tool::copy_memory(this->model_input->data, input_element, this->model_input->get_bytes());
                latency.end();
                if (this->print_info)
                    latency.print("image_preprocess", "copy");
            }

            // step2. normalize quantize
            uint8_t *input = (uint8_t *)this->model_input->get_element_ptr();
            feature_t *norm_quant_input = (feature_t *)this->model_input->get_element_ptr();
            latency.start();
            for (int i = 0; i < this->model_input->get_size(); i++)
            {
                norm_quant_input[i] = this->norm_lut[i % 3 * 256 + input[i]];
            }
            latency.end();
            if (this->print_info)
                latency.print("image_preprocess", "normalize");
        }

        template class ImagePreprocessor<int8_t>;
        template class ImagePreprocessor<int16_t>;
    } // namespace image
} // namespace dl