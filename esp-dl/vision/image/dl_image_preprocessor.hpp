#pragma once

#include "stdint.h"
#include "cmath"
#include "dl_image.hpp"
// #include "dl_detect_define.hpp"
#include "driver/ppa.h"
#include "esp_cache.h"
#include "esp_private/esp_cache_private.h"

namespace dl
{
    namespace image
    {
        /**
         * @brief rgb565->rgb888, crop, resize, normalize, quantize
         * NOTE: input should be (h, w, 3) or (h, w, 1) with value range in [0, 255].
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         */
        template <typename feature_t>
        class ImagePreprocessor
        {
        public:
            TensorBase *model_input;
        private:
            const std::vector<float> mean;
            const std::vector<float> std;
            bool rgb_swap;
            bool byte_swap;
            bool use_ppa;
            bool print_info = false;
            feature_t *norm_lut;
            int input_area_x_start;
            int input_area_y_start;
            int input_area_x_end;
            int input_area_y_end;
            float resize_scale_x;
            float resize_scale_y;
#if CONFIG_IDF_TARGET_ESP32P4
            ppa_client_handle_t ppa_client_srm_handle;
            ppa_client_config_t ppa_client_config;
            ppa_srm_oper_config_t srm_oper_config;
            size_t ppa_buffer_size;
            void *ppa_buffer;
#endif
            void create_norm_lut();

        public:
            ImagePreprocessor(TensorBase *model_input,
                              const std::vector<float> &mean,
                              const std::vector<float> &std,
                              bool byte_rgb = false,
                              bool byte_swap = false,
                              bool use_ppa = true);

            ~ImagePreprocessor();
            
            float get_resize_scale_x() {return this->resize_scale_x;};
            float get_resize_scale_y() {return this->resize_scale_y;};
            float get_top_left_x() {return this->input_area_x_start;};
            float get_top_left_y() {return this->input_area_y_start;};
            void set_print_info(bool print_info) {this->print_info = print_info;};

            void preprocess(uint8_t *input_element, const std::vector<int> &input_shape, const std::vector<int> &crop_area={});
            void preprocess(uint16_t *input_element, const std::vector<int> &input_shape, const std::vector<int> &crop_area={});
        };

    } // namespace layer
} // namespace dl