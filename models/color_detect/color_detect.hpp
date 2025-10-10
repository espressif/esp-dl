#pragma once
#include "dl_detect_base.hpp"
#if CONFIG_IDF_TARGET_ESP32P4
#include <opencv2/opencv.hpp>
#else
#undef EPS
#include <opencv2/opencv.hpp>
#define EPS 192
#endif

class ColorDetect : public dl::detect::Detect {
public:
    ColorDetect(uint16_t width, uint16_t height);
    ~ColorDetect();
    void register_color(const std::vector<uint8_t> &hsv_min,
                        const std::vector<uint8_t> &hsv_max,
                        const std::string &name,
                        int area_thr = 256);
    void delete_color(int idx);
    void delete_color(const std::string &name);
    void enable_morphology(int kernel_size = 5);
    std::string get_color_name(int idx);
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img) override;

private:
    void gen_xy_map();

    template <typename PixelCvt>
    void rgb2hsv_mask(const PixelCvt &pixel_cvt)
    {
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_hsv_mask);
        for (int i = 0; i < m_height; i++) {
            uint8_t *p_row = src + m_y[i];
            for (int j = 0; j < m_width; j++, dst++) {
                pixel_cvt(p_row + m_x[j], dst);
            }
        }
    }

    template <typename PixelCvt>
    void rgb2hsv_and_hsv_mask(const PixelCvt &pixel_cvt)
    {
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst_hsv = static_cast<uint8_t *>(m_hsv);
        uint8_t *dst_hsv_mask = static_cast<uint8_t *>(m_hsv_mask);
        for (int i = 0; i < m_height; i++) {
            uint8_t *p_row = src + m_y[i];
            for (int j = 0; j < m_width; j++, dst_hsv += 3, dst_hsv_mask++) {
                pixel_cvt(p_row + m_x[j], dst_hsv, dst_hsv_mask);
            }
        }
    }

    template <typename PixelCvt>
    void hsv2hsv_mask(const PixelCvt &pixel_cvt)
    {
        uint8_t *src = static_cast<uint8_t *>(m_hsv);
        uint8_t *dst = static_cast<uint8_t *>(m_hsv_mask);
        int n = m_height * m_width;
        for (int i = 0; i < n; i++, src += 3, dst++) {
            pixel_cvt(src, dst);
        }
    }

    dl::image::img_t m_src_img;
    uint16_t m_width;
    uint16_t m_height;
    void *m_hsv;
    void *m_hsv_mask;
    void *m_mask_label;
    std::vector<std::vector<uint8_t>> m_hsv_min;
    std::vector<std::vector<uint8_t>> m_hsv_max;
    std::vector<int> m_area_thr;
    std::vector<std::string> m_color_names;
    float m_inv_scale_x;
    float m_inv_scale_y;
    int *m_x;
    int *m_y;
    bool m_gen_xy_map;
    bool m_morphology;
    cv::Mat m_kernel;
    std::list<dl::detect::result_t> m_result;
};
