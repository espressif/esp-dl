#pragma once
#include "dl_detect_base.hpp"
#if CONFIG_IDF_TARGET_ESP32P4
#include <opencv2/opencv.hpp>
#else
#undef EPS
#include <opencv2/opencv.hpp>
#define EPS 192
#endif

class ColorDetectBase {
public:
    ColorDetectBase(uint16_t width, uint16_t height);
    virtual ~ColorDetectBase();
    void register_color(const std::array<uint8_t, 3> &hsv_min,
                        const std::array<uint8_t, 3> &hsv_max,
                        const std::string &name,
                        int area_thr = 256);
    void delete_color(int idx);
    void delete_color(const std::string &name);
    std::string get_color_name(int idx);
    int get_color_num();

protected:
    uint16_t m_width;
    uint16_t m_height;
    void *m_hsv;
    void *m_hsv_mask;
    cv::Mat m_hsv_mask_cvmat;
    std::vector<std::array<uint8_t, 3>> m_hsv_min;
    std::vector<std::array<uint8_t, 3>> m_hsv_max;
    std::vector<int> m_area_thr;
    std::vector<std::string> m_color_names;
    dl::image::ImageTransformer m_T;
};

class ColorDetect : public ColorDetectBase {
public:
    ColorDetect(uint16_t width, uint16_t height);
    ~ColorDetect();
    void enable_morphology(int kernel_size = 5);
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img);

private:
    void hsv_mask_process(
        int color_id, float inv_scale_x, float inv_scale_y, uint16_t limit_width, uint16_t limit_height);
    bool m_morphology;
    void *m_hsv_mask_label;
    cv::Mat m_hsv_mask_label_cvmat;
    cv::Mat m_kernel;
    std::list<dl::detect::result_t> m_result;
};

class ColorRotateDetect : public ColorDetectBase {
    typedef struct {
        int category;
        cv::RotatedRect rot_rect;
    } result_t;

public:
    ColorRotateDetect(uint16_t width, uint16_t height, int kernel_size = 5);
    std::vector<result_t> &run(const dl::image::img_t &img);

private:
    void hsv_mask_process(int color_id, float inv_scale_x, float inv_scale_y);
    cv::Mat m_kernel;
    std::vector<result_t> m_result;
};
