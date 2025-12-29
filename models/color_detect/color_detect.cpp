#include "color_detect.hpp"
#include "dl_image_color.hpp"

static const char *TAG = "ColorDetect";

ColorDetectBase::ColorDetectBase(uint16_t width, uint16_t height) :
    m_width(width),
    m_height(height),
    m_hsv(nullptr),
    m_hsv_mask(heap_caps_malloc(width * height, MALLOC_CAP_DEFAULT)),
    m_hsv_mask_cvmat(width, height, CV_8UC1, m_hsv_mask)
{
}

ColorDetectBase::~ColorDetectBase()
{
    heap_caps_free(m_hsv);
    heap_caps_free(m_hsv_mask);
}

void ColorDetectBase::register_color(const std::array<uint8_t, 3> &hsv_min,
                                     const std::array<uint8_t, 3> &hsv_max,
                                     const std::string &name,
                                     int area_thr)
{
    if (!(hsv_min[0] <= 180 && hsv_max[0] <= 180 && hsv_min[1] < hsv_max[1] && hsv_min[2] < hsv_max[2])) {
        ESP_LOGE(TAG, "Invalid hsv threshold.");
        return;
    }
    if (area_thr >= m_width * m_height) {
        ESP_LOGE(TAG, "Invalid area_thr.");
        return;
    }
    if (std::find(m_color_names.begin(), m_color_names.end(), name) != m_color_names.end()) {
        ESP_LOGE(TAG, "Color %s is already registered.", name.c_str());
        return;
    }
    if (m_hsv_min.size() == 1) {
        m_hsv = heap_caps_malloc(m_width * m_height * 3, MALLOC_CAP_DEFAULT);
        if (!m_hsv) {
            ESP_LOGE(TAG, "Failed to malloc memory.");
            return;
        }
    }
    m_hsv_min.emplace_back(hsv_min);
    m_hsv_max.emplace_back(hsv_max);
    m_area_thr.emplace_back(area_thr);
    m_color_names.emplace_back(name);
}

void ColorDetectBase::delete_color(int idx)
{
    assert(idx >= 0 && idx < m_hsv_min.size());
    m_hsv_min.erase(m_hsv_min.begin() + idx);
    m_hsv_max.erase(m_hsv_max.begin() + idx);
    m_area_thr.erase(m_area_thr.begin() + idx);
    m_color_names.erase(m_color_names.begin() + idx);
    if (m_hsv_min.size() == 1) {
        heap_caps_free(m_hsv);
        m_hsv = nullptr;
    }
}

void ColorDetectBase::delete_color(const std::string &name)
{
    auto it = std::find(m_color_names.begin(), m_color_names.end(), name);
    if (it == m_color_names.end()) {
        ESP_LOGE(TAG, "Failed to delete Color %s, it is not registered.", name.c_str());
        return;
    }
    delete_color(std::distance(m_color_names.begin(), it));
}

std::string ColorDetectBase::get_color_name(int idx)
{
    if (idx < 0 || idx >= m_hsv_min.size()) {
        ESP_LOGE(TAG, "Invalid index.");
        return {};
    }
    return m_color_names[idx];
}

int ColorDetectBase::get_color_num()
{
    return m_hsv_min.size();
}

ColorDetect::ColorDetect(uint16_t width, uint16_t height) :
    ColorDetectBase(width, height),
    m_morphology(false),
    m_hsv_mask_label(heap_caps_malloc(width * height * 2, MALLOC_CAP_DEFAULT)),
    m_hsv_mask_label_cvmat(width, height, CV_16U, m_hsv_mask_label)
{
}

ColorDetect::~ColorDetect()
{
    heap_caps_free(m_hsv_mask_label);
}

void ColorDetect::enable_morphology(int kernel_size)
{
    m_morphology = true;
    m_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
}

std::list<dl::detect::result_t> &ColorDetect::run(const dl::image::img_t &img)
{
    m_result.clear();
    int n = get_color_num();
    if (n == 0) {
        ESP_LOGE(TAG, "No color is registered. Please call register_color() first.");
        return m_result;
    }
    dl::image::img_t hsv_mask_img = {
        .data = m_hsv_mask, .width = m_width, .height = m_height, .pix_type = dl::image::DL_IMAGE_PIX_TYPE_HSV_MASK};
    if (n == 1) {
        m_T.reset().set_src_img(img).set_dst_img(hsv_mask_img).set_hsv_thr(m_hsv_min[0], m_hsv_max[0]).transform();
        hsv_mask_process(0, m_T.get_scale_x(true), m_T.get_scale_y(true), img.width, img.height);
    } else {
        dl::image::img_t hsv_img = {
            .data = m_hsv, .width = m_width, .height = m_height, .pix_type = dl::image::DL_IMAGE_PIX_TYPE_HSV};
        m_T.reset().set_src_img(img).set_dst_img(hsv_img).transform();
        float scale_x = m_T.get_scale_x(true);
        float scale_y = m_T.get_scale_y(true);
        m_T.reset().set_src_img(hsv_img).set_dst_img(hsv_mask_img);
        for (int i = 0; i < n; i++) {
            m_T.set_hsv_thr(m_hsv_min[i], m_hsv_max[i]).transform();
            hsv_mask_process(i, scale_x, scale_y, img.width, img.height);
        }
    }
    return m_result;
}

void ColorDetect::hsv_mask_process(
    int color_id, float inv_scale_x, float inv_scale_y, uint16_t limit_width, uint16_t limit_height)
{
    if (m_morphology) {
        cv::morphologyEx(m_hsv_mask_cvmat, m_hsv_mask_cvmat, cv::MORPH_OPEN, m_kernel);
        cv::morphologyEx(m_hsv_mask_cvmat, m_hsv_mask_cvmat, cv::MORPH_CLOSE, m_kernel);
    }
    cv::Mat stats, centroids;
    int num_labels =
        cv::connectedComponentsWithStats(m_hsv_mask_cvmat, m_hsv_mask_label_cvmat, stats, centroids, 8, CV_16U);
    for (int i = 1; i < num_labels; i++) {
        auto p_stats = stats.ptr<int32_t>(i);
        // auto p_centroids = centroids.ptr<double>(i);
        if (p_stats[cv::CC_STAT_AREA] < m_area_thr[color_id]) {
            continue;
        }
        int x1 = (int)(p_stats[cv::CC_STAT_LEFT] * inv_scale_x);
        int y1 = (int)(p_stats[cv::CC_STAT_TOP] * inv_scale_y);
        int x2 = (int)((p_stats[cv::CC_STAT_WIDTH] + p_stats[cv::CC_STAT_LEFT]) * inv_scale_x);
        int y2 = (int)((p_stats[cv::CC_STAT_HEIGHT] + p_stats[cv::CC_STAT_TOP]) * inv_scale_y);
        dl::detect::result_t res = {color_id, 1.f, {x1, y1, x2, y2}, {}};
        res.limit_box(limit_width, limit_height);
        m_result.push_back(res);
    }
}

ColorRotateDetect::ColorRotateDetect(uint16_t width, uint16_t height, int kernel_size) :
    ColorDetectBase(width, height),
    m_kernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)))
{
}

std::vector<ColorRotateDetect::result_t> &ColorRotateDetect::run(const dl::image::img_t &img)
{
    m_result.clear();
    int n = get_color_num();
    if (n == 0) {
        ESP_LOGE(TAG, "No color is registered. Please call register_color() first.");
        return m_result;
    }
    dl::image::img_t hsv_mask_img = {
        .data = m_hsv_mask, .width = m_width, .height = m_height, .pix_type = dl::image::DL_IMAGE_PIX_TYPE_HSV_MASK};
    if (n == 1) {
        m_T.reset().set_src_img(img).set_dst_img(hsv_mask_img).set_hsv_thr(m_hsv_min[0], m_hsv_max[0]).transform();
        hsv_mask_process(0, m_T.get_scale_x(true), m_T.get_scale_y(true));
    } else {
        dl::image::img_t hsv_img = {
            .data = m_hsv, .width = m_width, .height = m_height, .pix_type = dl::image::DL_IMAGE_PIX_TYPE_HSV};
        m_T.reset().set_src_img(img).set_dst_img(hsv_img).transform();
        float scale_x = m_T.get_scale_x(true);
        float scale_y = m_T.get_scale_y(true);
        m_T.reset().set_src_img(hsv_img).set_dst_img(hsv_mask_img);
        for (int i = 0; i < n; i++) {
            m_T.set_hsv_thr(m_hsv_min[i], m_hsv_max[i]).transform();
            hsv_mask_process(i, scale_x, scale_y);
        }
    }
    return m_result;
}

void ColorRotateDetect::hsv_mask_process(int color_id, float inv_scale_x, float inv_scale_y)
{
    cv::morphologyEx(m_hsv_mask_cvmat, m_hsv_mask_cvmat, cv::MORPH_OPEN, m_kernel);
    cv::morphologyEx(m_hsv_mask_cvmat, m_hsv_mask_cvmat, cv::MORPH_CLOSE, m_kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_hsv_mask_cvmat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);
        if (area < m_area_thr[color_id]) {
            continue;
        }
        cv::RotatedRect rot_rect = cv::minAreaRect(contour);
        double angle_rad = rot_rect.angle * CV_PI / 180;
        float sin = (float)std::sin(angle_rad);
        float cos = (float)std::cos(angle_rad);
        float width = std::sqrt(std::pow(rot_rect.size.width * cos * inv_scale_x, 2) +
                                std::pow(rot_rect.size.width * sin * inv_scale_y, 2));
        float height = std::sqrt(std::pow(rot_rect.size.height * sin * inv_scale_x, 2) +
                                 std::pow(rot_rect.size.height * cos * inv_scale_y, 2));
        m_result.emplace_back(color_id,
                              cv::RotatedRect{{rot_rect.center.x * inv_scale_x, rot_rect.center.y * inv_scale_y},
                                              {width, height},
                                              rot_rect.angle});
    }
}
