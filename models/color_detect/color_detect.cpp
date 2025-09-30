#include "color_detect.hpp"
#include "dl_image_color.hpp"

static const char *TAG = "ColorDetect";

ColorDetect::ColorDetect(uint16_t width, uint16_t height) :
    m_src_img(),
    m_width(width),
    m_height(height),
    m_hsv(nullptr),
    m_hsv_mask(heap_caps_malloc(width * height, MALLOC_CAP_DEFAULT)),
    m_mask_label(heap_caps_malloc(width * height * 2, MALLOC_CAP_DEFAULT)),
    m_inv_scale_x(0),
    m_inv_scale_y(0),
    m_x(nullptr),
    m_y(nullptr),
    m_gen_xy_map(false),
    m_morphology(false)
{
}

ColorDetect::~ColorDetect()
{
    auto check_and_free = [](void *ptr) {
        if (ptr) {
            heap_caps_free(ptr);
        }
    };
    check_and_free(m_x);
    check_and_free(m_y);
    check_and_free(m_hsv);
    heap_caps_free(m_hsv_mask);
    heap_caps_free(m_mask_label);
}

void ColorDetect::register_color(const std::vector<uint8_t> &hsv_min,
                                 const std::vector<uint8_t> &hsv_max,
                                 const std::string &name,
                                 int area_thr)
{
    assert(hsv_min.size() == 3 && hsv_max.size() == 3 && hsv_min[0] <= 180 && hsv_max[0] <= 180 &&
           hsv_min[1] < hsv_max[1] && hsv_min[2] < hsv_max[2] && area_thr < m_width * m_height);
    if (std::find(m_color_names.begin(), m_color_names.end(), name) != m_color_names.end()) {
        ESP_LOGE(TAG, "Color %s is already registered.", name.c_str());
        return;
    }
    m_hsv_min.emplace_back(hsv_min);
    m_hsv_max.emplace_back(hsv_max);
    m_area_thr.emplace_back(area_thr);
    m_color_names.emplace_back(name);
    if (m_hsv_min.size() == 2) {
        m_hsv = heap_caps_malloc(m_width * m_height * 3, MALLOC_CAP_DEFAULT);
    }
}

void ColorDetect::delete_color(int idx)
{
    assert(idx >= 0 && idx < m_hsv_min.size());
    m_hsv_min.erase(m_hsv_min.begin() + idx);
    m_hsv_max.erase(m_hsv_max.begin() + idx);
    m_area_thr.erase(m_area_thr.begin() + idx);
    m_color_names.erase(m_color_names.begin() + idx);
}

void ColorDetect::delete_color(const std::string &name)
{
    auto it = std::find(m_color_names.begin(), m_color_names.end(), name);
    if (it == m_color_names.end()) {
        ESP_LOGE(TAG, "Failed to delete Color %s, it is not registered.", name.c_str());
        return;
    }
    delete_color(std::distance(m_color_names.begin(), it));
}

void ColorDetect::enable_morphology(int kernel_size)
{
    m_morphology = true;
    m_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
}

std::string ColorDetect::get_color_name(int idx)
{
    assert(idx >= 0 && idx < m_hsv_min.size());
    return m_color_names[idx];
}

void ColorDetect::gen_xy_map()
{
    int col_step = get_pix_byte_size(m_src_img.pix_type);
    int row_step = m_src_img.width * get_pix_byte_size(m_src_img.pix_type);

    auto check_and_free = [](void *ptr) {
        if (ptr) {
            heap_caps_free(ptr);
        }
    };
    check_and_free(m_x);
    check_and_free(m_y);

    m_x = (int *)heap_caps_malloc(m_width * sizeof(int), MALLOC_CAP_DEFAULT);
    m_y = (int *)heap_caps_malloc(m_height * sizeof(int), MALLOC_CAP_DEFAULT);
    int src_width = m_src_img.width;
    int src_height = m_src_img.height;
    m_inv_scale_x = static_cast<float>(src_width) / m_width;
    m_inv_scale_y = static_cast<float>(src_height) / m_height;
    for (int i = 0; i < m_width; i++) {
        m_x[i] = std::min(static_cast<int>(i * m_inv_scale_x), src_width - 1) * col_step;
    }
    for (int i = 0; i < m_height; i++) {
        m_y[i] = std::min(static_cast<int>(i * m_inv_scale_y), src_height - 1) * row_step;
    }
}

std::list<dl::detect::result_t> &ColorDetect::run(const dl::image::img_t &img)
{
    m_result.clear();
    if (m_hsv_min.empty() || m_hsv_max.empty()) {
        ESP_LOGE(TAG, "No color is registered. Please call register_color() first.");
        return m_result;
    }
    if (!img.data || !img.height || !img.width ||
        !(img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB888 || img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB565)) {
        ESP_LOGE(TAG, "Invalid src img.");
        return m_result;
    }
    if ((img.width != m_src_img.width) || (img.height != m_src_img.height) ||
        get_pix_byte_size(img.pix_type) != get_pix_byte_size(m_src_img.pix_type)) {
        m_gen_xy_map = true;
    }
    m_src_img = img;
    if (m_gen_xy_map) {
        gen_xy_map();
        m_gen_xy_map = false;
    }
#if CONFIG_IDF_TARGET_ESP32P4
    constexpr bool big_endian = false;
#else
    constexpr bool big_endian = true;
#endif
    bool h_across_zero = m_hsv_min[0][0] > m_hsv_max[0][0];
    if (img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB888) {
        if (m_hsv_min.size() == 1) {
            if (h_across_zero) {
                rgb2hsv_mask(dl::image::RGB8882HSVMask<false, true>(m_hsv_min[0], m_hsv_max[0]));
            } else {
                rgb2hsv_mask(dl::image::RGB8882HSVMask<false, false>(m_hsv_min[0], m_hsv_max[0]));
            }
        } else {
            if (h_across_zero) {
                rgb2hsv_and_hsv_mask(dl::image::RGB8882HSVAndHSVMask<false, true>(m_hsv_min[0], m_hsv_max[0]));
            } else {
                rgb2hsv_and_hsv_mask(dl::image::RGB8882HSVAndHSVMask<false, false>(m_hsv_min[0], m_hsv_max[0]));
            }
        }
    } else if (img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB565) {
        if (m_hsv_min.size() == 1) {
            if (h_across_zero) {
                rgb2hsv_mask(dl::image::RGB5652HSVMask<big_endian, false, true>(m_hsv_min[0], m_hsv_max[0]));
            } else {
                rgb2hsv_mask(dl::image::RGB5652HSVMask<big_endian, false, false>(m_hsv_min[0], m_hsv_max[0]));
            }
        } else {
            if (h_across_zero) {
                rgb2hsv_and_hsv_mask(
                    dl::image::RGB5652HSVAndHSVMask<big_endian, false, true>(m_hsv_min[0], m_hsv_max[0]));
            } else {
                rgb2hsv_and_hsv_mask(
                    dl::image::RGB5652HSVAndHSVMask<big_endian, false, false>(m_hsv_min[0], m_hsv_max[0]));
            }
        }
    } else {
        ESP_LOGE(TAG, "Unsupported pix type.");
    }
    cv::Mat mask(m_height, m_width, CV_8UC1, m_hsv_mask);
    if (m_morphology) {
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, m_kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, m_kernel);
    }
    cv::Mat labels(m_height, m_width, CV_16U, m_mask_label);
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_16U);
    for (int i = 1; i < num_labels; i++) {
        auto p_stats = stats.ptr<int32_t>(i);
        // auto p_centroids = centroids.ptr<double>(i);
        if (p_stats[cv::CC_STAT_AREA] > m_area_thr[0]) {
            int x1 = (int)(p_stats[cv::CC_STAT_LEFT] * m_inv_scale_x);
            int y1 = (int)(p_stats[cv::CC_STAT_TOP] * m_inv_scale_y);
            int x2 = (int)((p_stats[cv::CC_STAT_WIDTH] + p_stats[cv::CC_STAT_LEFT]) * m_inv_scale_x);
            int y2 = (int)((p_stats[cv::CC_STAT_HEIGHT] + p_stats[cv::CC_STAT_TOP]) * m_inv_scale_y);
            m_result.push_back({0, 1.f, {x1, y1, x2, y2}, {}});
        }
    }
    for (int i = 1; i < m_hsv_min.size(); i++) {
        if (m_hsv_min[i][0] > m_hsv_max[i][0]) {
            hsv2hsv_mask(dl::image::HSV2HSVMask<true>(m_hsv_min[i], m_hsv_max[i]));
        } else {
            hsv2hsv_mask(dl::image::HSV2HSVMask<false>(m_hsv_min[i], m_hsv_max[i]));
        }
        if (m_morphology) {
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, m_kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, m_kernel);
        }
        int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_16U);
        for (int j = 1; j < num_labels; j++) {
            auto p_stats = stats.ptr<int32_t>(j);
            if (p_stats[cv::CC_STAT_AREA] > m_area_thr[i]) {
                int x1 = (int)(p_stats[cv::CC_STAT_LEFT] * m_inv_scale_x);
                int y1 = (int)(p_stats[cv::CC_STAT_TOP] * m_inv_scale_y);
                int x2 = (int)((p_stats[cv::CC_STAT_WIDTH] + p_stats[cv::CC_STAT_LEFT]) * m_inv_scale_x);
                int y2 = (int)((p_stats[cv::CC_STAT_HEIGHT] + p_stats[cv::CC_STAT_TOP]) * m_inv_scale_y);
                m_result.push_back({i, 1.f, {x1, y1, x2, y2}, {}});
            }
        }
    }
    for (auto &res : m_result) {
        res.limit_box(img.width, img.height);
    }
    return m_result;
}
