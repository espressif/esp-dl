#pragma once

#include "dl_image_process.hpp"
#include "dl_model_base.hpp"

namespace dl {
namespace image {
class ImagePreprocessor {
public:
    /**
     * @brief Construct a new Image Preprocessor object
     * @note Normalization parameters mean and std here is not in range [0, 1], but [0, 255].
     *
     * @param model Model object ptr.
     * @param mean Normalization mean in range [0, 255], usually 1 channel for gray image, 3 channel for color image.
     * @param std Normalization std in range [0, 255], usually 1 channel for gray image, 3 channel for color image.
     * @param rgb_swap Model accepts color image data in rgb(rgb_swap=false) or bgr(rgb_swap=true), only take affects
     * for color image.
     * @param input_name If there's other inputs besides the image, input_name of the image must be specified.
     */
    ImagePreprocessor(Model *model,
                      const std::vector<float> &mean,
                      const std::vector<float> &std,
                      bool rgb_swap = false,
                      const std::string &input_name = "");
    void enable_letterbox(const std::vector<uint8_t> &bg_value);
    float get_resize_scale_x(bool inv = false);
    float get_resize_scale_y(bool inv = false);
    int get_crop_area_top_left_x();
    int get_crop_area_top_left_y();
    int get_border_top();
    int get_border_left();
    TensorBase *get_model_input();
    void preprocess(const img_t &img, const std::vector<int> &crop_area = {});
    void preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv = false);

private:
    ImageTransformer m_image_transformer;
    TensorBase *m_model_input;

    // for letter box
    bool m_letter_box;
    std::vector<uint8_t> m_bg_value;
};

} // namespace image
} // namespace dl
