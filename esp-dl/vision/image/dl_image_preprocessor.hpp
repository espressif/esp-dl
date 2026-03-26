#pragma once

#include "dl_image_process.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

namespace dl {
namespace image {
class ImagePreprocessor {
public:
    /**
     * @brief Construct a new Image Preprocessor object for color Image
     * @note Normalization parameters mean and std here is not in range [0, 1], but [0, 255].
     *
     * @param model Model object ptr.
     * @param mean 3 channel normalization mean in range [0, 255]
     * @param std 3 channel normalization std in range [0, 255]
     * @param rgb_swap Model accepts color image data in rgb(rgb_swap=false) or bgr(rgb_swap=true)
     * @param input_name If there's other inputs besides the image, input_name of the image must be specified.
     */
    ImagePreprocessor(Model *model,
                      const std::array<float, 3> &mean,
                      const std::array<float, 3> &std,
                      bool rgb_swap = false,
                      const std::string &input_name = "");
    /**
     * @brief Construct a new Image Preprocessor object for gray image
     * @note Normalization parameters mean and std here is not in range [0, 1], but [0, 255].
     *
     * @param model Model object ptr.
     * @param mean 1 channel normalization mean in range [0, 255]
     * @param std 1 channel normalization std in range [0, 255]
     * @param input_name If there's other inputs besides the image, input_name of the image must be specified.
     */
    ImagePreprocessor(Model *model, float mean, float std, const std::string &input_name = "");
    void enable_letterbox(const std::array<uint8_t, 3> &bg_value = {});
    void enable_letterbox(uint8_t bg_value = 0);
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
    bool m_letter_box;
};

} // namespace image
} // namespace dl
