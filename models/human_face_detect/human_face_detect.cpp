#include "human_face_detect.hpp"

HumanFaceDetect::HumanFaceDetect()
{
    this->stage1_model = (void *)new dl::detect::MSR01<int8_t>(
        0.5,
        0.5,
        10,
        {{8, 8, 9, 9, {{16, 16}, {32, 32}}}, {16, 16, 9, 9, {{64, 64}, {128, 128}}}},
        {0, 0, 0},
        {1, 1, 1});
    this->stage2_model =
        (void *)new dl::detect::MNP01<int8_t>(0.5, 0.5, 10, {{1, 1, 0, 0, {{48, 48}}}}, {0, 0, 0}, {1, 1, 1});
}

HumanFaceDetect::~HumanFaceDetect()
{
    if (this->stage1_model) {
        delete (dl::detect::MSR01<int8_t> *)this->stage1_model;
        this->stage1_model = nullptr;
    }
    if (this->stage2_model) {
        delete (dl::detect::MNP01<int8_t> *)this->stage2_model;
        this->stage2_model = nullptr;
    }
}

template <typename T>
std::list<dl::detect::result_t> &HumanFaceDetect::run(T *input_element, std::vector<int> input_shape)
{
    std::list<dl::detect::result_t> &candidates =
        ((dl::detect::MSR01<int8_t> *)this->stage1_model)->run(input_element, input_shape);
    return ((dl::detect::MNP01<int8_t> *)this->stage2_model)->run(input_element, input_shape, candidates);
}
template std::list<dl::detect::result_t> &HumanFaceDetect::run(uint16_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &HumanFaceDetect::run(uint8_t *input_element, std::vector<int> input_shape);
