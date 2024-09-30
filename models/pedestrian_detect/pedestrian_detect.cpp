#include "pedestrian_detect.hpp"

PedestrianDetect::PedestrianDetect()
{
    this->model = (void *)new dl::detect::Pedestrian<int8_t>(
        0.5, 0.5, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}}, {0, 0, 0}, {1, 1, 1});
}

PedestrianDetect::~PedestrianDetect()
{
    if (this->model) {
        delete (dl::detect::Pedestrian<int8_t> *)this->model;
        this->model = nullptr;
    }
}

template <typename T>
std::list<dl::detect::result_t> &PedestrianDetect::run(T *input_element, std::vector<int> input_shape)
{
    return ((dl::detect::Pedestrian<int8_t> *)this->model)->run(input_element, input_shape);
}
template std::list<dl::detect::result_t> &PedestrianDetect::run(uint16_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &PedestrianDetect::run(uint8_t *input_element, std::vector<int> input_shape);

void PedestrianDetect::set_print_info(bool print_info)
{
    ((dl::detect::Pedestrian<int8_t> *)this->model)->set_print_info(print_info);
}
