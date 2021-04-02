#pragma once

#include <vector>
#include "detector.hpp"

class CatFaceDetector : public DetectorAnchorPoint<int16_t, int16_t>
{
public:
    CatFaceDetector(std::vector<int> input_shape, float resize_scale, const float score_threshold, const float nms_threshold, const int top_k);
    ~CatFaceDetector();
    void call();
};
