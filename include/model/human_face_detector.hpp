#pragma once

#include <vector>
#include "detector.hpp"

class HumanFaceDetector : public DetectorAnchorBox<int16_t, int16_t>
{
public:
    HumanFaceDetector(std::vector<int> input_shape, float resize_scale, const float score_threshold, const float nms_threshold, const int top_k);
    ~HumanFaceDetector();
    void call();
};
