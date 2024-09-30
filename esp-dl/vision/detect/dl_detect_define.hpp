#pragma once

#include "dl_define.hpp"
#include "dl_tool.hpp"
#include <vector>

namespace dl {
namespace detect {
typedef struct {
    int category;              /*<! category index */
    float score;               /*<! score of box */
    std::vector<int> box;      /*<! [left_up_x, left_up_y, right_down_x, right_down_y] */
    std::vector<int> keypoint; /*<! [x1, y1, x2, y2, ...] */
} result_t;

inline bool compare_greater_box(result_t a, result_t b)
{
    return a.score > b.score;
}

typedef struct {
    int stride_y;
    int stride_x;
    int offset_y;
    int offset_x;
} anchor_point_stage_t;

typedef struct {
    int stride_y;
    int stride_x;
    int offset_y;
    int offset_x;
    std::vector<std::vector<int>> anchor_shape;
} anchor_box_stage_t;
} // namespace detect
} // namespace dl
