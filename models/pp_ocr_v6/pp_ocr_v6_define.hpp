#pragma once
#include <array>
#include <cmath>
#include <limits>

namespace pp_ocr_v6 {
struct Point {
    float x;
    float y;
};

inline float euclidean_distance(const Point &a, const Point &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// TL, TR, BR, BL (PaddleOCR order_points_clockwise).
inline std::array<Point, 4> order_points_clockwise(std::array<Point, 4> pts)
{
    int idx_tl = 0;
    int idx_br = 0;
    float min_s = std::numeric_limits<float>::max();
    float max_s = -std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i) {
        float s = pts[i].x + pts[i].y;
        if (s < min_s) {
            min_s = s;
            idx_tl = i;
        }
        if (s > max_s) {
            max_s = s;
            idx_br = i;
        }
    }

    int idx_a = -1;
    int idx_b = -1;
    for (int i = 0; i < 4; ++i) {
        if (i == idx_tl || i == idx_br) {
            continue;
        }
        (idx_a < 0 ? idx_a : idx_b) = i;
    }

    if (idx_a < 0 || idx_b < 0) {
        return pts;
    }

    float diff_a = pts[idx_a].y - pts[idx_a].x;
    float diff_b = pts[idx_b].y - pts[idx_b].x;
    int idx_tr;
    int idx_bl;
    if (diff_a <= diff_b) {
        idx_tr = idx_a;
        idx_bl = idx_b;
    } else {
        idx_tr = idx_b;
        idx_bl = idx_a;
    }

    return {pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]};
}

} // namespace pp_ocr_v6
