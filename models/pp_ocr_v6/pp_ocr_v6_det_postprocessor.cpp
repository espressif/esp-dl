#include "pp_ocr_v6_det_postprocessor.hpp"
#include "dl_define.hpp"
#include "esp_log.h"
#include "pp_ocr_v6_define.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

static const char *TAG = "pp_ocr_v6";

namespace pp_ocr_v6 {
static float orient2d(const Point &o, const Point &a, const Point &b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

static float polygon_signed_area(const std::array<Point, 4> &poly)
{
    float area = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const auto &a = poly[i];
        const auto &b = poly[(i + 1) % 4];
        area += a.x * b.y - b.x * a.y;
    }
    return area * 0.5f;
}

static std::vector<Point> convex_hull(std::vector<Point> points)
{
    std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
        if (a.x == b.x) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    points.erase(std::unique(points.begin(),
                             points.end(),
                             [](const Point &a, const Point &b) { return a.x == b.x && a.y == b.y; }),
                 points.end());
    if (points.size() <= 2) {
        return points;
    }

    std::vector<Point> hull;
    hull.reserve(points.size() * 2);
    for (const auto &p : points) {
        while (hull.size() >= 2 && orient2d(hull[hull.size() - 2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    std::size_t lower_size = hull.size();
    for (int i = static_cast<int>(points.size()) - 2; i >= 0; --i) {
        const auto &p = points[i];
        while (hull.size() > lower_size && orient2d(hull[hull.size() - 2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    hull.pop_back();
    return hull;
}

// DBPostProcess.get_mini_boxes (approx. cv2.minAreaRect via hull).
static std::pair<std::array<Point, 4>, float> get_mini_boxes(const std::vector<Point> &points)
{
    auto hull = convex_hull(points);
    if (hull.empty()) {
        return {{}, 0.0f};
    }
    if (hull.size() == 1) {
        const auto &p = hull[0];
        std::array<Point, 4> box = {{{p.x, p.y}, {p.x + 1, p.y}, {p.x + 1, p.y + 1}, {p.x, p.y + 1}}};
        return {box, 1.0f};
    }

    float best_area = std::numeric_limits<float>::max();
    std::array<Point, 4> best_rect = {};

    for (std::size_t i = 0; i < hull.size(); ++i) {
        const auto &p0 = hull[i];
        const auto &p1 = hull[(i + 1) % hull.size()];
        float angle = std::atan2(p1.y - p0.y, p1.x - p0.x);
        float c = std::cos(angle);
        float s = std::sin(angle);
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();

        for (const auto &p : hull) {
            float rx = p.x * c + p.y * s;
            float ry = -p.x * s + p.y * c;
            min_x = std::min(min_x, rx);
            min_y = std::min(min_y, ry);
            max_x = std::max(max_x, rx);
            max_y = std::max(max_y, ry);
        }

        float area = (max_x - min_x) * (max_y - min_y);
        if (area < best_area) {
            best_area = area;
            std::array<Point, 4> rect = {{{min_x, min_y}, {max_x, min_y}, {max_x, max_y}, {min_x, max_y}}};
            for (auto &p : rect) {
                float x = p.x * c - p.y * s;
                float y = p.x * s + p.y * c;
                p = {x, y};
            }
            best_rect = order_points_clockwise(rect);
        }
    }

    float sside =
        std::min(euclidean_distance(best_rect[0], best_rect[1]), euclidean_distance(best_rect[1], best_rect[2]));
    return {best_rect, sside};
}

// DBPostProcess.unclip (JT_ROUND on convex quad).
static std::vector<Point> unclip(const std::array<Point, 4> &box, float unclip_ratio)
{
    float signed_area = polygon_signed_area(box);
    float area = std::abs(signed_area);

    std::array<Point, 4> poly = box;
    if (signed_area < 0) {
        std::reverse(poly.begin(), poly.end());
    }

    std::array<float, 4> edge_len = {};
    std::array<Point, 4> outward = {};
    float length = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const auto &p0 = poly[i];
        const auto &p1 = poly[(i + 1) % 4];
        float dx = p1.x - p0.x;
        float dy = p1.y - p0.y;
        edge_len[i] = std::sqrt(dx * dx + dy * dy);
        length += edge_len[i];
        if (edge_len[i] > 1e-6f) {
            outward[i] = {dy / edge_len[i], -dx / edge_len[i]};
        }
    }
    float distance = length > 0.0f ? area * unclip_ratio / length : 0.0f;
    if (distance <= 1e-6f) {
        return {poly.begin(), poly.end()};
    }

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kArcStep = kPi / 12.0f;

    std::vector<Point> expanded;
    expanded.reserve(64);
    for (int i = 0; i < 4; ++i) {
        int prev = (i + 3) % 4;
        if (edge_len[prev] < 1e-6f || edge_len[i] < 1e-6f) {
            continue;
        }
        const Point &n0 = outward[prev];
        const Point &n1 = outward[i];
        float ang0 = std::atan2(n0.y, n0.x);
        float ang1 = std::atan2(n1.y, n1.x);
        float dang = ang1 - ang0;
        while (dang <= 0.0f) {
            dang += 2.0f * kPi;
        }
        while (dang > 2.0f * kPi) {
            dang -= 2.0f * kPi;
        }
        if (dang > kPi + 1e-3f) {
            expanded.push_back({poly[i].x + n0.x * distance, poly[i].y + n0.y * distance});
            expanded.push_back({poly[i].x + n1.x * distance, poly[i].y + n1.y * distance});
            continue;
        }
        int steps = std::max(1, static_cast<int>(std::ceil(dang / kArcStep)));
        for (int s = 0; s <= steps; ++s) {
            float t = ang0 + dang * (static_cast<float>(s) / static_cast<float>(steps));
            expanded.push_back({poly[i].x + distance * std::cos(t), poly[i].y + distance * std::sin(t)});
        }
    }
    return expanded;
}

static bool point_in_polygon(float x, float y, const std::array<Point, 4> &poly)
{
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        const auto &pi = poly[i];
        const auto &pj = poly[j];
        if (((pi.y > y) != (pj.y > y)) && (x < (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y + 1e-6f) + pi.x)) {
            inside = !inside;
        }
    }
    return inside;
}

static float box_score_fast(const int8_t *pred, float scale, int height, int width, const std::array<Point, 4> &box)
{
    int xmin = width - 1;
    int xmax = 0;
    int ymin = height - 1;
    int ymax = 0;
    for (const auto &p : box) {
        xmin = std::min(xmin, static_cast<int>(std::floor(p.x)));
        xmax = std::max(xmax, static_cast<int>(std::ceil(p.x)));
        ymin = std::min(ymin, static_cast<int>(std::floor(p.y)));
        ymax = std::max(ymax, static_cast<int>(std::ceil(p.y)));
    }
    xmin = std::clamp(xmin, 0, width - 1);
    xmax = std::clamp(xmax, 0, width - 1);
    ymin = std::clamp(ymin, 0, height - 1);
    ymax = std::clamp(ymax, 0, height - 1);

    float score_sum = 0.0f;
    int count = 0;
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            if (point_in_polygon(x + 0.5f, y + 0.5f, box)) {
                score_sum += dl::dequantize(pred[y * width + x], scale);
                ++count;
            }
        }
    }
    if (count == 0) {
        return 0.0f;
    }
    return score_sum / count;
}

static std::array<Point, 4> map_box_to_image(
    const std::array<Point, 4> &box, int pred_w, int pred_h, int model_w, int model_h, float inv_resize_scale)
{
    std::array<Point, 4> mapped = {};
    float to_model_x = static_cast<float>(model_w) / pred_w;
    float to_model_y = static_cast<float>(model_h) / pred_h;
    for (int i = 0; i < 4; ++i) {
        mapped[i].x = std::round(box[i].x * to_model_x * inv_resize_scale);
        mapped[i].y = std::round(box[i].y * to_model_y * inv_resize_scale);
    }
    return mapped;
}

static std::array<Point, 4> clip_det_res(std::array<Point, 4> points, int img_height, int img_width)
{
    for (auto &p : points) {
        p.x = static_cast<float>(std::clamp(static_cast<int>(p.x), 0, img_width - 1));
        p.y = static_cast<float>(std::clamp(static_cast<int>(p.y), 0, img_height - 1));
    }
    return points;
}

static bool filter_tag_det_res(std::array<Point, 4> &box, int img_height, int img_width, int min_size)
{
    box = order_points_clockwise(box);
    box = clip_det_res(box, img_height, img_width);
    int rect_width = static_cast<int>(euclidean_distance(box[0], box[1]));
    int rect_height = static_cast<int>(euclidean_distance(box[0], box[3]));
    return rect_width > min_size && rect_height > min_size;
}

static void sorted_boxes(std::vector<TextBox> &boxes)
{
    std::sort(boxes.begin(), boxes.end(), [](const TextBox &a, const TextBox &b) {
        if (a.points[1] != b.points[1]) {
            return a.points[1] < b.points[1];
        }
        return a.points[0] < b.points[0];
    });
    for (std::size_t i = 1; i < boxes.size(); ++i) {
        for (std::size_t j = i; j > 0; --j) {
            if (std::abs(boxes[j].points[1] - boxes[j - 1].points[1]) < 10 &&
                boxes[j].points[0] < boxes[j - 1].points[0]) {
                std::swap(boxes[j], boxes[j - 1]);
            } else {
                break;
            }
        }
    }
}

DetPostprocessor::DetPostprocessor(
    dl::Model *model, float thresh, float box_thresh, float unclip_ratio, int min_size, int max_candidates) :
    m_model(model),
    m_thresh(thresh),
    m_box_thresh(box_thresh),
    m_unclip_ratio(unclip_ratio),
    m_min_size(min_size),
    m_max_candidates(max_candidates)
{
}

std::vector<TextBox> DetPostprocessor::postprocess(const dl::image::img_t &img,
                                                   float resize_scale,
                                                   int model_w,
                                                   int model_h)
{
    auto outputs = m_model->get_outputs();
    if (outputs.empty()) {
        ESP_LOGW(TAG, "Detection model has no outputs.");
        return {};
    }

    dl::TensorBase *pred_tensor = outputs.begin()->second;
    const auto &shape = pred_tensor->shape;
    if (!(shape.size() == 4 && shape[0] == 1 && shape[3] == 1) || shape[1] <= 0 || shape[2] <= 0) {
        ESP_LOGW(TAG, "Detection output must be NHWC [1,H,W,1], got shape %s.", dl::vector_to_string(shape).c_str());
        return {};
    }
    if (pred_tensor->dtype != dl::DATA_TYPE_INT8) {
        ESP_LOGW(TAG, "Detection output must be INT8, got %s.", dl::dtype_to_string(pred_tensor->dtype));
        return {};
    }

    const int height = shape[1];
    const int width = shape[2];
    const int8_t *pred = static_cast<const int8_t *>(pred_tensor->data);
    float scale = DL_SCALE(pred_tensor->exponent);
    const int8_t thr_q = dl::quantize<int8_t>(m_thresh, 1.f / scale);
    const float inv_scale = resize_scale > 0.0f ? 1.0f / resize_scale : 1.0f;

    std::vector<uint8_t> visited(height * width, 0);
    std::vector<TextBox> boxes;
    boxes.reserve(32);
    std::queue<int> queue;
    int candidate_count = 0;
    constexpr int kBfsDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    constexpr int kBfsDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    constexpr int kBoundDx[4] = {1, -1, 0, 0};
    constexpr int kBoundDy[4] = {0, 0, 1, -1};

    for (int y = 0; y < height && candidate_count < m_max_candidates; ++y) {
        for (int x = 0; x < width && candidate_count < m_max_candidates; ++x) {
            int start = y * width + x;
            if (visited[start] || pred[start] <= thr_q) {
                continue;
            }

            ++candidate_count;
            visited[start] = 1;
            queue.push(start);

            // Keep only boundary pixels during BFS to avoid a full-component
            // allocation for large connected regions.
            std::vector<Point> points;

            while (!queue.empty()) {
                int idx = queue.front();
                queue.pop();

                int cy = idx / width;
                int cx = idx % width;

                bool boundary = false;
                for (int k = 0; k < 4; ++k) {
                    int nx = cx + kBoundDx[k];
                    int ny = cy + kBoundDy[k];
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height || pred[ny * width + nx] <= thr_q) {
                        boundary = true;
                        break;
                    }
                }
                if (boundary) {
                    points.push_back({static_cast<float>(cx), static_cast<float>(cy)});
                }

                for (int k = 0; k < 8; ++k) {
                    int nx = cx + kBfsDx[k];
                    int ny = cy + kBfsDy[k];
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                        continue;
                    }
                    int nidx = ny * width + nx;
                    if (!visited[nidx] && pred[nidx] > thr_q) {
                        visited[nidx] = 1;
                        queue.push(nidx);
                    }
                }
            }

            if (points.empty()) {
                continue;
            }

            auto [box, sside] = get_mini_boxes(points);
            if (sside < static_cast<float>(m_min_size)) {
                continue;
            }
            float score = box_score_fast(pred, scale, height, width, box);
            if (m_box_thresh > score) {
                continue;
            }
            auto expanded = unclip(box, m_unclip_ratio);
            if (expanded.empty()) {
                continue;
            }
            auto [unclipped, unclipped_sside] = get_mini_boxes(expanded);
            if (unclipped_sside < static_cast<float>(m_min_size + 2)) {
                continue;
            }

            auto quad = map_box_to_image(unclipped, width, height, model_w, model_h, inv_scale);
            if (!filter_tag_det_res(quad, static_cast<int>(img.height), static_cast<int>(img.width), m_min_size)) {
                continue;
            }

            std::array<int, 8> normalized;
            for (int i = 0; i < 4; ++i) {
                normalized[i * 2] = static_cast<int>(std::round(quad[i].x));
                normalized[i * 2 + 1] = static_cast<int>(std::round(quad[i].y));
            }
            boxes.push_back({normalized, score});
        }
    }

    sorted_boxes(boxes);
    return boxes;
}

} // namespace pp_ocr_v6
