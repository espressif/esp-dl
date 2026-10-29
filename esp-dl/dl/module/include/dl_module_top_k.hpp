#pragma once

#include "dl_module_base.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief TopK operator as defined in ONNX.
 *
 * Reference: https://onnx.ai/onnx/operators/onnx__TopK.html
 */
class TopK : public Module {
private:
    int m_k;                    /*!< A positive, constant K */
    int m_axis;                 /*!< Dimension on which to do the sort */
    int m_largest;              /*!< Whether to return the top-K largest or smallest elements */
    int m_sorted;               /*!< Whether to return the elements in sorted order */
    void *m_scratch = nullptr;  /*!< Reusable per-module scratch buffer */
    size_t m_scratch_bytes = 0; /*!< Allocated bytes in m_scratch */

    // std::isnan is not reliable under -ffast-math; inspect IEEE-754 bits.
    template <typename T1>
    static bool is_nan_fastmath_safe(T1 value)
    {
        if constexpr (std::is_same_v<T1, float>) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
        } else if constexpr (std::is_same_v<T1, double>) {
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return (bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
        } else {
            return false;
        }
    }

    // NaN compares unequal to itself and loses every `<`/`>` test, so a plain `!=` guard would put
    // each NaN in its own rank class while ordering it identically to all others. That breaks the
    // transitivity of equivalence that std::sort / std::nth_element require.
    template <typename T1>
    static bool values_tie(T1 lhs, T1 rhs)
    {
        if constexpr (std::is_floating_point_v<T1>) {
            if (is_nan_fastmath_safe(lhs) && is_nan_fastmath_safe(rhs)) {
                return true;
            }
        }
        return lhs == rhs;
    }

    // Ranks NaN above every number, matching torch.topk / torch.sort: NaN wins under `largest`
    // and is pushed to the end otherwise.
    template <typename T1>
    static bool value_ranks_before(T1 lhs, T1 rhs, bool largest)
    {
        if constexpr (std::is_floating_point_v<T1>) {
            const bool lhs_nan = is_nan_fastmath_safe(lhs);
            const bool rhs_nan = is_nan_fastmath_safe(rhs);
            if (lhs_nan || rhs_nan) {
                return lhs_nan != rhs_nan ? (largest ? lhs_nan : rhs_nan) : false;
            }
        }
        return largest ? (lhs > rhs) : (lhs < rhs);
    }

    template <typename T1>
    struct RanksBefore {
        bool largest;

        bool operator()(const std::pair<T1, int> &lhs, const std::pair<T1, int> &rhs) const
        {
            if (values_tie(lhs.first, rhs.first)) {
                return lhs.second < rhs.second;
            }
            return value_ranks_before(lhs.first, rhs.first, largest);
        }
    };

    static constexpr int COUNTING_MIN_K = 5;
    static constexpr int COUNTING_K_AXIS_DIVISOR = 256;
    static constexpr int HEAP_MAX_K_FLOOR = 16;
    static constexpr int HEAP_K_AXIS_DIVISOR = 4;
    static constexpr int NTH_MAX_K_NUM = 3;
    static constexpr int NTH_MAX_K_DEN = 4;

    static bool is_supported_value_dtype(dtype_t dtype)
    {
        return dtype == DATA_TYPE_INT8 || dtype == DATA_TYPE_INT16 || dtype == DATA_TYPE_FLOAT ||
            dtype == DATA_TYPE_DOUBLE;
    }

    static bool is_supported_index_dtype(dtype_t dtype)
    {
        return dtype == DATA_TYPE_INT8 || dtype == DATA_TYPE_INT16 || dtype == DATA_TYPE_INT32 ||
            dtype == DATA_TYPE_INT64;
    }

    static bool is_supported_quant_type(quant_type_t type)
    {
        return type == QUANT_TYPE_NONE || type == QUANT_TYPE_FLOAT32 || type == QUANT_TYPE_SYMM_8BIT ||
            type == QUANT_TYPE_SYMM_16BIT;
    }

    static bool dtype_matches_quant_type(dtype_t dtype, quant_type_t type)
    {
        switch (type) {
        case QUANT_TYPE_NONE:
            return true;
        case QUANT_TYPE_FLOAT32:
            // ESP-DL has no separate FP64 quant type. Keep the existing double kernel usable for
            // unquantized models, whose generic exporter marks every floating-point op as F32.
            return dtype == DATA_TYPE_FLOAT || dtype == DATA_TYPE_DOUBLE;
        case QUANT_TYPE_SYMM_8BIT:
            return dtype == DATA_TYPE_INT8;
        case QUANT_TYPE_SYMM_16BIT:
            return dtype == DATA_TYPE_INT16;
        default:
            return false;
        }
    }

    static dtype_t recorded_tensor_dtype(fbs::FbsModel *fbs_model, const std::string &name)
    {
        if (name.empty()) {
            return DATA_TYPE_UNDEFINED;
        }
        return fbs_model->is_parameter(name) ? fbs_model->get_tensor_dtype(name)
                                             : fbs_model->get_value_info_dtype(name);
    }

    static bool recorded_per_tensor_exponent(fbs::FbsModel *fbs_model, const std::string &name, int *exponent_out)
    {
        if (name.empty()) {
            return false;
        }
        if (fbs_model->is_parameter(name)) {
            const std::vector<int> exponents = fbs_model->get_tensor_exponents(name);
            if (exponents.size() != 1) {
                return false;
            }
            *exponent_out = exponents[0];
            return true;
        }
        *exponent_out = fbs_model->get_value_info_exponent(name);
        return true;
    }

    static bool checked_element_count(const std::vector<int> &shape, const char *role, int *count_out)
    {
        if (shape.empty()) {
            ESP_LOGE("TopK", "%s rank must be at least 1", role);
            return false;
        }

        int64_t count = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                ESP_LOGE("TopK", "%s shape must contain only positive dimensions", role);
                return false;
            }
            if (count > static_cast<int64_t>(std::numeric_limits<int>::max()) / dim) {
                ESP_LOGE("TopK", "%s element count exceeds INT_MAX", role);
                return false;
            }
            count *= dim;
        }
        *count_out = static_cast<int>(count);
        return true;
    }

    static bool resolve_axis_value(int axis_attribute, int rank, int *axis_out)
    {
        int axis = axis_attribute;
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            ESP_LOGE("TopK", "axis=%d out of range for rank=%d", axis_attribute, rank);
            return false;
        }
        *axis_out = axis;
        return true;
    }

    static bool topk_parse_k(TensorBase *tensor, int *out)
    {
        if (!tensor || tensor->get_dtype() != DATA_TYPE_INT64) {
            ESP_LOGE("TopK", "K must be a constant int64 tensor");
            return false;
        }

        const std::vector<int> &shape = tensor->get_shape();
        if (shape.size() != 1 || shape[0] != 1 || tensor->get_size() != 1) {
            ESP_LOGE("TopK", "K must have shape [1]");
            return false;
        }

        const int64_t *data = static_cast<int64_t *>(tensor->get_element_ptr());
        if (!data) {
            ESP_LOGE("TopK", "K has no data");
            return false;
        }

        const int64_t raw = data[0];
        if (raw <= 0 || raw > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            ESP_LOGE("TopK", "k=%lld is not in (0, INT_MAX]", static_cast<long long>(raw));
            return false;
        }
        *out = static_cast<int>(raw);
        return true;
    }

    bool resolve_axis(int rank, int *axis_out) const { return resolve_axis_value(m_axis, rank, axis_out); }

    static bool validate_dtypes(
        dtype_t input_dtype, dtype_t values_dtype, dtype_t indices_dtype, quant_type_t qtype, bool allow_undefined)
    {
        if ((!allow_undefined || input_dtype != DATA_TYPE_UNDEFINED) && !is_supported_value_dtype(input_dtype)) {
            ESP_LOGE("TopK", "unsupported X dtype %s", dtype_to_string(input_dtype));
            return false;
        }
        if ((!allow_undefined || (input_dtype != DATA_TYPE_UNDEFINED && values_dtype != DATA_TYPE_UNDEFINED)) &&
            values_dtype != input_dtype) {
            ESP_LOGE("TopK",
                     "Values dtype %s must match X dtype %s",
                     dtype_to_string(values_dtype),
                     dtype_to_string(input_dtype));
            return false;
        }
        if ((!allow_undefined || indices_dtype != DATA_TYPE_UNDEFINED) && !is_supported_index_dtype(indices_dtype)) {
            ESP_LOGE("TopK", "unsupported Indices dtype %s", dtype_to_string(indices_dtype));
            return false;
        }
        if ((!allow_undefined || input_dtype != DATA_TYPE_UNDEFINED) && !dtype_matches_quant_type(input_dtype, qtype)) {
            ESP_LOGE("TopK",
                     "X dtype %s does not match quant_type %s",
                     dtype_to_string(input_dtype),
                     quant_type_to_string(qtype));
            return false;
        }
        return true;
    }

    static bool validate_shapes_and_index_range(const std::vector<int> &input_shape,
                                                const std::vector<int> &values_shape,
                                                const std::vector<int> &indices_shape,
                                                dtype_t indices_dtype,
                                                int axis_attr,
                                                int k,
                                                int *axis_out,
                                                int *input_count_out,
                                                int *output_count_out)
    {
        if (!checked_element_count(input_shape, "X", input_count_out)) {
            return false;
        }
        if (!resolve_axis_value(axis_attr, static_cast<int>(input_shape.size()), axis_out)) {
            return false;
        }
        const int axis = *axis_out;
        const int axis_size = input_shape[axis];
        if (k > axis_size) {
            ESP_LOGE("TopK", "k=%d out of range (0, axis_size=%d]", k, axis_size);
            return false;
        }
        if ((indices_dtype == DATA_TYPE_INT8 && axis_size > static_cast<int>(std::numeric_limits<int8_t>::max()) + 1) ||
            (indices_dtype == DATA_TYPE_INT16 &&
             axis_size > static_cast<int>(std::numeric_limits<int16_t>::max()) + 1)) {
            ESP_LOGE("TopK", "axis_size=%d exceeds %s index range", axis_size, dtype_to_string(indices_dtype));
            return false;
        }
        if (values_shape.size() != input_shape.size() || indices_shape.size() != input_shape.size()) {
            ESP_LOGE("TopK",
                     "output ranks (%d/%d) must match input rank %d",
                     static_cast<int>(values_shape.size()),
                     static_cast<int>(indices_shape.size()),
                     static_cast<int>(input_shape.size()));
            return false;
        }
        for (size_t d = 0; d < input_shape.size(); ++d) {
            const int expected_dim = (static_cast<int>(d) == axis) ? k : input_shape[d];
            if (values_shape[d] != expected_dim || indices_shape[d] != expected_dim) {
                ESP_LOGE("TopK",
                         "output dim[%d] (%d/%d) != expected %d",
                         static_cast<int>(d),
                         values_shape[d],
                         indices_shape[d],
                         expected_dim);
                return false;
            }
        }
        return checked_element_count(values_shape, "output", output_count_out);
    }

    static int counting_int8_prepare_column(
        const int8_t *column, int axis_size, int inner, int k, bool largest, bool sorted, int *hist, int *taken_out)
    {
        std::fill(hist, hist + 256, 0);
        for (int a = 0; a < axis_size; a++) {
            hist[static_cast<int>(column[static_cast<size_t>(a) * inner]) + 128]++;
        }

        int taken = 0;
        int cut = 0;
        if (largest) {
            for (cut = 255; cut > 0; cut--) {
                if (taken + hist[cut] >= k) {
                    break;
                }
                taken += hist[cut];
            }
        } else {
            for (cut = 0; cut < 255; cut++) {
                if (taken + hist[cut] >= k) {
                    break;
                }
                taken += hist[cut];
            }
        }

        if (sorted) {
            // Turn hist[] into starting output positions in-place so no second 256-int table is needed.
            int running = 0;
            if (largest) {
                for (int v = 255; v > cut; v--) {
                    const int count = hist[v];
                    hist[v] = running;
                    running += count;
                }
            } else {
                for (int v = 0; v < cut; v++) {
                    const int count = hist[v];
                    hist[v] = running;
                    running += count;
                }
            }
            hist[cut] = running;
        }

        *taken_out = taken;
        return cut;
    }

    template <typename T1>
    static void heap_select_column(
        std::pair<T1, int> *heap, const T1 *column, int axis_size, int inner, int k, bool largest, bool sorted)
    {
        const RanksBefore<T1> ranks_before{largest};
        using Elem = std::pair<T1, int>;

        auto sift_down = [&](int root, Elem moving) {
            for (;;) {
                int child = 2 * root + 1;
                if (child >= k) {
                    break;
                }
                if (child + 1 < k && ranks_before(heap[child], heap[child + 1])) {
                    child++;
                }
                if (ranks_before(heap[child], moving)) {
                    break;
                }
                heap[root] = heap[child];
                root = child;
            }
            heap[root] = moving;
        };

        for (int a = 0; a < k; a++) {
            heap[a] = {column[static_cast<size_t>(a) * inner], a};
        }
        for (int parent = k / 2 - 1; parent >= 0; parent--) {
            sift_down(parent, heap[parent]);
        }

        for (int a = k; a < axis_size; a++) {
            const Elem candidate{column[static_cast<size_t>(a) * inner], a};
            if (ranks_before(candidate, heap[0])) {
                sift_down(0, candidate);
            }
        }

        if (sorted) {
            std::sort(heap, heap + k, ranks_before);
        }
    }

    template <typename T1>
    static void nth_select_column(
        std::pair<T1, int> *buf, const T1 *column, int axis_size, int inner, int k, bool largest, bool sorted)
    {
        const RanksBefore<T1> ranks_before{largest};
        for (int a = 0; a < axis_size; a++) {
            buf[a] = {column[static_cast<size_t>(a) * inner], a};
        }

        if (sorted) {
            if (static_cast<int64_t>(k) * NTH_MAX_K_DEN <= static_cast<int64_t>(axis_size) * NTH_MAX_K_NUM) {
                std::nth_element(buf, buf + (k - 1), buf + axis_size, ranks_before);
                std::sort(buf, buf + k, ranks_before);
            } else {
                // RanksBefore breaks value/NaN ties by unique axis index, so std::sort yields
                // the exact same order as std::stable_sort without allocating a temp buffer.
                std::sort(buf, buf + axis_size, ranks_before);
            }
        } else if (k < axis_size) {
            std::nth_element(buf, buf + (k - 1), buf + axis_size, ranks_before);
        }
    }

    static void topk_zero_tensor(TensorBase *tensor)
    {
        if (tensor != nullptr && tensor->data != nullptr && tensor->get_size() > 0 && tensor->get_bytes() > 0) {
            std::memset(tensor->data, 0, tensor->get_bytes());
        }
    }

    static void topk_zero_outputs(TensorBase *values, TensorBase *indices)
    {
        topk_zero_tensor(values);
        topk_zero_tensor(indices);
    }

    template <typename Elem>
    Elem *ensure_scratch(size_t count)
    {
        if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(Elem)) {
            return nullptr;
        }
        const size_t bytes = count * sizeof(Elem);
        if (m_scratch_bytes < bytes) {
            void *next = std::realloc(m_scratch, bytes);
            if (!next) {
                ESP_LOGE("TopK", "failed to allocate %d-byte scratch buffer", static_cast<int>(bytes));
                return nullptr;
            }
            m_scratch = next;
            m_scratch_bytes = bytes;
        }
        return static_cast<Elem *>(m_scratch);
    }

public:
    /**
     * @brief Construct a new TopK object
     *
     * @param k             Number of top elements to retrieve
     * @param axis          Dimension on which to do the sort
     * @param largest       Whether to return the top-K largest or smallest elements
     * @param sorted        Whether to return the elements in sorted order
     * @param name          Op name
     * @param inplace       The inplace type
     * @param quant_type    Quantization type
     */
    TopK(int k = 1,
         int axis = -1,
         int largest = 1,
         int sorted = 1,
         const char *name = NULL,
         module_inplace_t inplace = MODULE_NON_INPLACE,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_k(k), m_axis(axis), m_largest(largest), m_sorted(sorted)
    {
    }

    /**
     * @brief Destroy the TopK object.
     */
    ~TopK() override { std::free(m_scratch); }

    void reset() override
    {
        std::free(m_scratch);
        m_scratch = nullptr;
        m_scratch_bytes = 0;
    }

    TopK(const TopK &) = delete;
    TopK &operator=(const TopK &) = delete;
    TopK(TopK &&) = delete;
    TopK &operator=(TopK &&) = delete;

    /**
     * @brief Calculate the output shape.
     *
     * @param input_shapes The shapes of inputs
     * @return output shapes [Values_shape, Indices_shape]
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        if (m_k <= 0 || (m_largest != 0 && m_largest != 1) || (m_sorted != 0 && m_sorted != 1)) {
            ESP_LOGE("TopK", "invalid attributes: k=%d, largest=%d, sorted=%d", m_k, m_largest, m_sorted);
            return {};
        }
        if ((input_shapes.size() != 1 && input_shapes.size() != 2) || input_shapes[0].empty()) {
            ESP_LOGE("TopK", "missing input shape");
            return {};
        }

        int input_count = 0;
        if (!checked_element_count(input_shapes[0], "X", &input_count)) {
            return {};
        }

        int axis = 0;
        if (!resolve_axis(static_cast<int>(input_shapes[0].size()), &axis)) {
            return {};
        }

        const int k = m_k;
        if (k > input_shapes[0][axis]) {
            ESP_LOGE("TopK", "k=%d out of range (0, axis_size=%d]", k, input_shapes[0][axis]);
            return {};
        }

        std::vector<int> out_shape = input_shapes[0];
        out_shape[axis] = k;
        int output_count = 0;
        if (!checked_element_count(out_shape, "output", &output_count)) {
            return {};
        }
        return {out_shape, out_shape};
    }

    template <typename T1, typename T2>
    void forward_argbest(T1 *values_out, T2 *indices_out, const T1 *input, int outer, int axis_size, int inner)
    {
        const bool largest = (m_largest != 0);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                const T1 *row = input + static_cast<size_t>(o) * axis_size;
                T1 best_value = row[0];
                int best_index = 0;
                for (int a = 1; a < axis_size; a++) {
                    if (value_ranks_before(row[a], best_value, largest)) {
                        best_value = row[a];
                        best_index = a;
                    }
                }
                values_out[o] = best_value;
                indices_out[o] = static_cast<T2>(best_index);
            }
            return;
        }

        for (int o = 0; o < outer; o++) {
            const T1 *base = input + static_cast<size_t>(o) * axis_size * inner;
            T1 *out_val_row = values_out + static_cast<size_t>(o) * inner;
            T2 *out_idx_row = indices_out + static_cast<size_t>(o) * inner;
            for (int i = 0; i < inner; i++) {
                out_val_row[i] = base[i];
                out_idx_row[i] = static_cast<T2>(0);
            }
            for (int a = 1; a < axis_size; a++) {
                const T1 *row = base + static_cast<size_t>(a) * inner;
                for (int i = 0; i < inner; i++) {
                    if (value_ranks_before(row[i], out_val_row[i], largest)) {
                        out_val_row[i] = row[i];
                        out_idx_row[i] = static_cast<T2>(a);
                    }
                }
            }
        }
    }

    // Counting sort needs no final sort pass: in-place prefix positions in hist[] hand each value a
    // slot range accumulated in rank order, and the scatter loop walks the axis forwards, so equal
    // values land in ascending index order. That is the exact order RanksBefore defines.
    template <typename T2>
    bool forward_counting_int8(
        int8_t *values_out, T2 *indices_out, const int8_t *input, int outer, int axis_size, int inner, int k)
    {
        const bool largest = (m_largest != 0);
        const bool sorted = (m_sorted != 0);
        // Keep the 1 KB histogram off the 4 KB FreeRTOS task stack while reusing hist[] in-place for positions.
        int *hist = ensure_scratch<int>(256);
        if (!hist) {
            return false;
        }

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                const int8_t *column = input + static_cast<size_t>(o) * axis_size * inner + i;
                int taken = 0;
                const int cut =
                    counting_int8_prepare_column(column, axis_size, inner, k, largest, sorted, hist, &taken);

                if (sorted) {
                    for (int a = 0; a < axis_size; a++) {
                        const int8_t raw = column[static_cast<size_t>(a) * inner];
                        const int v = static_cast<int>(raw) + 128;
                        const bool strictly_better = largest ? (v > cut) : (v < cut);
                        if (strictly_better || (v == cut && hist[cut] < k)) {
                            const size_t dst = static_cast<size_t>((o * k + hist[v]++) * inner + i);
                            values_out[dst] = raw;
                            indices_out[dst] = static_cast<T2>(a);
                        }
                    }
                } else {
                    const int cutoff_count = k - taken;
                    int selected_cutoff = 0;
                    int output_position = 0;
                    for (int a = 0; a < axis_size; a++) {
                        const int8_t raw = column[static_cast<size_t>(a) * inner];
                        const int v = static_cast<int>(raw) + 128;
                        const bool strictly_better = largest ? (v > cut) : (v < cut);
                        const bool take_cutoff = v == cut && selected_cutoff < cutoff_count;
                        if (strictly_better || take_cutoff) {
                            if (take_cutoff) {
                                selected_cutoff++;
                            }
                            const size_t dst = static_cast<size_t>((o * k + output_position++) * inner + i);
                            values_out[dst] = raw;
                            indices_out[dst] = static_cast<T2>(a);
                        }
                    }
                }
            }
        }
        return true;
    }

    template <typename T1, typename T2>
    bool forward_heap_select(
        T1 *values_out, T2 *indices_out, const T1 *input, int outer, int axis_size, int inner, int k)
    {
        const bool largest = (m_largest != 0);
        const bool sorted = (m_sorted != 0);
        using Elem = std::pair<T1, int>;
        constexpr bool use_stack_buf = sizeof(T1) <= sizeof(float);
        Elem stack_candidates[use_stack_buf ? HEAP_MAX_K_FLOOR : 1];
        Elem *heap = stack_candidates;
        if (!use_stack_buf || k > HEAP_MAX_K_FLOOR) {
            heap = ensure_scratch<Elem>(static_cast<size_t>(k));
            if (!heap) {
                return false;
            }
        }

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                const T1 *column = input + static_cast<size_t>(o) * axis_size * inner + i;
                heap_select_column(heap, column, axis_size, inner, k, largest, sorted);

                for (int t = 0; t < k; t++) {
                    const size_t dst = static_cast<size_t>((o * k + t) * inner + i);
                    values_out[dst] = heap[t].first;
                    indices_out[dst] = static_cast<T2>(heap[t].second);
                }
            }
        }
        return true;
    }

    template <typename T1, typename T2>
    bool forward_nth_select(
        T1 *values_out, T2 *indices_out, const T1 *input, int outer, int axis_size, int inner, int k)
    {
        const bool largest = (m_largest != 0);
        const bool sorted = (m_sorted != 0);
        using Elem = std::pair<T1, int>;
        Elem *buf = ensure_scratch<Elem>(static_cast<size_t>(axis_size));
        if (!buf) {
            return false;
        }

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                const T1 *column = input + static_cast<size_t>(o) * axis_size * inner + i;
                nth_select_column(buf, column, axis_size, inner, k, largest, sorted);

                for (int t = 0; t < k; t++) {
                    const size_t dst = static_cast<size_t>((o * k + t) * inner + i);
                    values_out[dst] = buf[t].first;
                    indices_out[dst] = static_cast<T2>(buf[t].second);
                }
            }
        }
        return true;
    }

    template <typename T1, typename T2>
    bool forward_template(T1 *values_out, T2 *indices_out, const T1 *input, int outer, int axis_size, int inner, int k)
    {
        if (k == 1) {
            forward_argbest(values_out, indices_out, input, outer, axis_size, inner);
            return true;
        }

        if constexpr (std::is_same_v<T1, int8_t>) {
            if (k >= std::max(COUNTING_MIN_K, axis_size / COUNTING_K_AXIS_DIVISOR)) {
                return forward_counting_int8(values_out, indices_out, input, outer, axis_size, inner, k);
            }
        }

        if (k <= std::max(HEAP_MAX_K_FLOOR, axis_size / HEAP_K_AXIS_DIVISOR)) {
            return forward_heap_select(values_out, indices_out, input, outer, axis_size, inner, k);
        }

        return forward_nth_select(values_out, indices_out, input, outer, axis_size, inner, k);
    }

    template <typename T>
    bool forward_with_index_dtype(
        T *values_out, TensorBase *indices, const T *input, int outer, int axis_size, int inner, int k)
    {
        if (indices->get_dtype() == DATA_TYPE_INT64) {
            return forward_template<T, int64_t>(
                values_out, static_cast<int64_t *>(indices->get_element_ptr()), input, outer, axis_size, inner, k);
        } else if (indices->get_dtype() == DATA_TYPE_INT32) {
            return forward_template<T, int32_t>(
                values_out, static_cast<int32_t *>(indices->get_element_ptr()), input, outer, axis_size, inner, k);
        } else if (indices->get_dtype() == DATA_TYPE_INT16) {
            return forward_template<T, int16_t>(
                values_out, static_cast<int16_t *>(indices->get_element_ptr()), input, outer, axis_size, inner, k);
        } else if (indices->get_dtype() == DATA_TYPE_INT8) {
            return forward_template<T, int8_t>(
                values_out, static_cast<int8_t *>(indices->get_element_ptr()), input, outer, axis_size, inner, k);
        }
        return false;
    }

    /**
     * @brief Forward pass for TopK.
     *
     * @param context Model context
     * @param mode    Runtime mode
     */
    void forward(ModelContext *context, runtime_mode_t /*mode*/ = RUNTIME_MODE_AUTO) override
    {
        if (!context) {
            ESP_LOGE("TopK", "null model context");
            return;
        }
        TensorBase *values = m_outputs_index.size() > 0 ? context->get_tensor(m_outputs_index[0]) : nullptr;
        TensorBase *indices = m_outputs_index.size() > 1 ? context->get_tensor(m_outputs_index[1]) : nullptr;
        if ((m_inputs_index.size() != 1 && m_inputs_index.size() != 2) || m_outputs_index.size() != 2) {
            ESP_LOGE("TopK",
                     "expected 1 data input, an optional serialized K input, and 2 outputs; got %d inputs/%d outputs",
                     static_cast<int>(m_inputs_index.size()),
                     static_cast<int>(m_outputs_index.size()));
            topk_zero_outputs(values, indices);
            return;
        }
        if (m_k <= 0 || (m_largest != 0 && m_largest != 1) || (m_sorted != 0 && m_sorted != 1)) {
            ESP_LOGE("TopK", "invalid attributes: k=%d, largest=%d, sorted=%d", m_k, m_largest, m_sorted);
            topk_zero_outputs(values, indices);
            return;
        }
        if (!is_supported_quant_type(quant_type)) {
            ESP_LOGE("TopK", "unsupported quant_type %d", static_cast<int>(quant_type));
            topk_zero_outputs(values, indices);
            return;
        }

        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        if (!input || !values || !indices) {
            ESP_LOGE("TopK", "missing input or output tensor");
            topk_zero_outputs(values, indices);
            return;
        }
        if (!input->get_element_ptr() || !values->get_element_ptr() || !indices->get_element_ptr()) {
            ESP_LOGE("TopK", "null tensor data pointer");
            topk_zero_outputs(values, indices);
            return;
        }
        if (values->get_element_ptr() == indices->get_element_ptr()) {
            ESP_LOGE("TopK", "Values and Indices output buffers must not alias");
            // Both views share one buffer, but their byte spans differ when the
            // dtypes differ (e.g. int8 Values vs int64 Indices), so zero both to
            // make sure no stale byte survives either view.
            topk_zero_tensor(values);
            topk_zero_tensor(indices);
            return;
        }
        if (values->get_element_ptr() == input->get_element_ptr() ||
            indices->get_element_ptr() == input->get_element_ptr()) {
            ESP_LOGE("TopK", "output buffers must not alias input X buffer");
            if (values->get_element_ptr() != input->get_element_ptr()) {
                topk_zero_tensor(values);
            }
            if (indices->get_element_ptr() != input->get_element_ptr()) {
                topk_zero_tensor(indices);
            }
            return;
        }

        if (!validate_dtypes(
                input->get_dtype(), values->get_dtype(), indices->get_dtype(), quant_type, /*allow_undefined=*/false)) {
            topk_zero_outputs(values, indices);
            return;
        }
        if (input->get_dtype() == DATA_TYPE_INT8 || input->get_dtype() == DATA_TYPE_INT16) {
            if (!input->exponent.is_valid() || !values->exponent.is_valid() || input->exponent.is_per_channel() ||
                values->exponent.is_per_channel() || input->exponent != values->exponent) {
                ESP_LOGE("TopK", "X and Values require matching per-tensor exponents");
                topk_zero_outputs(values, indices);
                return;
            }
        }

        const int k = m_k;
        const std::vector<int> &input_shape = input->get_shape();
        int axis = 0;
        int input_count = 0;
        int output_count = 0;
        if (!validate_shapes_and_index_range(input_shape,
                                             values->get_shape(),
                                             indices->get_shape(),
                                             indices->get_dtype(),
                                             m_axis,
                                             k,
                                             &axis,
                                             &input_count,
                                             &output_count) ||
            input->get_size() != input_count || values->get_size() != output_count ||
            indices->get_size() != output_count) {
            if (input->get_size() != input_count || values->get_size() != output_count ||
                indices->get_size() != output_count) {
                ESP_LOGE("TopK",
                         "tensor size mismatch X=%d(%d) Values=%d(%d) Indices=%d(%d)",
                         input->get_size(),
                         input_count,
                         values->get_size(),
                         output_count,
                         indices->get_size(),
                         output_count);
            }
            topk_zero_outputs(values, indices);
            return;
        }
        const int axis_size = input_shape[axis];

        int64_t outer_count = 1;
        int64_t inner_count = 1;
        for (int i = 0; i < axis; i++) {
            outer_count *= input_shape[i];
        }
        for (int i = axis + 1; i < static_cast<int>(input_shape.size()); i++) {
            inner_count *= input_shape[i];
        }
        const int outer = static_cast<int>(outer_count);
        const int inner = static_cast<int>(inner_count);

        bool ok = false;
        if (input->get_dtype() == DATA_TYPE_INT8) {
            ok = forward_with_index_dtype(static_cast<int8_t *>(values->get_element_ptr()),
                                          indices,
                                          static_cast<int8_t *>(input->get_element_ptr()),
                                          outer,
                                          axis_size,
                                          inner,
                                          k);
        } else if (input->get_dtype() == DATA_TYPE_INT16) {
            ok = forward_with_index_dtype(static_cast<int16_t *>(values->get_element_ptr()),
                                          indices,
                                          static_cast<int16_t *>(input->get_element_ptr()),
                                          outer,
                                          axis_size,
                                          inner,
                                          k);
        } else if (input->get_dtype() == DATA_TYPE_FLOAT) {
            ok = forward_with_index_dtype(static_cast<float *>(values->get_element_ptr()),
                                          indices,
                                          static_cast<float *>(input->get_element_ptr()),
                                          outer,
                                          axis_size,
                                          inner,
                                          k);
        } else if (input->get_dtype() == DATA_TYPE_DOUBLE) {
            ok = forward_with_index_dtype(static_cast<double *>(values->get_element_ptr()),
                                          indices,
                                          static_cast<double *>(input->get_element_ptr()),
                                          outer,
                                          axis_size,
                                          inner,
                                          k);
        }
        if (!ok) {
            topk_zero_outputs(values, indices);
        }
    }

    /**
     * @brief deserialize TopK module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        if (!fbs_model) {
            ESP_LOGE("TopK", "%s: null FbsModel", node_name.c_str());
            return nullptr;
        }

        int axis = -1;
        int largest = 1;
        int sorted = 1;
        quant_type_t quant_type = QUANT_TYPE_NONE;
        fbs_model->get_operation_attribute(node_name, "axis", axis);
        fbs_model->get_operation_attribute(node_name, "largest", largest);
        fbs_model->get_operation_attribute(node_name, "sorted", sorted);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        if (largest != 0 && largest != 1) {
            ESP_LOGE("TopK", "%s: largest must be 0 or 1, got %d", node_name.c_str(), largest);
            return nullptr;
        }
        if (sorted != 0 && sorted != 1) {
            ESP_LOGE("TopK", "%s: sorted must be 0 or 1, got %d", node_name.c_str(), sorted);
            return nullptr;
        }
        if (!is_supported_quant_type(quant_type)) {
            ESP_LOGE("TopK", "%s: unsupported quant_type %d", node_name.c_str(), static_cast<int>(quant_type));
            return nullptr;
        }

        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        if (fbs_model->get_operation_inputs_and_outputs(node_name, inputs, outputs) != ESP_OK || inputs.size() != 2 ||
            outputs.size() != 2) {
            ESP_LOGE("TopK",
                     "%s: expected exactly 2 inputs and 2 outputs, got %d/%d",
                     node_name.c_str(),
                     static_cast<int>(inputs.size()),
                     static_cast<int>(outputs.size()));
            return nullptr;
        }

        TensorBase *k_tensor = fbs_model->get_operation_parameter(node_name, 1);
        if (!k_tensor) {
            ESP_LOGE("TopK", "%s: K must be a constant parameter, not a runtime input", node_name.c_str());
            return nullptr;
        }

        int k = 0;
        if (!topk_parse_k(k_tensor, &k)) {
            delete k_tensor;
            return nullptr;
        }
        delete k_tensor;

        const dtype_t input_dtype = recorded_tensor_dtype(fbs_model, inputs[0]);
        const dtype_t values_dtype = recorded_tensor_dtype(fbs_model, outputs[0]);
        const dtype_t indices_dtype = recorded_tensor_dtype(fbs_model, outputs[1]);
        if (!validate_dtypes(input_dtype, values_dtype, indices_dtype, quant_type, /*allow_undefined=*/true)) {
            return nullptr;
        }
        if (input_dtype == DATA_TYPE_INT8 || input_dtype == DATA_TYPE_INT16) {
            int input_exponent = 0;
            int values_exponent = 0;
            if (!recorded_per_tensor_exponent(fbs_model, inputs[0], &input_exponent) ||
                !recorded_per_tensor_exponent(fbs_model, outputs[0], &values_exponent)) {
                ESP_LOGE("TopK", "%s: X and Values require per-tensor exponents", node_name.c_str());
                return nullptr;
            }
            if (input_exponent != values_exponent) {
                ESP_LOGE("TopK", "%s: Values exponent must match X exponent", node_name.c_str());
                return nullptr;
            }
        }

        std::vector<int> input_shape;
        std::vector<int> values_shape;
        std::vector<int> indices_shape;
        if (fbs_model->get_operation_input_shape(node_name, 0, input_shape) != ESP_OK ||
            fbs_model->get_operation_output_shape(node_name, 0, values_shape) != ESP_OK ||
            fbs_model->get_operation_output_shape(node_name, 1, indices_shape) != ESP_OK) {
            ESP_LOGE("TopK", "%s: missing input or output shape", node_name.c_str());
            return nullptr;
        }

        int normalized_axis = 0;
        int input_count = 0;
        int output_count = 0;
        if (!validate_shapes_and_index_range(input_shape,
                                             values_shape,
                                             indices_shape,
                                             indices_dtype,
                                             axis,
                                             k,
                                             &normalized_axis,
                                             &input_count,
                                             &output_count)) {
            return nullptr;
        }

        return new TopK(k, axis, largest, sorted, node_name.c_str(), MODULE_NON_INPLACE, quant_type);
    }

    /**
     * @brief Print module information.
     */
    void print() override
    {
        ESP_LOGI("TopK",
                 "quant_type: %s, axis: %d, largest: %d, sorted: %d, k: %d.",
                 quant_type_to_string(quant_type),
                 m_axis,
                 m_largest,
                 m_sorted,
                 m_k);
    }
};

} // namespace module
} // namespace dl
