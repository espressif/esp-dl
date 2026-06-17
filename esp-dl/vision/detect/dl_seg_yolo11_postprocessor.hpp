#pragma once
#include "dl_detect_postprocessor.hpp"

namespace dl {
namespace detect {
class yolo11segPostProcessor : public AnchorPointDetectPostprocessor {
private:
    static constexpr int m_nm = 32;

    template <typename T>
    void parse_stage(TensorBase *score, TensorBase *box, TensorBase *mc, const int stage_index);

    template <typename T>
    void synthesize_masks(TensorBase *proto);

public:
    void postprocess() override;
    using AnchorPointDetectPostprocessor::AnchorPointDetectPostprocessor;
};

} // namespace detect
} // namespace dl
