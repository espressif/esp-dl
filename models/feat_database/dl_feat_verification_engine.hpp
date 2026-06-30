#pragma once
#include "dl_feat_verification_define.hpp"
#include "esp_check.h"
#include "esp_system.h"
#include <vector>

namespace dl {
namespace feat {

/**
 * @brief Feature subspace for embedding verification
 *
 * Computes and stores a per-identity subspace from enrollment embeddings.
 * Supports verification using subspace (residual) distance.
 */
class FeatSubspace {
public:
    FeatSubspace();
    ~FeatSubspace();
    void clear();

    // Build the subspace from enrollment embeddings
    void build(const std::vector<const float *> &enroll_embeds, int embedding_dim);

    // Compute negative distance to the subspace (higher score = better match)
    // Note: target_embed should be raw (not normalized), inv_norm = 1/||target_embed||
    float compute_neg_distance(const float *target_embed, float inv_norm) const;

    // Check if subspace data is ready
    bool is_ready() const { return m_is_ready; }
    void set_ready(bool ready) { m_is_ready = ready; }

    // Getters
    int get_dimension() const { return m_num_embeddings; }
    const float *get_mean() const { return m_mean.data(); }
    const float *get_basis() const { return m_basis.data(); }
    const float *get_variances() const { return m_variances.data(); }

    // Load subspace from storage
    void load(bool ready,
              int embedding_dim,
              int num_embeddings,
              const float *mean,
              const float *basis,
              const float *variances);

private:
    bool m_is_ready;
    int m_embedding_dim;
    int m_num_embeddings;
    std::vector<float> m_mean;
    std::vector<float> m_basis;
    std::vector<float> m_variances;
};

} // namespace feat
} // namespace dl
