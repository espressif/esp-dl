#pragma once
#include <cstdint>

namespace dl {
namespace feat {

constexpr int FEAT_LABEL_MAX = 32;

// Database metadata
struct FeatDatabaseMeta {
    uint16_t total_embeddings;
    uint16_t valid_embeddings;
    uint16_t embedding_dim;
};

// One enrollment record
struct FeatEmbedding {
    uint16_t id;
    uint16_t identity_id;
    char label[FEAT_LABEL_MAX];
    float *embedding;
};

// Verification result
struct VerificationResult {
    float score;
    bool is_match;
    uint16_t identity_id;
    char label[FEAT_LABEL_MAX];
};

} // namespace feat
} // namespace dl
