#pragma once
#include <cstdint>

namespace dl {
namespace audio {

constexpr int AUDIO_SPK_NAME_MAX = 32;

// Database metadata
struct AudioDatabaseMeta {
    uint16_t total_embeddings;
    uint16_t valid_embeddings;
    uint16_t embedding_dim;
};

// One enrollment record
struct SpeakerEmbedding {
    uint16_t id;
    uint16_t speaker_id;
    char speaker_name[AUDIO_SPK_NAME_MAX];
    float *embedding;
};

// Verification result
struct VerificationResult {
    float score;
    bool is_same_speaker;
    uint16_t speaker_id;
    char speaker_name[AUDIO_SPK_NAME_MAX];
};

} // namespace audio
} // namespace dl
