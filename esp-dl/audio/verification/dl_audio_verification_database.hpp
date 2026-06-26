#pragma once
#include "dl_audio_verification_define.hpp"
#include "dl_audio_verification_engine.hpp"
#include "dl_tensor_base.hpp"
#include "esp_check.h"
#include "esp_system.h"
#include <algorithm>
#include <list>
#include <string>
#include <vector>

namespace dl {
namespace audio {

/**
 * @brief Audio Verification Database with Multi-Speaker Support
 */
class AudioVerificationDatabase {
public:
    AudioVerificationDatabase(const std::string &db_path, int embedding_dim);
    ~AudioVerificationDatabase();
    bool is_valid() const { return m_init_error == ESP_OK; }
    esp_err_t get_init_error() const { return m_init_error; }
    esp_err_t clear();

    esp_err_t enroll(const std::string &speaker_name, const float *embedding);
    esp_err_t delete_embedding(const std::string &speaker_name, int idx);
    esp_err_t build(int min_samples = 2);
    esp_err_t verify_max_cosine(const float *embedding, float threshold);
    esp_err_t verify_subspace(const float *embedding, float threshold);

    void print();

private:
    // Data structures
    struct speaker_bucket_t {
        uint16_t speaker_id;
        std::string speaker_name;
        std::vector<std::vector<float>> embeddings;
        SpeakerSubspace subspace;
    };

    std::string m_db_path;
    AudioDatabaseMeta m_meta;
    std::vector<speaker_bucket_t> m_speakers;
    uint16_t m_next_speaker_id;
    esp_err_t m_init_error;

    // Storage operations
    esp_err_t create_empty_database_in_storage(int embedding_dim);
    esp_err_t load_database_from_storage(int embedding_dim);
    esp_err_t save_database_to_storage();

    // Speaker management
    speaker_bucket_t *find_speaker(const std::string &name);
    speaker_bucket_t *get_or_create_speaker(const std::string &name);
    std::vector<std::string> get_speaker_names() const;
    std::vector<std::vector<float>> get_speaker_embeddings(const std::string &speaker_name);
    esp_err_t build_per_speaker(const std::string &speaker_name, int min_samples = 2);
};

} // namespace audio
} // namespace dl
