#pragma once
#include "dl_feat_verification_define.hpp"
#include "dl_feat_verification_engine.hpp"
#include "esp_check.h"
#include "esp_system.h"
#include <algorithm>
#include <list>
#include <string>
#include <vector>

namespace dl {
namespace feat {

/**
 * @brief Feature Verification Database with multi-identity support
 *
 * Stores L2-normalized embeddings grouped by identity label and verifies a query
 * embedding with max cosine similarity or per-identity subspace distance. Model
 * agnostic: any fixed-dimension embedding (speaker, face, keyword, ...).
 */
class FeatVerificationDatabase {
public:
    FeatVerificationDatabase(const std::string &db_path, int embedding_dim);
    ~FeatVerificationDatabase();
    bool is_valid() const { return m_init_error == ESP_OK; }
    esp_err_t get_init_error() const { return m_init_error; }
    esp_err_t clear();

    esp_err_t enroll(const std::string &label, const float *embedding);
    esp_err_t delete_embedding(const std::string &label, int idx);
    esp_err_t build(int min_samples = 2);
    esp_err_t verify_max_cosine(const float *embedding, float threshold);
    esp_err_t verify_subspace(const float *embedding, float threshold);

    void print();

private:
    // Data structures
    struct identity_bucket_t {
        uint16_t identity_id;
        std::string label;
        std::vector<std::vector<float>> embeddings;
        FeatSubspace subspace;
    };

    std::string m_db_path;
    FeatDatabaseMeta m_meta;
    std::vector<identity_bucket_t> m_identities;
    uint16_t m_next_identity_id;
    esp_err_t m_init_error;

    // Storage operations
    esp_err_t create_empty_database_in_storage(int embedding_dim);
    esp_err_t load_database_from_storage(int embedding_dim);
    esp_err_t save_database_to_storage();

    // Identity management
    identity_bucket_t *find_identity(const std::string &label);
    identity_bucket_t *get_or_create_identity(const std::string &label);
    std::vector<std::string> get_labels() const;
    std::vector<std::vector<float>> get_embeddings(const std::string &label);
    esp_err_t build_per_identity(const std::string &label, int min_samples = 2);
};

} // namespace feat
} // namespace dl
