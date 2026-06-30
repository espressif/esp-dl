#include "dl_feat_verification_database.hpp"
#include "dl_feat_verification_define.hpp"
#include "esp_log.h"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static const char *TAG = "dl::feat::VerificationDB";

namespace dl {
namespace feat {

FeatVerificationDatabase::FeatVerificationDatabase(const std::string &db_path, int embedding_dim) :
    m_db_path(db_path), m_next_identity_id(1), m_init_error(ESP_OK)
{
    m_meta.embedding_dim = embedding_dim;
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
    struct stat st;
    if (stat(db_path.c_str(), &st) == 0) {
        ESP_LOGI(TAG, "Loading database from %s.", db_path.c_str());
        m_init_error = load_database_from_storage(embedding_dim);
    } else {
        ESP_LOGW(TAG, "Database file %s not found, creating new database in Flash.", db_path.c_str());
        m_init_error = create_empty_database_in_storage(embedding_dim);
    }
    if (m_init_error != ESP_OK) {
        m_identities.clear();
        m_meta.embedding_dim = embedding_dim;
        m_meta.total_embeddings = 0;
        m_meta.valid_embeddings = 0;
        m_next_identity_id = 1;
        ESP_LOGE(TAG, "Database initialization failed for %s (err=0x%x).", db_path.c_str(), m_init_error);
    }
}

FeatVerificationDatabase::~FeatVerificationDatabase()
{
    m_identities.clear();
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
}

esp_err_t FeatVerificationDatabase::clear()
{
    if (!is_valid()) {
        return m_init_error;
    }
    for (auto &ident : m_identities) {
        ident.subspace.clear();
    }
    m_identities.clear();
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
    m_next_identity_id = 1;
    ESP_LOGI(TAG, "Database cleared.");
    return create_empty_database_in_storage(m_meta.embedding_dim);
}

esp_err_t FeatVerificationDatabase::create_empty_database_in_storage(int embedding_dim)
{
    FILE *f = fopen(m_db_path.c_str(), "wb");
    size_t size = 0;
    if (!f) {
        ESP_LOGE(TAG, "Failed to open database.");
        return ESP_FAIL;
    }
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
    m_meta.embedding_dim = embedding_dim;
    size = fwrite(&m_meta, sizeof(FeatDatabaseMeta), 1, f);
    if (size != 1) {
        ESP_LOGE(TAG, "Failed to write database meta.");
        fclose(f);
        return ESP_FAIL;
    }
    fclose(f);
    ESP_LOGI(TAG, "Created empty database in storage.");
    return ESP_OK;
}

esp_err_t FeatVerificationDatabase::load_database_from_storage(int embedding_dim)
{
    FILE *f = fopen(m_db_path.c_str(), "rb");
    size_t size = 0;
    if (!f) {
        ESP_LOGE(TAG, "Failed to open database.");
        return ESP_FAIL;
    }

    size = fread(&m_meta, sizeof(FeatDatabaseMeta), 1, f);
    if (size != 1) {
        ESP_LOGE(TAG, "Failed to read database meta.");
        fclose(f);
        return ESP_FAIL;
    }

    if (embedding_dim != m_meta.embedding_dim) {
        ESP_LOGE(TAG, "Embedding dimension mismatch with database.");
        fclose(f);
        return ESP_FAIL;
    }

    // Read identities
    while (true) {
        uint16_t identity_id, num_embeds;
        char label[FEAT_LABEL_MAX];

        // Read identity header
        size = fread(&identity_id, sizeof(uint16_t), 1, f);
        if (size != 1)
            break; // End of file

        size = fread(&num_embeds, sizeof(uint16_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read num embeddings.");
            fclose(f);
            return ESP_FAIL;
        }

        size = fread(label, sizeof(char), FEAT_LABEL_MAX, f);
        if (size != FEAT_LABEL_MAX) {
            ESP_LOGE(TAG, "Failed to read identity label.");
            fclose(f);
            return ESP_FAIL;
        }
        // Force null-termination so a corrupted (non-terminated) label can't cause
        // an out-of-bounds read when constructing the std::string below.
        label[FEAT_LABEL_MAX - 1] = '\0';

        // Read subspace flag
        uint8_t subspace_ready;
        size = fread(&subspace_ready, sizeof(uint8_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read subspace flag.");
            fclose(f);
            return ESP_FAIL;
        }

        // Create identity bucket
        identity_bucket_t bucket;
        bucket.identity_id = identity_id;
        bucket.label = label;

        // Load subspace data
        int subspace_dim;
        size = fread(&subspace_dim, sizeof(int), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read subspace dim.");
            fclose(f);
            return ESP_FAIL;
        }
        if (subspace_dim < 0 || subspace_dim > m_meta.embedding_dim) {
            ESP_LOGE(TAG, "Invalid subspace dim %d for embedding dim %d.", subspace_dim, m_meta.embedding_dim);
            fclose(f);
            return ESP_FAIL;
        }
        if (subspace_dim > 0 && m_meta.embedding_dim > INT_MAX / subspace_dim) {
            ESP_LOGE(TAG,
                     "Subspace basis size overflow: subspace_dim=%d embedding_dim=%d.",
                     subspace_dim,
                     m_meta.embedding_dim);
            fclose(f);
            return ESP_FAIL;
        }

        // Read mean
        std::vector<float> mean(m_meta.embedding_dim);
        size = fread(mean.data(), sizeof(float), m_meta.embedding_dim, f);
        if (size != m_meta.embedding_dim) {
            ESP_LOGE(TAG, "Failed to read subspace mean.");
            fclose(f);
            return ESP_FAIL;
        }

        // Read basis
        int basis_size = subspace_dim > 0 ? subspace_dim * m_meta.embedding_dim : m_meta.embedding_dim;
        std::vector<float> basis(basis_size);
        size = fread(basis.data(), sizeof(float), basis_size, f);
        if (size != basis_size) {
            ESP_LOGE(TAG, "Failed to read subspace basis.");
            fclose(f);
            return ESP_FAIL;
        }

        // Read variances (always read at least 1 value)
        int var_size = subspace_dim > 0 ? subspace_dim : 1;
        std::vector<float> variances(var_size);
        size = fread(variances.data(), sizeof(float), var_size, f);
        if (size != var_size) {
            ESP_LOGE(TAG, "Failed to read subspace variances.");
            fclose(f);
            return ESP_FAIL;
        }

        // Load subspace data
        bucket.subspace.load(
            subspace_ready, m_meta.embedding_dim, subspace_dim, mean.data(), basis.data(), variances.data());
        if (subspace_ready && subspace_dim > 0) {
            ESP_LOGD(TAG, "Loaded subspace for '%s' (dim=%d, ready=%d).", label, subspace_dim, subspace_ready);
        }

        // Read embeddings
        for (uint16_t i = 0; i < num_embeds; i++) {
            uint16_t id;
            size = fread(&id, sizeof(uint16_t), 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to read embedding id.");
                fclose(f);
                return ESP_FAIL;
            }

            std::vector<float> embedding(m_meta.embedding_dim);
            size = fread(embedding.data(), sizeof(float), m_meta.embedding_dim, f);
            if (size != m_meta.embedding_dim) {
                ESP_LOGE(TAG, "Failed to read embedding data.");
                fclose(f);
                return ESP_FAIL;
            }

            bucket.embeddings.push_back(std::move(embedding));
        }

        m_identities.push_back(std::move(bucket));
    }

    fclose(f);

    // Update valid embeddings count and recover the next assignable identity id.
    m_meta.valid_embeddings = 0;
    uint16_t max_id = 0;
    for (const auto &ident : m_identities) {
        m_meta.valid_embeddings += ident.embeddings.size();
        if (ident.identity_id > max_id) {
            max_id = ident.identity_id;
        }
    }
    // Recover the next assignable identity id. Identity ids start from 1;
    // m_next_identity_id == 0 indicates that the id space is exhausted.
    if (max_id == UINT16_MAX) {
        // No more identity ids can be allocated.
        m_next_identity_id = 0;
    } else {
        m_next_identity_id = static_cast<uint16_t>(max_id + 1);
    }

    ESP_LOGI(TAG, "Loaded %zu identities, %d embeddings.", m_identities.size(), m_meta.valid_embeddings);
    return ESP_OK;
}

esp_err_t FeatVerificationDatabase::save_database_to_storage()
{
    if (!is_valid()) {
        return m_init_error;
    }

    // Write to a temp file and rename only after a full successful write, so a
    // failed save never corrupts the existing database.
    const std::string tmp_path = m_db_path + ".tmp";
    FILE *f = fopen(tmp_path.c_str(), "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open temp database.");
        return ESP_FAIL;
    }

    uint32_t count = 0;

// Abort to the cleanup label on any short write.
#define WRITE_OR_FAIL(ptr, sz, cnt)                           \
    do {                                                      \
        if (fwrite((ptr), (sz), (cnt), f) != (size_t)(cnt)) { \
            goto write_fail;                                  \
        }                                                     \
    } while (0)

    // Write metadata
    WRITE_OR_FAIL(&m_meta, sizeof(FeatDatabaseMeta), 1);

    // Write all identities with embeddings and subspace
    for (const auto &ident : m_identities) {
        // Write identity header
        uint16_t num_embeds = ident.embeddings.size();
        WRITE_OR_FAIL(&ident.identity_id, sizeof(uint16_t), 1);
        WRITE_OR_FAIL(&num_embeds, sizeof(uint16_t), 1);
        char label_buf[FEAT_LABEL_MAX] = {};
        snprintf(label_buf, FEAT_LABEL_MAX, "%s", ident.label.c_str());
        WRITE_OR_FAIL(label_buf, sizeof(char), FEAT_LABEL_MAX);

        // Write subspace flag (1 = ready, 0 = not ready)
        uint8_t subspace_ready = ident.subspace.is_ready() ? 1 : 0;
        WRITE_OR_FAIL(&subspace_ready, sizeof(uint8_t), 1);

        // Always write subspace data (use defaults if not ready)
        int subspace_dim = subspace_ready ? ident.subspace.get_dimension() : 0;
        WRITE_OR_FAIL(&subspace_dim, sizeof(int), 1);

        // Write mean (or zeros if not ready)
        if (subspace_ready) {
            WRITE_OR_FAIL(ident.subspace.get_mean(), sizeof(float), m_meta.embedding_dim);
        } else {
            std::vector<float> zero_mean(m_meta.embedding_dim, 0.0f);
            WRITE_OR_FAIL(zero_mean.data(), sizeof(float), m_meta.embedding_dim);
        }

        // Write basis (or zeros if not ready)
        int basis_size = subspace_dim * m_meta.embedding_dim;
        if (subspace_ready && basis_size > 0) {
            WRITE_OR_FAIL(ident.subspace.get_basis(), sizeof(float), basis_size);
        } else {
            std::vector<float> zero_basis(basis_size > 0 ? basis_size : m_meta.embedding_dim, 0.0f);
            WRITE_OR_FAIL(zero_basis.data(), sizeof(float), zero_basis.size());
        }

        // Write variances (or zeros if not ready)
        if (subspace_ready && subspace_dim > 0) {
            WRITE_OR_FAIL(ident.subspace.get_variances(), sizeof(float), subspace_dim);
        } else {
            float zero_var = 0.0f;
            WRITE_OR_FAIL(&zero_var, sizeof(float), 1);
        }

        // Write embeddings
        for (const auto &embed : ident.embeddings) {
            // Embedding ids are stored as uint16_t; guard against truncation.
            if (count >= UINT16_MAX) {
                ESP_LOGE(TAG, "Too many embeddings to store (id overflows uint16_t).");
                goto write_fail;
            }
            uint16_t id = ++count;
            WRITE_OR_FAIL(&id, sizeof(uint16_t), 1);
            WRITE_OR_FAIL(embed.data(), sizeof(float), m_meta.embedding_dim);
        }
    }

#undef WRITE_OR_FAIL

    // Call both unconditionally so a failed fflush never leaks the FILE handle.
    {
        int flush_ret = fflush(f);
        int close_ret = fclose(f);
        if (flush_ret != 0 || close_ret != 0) {
            ESP_LOGE(TAG, "Failed to finalize temp database.");
            remove(tmp_path.c_str());
            return ESP_FAIL;
        }
    }

    // SPIFFS (and some other embedded filesystems) cannot rename onto an existing
    // file -- the rename fails with EIO. Remove the previous database first; the
    // temp file already holds the complete new database. The only unsafe window is
    // a power loss between this remove and the rename (rare; on failure the .tmp
    // file is left behind for recovery).
    remove(m_db_path.c_str());
    if (rename(tmp_path.c_str(), m_db_path.c_str()) != 0) {
        ESP_LOGE(TAG, "Failed to commit database file (errno=%d: %s).", errno, strerror(errno));
        remove(tmp_path.c_str());
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Saved %d embeddings to storage.", count);
    return ESP_OK;

write_fail:
    ESP_LOGE(TAG, "Failed to write database content.");
    fclose(f);
    remove(tmp_path.c_str());
    return ESP_FAIL;
}

FeatVerificationDatabase::identity_bucket_t *FeatVerificationDatabase::find_identity(const std::string &label)
{
    for (auto &ident : m_identities) {
        if (ident.label == label) {
            return &ident;
        }
    }
    return nullptr;
}

FeatVerificationDatabase::identity_bucket_t *FeatVerificationDatabase::get_or_create_identity(const std::string &label)
{
    auto *ident = find_identity(label);
    if (ident) {
        return ident;
    }
    // m_next_identity_id == 0 means the 1..UINT16_MAX id space is exhausted.
    // Assigning here would wrap and collide with an existing identity id.
    if (m_next_identity_id == 0) {
        ESP_LOGE(TAG, "Identity ID space exhausted; cannot create new identity '%s'.", label.c_str());
        return nullptr;
    }
    identity_bucket_t bucket;
    bucket.identity_id = m_next_identity_id++; // wraps to 0 (exhausted) after assigning UINT16_MAX
    bucket.label = label;
    m_identities.push_back(bucket);
    return &m_identities.back();
}

std::vector<std::string> FeatVerificationDatabase::get_labels() const
{
    std::vector<std::string> labels;
    for (const auto &ident : m_identities) {
        labels.push_back(ident.label);
    }
    return labels;
}

std::vector<std::vector<float>> FeatVerificationDatabase::get_embeddings(const std::string &label)
{
    auto *ident = find_identity(label);
    if (ident) {
        return ident->embeddings;
    }
    return {};
}

esp_err_t FeatVerificationDatabase::enroll(const std::string &label, const float *embedding)
{
    if (!is_valid()) {
        return m_init_error;
    }
    if (!embedding) {
        ESP_LOGE(TAG, "Null embedding.");
        return ESP_FAIL;
    }

    // L2-normalize embedding
    float norm = 0.0f;
    for (int i = 0; i < m_meta.embedding_dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f) {
        ESP_LOGE(TAG, "Embedding has zero norm.");
        return ESP_FAIL;
    }

    std::vector<float> normalized_embedding(m_meta.embedding_dim);
    for (int i = 0; i < m_meta.embedding_dim; i++) {
        normalized_embedding[i] = embedding[i] / norm;
    }

    // Get or create identity bucket
    auto *ident = get_or_create_identity(label);
    if (!ident) {
        ESP_LOGE(TAG, "Failed to enroll '%s': identity ID space exhausted.", label.c_str());
        return ESP_ERR_NO_MEM;
    }
    ident->embeddings.push_back(std::move(normalized_embedding));
    ident->subspace.set_ready(false);

    // Update metadata
    m_meta.total_embeddings++;
    m_meta.valid_embeddings++;

    ESP_LOGI(TAG, "Enrolled embedding for identity '%s'.", label.c_str());
    return ESP_OK;
}

esp_err_t FeatVerificationDatabase::delete_embedding(const std::string &label, int idx)
{
    if (!is_valid()) {
        return m_init_error;
    }
    auto *ident = find_identity(label);
    if (!ident) {
        ESP_LOGE(TAG, "Identity '%s' not found.", label.c_str());
        return ESP_ERR_NOT_FOUND;
    }

    if (idx < 0 || idx >= static_cast<int>(ident->embeddings.size())) {
        ESP_LOGE(TAG,
                 "Invalid index %d for identity '%s' (have %zu embeddings).",
                 idx,
                 label.c_str(),
                 ident->embeddings.size());
        return ESP_ERR_INVALID_ARG;
    }

    // Remove embedding and update metadata
    ident->embeddings.erase(ident->embeddings.begin() + idx);
    ident->subspace.set_ready(false);
    m_meta.total_embeddings--;
    m_meta.valid_embeddings = 0;
    for (const auto &s : m_identities) {
        m_meta.valid_embeddings += s.embeddings.size();
    }

    ESP_LOGI(TAG,
             "Deleted embedding %d from identity '%s'. Remaining: %zu embeddings.",
             idx,
             label.c_str(),
             ident->embeddings.size());

    // Automatically build and save to Flash after deletion
    return build();
}

esp_err_t FeatVerificationDatabase::build_per_identity(const std::string &label, int min_samples)
{
    if (!is_valid()) {
        return m_init_error;
    }
    min_samples = std::max(min_samples, 2);
    auto embeddings = get_embeddings(label);
    if (static_cast<int>(embeddings.size()) < min_samples) {
        ESP_LOGW(TAG,
                 "'%s' has only %zu sample(s), default to cosine similarity for verification.",
                 label.c_str(),
                 embeddings.size());
        return ESP_ERR_INVALID_STATE;
    }

    auto *ident = find_identity(label);
    if (ident == nullptr) {
        ESP_LOGE(TAG, "Identity '%s' not found during build.", label.c_str());
        return ESP_ERR_NOT_FOUND;
    }
    if (ident->subspace.is_ready()) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Building subspace for '%s' with %zu samples...", label.c_str(), embeddings.size());
    std::vector<const float *> enroll_ptrs;
    for (const auto &embed : embeddings) {
        enroll_ptrs.push_back(embed.data());
    }
    ident->subspace.build(enroll_ptrs, m_meta.embedding_dim);
    ESP_LOGI(TAG, "Subspace is ready for '%s'.", label.c_str());

    return ESP_OK;
}

esp_err_t FeatVerificationDatabase::build(int min_samples)
{
    if (!is_valid()) {
        return m_init_error;
    }
    // A meaningful subspace needs >= 2 samples.
    min_samples = std::max(min_samples, 2);
    auto labels = get_labels();
    if (labels.empty()) {
        ESP_LOGW(TAG, "No enrolled identities found.");
        return ESP_ERR_NOT_FOUND;
    }
    for (const auto &label : labels) {
        build_per_identity(label, min_samples);
    }
    return save_database_to_storage();
}

esp_err_t FeatVerificationDatabase::verify_max_cosine(const float *embedding, float threshold)
{
    if (!is_valid()) {
        return m_init_error;
    }
    if (!embedding || m_identities.empty()) {
        ESP_LOGW(TAG, "Invalid target embedding or no enrolled identities found.");
        return ESP_ERR_NOT_FOUND;
    }

    // Compute target embedding norm
    float norm = 0.0f;
    for (int i = 0; i < m_meta.embedding_dim; ++i) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f) {
        ESP_LOGW(TAG, "Target embedding has zero norm.");
        return ESP_ERR_INVALID_STATE;
    }
    float inv_norm = 1.0f / norm;

    for (const auto &ident : m_identities) {
        if (ident.embeddings.empty()) {
            continue;
        }

        // Find max cosine similarity
        float score = -1.0f;
        for (const auto &en : ident.embeddings) {
            float sim = 0.0f;
            for (int i = 0; i < m_meta.embedding_dim; ++i) {
                sim += (embedding[i] * inv_norm) * en[i];
            }
            if (sim > score)
                score = sim;
        }

        bool is_match = (score > threshold);
        ESP_LOGI(TAG,
                 "[MAX-COSINE] %s: score=%.4f (threshold=%.4f) -> %s",
                 ident.label.c_str(),
                 score,
                 threshold,
                 is_match ? "MATCH" : "NO MATCH");
    }

    return ESP_OK;
}

esp_err_t FeatVerificationDatabase::verify_subspace(const float *embedding, float threshold)
{
    if (!is_valid()) {
        return m_init_error;
    }
    if (!embedding || m_identities.empty()) {
        ESP_LOGW(TAG, "Invalid target embedding or no enrolled identities found.");
        return ESP_ERR_NOT_FOUND;
    }

    // Compute target embedding norm
    float norm = 0.0f;
    for (int i = 0; i < m_meta.embedding_dim; ++i) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f) {
        ESP_LOGW(TAG, "Target embedding has zero norm.");
        return ESP_ERR_INVALID_STATE;
    }
    const float inv_norm = 1.0f / norm;

    for (const auto &ident : m_identities) {
        if (!ident.subspace.is_ready() || ident.embeddings.size() < 2) {
            continue;
        }
        float score = ident.subspace.compute_neg_distance(embedding, inv_norm);
        bool is_match = (score > threshold);
        ESP_LOGI(TAG,
                 "[SUBSPACE] %s: score=%.4f (threshold=%.4f) -> %s",
                 ident.label.c_str(),
                 score,
                 threshold,
                 is_match ? "MATCH" : "NO MATCH");
    }

    return ESP_OK;
}

void FeatVerificationDatabase::print()
{
    ESP_LOGI(TAG, "============ Database Summary ============");
    ESP_LOGI(TAG, "%-5s | %-20s | %-10s", "ID", "Label", "#Embeddings");
    ESP_LOGI(TAG, "------------------------------------------");
    for (const auto &ident : m_identities) {
        ESP_LOGI(TAG, "%-5d | %-20s | %-10zu", ident.identity_id, ident.label.c_str(), ident.embeddings.size());
    }
    ESP_LOGI(TAG, "==========================================");
}

} // namespace feat
} // namespace dl
