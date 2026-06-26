#include "dl_audio_verification_database.hpp"
#include "dl_audio_verification_define.hpp"
#include "esp_log.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static const char *TAG = "dl::audio::VerificationDB";

namespace dl {
namespace audio {

AudioVerificationDatabase::AudioVerificationDatabase(const std::string &db_path, int embedding_dim) :
    m_db_path(db_path), m_next_speaker_id(1), m_init_error(ESP_OK)
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
        m_speakers.clear();
        m_meta.embedding_dim = embedding_dim;
        m_meta.total_embeddings = 0;
        m_meta.valid_embeddings = 0;
        m_next_speaker_id = 1;
        ESP_LOGE(TAG, "Database initialization failed for %s (err=0x%x).", db_path.c_str(), m_init_error);
    }
}

AudioVerificationDatabase::~AudioVerificationDatabase()
{
    m_speakers.clear();
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
}

esp_err_t AudioVerificationDatabase::clear()
{
    if (!is_valid()) {
        return m_init_error;
    }
    for (auto &spk : m_speakers) {
        spk.subspace.clear();
    }
    m_speakers.clear();
    m_meta.total_embeddings = 0;
    m_meta.valid_embeddings = 0;
    m_next_speaker_id = 1;
    ESP_LOGI(TAG, "Database cleared.");
    return create_empty_database_in_storage(m_meta.embedding_dim);
}

esp_err_t AudioVerificationDatabase::create_empty_database_in_storage(int embedding_dim)
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
    size = fwrite(&m_meta, sizeof(AudioDatabaseMeta), 1, f);
    if (size != 1) {
        ESP_LOGE(TAG, "Failed to write database meta.");
        fclose(f);
        return ESP_FAIL;
    }
    fclose(f);
    ESP_LOGI(TAG, "Created empty database in storage.");
    return ESP_OK;
}

esp_err_t AudioVerificationDatabase::load_database_from_storage(int embedding_dim)
{
    FILE *f = fopen(m_db_path.c_str(), "rb");
    size_t size = 0;
    if (!f) {
        ESP_LOGE(TAG, "Failed to open database.");
        return ESP_FAIL;
    }

    size = fread(&m_meta, sizeof(AudioDatabaseMeta), 1, f);
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

    // Read speakers
    while (true) {
        uint16_t speaker_id, num_embeds;
        char speaker_name[AUDIO_SPK_NAME_MAX];

        // Read speaker header
        size = fread(&speaker_id, sizeof(uint16_t), 1, f);
        if (size != 1)
            break; // End of file

        size = fread(&num_embeds, sizeof(uint16_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read num embeddings.");
            fclose(f);
            return ESP_FAIL;
        }

        size = fread(speaker_name, sizeof(char), AUDIO_SPK_NAME_MAX, f);
        if (size != AUDIO_SPK_NAME_MAX) {
            ESP_LOGE(TAG, "Failed to read speaker name.");
            fclose(f);
            return ESP_FAIL;
        }
        // Force null-termination so a corrupted (non-terminated) name can't cause
        // an out-of-bounds read when constructing the std::string below.
        speaker_name[AUDIO_SPK_NAME_MAX - 1] = '\0';

        // Read subspace flag
        uint8_t subspace_ready;
        size = fread(&subspace_ready, sizeof(uint8_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read subspace flag.");
            fclose(f);
            return ESP_FAIL;
        }

        // Create speaker bucket
        speaker_bucket_t bucket;
        bucket.speaker_id = speaker_id;
        bucket.speaker_name = speaker_name;

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
            ESP_LOGD(TAG, "Loaded subspace for '%s' (dim=%d, ready=%d).", speaker_name, subspace_dim, subspace_ready);
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

        m_speakers.push_back(std::move(bucket));

        if (speaker_id >= m_next_speaker_id) {
            m_next_speaker_id = speaker_id + 1;
        }
    }

    fclose(f);

    // Update valid embeddings count
    m_meta.valid_embeddings = 0;
    for (const auto &spk : m_speakers) {
        m_meta.valid_embeddings += spk.embeddings.size();
    }

    ESP_LOGI(TAG, "Loaded %zu speakers, %d embeddings.", m_speakers.size(), m_meta.valid_embeddings);
    return ESP_OK;
}

esp_err_t AudioVerificationDatabase::save_database_to_storage()
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
    WRITE_OR_FAIL(&m_meta, sizeof(AudioDatabaseMeta), 1);

    // Write all speakers with embeddings and subspace
    for (const auto &spk : m_speakers) {
        // Write speaker header
        uint16_t num_embeds = spk.embeddings.size();
        WRITE_OR_FAIL(&spk.speaker_id, sizeof(uint16_t), 1);
        WRITE_OR_FAIL(&num_embeds, sizeof(uint16_t), 1);
        char spk_name[AUDIO_SPK_NAME_MAX] = {};
        snprintf(spk_name, AUDIO_SPK_NAME_MAX, "%s", spk.speaker_name.c_str());
        WRITE_OR_FAIL(spk_name, sizeof(char), AUDIO_SPK_NAME_MAX);

        // Write subspace flag (1 = ready, 0 = not ready)
        uint8_t subspace_ready = spk.subspace.is_ready() ? 1 : 0;
        WRITE_OR_FAIL(&subspace_ready, sizeof(uint8_t), 1);

        // Always write subspace data (use defaults if not ready)
        int subspace_dim = subspace_ready ? spk.subspace.get_dimension() : 0;
        WRITE_OR_FAIL(&subspace_dim, sizeof(int), 1);

        // Write mean (or zeros if not ready)
        if (subspace_ready) {
            WRITE_OR_FAIL(spk.subspace.get_mean(), sizeof(float), m_meta.embedding_dim);
        } else {
            std::vector<float> zero_mean(m_meta.embedding_dim, 0.0f);
            WRITE_OR_FAIL(zero_mean.data(), sizeof(float), m_meta.embedding_dim);
        }

        // Write basis (or zeros if not ready)
        int basis_size = subspace_dim * m_meta.embedding_dim;
        if (subspace_ready && basis_size > 0) {
            WRITE_OR_FAIL(spk.subspace.get_basis(), sizeof(float), basis_size);
        } else {
            std::vector<float> zero_basis(basis_size > 0 ? basis_size : m_meta.embedding_dim, 0.0f);
            WRITE_OR_FAIL(zero_basis.data(), sizeof(float), zero_basis.size());
        }

        // Write variances (or zeros if not ready)
        if (subspace_ready && subspace_dim > 0) {
            WRITE_OR_FAIL(spk.subspace.get_variances(), sizeof(float), subspace_dim);
        } else {
            float zero_var = 0.0f;
            WRITE_OR_FAIL(&zero_var, sizeof(float), 1);
        }

        // Write embeddings
        for (const auto &embed : spk.embeddings) {
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

    if (rename(tmp_path.c_str(), m_db_path.c_str()) != 0) {
        ESP_LOGE(TAG, "Failed to commit database file.");
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

AudioVerificationDatabase::speaker_bucket_t *AudioVerificationDatabase::find_speaker(const std::string &name)
{
    for (auto &spk : m_speakers) {
        if (spk.speaker_name == name) {
            return &spk;
        }
    }
    return nullptr;
}

AudioVerificationDatabase::speaker_bucket_t *AudioVerificationDatabase::get_or_create_speaker(const std::string &name)
{
    auto *spk = find_speaker(name);
    if (spk) {
        return spk;
    }
    speaker_bucket_t bucket;
    bucket.speaker_id = m_next_speaker_id++;
    bucket.speaker_name = name;
    m_speakers.push_back(bucket);
    return &m_speakers.back();
}

std::vector<std::string> AudioVerificationDatabase::get_speaker_names() const
{
    std::vector<std::string> names;
    for (const auto &spk : m_speakers) {
        names.push_back(spk.speaker_name);
    }
    return names;
}

std::vector<std::vector<float>> AudioVerificationDatabase::get_speaker_embeddings(const std::string &speaker_name)
{
    auto *spk = find_speaker(speaker_name);
    if (spk) {
        return spk->embeddings;
    }
    return {};
}

esp_err_t AudioVerificationDatabase::enroll(const std::string &speaker_name, const float *embedding)
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

    // Get or create speaker bucket
    auto *spk = get_or_create_speaker(speaker_name);
    spk->embeddings.push_back(std::move(normalized_embedding));
    spk->subspace.set_ready(false);

    // Update metadata
    m_meta.total_embeddings++;
    m_meta.valid_embeddings++;

    ESP_LOGI(TAG, "Enrolled embedding for speaker '%s'.", speaker_name.c_str());
    return ESP_OK;
}

esp_err_t AudioVerificationDatabase::delete_embedding(const std::string &speaker_name, int idx)
{
    if (!is_valid()) {
        return m_init_error;
    }
    auto *spk = find_speaker(speaker_name);
    if (!spk) {
        ESP_LOGE(TAG, "Speaker '%s' not found.", speaker_name.c_str());
        return ESP_ERR_NOT_FOUND;
    }

    if (idx < 0 || idx >= static_cast<int>(spk->embeddings.size())) {
        ESP_LOGE(TAG,
                 "Invalid index %d for speaker '%s' (have %zu embeddings).",
                 idx,
                 speaker_name.c_str(),
                 spk->embeddings.size());
        return ESP_ERR_INVALID_ARG;
    }

    // Remove embedding and update metadata
    spk->embeddings.erase(spk->embeddings.begin() + idx);
    spk->subspace.set_ready(false);
    m_meta.total_embeddings--;
    m_meta.valid_embeddings = 0;
    for (const auto &s : m_speakers) {
        m_meta.valid_embeddings += s.embeddings.size();
    }

    ESP_LOGI(TAG,
             "Deleted embedding %d from speaker '%s'. Remaining: %zu embeddings.",
             idx,
             speaker_name.c_str(),
             spk->embeddings.size());

    // Automatically build and save to Flash after deletion
    return build();
}

esp_err_t AudioVerificationDatabase::build_per_speaker(const std::string &speaker_name, int min_samples)
{
    if (!is_valid()) {
        return m_init_error;
    }
    auto embeddings = get_speaker_embeddings(speaker_name);
    if (static_cast<int>(embeddings.size()) < min_samples) {
        ESP_LOGW(TAG,
                 "'%s' has only %zu sample(s), default to cosine similarity for verification.",
                 speaker_name.c_str(),
                 embeddings.size());
        return ESP_ERR_INVALID_STATE;
    }

    auto *spk = find_speaker(speaker_name);
    if (spk == nullptr) {
        ESP_LOGE(TAG, "Speaker '%s' not found during build.", speaker_name.c_str());
        return ESP_ERR_NOT_FOUND;
    }
    if (spk->subspace.is_ready()) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Building subspace for '%s' with %zu samples...", speaker_name.c_str(), embeddings.size());
    std::vector<const float *> enroll_ptrs;
    for (const auto &embed : embeddings) {
        enroll_ptrs.push_back(embed.data());
    }
    spk->subspace.build(enroll_ptrs, m_meta.embedding_dim);
    ESP_LOGI(TAG, "Subspace is ready for '%s'.", speaker_name.c_str());

    return ESP_OK;
}

esp_err_t AudioVerificationDatabase::build(int min_samples)
{
    if (!is_valid()) {
        return m_init_error;
    }
    auto speakers = get_speaker_names();
    if (speakers.empty()) {
        ESP_LOGW(TAG, "No enrolled speakers found.");
        return ESP_ERR_NOT_FOUND;
    }
    for (const auto &speaker : speakers) {
        build_per_speaker(speaker, min_samples);
    }
    return save_database_to_storage();
}

esp_err_t AudioVerificationDatabase::verify_max_cosine(const float *embedding, float threshold)
{
    if (!is_valid()) {
        return m_init_error;
    }
    if (!embedding || m_speakers.empty()) {
        ESP_LOGW(TAG, "Invalid target embedding or no enrolled speakers found.");
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

    for (const auto &spk : m_speakers) {
        if (spk.embeddings.empty()) {
            continue;
        }

        // Find max cosine similarity
        float score = -1.0f;
        for (const auto &en : spk.embeddings) {
            float sim = 0.0f;
            for (int i = 0; i < m_meta.embedding_dim; ++i) {
                sim += (embedding[i] * inv_norm) * en[i];
            }
            if (sim > score)
                score = sim;
        }

        bool is_same = (score > threshold);
        ESP_LOGI(TAG,
                 "[MAX-COSINE] %s: score=%.4f (threshold=%.4f) -> %s",
                 spk.speaker_name.c_str(),
                 score,
                 threshold,
                 is_same ? "SAME speaker" : "DIFFERENT speaker");
    }

    return ESP_OK;
}

esp_err_t AudioVerificationDatabase::verify_subspace(const float *embedding, float threshold)
{
    if (!is_valid()) {
        return m_init_error;
    }
    if (!embedding || m_speakers.empty()) {
        ESP_LOGW(TAG, "Invalid target embedding or no enrolled speakers found.");
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

    for (const auto &spk : m_speakers) {
        if (!spk.subspace.is_ready() || spk.embeddings.size() < 2) {
            continue;
        }
        float score = spk.subspace.compute_neg_distance(embedding, inv_norm);
        bool is_same = (score > threshold);
        ESP_LOGI(TAG,
                 "[SUBSPACE] %s: score=%.4f (threshold=%.4f) -> %s",
                 spk.speaker_name.c_str(),
                 score,
                 threshold,
                 is_same ? "SAME speaker" : "DIFFERENT speaker");
    }

    return ESP_OK;
}

void AudioVerificationDatabase::print()
{
    ESP_LOGI(TAG, "============ Database Summary ============");
    ESP_LOGI(TAG, "%-5s | %-20s | %-10s", "ID", "Name", "#Embeddings");
    ESP_LOGI(TAG, "------------------------------------------");
    for (const auto &spk : m_speakers) {
        ESP_LOGI(TAG, "%-5d | %-20s | %-10zu", spk.speaker_id, spk.speaker_name.c_str(), spk.embeddings.size());
    }
    ESP_LOGI(TAG, "==========================================");
}

} // namespace audio
} // namespace dl
