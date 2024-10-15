#include "dl_recognition_database_base.hpp"

static const char *TAG = "dl::recognition::DataBase";

namespace dl {
namespace recognition {
void DataBase::init(int feat_len)
{
    ESP_ERROR_CHECK(this->mount());
    FILE *f = fopen(this->db_path, "rb");
    if (f == NULL) {
        this->create_empty_database_in_storage(feat_len);
    } else {
        this->load_database_from_storage(feat_len);
    }
}

void DataBase::deinit()
{
    this->clear_all_feats_in_memory();
    ESP_ERROR_CHECK(this->unmount());
}

esp_err_t DataBase::create_empty_database_in_storage(int feat_len)
{
    FILE *f = fopen(this->db_path, "wb");
    size_t size = 0;
    if (f) {
        this->meta.num_feats_total = 0;
        this->meta.num_feats_valid = 0;
        this->meta.feat_len = feat_len;
        size = fwrite(&this->meta, sizeof(database_meta), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write db meta data.");
            return ESP_FAIL;
        }
        fclose(f);
        return ESP_OK;
    } else {
        ESP_LOGE(TAG, "Failed to open db.");
        return ESP_FAIL;
    }
}

esp_err_t DataBase::clear_all_feats()
{
    if (remove(this->db_path) == -1) {
        ESP_LOGE(TAG, "Failed to remove db.");
        return ESP_FAIL;
    }
    ESP_RETURN_ON_ERROR(
        this->create_empty_database_in_storage(this->meta.feat_len), TAG, "Failed to create empty db in storage.");
    this->clear_all_feats_in_memory();
    return ESP_OK;
}

void DataBase::clear_all_feats_in_memory()
{
    for (auto it = this->feats.begin(); it != this->feats.end(); it++) {
        heap_caps_free(it->feat);
    }
    this->feats.clear();
    this->meta.num_feats_total = 0;
    this->meta.num_feats_valid = 0;
}

esp_err_t DataBase::load_database_from_storage(int feat_len)
{
    this->clear_all_feats_in_memory();
    FILE *f = fopen(this->db_path, "rb");
    size_t size = 0;
    if (f) {
        size = fread(&this->meta, sizeof(database_meta), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to read database meta.");
            fclose(f);
            return ESP_FAIL;
        }
        if (feat_len != this->meta.feat_len) {
            ESP_LOGE(TAG, "Feature len in storage does not match feature len in db.");
            fclose(f);
            return ESP_FAIL;
        }
        uint16_t id;
        for (int i = 0; i < this->meta.num_feats_total; i++) {
            size = fread(&id, sizeof(uint16_t), 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to read feature id.");
                fclose(f);
                return ESP_FAIL;
            }
            if (id == 0) {
                if (fseek(f, sizeof(float) * this->meta.feat_len, SEEK_CUR) == -1) {
                    ESP_LOGE(TAG, "Failed to seek db file.");
                    fclose(f);
                    return ESP_FAIL;
                }
                continue;
            }
            float *feat = (float *)heap_caps_malloc(sizeof(float) * this->meta.feat_len, MALLOC_CAP_SPIRAM);
            size = fread(feat, sizeof(float), this->meta.feat_len, f);
            if (size != this->meta.feat_len) {
                ESP_LOGE(TAG, "Failed to read feature data.");
                fclose(f);
                return ESP_FAIL;
            }
            this->feats.emplace_back(id, feat);
        }
        if (this->feats.size() != this->meta.num_feats_valid) {
            ESP_LOGE(TAG, "Incorrect valid feature num.");
            fclose(f);
            return ESP_FAIL;
        }
        fclose(f);
    } else {
        ESP_LOGE(TAG, "Failed to open db.");
        return ESP_FAIL;
    }
    return ESP_OK;
}

esp_err_t DataBase::enroll_feat(TensorBase *feat)
{
    ESP_RETURN_ON_ERROR(this->check_enough_free_space(), TAG, "No more space in storage.");
    if (feat->dtype != DATA_TYPE_FLOAT) {
        ESP_LOGE(TAG, "Only support float feature.");
        return ESP_FAIL;
    }
    if (feat->size != this->meta.feat_len) {
        ESP_LOGE(TAG, "Feature len to enroll does not match feature len in db.");
        return ESP_FAIL;
    }
    float *feat_copy = (float *)heap_caps_malloc(sizeof(float) * this->meta.feat_len, MALLOC_CAP_SPIRAM);
    memcpy(feat_copy, feat->data, feat->get_bytes());

    this->feats.emplace_back(this->meta.num_feats_total + 1, feat_copy);
    this->meta.num_feats_total++;
    this->meta.num_feats_valid++;

    size_t size = 0;
    FILE *f = fopen(this->db_path, "rb+");
    if (f) {
        size = fwrite(&this->meta, sizeof(database_meta), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write database meta.");
            fclose(f);
            return ESP_FAIL;
        }
        if (fseek(f, 0, SEEK_END) == 0) {
            size = fwrite(&this->feats.back().id, sizeof(uint16_t), 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write feature id.");
                fclose(f);
                return ESP_FAIL;
            }
            size = fwrite(this->feats.back().feat, sizeof(float), this->meta.feat_len, f);
            if (size != this->meta.feat_len) {
                ESP_LOGE(TAG, "Failed to write feature.");
                fclose(f);
                return ESP_FAIL;
            }
        } else {
            ESP_LOGE(TAG, "Failed to seek db file.");
            fclose(f);
            return ESP_FAIL;
        }
    } else {
        ESP_LOGE(TAG, "Failed to open db.");
        fclose(f);
        return ESP_FAIL;
    }
    fclose(f);
    return ESP_OK;
}

esp_err_t DataBase::delete_feat(uint16_t id)
{
    bool invalid_id = true;
    for (auto it = this->feats.begin(); it != this->feats.end(); it++) {
        if (it->id != id) {
            continue;
        } else {
            heap_caps_free(it->feat);
            it = this->feats.erase(it);
            this->meta.num_feats_valid--;
            invalid_id = false;
            break;
        }
    }
    if (invalid_id) {
        ESP_LOGW(TAG, "Invalid id to delete.");
        return ESP_FAIL;
    }
    size_t size = 0;
    FILE *f = fopen(this->db_path, "rb+");
    if (f) {
        long int offset = sizeof(database_meta) + (sizeof(uint16_t) + sizeof(float) * this->meta.feat_len) * (id - 1);
        uint16_t id = 0;
        if (fseek(f, offset, SEEK_SET) == 0) {
            size = fwrite(&id, sizeof(uint16_t), 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write feature id.");
                fclose(f);
                return ESP_FAIL;
            }
        } else {
            ESP_LOGE(TAG, "Failed to seek db file.");
            fclose(f);
            return ESP_FAIL;
        }

        offset = sizeof(uint16_t);
        if (fseek(f, offset, SEEK_SET) == 0) {
            size = fwrite(&this->meta.num_feats_valid, sizeof(uint16_t), 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write valid feature num.");
                fclose(f);
                return ESP_FAIL;
            }
        } else {
            ESP_LOGE(TAG, "Failed to seek db file.");
            fclose(f);
            return ESP_FAIL;
        }
    } else {
        ESP_LOGE(TAG, "Failed to open db.");
        fclose(f);
        return ESP_FAIL;
    }
    fclose(f);
    return ESP_OK;
}

esp_err_t DataBase::delete_last_feat()
{
    if (!this->feats.empty()) {
        uint16_t id = this->feats.back().id;
        return this->delete_feat(id);
    } else {
        ESP_LOGW(TAG, "Empty db, nothing to delete");
        return ESP_FAIL;
    }
}

float DataBase::cal_similarity(float *feat1, float *feat2)
{
    float sum = 0;
    for (int i = 0; i < this->meta.feat_len; i++) {
        sum += feat1[i] * feat2[i];
    }
    return sum;
}

std::list<query_info> DataBase::query_feat(TensorBase *feat, float thr, int top_k)
{
    std::list<query_info> res;
    if (top_k < 1) {
        ESP_LOGW(TAG, "Top_k should be greater than 0.");
        return res;
    }
    float sim;
    for (auto it = this->feats.begin(); it != this->feats.end(); it++) {
        sim = this->cal_similarity(it->feat, (float *)feat->data);
        if (sim <= thr) {
            continue;
        }
        query_info q = {it->id, sim};
        res.insert(std::upper_bound(res.begin(), res.end(), q, greater_query_info), q);
        if (res.size() > top_k)
            res.pop_back();
    }
    return res;
}

void DataBase::print()
{
    printf("\n");
    printf("[db meta]\nnum_feats_total: %d, num_feats_valid: %d, feat_len: %d\n",
           this->meta.num_feats_total,
           this->meta.num_feats_valid,
           this->meta.feat_len);
    printf("[feats]\n");
    for (auto it : this->feats) {
        printf("id: %d feat: ", it.id);
        for (int i = 0; i < this->meta.feat_len; i++) {
            printf("%f, ", it.feat[i]);
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace recognition
} // namespace dl
