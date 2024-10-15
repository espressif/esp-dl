#pragma once
#include "dl_recognition_define.hpp"
#include "dl_tensor_base.hpp"
#include "esp_check.h"
#include "esp_system.h"
#include <algorithm>
#include <list>

namespace dl {
namespace recognition {
class DataBase {
public:
    DataBase(const char *name) : name(name) {}
    virtual ~DataBase() {}
    esp_err_t clear_all_feats();
    esp_err_t enroll_feat(TensorBase *feat);
    esp_err_t delete_feat(uint16_t id);
    esp_err_t delete_last_feat();
    std::list<query_info> query_feat(TensorBase *feat, float thr, int top_k);
    void print();

protected:
    char db_path[50];
    const char *name;
    database_meta meta;
    void init(int feat_len);
    void deinit();

private:
    std::list<database_feat> feats;

    esp_err_t create_empty_database_in_storage(int feat_len);
    esp_err_t load_database_from_storage(int feat_len);
    void clear_all_feats_in_memory();
    virtual esp_err_t mount() = 0;
    virtual esp_err_t unmount() = 0;
    virtual esp_err_t check_enough_free_space() = 0;
    float cal_similarity(float *feat1, float *feat2);
};
} // namespace recognition
} // namespace dl
