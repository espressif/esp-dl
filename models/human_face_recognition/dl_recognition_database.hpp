#pragma once

#include "dl_recognition_database_base.hpp"
#include "esp_spiffs.h"
#include "esp_vfs.h"
#include "esp_vfs_fat.h"
#include "bsp/esp-bsp.h"

namespace dl {
namespace recognition {
class FatFsFlashDataBase : public DataBase {
private:
    wl_handle_t m_wl_handle;
    esp_err_t mount() override;
    esp_err_t unmount() override;
    esp_err_t check_enough_free_space() override;

public:
    FatFsFlashDataBase(int feat_len, const char *name);
    ~FatFsFlashDataBase();
};

class FatFsSDCardDataBase : public DataBase {
private:
    esp_err_t mount() override;
    esp_err_t unmount() override;
    esp_err_t check_enough_free_space() override;

public:
    FatFsSDCardDataBase(int feat_len, const char *name);
    ~FatFsSDCardDataBase();
};

class SPIFFSDataBase : public DataBase {
private:
    esp_err_t mount() override;
    esp_err_t unmount() override;
    esp_err_t check_enough_free_space() override;

public:
    SPIFFSDataBase(int feat_len, const char *name);
    ~SPIFFSDataBase();
};

class DB {
public:
    DB(db_type_t db_type, int feat_len, const char *name);
    ~DB();
    esp_err_t clear_all_feats() { return m_db->clear_all_feats(); }
    esp_err_t enroll_feat(TensorBase *feat) { return m_db->enroll_feat(feat); }
    // esp_err_t delete_feat(uint16_t id) { return m_db->delete_feat(id); }
    esp_err_t delete_last_feat() { return m_db->delete_last_feat(); }
    std::vector<result_t> query_feat(TensorBase *feat, float thr, int top_k)
    {
        return m_db->query_feat(feat, thr, top_k);
    }
    int get_num_feats() { return m_db->m_meta.num_feats_valid; }

private:
    DataBase *m_db;
};
} // namespace recognition
} // namespace dl
