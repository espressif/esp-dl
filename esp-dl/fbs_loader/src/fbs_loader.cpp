#include "fbs_loader.hpp"
#include "esp_idf_version.h"
#include "mbedtls/sha256.h"

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#include "psa/crypto.h"
#else
#include "mbedtls/aes.h"
#endif

static const char *TAG = "FbsLoader";

namespace fbs {

// PDL3 header byte layout
#define PDL3_VERSION_OFFSET 4
#define PDL3_VERSION_SIZE 16
#define PDL3_PACKAGE_SIZE_OFFSET 20
#define PDL3_SHA256_OFFSET 24
#define PDL3_SHA256_SIZE 32
#define PDL3_MODEL_NUM_OFFSET 56
#define PDL3_HEADER_SIZE 60

/**
 * @brief This function is used to decrypt the AES 128-bit CTR mode encrypted data.
 * AES (Advanced Encryption Standard) is a widely-used symmetric encryption algorithm that provides strong security for
 * data protection CTR mode converts the block cipher into a stream cipher, allowing it to encrypt data of any length
 * without the need for padding
 *
 * @param ciphertext   Input Fbs data encrypted by AES 128-bit CTR mode
 * @param plaintext    Decrypted data
 * @param size         Size of input data
 * @param key          128-bit AES key
 */
void fbs_aes_crypt_ctr(const uint8_t *ciphertext, uint8_t *plaintext, size_t size, const uint8_t *key)
{
    uint8_t nonce[16] = {
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    psa_key_attributes_t key_attributes = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t key_id = PSA_KEY_ID_NULL;
    psa_cipher_operation_t operation = PSA_CIPHER_OPERATION_INIT;
    size_t output_length = 0;

    psa_set_key_usage_flags(&key_attributes, PSA_KEY_USAGE_ENCRYPT);
    psa_set_key_algorithm(&key_attributes, PSA_ALG_CTR);
    psa_set_key_type(&key_attributes, PSA_KEY_TYPE_AES);
    psa_set_key_bits(&key_attributes, 128);
    psa_import_key(&key_attributes, key, 16, &key_id);
    psa_reset_key_attributes(&key_attributes);

    psa_cipher_encrypt_setup(&operation, key_id, PSA_ALG_CTR);
    psa_cipher_set_iv(&operation, nonce, sizeof(nonce));
    psa_cipher_update(&operation, ciphertext, size, plaintext, size, &output_length);

    uint8_t finish_buf[16];
    size_t finish_len;
    psa_cipher_finish(&operation, finish_buf, sizeof(finish_buf), &finish_len);
    psa_destroy_key(key_id);
#else
    mbedtls_aes_context aes_ctx;
    size_t offset = 0;
    uint8_t stream_block[16];
    mbedtls_aes_init(&aes_ctx);
    mbedtls_aes_setkey_enc(&aes_ctx, key, 128); // 128-bit key
    mbedtls_aes_crypt_ctr(&aes_ctx, size, &offset, nonce, stream_block, ciphertext, plaintext);
    mbedtls_aes_free(&aes_ctx);
#endif
}

static inline bool is_packed_format(fbs_file_format_t format)
{
    return format == FBS_FILE_FORMAT_PDL1 || format == FBS_FILE_FORMAT_PDL2 || format == FBS_FILE_FORMAT_PDL3;
}

// uint32 word index of model_num within the packed header
static inline uint32_t pack_model_num_word(fbs_file_format_t format)
{
    return (format == FBS_FILE_FORMAT_PDL3) ? (PDL3_MODEL_NUM_OFFSET / 4) : 1;
}

// uint32 word index of the first model entry within the packed header
static inline uint32_t pack_entry_base_word(fbs_file_format_t format)
{
    return (format == FBS_FILE_FORMAT_PDL3) ? (PDL3_HEADER_SIZE / 4) : 2;
}

// byte offset of model_num within the packed header
static inline uint32_t pack_model_num_byte(fbs_file_format_t format)
{
    return (format == FBS_FILE_FORMAT_PDL3) ? PDL3_MODEL_NUM_OFFSET : 4;
}

// byte offset of the first model entry within the packed header
static inline uint32_t pack_entry_base_byte(fbs_file_format_t format)
{
    return (format == FBS_FILE_FORMAT_PDL3) ? PDL3_HEADER_SIZE : 8;
}

fbs_file_format_t get_model_format(const char *fbs_buf, model_location_type_t model_location)
{
    char str[5];
    if (model_location != MODEL_LOCATION_IN_SDCARD) {
        memcpy(str, fbs_buf, 4);
        str[4] = '\0';
    } else {
        FILE *f = fopen(fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", fbs_buf);
            return FBS_FILE_FORMAT_UNK;
        }
        fread(str, 4, 1, f);
        str[4] = '\0';
        fclose(f);
    }

    if (strcmp(str, "EDL1") == 0) {
        return FBS_FILE_FORMAT_EDL1;
    } else if (strcmp(str, "PDL1") == 0) {
        return FBS_FILE_FORMAT_PDL1;
    } else if (strcmp(str, "EDL2") == 0) {
        return FBS_FILE_FORMAT_EDL2;
    } else if (strcmp(str, "PDL2") == 0) {
        return FBS_FILE_FORMAT_PDL2;
    } else if (strcmp(str, "PDL3") == 0) {
        return FBS_FILE_FORMAT_PDL3;
    } else {
        return FBS_FILE_FORMAT_UNK;
    }
}

esp_err_t get_model_offset_by_index(const char *fbs_buf,
                                    model_location_type_t model_location,
                                    fbs_file_format_t format,
                                    uint32_t index,
                                    uint32_t &offset)
{
    uint32_t num_word = pack_model_num_word(format);
    uint32_t entry_word = pack_entry_base_word(format);
    uint32_t num_byte = pack_model_num_byte(format);
    uint32_t entry_byte = pack_entry_base_byte(format);
    if (model_location != MODEL_LOCATION_IN_SDCARD) {
        const uint32_t *header = (const uint32_t *)fbs_buf;
        uint32_t model_num = header[num_word];
        if (index >= model_num) {
            ESP_LOGE(TAG, "The model index is out of range.");
            return ESP_FAIL;
        }
        offset = header[entry_word + index * 3];
        return ESP_OK;
    } else {
        FILE *f = fopen(fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", fbs_buf);
            return ESP_FAIL;
        }
        fseek(f, num_byte, SEEK_SET);
        uint32_t model_num;
        fread(&model_num, 4, 1, f);
        if (index >= model_num) {
            ESP_LOGE(TAG, "The model index is out of range.");
            fclose(f);
            return ESP_FAIL;
        }
        fseek(f, entry_byte + 12 * index, SEEK_SET);
        fread(&offset, 4, 1, f);
        fclose(f);
        return ESP_OK;
    }
}

esp_err_t get_model_offset_by_name(const char *fbs_buf,
                                   model_location_type_t model_location,
                                   fbs_file_format_t format,
                                   const char *name,
                                   uint32_t &offset)
{
    uint32_t num_word = pack_model_num_word(format);
    uint32_t entry_word = pack_entry_base_word(format);
    uint32_t num_byte = pack_model_num_byte(format);
    uint32_t entry_byte = pack_entry_base_byte(format);
    if (model_location != MODEL_LOCATION_IN_SDCARD) {
        const uint32_t *header = (const uint32_t *)fbs_buf;
        uint32_t model_num = header[num_word];
        uint32_t name_offset, name_length;
        for (int i = 0; i < model_num; i++) {
            name_offset = header[entry_word + 3 * i + 1];
            name_length = header[entry_word + 3 * i + 2];
            std::string model_name(fbs_buf + name_offset, name_length);
            if (model_name == std::string(name)) {
                offset = header[entry_word + 3 * i];
                return ESP_OK;
            }
        }
        ESP_LOGE(TAG, "Model %s is not found.", name);
        return ESP_FAIL;
    } else {
        FILE *f = fopen(fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", fbs_buf);
            return ESP_FAIL;
        }
        fseek(f, num_byte, SEEK_SET);
        uint32_t model_num;
        fread(&model_num, 4, 1, f);
        uint32_t name_offset, name_length;
        for (int i = 0; i < model_num; i++) {
            fseek(f, entry_byte + 12 * i + 4, SEEK_SET);
            fread(&name_offset, 4, 1, f);
            fread(&name_length, 4, 1, f);
            std::string model_name(name_length, '\0');
            fseek(f, name_offset, SEEK_SET);
            fread(model_name.data(), name_length, 1, f);
            if (model_name == std::string(name)) {
                fseek(f, entry_byte + 12 * i, SEEK_SET);
                fread(&offset, 4, 1, f);
                fclose(f);
                return ESP_OK;
            }
        }
        ESP_LOGE(TAG, "Model %s is not found.", name);
        fclose(f);
        return ESP_FAIL;
    }
}

FbsModel *create_fbs_model(const char *fbs_buf,
                           fbs_file_format_t format,
                           model_location_type_t model_location,
                           uint32_t offset,
                           const uint8_t *key,
                           bool param_copy)
{
    if (fbs_buf == nullptr) {
        ESP_LOGE(TAG, "Model's flatbuffers is empty or broken.");
        return nullptr;
    }

    char *model_buf;
    uint32_t mode, size;
    if (model_location != MODEL_LOCATION_IN_SDCARD) {
        model_buf = const_cast<char *>(fbs_buf + offset);
        uint32_t *header = (uint32_t *)model_buf;
        mode = header[1]; // cryptographic mode, 0: without encryption, 1: aes encryption
        size = header[2];
        if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_PDL1) {
            model_buf += 12;
        } else {
            model_buf += 16;
        }
    } else {
        FILE *f = fopen(fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", fbs_buf);
            return nullptr;
        }
        fseek(f, offset + 4, SEEK_SET);
        fread(&mode, 4, 1, f);
        fread(&size, 4, 1, f);
        model_buf = (char *)dl::tool::malloc_aligned(size, MALLOC_CAP_DEFAULT);
        if (!model_buf) {
            ESP_LOGE(
                TAG,
                "Failed to alloc %.2fKB RAM, largest available PSRAM block size %.2fKB, internal RAM block size %.2fKB",
                size / 1024.f,
                heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.f,
                heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.f);
            return nullptr;
        }
        if (format == FBS_FILE_FORMAT_EDL2 || format == FBS_FILE_FORMAT_PDL2 || format == FBS_FILE_FORMAT_PDL3) {
            fseek(f, 4, SEEK_CUR);
        }
        fread(model_buf, size, 1, f);
        fclose(f);
    }

    assert(mode == 0 || mode == 1);
    if (mode != 0 && key == NULL) {
        ESP_LOGE(TAG, "This is a cryptographic model, please enter the secret key!");
        return nullptr;
    }

    bool rodata_move = false;
    if (model_location == MODEL_LOCATION_IN_FLASH_RODATA &&
        dl::tool::memory_addr_type(model_buf) == dl::MEMORY_ADDR_PSRAM) {
        ESP_LOGW(TAG,
                 "CONFIG_SPIRAM_RODATA or CONFIG_SPIRAM_XIP_FROM_PSRAM option is on, fbs model is copied to PSRAM.");
        rodata_move = true;
    }

    bool auto_free;
    if (mode == 0) { // without encryption
        auto_free = (model_location == MODEL_LOCATION_IN_SDCARD) ? true : false;
        bool address_align = !(reinterpret_cast<uintptr_t>(model_buf) & 0xf);
        if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_PDL1) {
            param_copy = true;
        } else if (!address_align) {
            ESP_LOGW(TAG, "The address of fbs model in flash is not aligned with 16 bytes.");
            param_copy = true;
        } else {
            if (model_location == MODEL_LOCATION_IN_SDCARD) {
                param_copy = false;
            } else if (dl::tool::memory_addr_type(model_buf) == dl::MEMORY_ADDR_PSRAM) {
                param_copy = false;
            }
        }
    } else { // 128-bit AES encryption
        auto_free = true;
        param_copy = (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_PDL1) ? true : false;
        uint8_t *model_buf_decrypt;
        if (model_location == MODEL_LOCATION_IN_SDCARD) {
            model_buf_decrypt = (uint8_t *)model_buf;
        } else {
            model_buf_decrypt = (uint8_t *)dl::tool::malloc_aligned(size, MALLOC_CAP_DEFAULT);
            if (!model_buf_decrypt) {
                ESP_LOGE(TAG,
                         "Failed to alloc %.2fKB RAM, largest available PSRAM block size %.2fKB, internal RAM block "
                         "size %.2fKB",
                         size / 1024.f,
                         heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.f,
                         heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.f);
                return nullptr;
            }
        }
        fbs_aes_crypt_ctr((const uint8_t *)model_buf, model_buf_decrypt, size, key);
        model_buf = (char *)model_buf_decrypt;
    }

    return new FbsModel(model_buf, size, model_location, mode, rodata_move, auto_free, param_copy);
}

FbsLoader::FbsLoader(const char *name, model_location_type_t location) :
    m_mmap_handle(nullptr), m_location(location), m_fbs_buf(nullptr), m_format(FBS_FILE_FORMAT_UNK)
{
    if (name == nullptr) {
        return;
    }

    if (m_location == MODEL_LOCATION_IN_FLASH_RODATA || m_location == MODEL_LOCATION_IN_SDCARD) {
        m_fbs_buf = (const void *)name;
    } else if (m_location == MODEL_LOCATION_IN_FLASH_PARTITION) {
        const esp_partition_t *partition =
            esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, name);
        if (partition) {
            int free_pages = spi_flash_mmap_get_free_pages(SPI_FLASH_MMAP_DATA);
            uint32_t storage_size = free_pages * 64 * 1024; // Byte
            ESP_LOGI(TAG, "The storage free size is %ld KB", storage_size / 1024);
            ESP_LOGI(TAG, "The partition size is %ld KB", partition->size / 1024);
            if (storage_size < partition->size) {
                ESP_LOGE(TAG,
                         "The storage free size of this board is less than %s partition required size",
                         partition->label);
            }
            this->m_mmap_handle = (esp_partition_mmap_handle_t *)malloc(sizeof(esp_partition_mmap_handle_t));
            ESP_ERROR_CHECK(esp_partition_mmap(partition,
                                               0,
                                               partition->size,
                                               ESP_PARTITION_MMAP_DATA,
                                               &this->m_fbs_buf,
                                               static_cast<esp_partition_mmap_handle_t *>(this->m_mmap_handle)));
        } else {
            ESP_LOGE(TAG, "Can not find %s in partition table", name);
        }
    }

    if (m_fbs_buf != nullptr) {
        m_format = fbs::get_model_format((const char *)m_fbs_buf, m_location);
    }
}

FbsLoader::~FbsLoader()
{
    if (m_location == MODEL_LOCATION_IN_FLASH_PARTITION) {
        esp_partition_munmap(*static_cast<esp_partition_mmap_handle_t *>(this->m_mmap_handle)); // support esp-idf v5
        if (this->m_mmap_handle) {
            free(this->m_mmap_handle);
            this->m_mmap_handle = nullptr;
        }
    }
}

FbsModel *FbsLoader::load(const int model_index, const uint8_t *key, bool param_copy)
{
    if (m_format == FBS_FILE_FORMAT_UNK) {
        ESP_LOGE(TAG, "Model's flatbuffers is empty or broken.");
        return nullptr;
    }

    uint32_t offset = 0;
    fbs_file_format_t format = m_format;
    if (is_packed_format(format)) {
        // packed multiple espdl models
        if (get_model_offset_by_index((const char *)m_fbs_buf, m_location, format, model_index, offset) != ESP_OK) {
            return nullptr;
        }
    } else if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_EDL2) {
        // single espdl model
        if (model_index > 0) {
            ESP_LOGW(TAG, "There is only one model in the flatbuffers, ignore the input model index!");
        }
        offset = 0;
    } else {
        ESP_LOGE(TAG, "Unsupported format, or the model file is corrupted!");
        return nullptr;
    }
    return create_fbs_model((const char *)m_fbs_buf, format, m_location, offset, key, param_copy);
}

FbsModel *FbsLoader::load(const uint8_t *key, bool param_copy)
{
    return this->load(0, key, param_copy);
}

FbsModel *FbsLoader::load(const char *model_name, const uint8_t *key, bool param_copy)
{
    if (m_format == FBS_FILE_FORMAT_UNK) {
        ESP_LOGE(TAG, "Model's flatbuffers is empty or broken.");
        return nullptr;
    }

    uint32_t offset = 0;
    fbs_file_format_t format = m_format;
    if (is_packed_format(format)) {
        // packed multiple espdl models
        if (get_model_offset_by_name((const char *)m_fbs_buf, m_location, format, model_name, offset) != ESP_OK) {
            return nullptr;
        }
    } else if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_EDL2) {
        // single espdl model
        if (model_name) {
            ESP_LOGW(TAG, "There is only one model in the flatbuffers, ignore the input model name!");
        }
        offset = 0;
    } else {
        ESP_LOGE(TAG, "Unsupported format, or the model file is corrupted!");
        return nullptr;
    }
    return create_fbs_model((const char *)m_fbs_buf, format, m_location, offset, key, param_copy);
}

int FbsLoader::get_model_num()
{
    if (m_format == FBS_FILE_FORMAT_UNK) {
        return 0;
    }

    fbs_file_format_t format = m_format;
    if (is_packed_format(format)) {
        // packed multiple espdl models
        uint32_t model_num;
        if (m_location != MODEL_LOCATION_IN_SDCARD) {
            uint32_t *header = (uint32_t *)m_fbs_buf;
            model_num = header[pack_model_num_word(format)];
        } else {
            FILE *f = fopen((const char *)m_fbs_buf, "rb");
            if (!f) {
                ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
                return 0;
            }
            fseek(f, pack_model_num_byte(format), SEEK_SET);
            fread(&model_num, 4, 1, f);
            fclose(f);
        }
        return model_num;
    } else if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_EDL2) {
        // single espdl model
        return 1;
    } else {
        ESP_LOGE(TAG, "Unsupported format, or the model file is corrupted!");
        return 0;
    }

    return 0;
}

void FbsLoader::list_models()
{
    if (m_format == FBS_FILE_FORMAT_UNK) {
        ESP_LOGE(TAG, "Model's flatbuffers is empty or broken.");
        return;
    }

    fbs_file_format_t format = m_format;
    if (is_packed_format(format)) {
        // packed multiple espdl models
        uint32_t entry_word = pack_entry_base_word(format);
        uint32_t entry_byte = pack_entry_base_byte(format);
        if (m_location != MODEL_LOCATION_IN_SDCARD) {
            uint32_t *header = (uint32_t *)m_fbs_buf;
            uint32_t model_num = header[pack_model_num_word(format)];
            for (int i = 0; i < model_num; i++) {
                uint32_t name_offset = header[entry_word + 3 * i + 1];
                uint32_t name_length = header[entry_word + 3 * i + 2];
                std::string name((const char *)m_fbs_buf + name_offset, name_length);
                ESP_LOGI(TAG, "model name: %s, index:%d", name.c_str(), i);
            }
        } else {
            FILE *f = fopen((const char *)m_fbs_buf, "rb");
            if (!f) {
                ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
                return;
            }
            fseek(f, pack_model_num_byte(format), SEEK_SET);
            uint32_t model_num;
            fread(&model_num, 4, 1, f);
            uint32_t name_offset, name_length;
            for (int i = 0; i < model_num; i++) {
                fseek(f, entry_byte + 12 * i + 4, SEEK_SET);
                fread(&name_offset, 4, 1, f);
                fread(&name_length, 4, 1, f);
                std::string name(name_length, '\0');
                fseek(f, name_offset, SEEK_SET);
                fread(name.data(), name_length, 1, f);
                ESP_LOGI(TAG, "model name: %s, index:%d", name.c_str(), i);
            }
            fclose(f);
        }
    } else if (format == FBS_FILE_FORMAT_EDL1 || format == FBS_FILE_FORMAT_EDL2) {
        ESP_LOGI(TAG, "There is only one model in the flatbuffers without model name.");
    }
}

esp_err_t FbsLoader::get_package_version(char *out_version, size_t out_size)
{
    if (out_version == nullptr || out_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (m_format == FBS_FILE_FORMAT_UNK) {
        ESP_LOGE(TAG, "Model's flatbuffers is empty or broken.");
        return ESP_ERR_INVALID_STATE;
    }
    fbs_file_format_t format = m_format;
    if (format != FBS_FILE_FORMAT_PDL3) {
        ESP_LOGW(TAG, "get_package_version is only supported for PDL3 packages.");
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (out_size < PDL3_VERSION_SIZE) {
        return ESP_ERR_INVALID_SIZE;
    }

    char version[PDL3_VERSION_SIZE];
    if (m_location != MODEL_LOCATION_IN_SDCARD) {
        memcpy(version, (const char *)m_fbs_buf + PDL3_VERSION_OFFSET, PDL3_VERSION_SIZE);
    } else {
        FILE *f = fopen((const char *)m_fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
            return ESP_FAIL;
        }
        fseek(f, PDL3_VERSION_OFFSET, SEEK_SET);
        fread(version, PDL3_VERSION_SIZE, 1, f);
        fclose(f);
    }
    version[PDL3_VERSION_SIZE - 1] = '\0'; // guarantee termination
    strlcpy(out_version, version, out_size);
    return ESP_OK;
}

uint32_t FbsLoader::get_package_size()
{
    if (m_format == FBS_FILE_FORMAT_UNK) {
        return 0;
    }
    fbs_file_format_t format = m_format;
    if (format != FBS_FILE_FORMAT_PDL3) {
        return 0;
    }

    uint32_t package_size = 0;
    if (m_location != MODEL_LOCATION_IN_SDCARD) {
        memcpy(&package_size, (const char *)m_fbs_buf + PDL3_PACKAGE_SIZE_OFFSET, sizeof(uint32_t));
    } else {
        FILE *f = fopen((const char *)m_fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
            return 0;
        }
        fseek(f, PDL3_PACKAGE_SIZE_OFFSET, SEEK_SET);
        fread(&package_size, sizeof(uint32_t), 1, f);
        fclose(f);
    }
    return package_size;
}

esp_err_t FbsLoader::get_package_sha256(uint8_t out_sha256[32])
{
    if (out_sha256 == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    if (m_format == FBS_FILE_FORMAT_UNK) {
        return ESP_ERR_INVALID_STATE;
    }
    fbs_file_format_t format = m_format;
    if (format != FBS_FILE_FORMAT_PDL3) {
        ESP_LOGW(TAG, "get_package_sha256 is only supported for PDL3 packages.");
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (m_location != MODEL_LOCATION_IN_SDCARD) {
        memcpy(out_sha256, (const char *)m_fbs_buf + PDL3_SHA256_OFFSET, PDL3_SHA256_SIZE);
    } else {
        FILE *f = fopen((const char *)m_fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
            return ESP_FAIL;
        }
        fseek(f, PDL3_SHA256_OFFSET, SEEK_SET);
        fread(out_sha256, PDL3_SHA256_SIZE, 1, f);
        fclose(f);
    }
    return ESP_OK;
}

esp_err_t FbsLoader::calc_package_sha256(uint8_t out_sha256[32])
{
    if (out_sha256 == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    if (m_format == FBS_FILE_FORMAT_UNK) {
        return ESP_ERR_INVALID_STATE;
    }
    fbs_file_format_t format = m_format;
    if (format != FBS_FILE_FORMAT_PDL3) {
        ESP_LOGW(TAG, "calc_package_sha256 is only supported for PDL3 packages.");
        return ESP_ERR_NOT_SUPPORTED;
    }

    uint32_t package_size = this->get_package_size();
    if (package_size <= PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE) {
        ESP_LOGE(TAG, "Invalid PDL3 package_size: %lu", (unsigned long)package_size);
        return ESP_ERR_INVALID_SIZE;
    }

    const uint8_t zero_digest[PDL3_SHA256_SIZE] = {0};
    mbedtls_sha256_context ctx;
    mbedtls_sha256_init(&ctx);
    esp_err_t ret = ESP_OK;
    // 0 selects SHA-256 (as opposed to SHA-224).
    if (mbedtls_sha256_starts(&ctx, 0) != 0) {
        mbedtls_sha256_free(&ctx);
        return ESP_FAIL;
    }

    if (m_location != MODEL_LOCATION_IN_SDCARD) {
        const uint8_t *buf = (const uint8_t *)m_fbs_buf;
        // [0, PDL3_SHA256_OFFSET): magic + version + package_size
        if (mbedtls_sha256_update(&ctx, buf, PDL3_SHA256_OFFSET) != 0 ||
            // the package_sha256 field is hashed as 32 zero bytes
            mbedtls_sha256_update(&ctx, zero_digest, PDL3_SHA256_SIZE) != 0 ||
            // remaining bytes up to package_size
            mbedtls_sha256_update(&ctx,
                                  buf + PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE,
                                  package_size - PDL3_SHA256_OFFSET - PDL3_SHA256_SIZE) != 0) {
            ret = ESP_FAIL;
        }
    } else {
        FILE *f = fopen((const char *)m_fbs_buf, "rb");
        if (!f) {
            ESP_LOGE(TAG, "Failed to open %s.", (const char *)m_fbs_buf);
            mbedtls_sha256_free(&ctx);
            return ESP_FAIL;
        }
        const size_t chunk_size = 1024;
        uint8_t *chunk = (uint8_t *)malloc(chunk_size);
        if (!chunk) {
            fclose(f);
            mbedtls_sha256_free(&ctx);
            return ESP_ERR_NO_MEM;
        }
        uint32_t remaining = package_size;
        uint32_t consumed = 0;
        while (remaining > 0 && ret == ESP_OK) {
            size_t to_read = remaining < chunk_size ? remaining : chunk_size;
            if (fread(chunk, 1, to_read, f) != to_read) {
                ret = ESP_FAIL;
                break;
            }
            // mask out the package_sha256 field with zeros within this chunk
            for (size_t i = 0; i < to_read; i++) {
                uint32_t abs_off = consumed + i;
                if (abs_off >= PDL3_SHA256_OFFSET && abs_off < PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE) {
                    chunk[i] = 0;
                }
            }
            if (mbedtls_sha256_update(&ctx, chunk, to_read) != 0) {
                ret = ESP_FAIL;
                break;
            }
            consumed += to_read;
            remaining -= to_read;
        }
        free(chunk);
        fclose(f);
    }

    if (ret == ESP_OK && mbedtls_sha256_finish(&ctx, out_sha256) != 0) {
        ret = ESP_FAIL;
    }
    mbedtls_sha256_free(&ctx);
    return ret;
}

bool FbsLoader::verify_package_sha256()
{
    uint8_t stored[PDL3_SHA256_SIZE];
    uint8_t computed[PDL3_SHA256_SIZE];
    if (this->get_package_sha256(stored) != ESP_OK) {
        return false;
    }
    if (this->calc_package_sha256(computed) != ESP_OK) {
        return false;
    }
    if (memcmp(stored, computed, PDL3_SHA256_SIZE) != 0) {
        ESP_LOGE(TAG, "PDL3 package SHA256 verification failed.");
        return false;
    }
    return true;
}

fbs_file_format_t FbsLoader::get_model_format()
{
    return m_format;
}

const char *FbsLoader::get_model_location_string()
{
    switch (m_location) {
    case MODEL_LOCATION_IN_FLASH_RODATA:
        return "MODEL LOCATION IN FLASH RODATA";
    case MODEL_LOCATION_IN_FLASH_PARTITION:
        return "MODEL LOCATION IN FLASH PARTITION";
    case MODEL_LOCATION_IN_SDCARD:
        return "MODEL LOCATION IN SDCARD";
    default:
        return "MODEL LOCATION UNK";
    }
    return "MODEL LOCATION UNK";
}

} // namespace fbs
