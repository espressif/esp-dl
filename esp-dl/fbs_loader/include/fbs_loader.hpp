#pragma once

#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_partition.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#include "spi_flash_mmap.h"
#endif
#include "fbs_model.hpp"

namespace fbs {

/**
    FBS_FILE_FORMAT_EDL1:
    {
        char[4]: "EDL1",
        uint32:  the mode of entru
        uint32:  the length of data
        uint8[]:  the data
    }

    FBS_FILE_FORMAT_PDL1:
    {
        "PDL1": char[4]
        model_num: uint32
        model1_data_offset: uint32
        model1_name_offset: uint32
        model1_name_length: uint32
        model2_data_offset: uint32
        model2_name_offset: uint32
        model2_name_length: uint32
        ...
        model1_name,
        model2_name,
        ...
        model1_data(format:FBS_FILE_FORMAT_EDL1),
        model2_data(format:FBS_FILE_FORMAT_EDL1),
        ...
    }

    FBS_FILE_FORMAT_EDL2:
    {
        char[4]: "EDL2",
        uint32:  the mode of entru
        uint32:  the length of data
        uint32:  zero padding
        uint8[]:  the data
        zero padding
    }

    FBS_FILE_FORMAT_PDL2:
    {
        "PDL2": char[4]
        model_num: uint32
        model1_data_offset: uint32
        model1_name_offset: uint32
        model1_name_length: uint32
        model2_data_offset: uint32
        model2_name_offset: uint32
        model2_name_length: uint32
        ...
        model1_name,
        model2_name,
        ...
        zero padding
        model1_data(format:FBS_FILE_FORMAT_EDL2),
        model2_data(format:FBS_FILE_FORMAT_EDL2),
        ...
    }

    FBS_FILE_FORMAT_PDL3:
    {
        "PDL3": char[4]
        package_version: char[16]   // ASCII, '\0' terminated, max 15 chars
        package_size: uint32        // valid byte count of the whole package
        package_sha256: uint8[32]   // integrity digest, see SHA256 rule below
        model_num: uint32
        model1_data_offset: uint32  // 16-byte aligned
        model1_name_offset: uint32
        model1_name_length: uint32
        model2_data_offset: uint32
        model2_name_offset: uint32
        model2_name_length: uint32
        ...
        model1_name,
        model2_name,
        ...
        zero padding
        model1_data(format:FBS_FILE_FORMAT_EDL2),
        model2_data(format:FBS_FILE_FORMAT_EDL2),
        ...
    }

    PDL3 SHA256 rule:
        package_sha256 = SHA256(package[0, package_size)) where the 32 bytes of
        the package_sha256 field itself are treated as all zeros.
*/
typedef enum {
    FBS_FILE_FORMAT_UNK = 0,  // Unknown format
    FBS_FILE_FORMAT_EDL1 = 1, // EDL1 format
    FBS_FILE_FORMAT_PDL1 = 2, // PDL1 format
    FBS_FILE_FORMAT_EDL2 = 3, // EDL2 format
    FBS_FILE_FORMAT_PDL2 = 4, // PDL2 format
    FBS_FILE_FORMAT_PDL3 = 5  // PDL3 format
} fbs_file_format_t;

/**
 * @brief Class for parser the flatbuffers.
 *
 */
class FbsLoader {
public:
    /**
     * @brief Construct a new FbsLoader object.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param location  The model location.
     */
    FbsLoader(const char *rodata_address_or_partition_label_or_path = nullptr,
              model_location_type_t location = MODEL_LOCATION_IN_FLASH_RODATA);

    /**
     * @brief Destroy the FbsLoader object.
     */
    ~FbsLoader();

    /**
     * @brief Load the model. If there are multiple sub-models, the first sub-model will be loaded.
     *
     * @param key   NULL or a 128-bit AES key, like {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
     * 0x0b, 0x0c, 0x0d, 0x0e, 0x0f}
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     *
     * @return  Return nullptr if loading fails. Otherwise return the pointer of FbsModel.
     */
    FbsModel *load(const uint8_t *key = nullptr, bool param_copy = true);

    /**
     * @brief Load the model by model index.
     *
     * @param model_index  The index of model.
     * @param key   NULL or a 128-bit AES key, like {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
     * 0x0b, 0x0c, 0x0d, 0x0e, 0x0f}.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     *
     * @return  Return nullptr if loading fails. Otherwise return the pointer of FbsModel.
     */
    FbsModel *load(const int model_index, const uint8_t *key = nullptr, bool param_copy = true);

    /**
     * @brief Load the model by model name.
     *
     * @param model_name  The name of model.
     * @param key   NULL or a 128-bit AES key, like {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
     * 0x0b, 0x0c, 0x0d, 0x0e, 0x0f}
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     *
     * @return  Return nullptr if loading fails. Otherwise return the pointer of FbsModel.
     */
    FbsModel *load(const char *model_name, const uint8_t *key = nullptr, bool param_copy = true);

    /**
     * @brief Get the number of models.
     *
     * @return The number of models
     */
    int get_model_num();

    /**
     * @brief List all model's name
     */
    void list_models();

    /**
     * @brief Get the model location string.
     *
     * @return The model location string.
     */
    const char *get_model_location_string();

    /**
     * @brief Get the format of the loaded model/package.
     *
     * @return The format of the model, or FBS_FILE_FORMAT_UNK if the flatbuffers is empty or the format is
     * unrecognized.
     */
    fbs_file_format_t get_model_format();

    /**
     * @brief Get the package version string of a PDL3 package.
     *
     * @note Only valid for the PDL3 format.
     *
     * @param out_version  Output buffer that receives the '\0' terminated version string.
     * @param out_size     Size of the output buffer in bytes. Must be at least 16 bytes to hold the full field.
     *
     * @return ESP_OK on success.
     *         ESP_ERR_NOT_SUPPORTED if the package is not PDL3.
     *         ESP_ERR_INVALID_ARG / ESP_ERR_INVALID_SIZE on bad arguments.
     */
    esp_err_t get_package_version(char *out_version, size_t out_size);

    /**
     * @brief Get the package_size field of a PDL3 package.
     *
     * @note Only valid for the PDL3 format.
     *
     * @return The valid byte count of the PDL3 package, or 0 if the package is not PDL3.
     */
    uint32_t get_package_size();

    /**
     * @brief Get the package_sha256 field stored in a PDL3 package header.
     *
     * @note Only valid for the PDL3 format. This returns the digest stored in the package, it does not recompute it.
     *
     * @param out_sha256  Output buffer that receives the 32-byte digest.
     *
     * @return ESP_OK on success.
     *         ESP_ERR_NOT_SUPPORTED if the package is not PDL3.
     *         ESP_ERR_INVALID_ARG on bad arguments.
     */
    esp_err_t get_package_sha256(uint8_t out_sha256[32]);

    /**
     * @brief Recompute the SHA256 digest of a PDL3 package.
     *
     * The digest is computed over the package bytes [0, package_size), where the 32 bytes of the package_sha256 field
     * are treated as all zeros. Only the PDL3 header and the data within package_size are read.
     *
     * @param out_sha256  Output buffer that receives the recomputed 32-byte digest.
     *
     * @return ESP_OK on success.
     *         ESP_ERR_NOT_SUPPORTED if the package is not PDL3.
     *         ESP_ERR_INVALID_ARG on bad arguments.
     */
    esp_err_t calc_package_sha256(uint8_t out_sha256[32]);

    /**
     * @brief Verify the integrity of a PDL3 package.
     *
     * Recomputes the SHA256 digest (see calc_package_sha256) and compares it with the package_sha256 field stored in
     * the header.
     *
     * @return true if the package is PDL3 and the recomputed digest matches the stored digest, false otherwise.
     */
    bool verify_package_sha256();

private:
    void *m_mmap_handle;
    model_location_type_t m_location;
    const void *m_fbs_buf;
    fbs_file_format_t m_format;
};

/**
 * @brief Get the format of model.
 *
 * @param fbs_buf   The buffer of fbs model.
 * @param model_location    The location of fbs model.
 *
 * @return The format of model.
 */
fbs_file_format_t get_model_format(const char *fbs_buf, model_location_type_t model_location);

/**
 * @brief Create a FbsModel object.
 *
 * @param fbs_buf   The buffer of fbs model.
 * @param format    The format of fbs model.
 * @param model_location    The location of fbs model.
 * @param offset    The offset of fbs model.
 * @param key       The key of encrypted model.
 * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
 */
FbsModel *create_fbs_model(const char *fbs_buf,
                           fbs_file_format_t format,
                           model_location_type_t model_location,
                           uint32_t offset,
                           const uint8_t *key,
                           bool param_copy);

} // namespace fbs
