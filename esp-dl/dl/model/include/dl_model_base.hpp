#pragma once

#include "dl_memory_manager.hpp"
#include "dl_model_context.hpp"
#include "dl_module_base.hpp"
#include "esp_log.h"
#include "fbs_loader.hpp"
#include "fbs_model.hpp"

#if DL_LOG_INFER_LATENCY
#define DL_LOG_INFER_LATENCY_INIT_WITH_SIZE(size) DL_LOG_LATENCY_INIT_WITH_SIZE(size)
#define DL_LOG_INFER_LATENCY_INIT() DL_LOG_LATENCY_INIT()
#define DL_LOG_INFER_LATENCY_START() DL_LOG_LATENCY_START()
#define DL_LOG_INFER_LATENCY_END() DL_LOG_LATENCY_END()
#define DL_LOG_INFER_LATENCY_PRINT(prefix, key) DL_LOG_LATENCY_PRINT(prefix, key)
#define DL_LOG_INFER_LATENCY_END_PRINT(prefix, key) DL_LOG_LATENCY_END_PRINT(prefix, key)
#define DL_LOG_INFER_LATENCY_ARRAY_INIT_WITH_SIZE(n, size) DL_LOG_LATENCY_ARRAY_INIT_WITH_SIZE(n, size)
#define DL_LOG_INFER_LATENCY_ARRAY_INIT(n) DL_LOG_LATENCY_ARRAY_INIT(n)
#define DL_LOG_INFER_LATENCY_ARRAY_START(i) DL_LOG_LATENCY_ARRAY_START(i)
#define DL_LOG_INFER_LATENCY_ARRAY_END(i) DL_LOG_LATENCY_ARRAY_END(i)
#define DL_LOG_INFER_LATENCY_ARRAY_PRINT(i, prefix, key) DL_LOG_LATENCY_ARRAY_PRINT(i, prefix, key)
#define DL_LOG_INFER_LATENCY_ARRAY_END_PRINT(i, prefix, key) DL_LOG_LATENCY_ARRAY_END_PRINT(i, prefix, key)
#else
#define DL_LOG_INFER_LATENCY_INIT_WITH_SIZE(size)
#define DL_LOG_INFER_LATENCY_INIT()
#define DL_LOG_INFER_LATENCY_START()
#define DL_LOG_INFER_LATENCY_END()
#define DL_LOG_INFER_LATENCY_PRINT(prefix, key)
#define DL_LOG_INFER_LATENCY_END_PRINT(prefix, key)
#define DL_LOG_INFER_LATENCY_ARRAY_INIT_WITH_SIZE(n, size)
#define DL_LOG_INFER_LATENCY_ARRAY_INIT(n)
#define DL_LOG_INFER_LATENCY_ARRAY_START(i)
#define DL_LOG_INFER_LATENCY_ARRAY_END(i)
#define DL_LOG_INFER_LATENCY_ARRAY_PRINT(i, prefix, key)
#define DL_LOG_INFER_LATENCY_ARRAY_END_PRINT(i, prefix, key)
#endif

namespace dl {

// currently only support MEMORY_MANAGER_GREEDY
typedef enum { MEMORY_MANAGER_GREEDY = 0, LINEAR_MEMORY_MANAGER = 1 } memory_manager_t;

/**
 * @brief Neural Network Model.
 */
class Model {
private:
    fbs::FbsLoader *m_fbs_loader = nullptr; /*!< The instance of flatbuffers Loader */
    fbs::FbsModel *m_fbs_model = nullptr;   /*!< The instance of flatbuffers Model */
    std::vector<dl::module::Module *>
        m_execution_plan; /*!< This represents a valid topological sort (dependency ordered) execution plan. */
    ModelContext *m_model_context = nullptr;       /*!< The pointer of model context */
    std::map<std::string, TensorBase *> m_inputs;  /*!< The map of model input's name and TensorBase */
    std::map<std::string, TensorBase *> m_outputs; /*!< The map of model output's name and TensorBase */
    std::string m_name;                            /*!< The name of model */
    int64_t m_version;                             /*!< The version of model */
    std::string m_doc_string;                      /*!< doc string of model */
    size_t m_internal_size;                        /*!< Internal RAM usage */
    size_t m_psram_size;                           /*!< PSRAM usage */

public:
    Model() {}

    /**
     * @brief Create the Model object by rodata address or partition label.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param location      The model location.
     * @param max_internal_size  In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM, and
     you want to alloc memory on internal RAM first.
     * @param mm_type        Type of memory manager
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     */
    Model(const char *rodata_address_or_partition_label_or_path,
          fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
          int max_internal_size = 0,
          memory_manager_t mm_type = MEMORY_MANAGER_GREEDY,
          const uint8_t *key = nullptr,
          bool param_copy = true);

    /**
     * @brief Create the Model object by rodata address or partition label.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param model_index   The model index of packed models.
     * @param location      The model location.
     * @param max_internal_size  In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM, and
     you want to alloc memory on internal RAM first.
     * @param mm_type        Type of memory manager
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     */
    Model(const char *rodata_address_or_partition_label_or_path,
          int model_index,
          fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
          int max_internal_size = 0,
          memory_manager_t mm_type = MEMORY_MANAGER_GREEDY,
          const uint8_t *key = nullptr,
          bool param_copy = true);

    /**
     * @brief Create the Model object by rodata address or partition label.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param model_name   The model name of packed models.
     * @param location      The model location.
     * @param max_internal_size  In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM, and
     you want to alloc memory on internal RAM first.
     * @param mm_type        Type of memory manager
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     */
    Model(const char *rodata_address_or_partition_label_or_path,
          const char *model_name,
          fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
          int max_internal_size = 0,
          memory_manager_t mm_type = MEMORY_MANAGER_GREEDY,
          const uint8_t *key = nullptr,
          bool param_copy = true);

    /**
     * @brief Create the Model object by fbs_model.
     *
     * @param fbs_model      The fbs model.
     * @param internal_size  Internal ram size, in bytes
     * @param mm_type        Type of memory manager
     */
    Model(fbs::FbsModel *fbs_model, int internal_size = 0, memory_manager_t mm_type = MEMORY_MANAGER_GREEDY);

    /**
     * @brief Destroy the Model object.
     */
    virtual ~Model();

    /**
     * @brief Load model graph and parameters from FLASH or sdcard.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param location      The model location.
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     * @return
     *      - ESP_OK       Success
     *      - ESP_FAIL     Failed
     */
    virtual esp_err_t load(const char *rodata_address_or_partition_label_or_path,
                           fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                           const uint8_t *key = nullptr,
                           bool param_copy = true);

    /**
     * @brief Load model graph and parameters from FLASH or sdcard.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param location      The model location.
     * @param model_index   The model index of packed models.
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     * @return
     *      - ESP_OK       Success
     *      - ESP_FAIL     Failed
     */
    virtual esp_err_t load(const char *rodata_address_or_partition_label_or_path,
                           fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                           int model_index = 0,
                           const uint8_t *key = nullptr,
                           bool param_copy = true);

    /**
     * @brief Load model graph and parameters from FLASH or sdcard.
     *
     * @param rodata_address_or_partition_label_or_path
     *                                     The address of model data while location is MODEL_LOCATION_IN_FLASH_RODATA.
     *                                     The label of partition while location is MODEL_LOCATION_IN_FLASH_PARTITION.
     *                                     The path of model while location is MODEL_LOCATION_IN_SDCARD.
     * @param location      The model location.
     * @param model_name    The model name of packed models.
     * @param key           The key of encrypted model.
     * @param param_copy    Set to false to avoid copy model parameters from FLASH to PSRAM.
     *                      Only set this param to false when your PSRAM resource is very tight. This saves PSRAM and
     *                      sacrifices the performance of model inference because the frequency of PSRAM is higher than
     * FLASH. Only takes effect when MODEL_LOCATION_IN_FLASH_RODATA(CONFIG_SPIRAM_RODATA not set) or
     * MODEL_LOCATION_IN_FLASH_PARTITION.
     * @return
     *      - ESP_OK       Success
     *      - ESP_FAIL     Failed
     */
    virtual esp_err_t load(const char *rodata_address_or_partition_label_or_path,
                           fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                           const char *model_name = nullptr,
                           const uint8_t *key = nullptr,
                           bool param_copy = true);

    /**
     * @brief Load model graph and parameters from Flatbuffers model
     *
     * @param fbs_model          The FlatBuffers model
     * @return
     *      - ESP_OK       Success
     *      - ESP_FAIL     Failed
     */
    virtual esp_err_t load(fbs::FbsModel *fbs_model);

    /**
     * @brief Allocate memory for the model.
     *
     * @param max_internal_size  In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM, and
     you want to alloc memory on internal RAM first.
     * @param mm_type        Type of memory manager
     * @param preload        Whether to preload the model's parameters to internal ram (not implemented yet)
     */
    virtual void build(size_t max_internal_size,
                       memory_manager_t mm_type = MEMORY_MANAGER_GREEDY,
                       bool preload = false);

    /**
     * @brief Run the model module by module.
     *
     * @param mode  Runtime mode.
     */
    virtual void run(runtime_mode_t mode = RUNTIME_MODE_SINGLE_CORE);

    /**
     * @brief Run the model module by module.
     *
     * @param input  The model input.
     * @param mode   Runtime mode.
     */
    virtual void run(TensorBase *input, runtime_mode_t mode = RUNTIME_MODE_SINGLE_CORE);

    /**
     * @brief Run the model module by module.
     *
     * @param user_inputs   The model inputs.
     * @param mode          Runtime mode.
     * @param user_outputs  It's for debug to pecify the output of the intermediate layer; Under normal use, there is no
     *                      need to pass a value to this parameter. If no parameter is passed, the default is the
     * graphical output, which can be obtained through Model::get_outputs().
     */
    virtual void run(std::map<std::string, TensorBase *> &user_inputs,
                     runtime_mode_t mode = RUNTIME_MODE_SINGLE_CORE,
                     std::map<std::string, TensorBase *> user_outputs = {});

    /**
     * @brief Minimize the model.
     */
    void minimize();

    /**
     * @brief Test whether the model inference result is correct.
     * The model should contain test_inputs and test_outputs.
     * Enable export_test_values option in esp-ppq to use this api.
     *
     * @return esp_err_t
     */
    esp_err_t test();

    /**
     * @brief Get memory info
     *
     * @return Memory usage statistics on internal and PSRAM.
     */
    std::map<std::string, mem_info_t> get_memory_info();

    /**
     * @brief Get module info
     *
     * @return return Type and latency of each module.
     */
    std::map<std::string, module_info> get_module_info();

    /**
     * @brief Print the module info obtained by get_module_info function.
     *
     * @param info
     * @param sort_module_by_latency
     */
    void print_module_info(const std::map<std::string, module_info> &info, bool sort_module_by_latency = false);

    /**
     * @brief Print model memory summary.
     *
     */
    void profile_memory();

    /**
     * @brief Print module info summary. (Name, Type, Latency)
     *
     * @param sort_module_by_latency True The module is printed in latency decreasing sort.
     *                               False The module is printed in ONNX topological sort.
     */
    void profile_module(bool sort_module_by_latency = false);

    /**
     * @brief Combination of profile_memory & profile_module.
     *
     * @param sort_module_by_latency True The module is printed in latency decreasing sort.
     *                               False The module is printed in ONNX topological sort.
     */
    void profile(bool sort_module_by_latency = false);

    /**
     * @brief Get inputs of model
     *
     * @return The map of model input's name and TensorBase*
     */
    virtual std::map<std::string, TensorBase *> &get_inputs();

    /**
     * @brief Get the only input of model.
     *
     * @return TensorBase*
     */
    virtual TensorBase *get_input();

    /**
     * @brief Get input of model by name.
     *
     * @param name input name
     * @return TensorBase*
     */
    virtual TensorBase *get_input(const std::string &name);

    /**
     * @brief Get intermediate TensorBase of model
     * @note   When using memory manager, the content of TensorBase's data may be overwritten by the outputs of other
     * @param name The name of intermediate Tensor.
     * operators.
     * @return The intermediate TensorBase*.
     */
    virtual TensorBase *get_intermediate(const std::string &name);

    /**
     * @brief Get outputs of model
     *
     * @return The map of model output's name and TensorBase*
     */
    virtual std::map<std::string, TensorBase *> &get_outputs();

    /**
     * @brief Get the only output of model.
     *
     * @return TensorBase*
     */
    virtual TensorBase *get_output();

    /**
     * @brief Get output of model by name.
     *
     * @param name output name
     * @return TensorBase*
     */
    virtual TensorBase *get_output(const std::string &name);

    /**
     * @brief Get the model's metadata prop
     *
     * @param key   The key of metadata prop
     * @return The value of metadata prop
     */
    std::string get_metadata_prop(const std::string &key);

    /**
     * @brief Print the model.
     */
    virtual void print();

    /**
     * @brief Get the fbs model instance.
     *
     * @return fbs::FbsModel *
     */
    virtual fbs::FbsModel *get_fbs_model() { return m_fbs_model; }
};

} // namespace dl
