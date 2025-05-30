#pragma once

#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include <limits>
#include <map>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace fbs {
typedef enum {
    MODEL_LOCATION_IN_FLASH_RODATA = 0,    // The model in FLASH .rodata section
    MODEL_LOCATION_IN_FLASH_PARTITION = 1, // The model in SPIFFS
    MODEL_LOCATION_IN_SDCARD = 2,          // The model in SDCard
    MODEL_LOCATION_MAX = MODEL_LOCATION_IN_SDCARD,
} model_location_type_t;

/**
 * @brief Flatbuffer model object.
 */
class FbsModel {
public:
    /**
     * @brief Construct a new FbsModel object.
     *
     * @param data          The data of model flatbuffers.
     * @param size          The size of model flatbuffers in bytes.
     * @param location      The location of model flatbuffers.
     * @param encrypt       Whether the model flatbuffers is encrypted or not.
     * @param rodata_move   Whether the model flatbuffers is moved from FLASH rodata to PSRAM.
     * @param auto_free     Whether to free the model flatbuffers data when destroy this class instance.
     * @param param_copy    Whether to copy the parameter in flatbuffers.
     */
    FbsModel(const void *data,
             size_t size,
             model_location_type_t location,
             bool encrypt,
             bool rodata_move,
             bool auto_free,
             bool param_copy);

    /**
     * @brief Destroy the FbsModel object.
     */
    ~FbsModel();

    /**
     * @brief Print the model information.
     */
    void print();

    /**
     * @brief Return vector of node name in the order of execution.
     *
     * @return topological sort of node name.
     */
    std::vector<std::string> topological_sort();

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, int &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, float &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, std::string &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, std::vector<int> &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, std::vector<float> &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, dl::quant_type_t &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name,
                                      std::string attribute_name,
                                      dl::activation_type_t &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, dl::resize_mode_t &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param attribute_name    The name of attribute.
     * @param ret_value         The attribute value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_attribute(std::string node_name, std::string attribute_name, dl::TensorBase *&ret_value);

    /**
     * @brief Get operation output shape
     *
     * @param node_name         The name of operation.
     * @param index             The index of outputs
     * @param ret_value         Return shape value.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_output_shape(std::string node_name, int index, std::vector<int> &ret_value);

    /**
     * @brief Get the attribute of node.
     *
     * @param node_name         The name of operation.
     * @param inputs            The vector of operation inputs.
     * @param outputs           The vector of operation outputs.
     *
     * @return esp_err_t        Return ESP_OK if get successfully. Otherwise return ESP_FAIL.
     */
    esp_err_t get_operation_inputs_and_outputs(std::string node_name,
                                               std::vector<std::string> &inputs,
                                               std::vector<std::string> &outputs);

    /**
     * @brief Get operation type, "Conv", "Linear" etc
     *
     * @param node_name  The name of operation
     *
     * @return The type of operation.
     */
    std::string get_operation_type(std::string node_name);

    /**
     * @brief Return if the variable is a parameter
     *
     * @param node_name  The name of operation
     * @param index      The index of the variable
     * @param caps       Bitwise OR of MALLOC_CAP_* flags indicating the type of memory to be returned
     *
     * @return dl::TensorBase*
     */
    dl::TensorBase *get_operation_parameter(std::string node_name, int index = 1, uint32_t caps = MALLOC_CAP_DEFAULT);

    /**
     * @brief Get LUT(Look Up Table) if the operation has LUT
     *
     * @param node_name   The name of operation
     * @param caps       Bitwise OR of MALLOC_CAP_* flags indicating the type of memory to be returned
     * @param attribute_name The name of LUT attribute
     * @return dl::TensorBase*
     */
    dl::TensorBase *get_operation_lut(std::string node_name,
                                      uint32_t caps = MALLOC_CAP_DEFAULT,
                                      std::string attribute_name = "lut");

    /**
     * @brief return true if the variable is a parameter
     *
     * @param name Variable name
     *
     * @return true if the variable is a parameter else false
     */
    bool is_parameter(std::string name);

    /**
     * @brief Get the raw data of FlatBuffers::Dl::Tensor.
     *
     * @param tensor_name   The name of Tensor.
     *
     * @return uint8_t *    The pointer of raw data.
     */
    const void *get_tensor_raw_data(std::string tensor_name);

    /**
     * @brief Get the element type of tensor tensor.
     *
     * @param tensor_name    The tensor name.
     *
     * @return FlatBuffers::Dl::TensorDataType
     */
    dl::dtype_t get_tensor_dtype(std::string tensor_name);

    /**
     * @brief Get the shape of tensor.
     *
     * @param tensor_name       The name of tensor.
     *
     * @return std::vector<int>  The shape of tensor.
     */
    std::vector<int> get_tensor_shape(std::string tensor_name);

    /**
     * @brief Get the exponents of tensor.
     *
     * @warning When quantization is PER_CHANNEL, the size of exponents is same as out_channels.
     *          When quantization is PER_TENSOR, the size of exponents is 1.
     *
     * @param tensor_name       The name of tensor.
     *
     * @return  The exponents of tensor.
     */
    std::vector<int> get_tensor_exponents(std::string tensor_name);

    /**
     * @brief Get the element type of value_info.
     *
     * @param var_name    The value_info name.
     *
     * @return dl::dtype_t
     */
    dl::dtype_t get_value_info_dtype(std::string var_name);

    /**
     * @brief Get the shape of value_info.
     *
     * @param var_name      The value_info name.
     *
     * @return the shape of value_info.
     */
    std::vector<int> get_value_info_shape(std::string var_name);

    /**
     * @brief Get the exponent of value_info. Only support PER_TENSOR quantization.
     *
     * @param var_name      The value_info name.
     *
     * @return the exponent of value_info
     */
    int get_value_info_exponent(std::string var_name);

    /**
     * @brief Get the raw data of test input tensor.
     *
     * @param tensor_name   The name of test input tensor.
     *
     * @return uint8_t *    The pointer of raw data.
     */
    const void *get_test_input_tensor_raw_data(std::string tensor_name);

    /**
     * @brief Get the raw data of test output tensor.
     *
     * @param tensor_name   The name of test output tensor.
     *
     * @return uint8_t *    The pointer of raw data.
     */
    const void *get_test_output_tensor_raw_data(std::string tensor_name);

    /**
     * @brief Get the test input tensor.
     *
     * @param tensor_name   The name of test input tensor.
     * @return  The pointer of tensor.
     */
    dl::TensorBase *get_test_input_tensor(std::string tensor_name);

    /**
     * @brief Get the test output tensor.
     *
     * @param tensor_name   The name of test output tensor.
     * @return The pointer of tensor.
     */
    dl::TensorBase *get_test_output_tensor(std::string tensor_name);

    /**
     * @brief Get the name of test outputs.
     *
     * @return the name of test outputs
     */
    std::vector<std::string> get_test_outputs_name();

    /**
     * @brief Get the graph inputs.
     *
     * @return the name of inputs
     */
    std::vector<std::string> get_graph_inputs();

    /**
     * @brief Get the graph outputs.
     *
     * @return the name of ounputs
     */
    std::vector<std::string> get_graph_outputs();

    /**
     * @brief Clear all map
     */
    void clear_map();

    /**
     * @brief Load all map
     */
    void load_map();

    /**
     * @brief Get the model name
     *
     * @return the name of model
     */
    std::string get_model_name();

    /**
     * @brief Get the model version
     *
     * @return The version of model
     */
    int64_t get_model_version();

    /**
     * @brief Get the model doc string
     *
     * @return The doc string of model
     */
    std::string get_model_doc_string();

    /**
     * @brief Get the model's metadata prop
     *
     * @param key   The key of metadata prop
     * @return The value of metadata prop
     */
    std::string get_model_metadata_prop(const std::string &key);

    /**
     * @brief Get the model size
     *
     * @param internal_size        Flatbuffers model internal RAM usage
     * @param psram_size           Flatbuffers model PSRAM usage
     * @param psram_rodata_size    Flatbuffers model PSRAM rodate usage. If CONFIG_SPIRAM_RODATA option is on, \
     *                             Flatbuffers model in FLASH rodata will be copied to PSRAM
     * @param flash_size           Flatbuffers model FLASH usage
     */
    void get_model_size(size_t *internal_size, size_t *psram_size, size_t *psram_rodata_size, size_t *flash_size);

    bool m_param_copy; ///< copy flatbuffers param or not.

private:
    model_location_type_t m_location;
    bool m_encrypt;
    bool m_rodata_move;
    bool m_auto_free;
    size_t m_size;
    const uint8_t *m_data;
    const void *m_model;
    std::map<std::string, const void *> m_name_to_node_map;
    std::map<std::string, const void *> m_name_to_initial_tensor_map;
    std::map<std::string, const void *> m_name_to_value_info_map;
    std::unordered_map<std::string, const void *> m_name_to_test_inputs_value_map;
    std::unordered_map<std::string, const void *> m_name_to_test_outputs_value_map;
};
} // namespace fbs
