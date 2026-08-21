#include "fbs_model.hpp"
#include "schema_generated.h"
#include "flatbuffers/flatbuffers.h"

static const char *TAG = "FbsModel";

namespace fbs {

dl::dtype_t fbs_dtype_to_dl_dtype(FlatBuffers::Dl::TensorDataType type)
{
    switch (type) {
    case FlatBuffers::Dl::TensorDataType_UNDEFINED:
        return dl::DATA_TYPE_UNDEFINED;
    case FlatBuffers::Dl::TensorDataType_FLOAT:
        return dl::DATA_TYPE_FLOAT;
    case FlatBuffers::Dl::TensorDataType_UINT8:
        return dl::DATA_TYPE_UINT8;
    case FlatBuffers::Dl::TensorDataType_INT8:
        return dl::DATA_TYPE_INT8;
    case FlatBuffers::Dl::TensorDataType_UINT16:
        return dl::DATA_TYPE_UINT16;
    case FlatBuffers::Dl::TensorDataType_INT16:
        return dl::DATA_TYPE_INT16;
    case FlatBuffers::Dl::TensorDataType_INT32:
        return dl::DATA_TYPE_INT32;
    case FlatBuffers::Dl::TensorDataType_INT64:
        return dl::DATA_TYPE_INT64;
    case FlatBuffers::Dl::TensorDataType_STRING:
        return dl::DATA_TYPE_STRING;
    case FlatBuffers::Dl::TensorDataType_BOOL:
        return dl::DATA_TYPE_BOOL;
    case FlatBuffers::Dl::TensorDataType_FLOAT16:
        return dl::DATA_TYPE_FLOAT16;
    case FlatBuffers::Dl::TensorDataType_DOUBLE:
        return dl::DATA_TYPE_DOUBLE;
    case FlatBuffers::Dl::TensorDataType_UINT32:
        return dl::DATA_TYPE_UINT32;
    case FlatBuffers::Dl::TensorDataType_UINT64:
        return dl::DATA_TYPE_UINT64;
    default:
        return dl::DATA_TYPE_UNDEFINED;
    }
    return dl::DATA_TYPE_UNDEFINED;
}

const void *get_tensor_raw_data(const FlatBuffers::Dl::Tensor *pTensor)
{
    const void *ret = nullptr;

    if (pTensor) {
        if (pTensor->raw_data() != nullptr) {
            // ret = reinterpret_cast<const void *>(pTensor->raw_data()->Get(0)->bytes()->Data());
            ret = reinterpret_cast<const void *>(pTensor->raw_data()->Data());
        } else if (pTensor->float_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->float_data()->Data());
        } else if (pTensor->int32_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->int32_data()->Data());
        } else if (pTensor->string_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->string_data()->Data());
        } else if (pTensor->int64_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->int64_data()->Data());
        } else if (pTensor->external_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->external_data()->Data());
        } else if (pTensor->double_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->double_data()->Data());
        } else if (pTensor->uint64_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->uint64_data()->Data());
        } else {
            ESP_LOGW(TAG, "The tensor(%s) raw data is nullptr", pTensor->name()->c_str());
        }
    }
    return ret;
}

dl::dtype_t get_tensor_dtype(const FlatBuffers::Dl::Tensor *pTensor)
{
    dl::dtype_t ret = dl::DATA_TYPE_UNDEFINED;
    if (pTensor) {
        ret = fbs_dtype_to_dl_dtype(pTensor->data_type());
    }
    return ret;
}

std::vector<int> get_tensor_shape(const FlatBuffers::Dl::Tensor *pTensor)
{
    std::vector<int> shape;

    if (pTensor) {
        shape.reserve(pTensor->dims()->size());
        for (auto iter = pTensor->dims()->begin(); iter != pTensor->dims()->end(); iter++) {
            shape.push_back(static_cast<int>(*iter));
        }
    }
    return shape;
}

std::vector<int> get_tensor_exponents(const FlatBuffers::Dl::Tensor *pTensor)
{
    std::vector<int> exponents;

    if (pTensor) {
        exponents.reserve(pTensor->exponents()->size());
        for (auto iter = pTensor->exponents()->begin(); iter != pTensor->exponents()->end(); iter++) {
            exponents.push_back(static_cast<int>(*iter));
        }
    }

    return exponents;
}

/**
 * @brief Create a TensorBase, or nullptr when its per-channel exponents could not be allocated.
 *
 * A tensor without its exponent array carries no scale that describes its quantized data, so it must
 * not reach an operator: the failure is reported to the caller instead.
 */
static dl::TensorBase *create_tensor(const std::vector<int> &shape,
                                     const void *raw_data,
                                     const std::vector<int> &exponents,
                                     dl::dtype_t dtype,
                                     bool deep,
                                     uint32_t caps,
                                     const char *tensor_name)
{
    dl::TensorBase *tensor = new dl::TensorBase(shape, raw_data, exponents, dtype, deep, caps);
    if (!tensor->exponent.is_valid()) {
        ESP_LOGE(TAG, "Failed to load the %d per-channel exponents of tensor %s", (int)exponents.size(), tensor_name);
        delete tensor;
        return nullptr;
    }
    return tensor;
}

FbsModel::FbsModel(const void *data,
                   size_t size,
                   model_location_type_t location,
                   bool encrypt,
                   bool rodata_move,
                   bool auto_free,
                   bool param_copy) :
    m_param_copy(param_copy),
    m_location(location),
    m_encrypt(encrypt),
    m_rodata_move(rodata_move),
    m_auto_free(auto_free),
    m_size(size)
{
    if (data == nullptr) {
        m_data = nullptr;
        m_model = nullptr;
        ESP_LOGW(TAG, "Model data is null");
    } else {
        m_data = (const uint8_t *)data;
        m_model = FlatBuffers::Dl::GetModel(m_data);
        load_map();
    }
}

void FbsModel::load_map()
{
    if (m_model && m_name_to_node_map.empty()) {
        const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
        clear_map();
        for (auto node_iter = fbs_model->graph()->node()->begin(); node_iter != fbs_model->graph()->node()->end();
             node_iter++) {
            m_name_to_node_map.emplace((*node_iter)->name()->str(), (*node_iter));
        }
        for (auto initializer_iter = fbs_model->graph()->initializer()->begin();
             initializer_iter != fbs_model->graph()->initializer()->end();
             initializer_iter++) {
            m_name_to_initial_tensor_map.emplace((*initializer_iter)->name()->str(), (*initializer_iter));
        }
        for (auto value_info_iter = fbs_model->graph()->value_info()->begin();
             value_info_iter != fbs_model->graph()->value_info()->end();
             value_info_iter++) {
            m_name_to_value_info_map.emplace((*value_info_iter)->name()->str(), (*value_info_iter));
        }
        for (auto test_inputs_value_iter = fbs_model->graph()->test_inputs_value()->begin();
             test_inputs_value_iter != fbs_model->graph()->test_inputs_value()->end();
             test_inputs_value_iter++) {
            m_name_to_test_inputs_value_map.emplace((*test_inputs_value_iter)->name()->str(),
                                                    (*test_inputs_value_iter));
        }
        for (auto test_outputs_value_iter = fbs_model->graph()->test_outputs_value()->begin();
             test_outputs_value_iter != fbs_model->graph()->test_outputs_value()->end();
             test_outputs_value_iter++) {
            m_name_to_test_outputs_value_map.emplace((*test_outputs_value_iter)->name()->str(),
                                                     (*test_outputs_value_iter));
        }
    }
}

void FbsModel::clear_map()
{
    if (!m_name_to_node_map.empty()) {
        {
            std::map<std::string, const void *> temp;
            m_name_to_node_map.swap(temp);
        }
        {
            std::map<std::string, const void *> temp;
            m_name_to_initial_tensor_map.swap(temp);
        }
        {
            std::map<std::string, const void *> temp;
            m_name_to_value_info_map.swap(temp);
        }
        {
            std::unordered_map<std::string, const void *> temp;
            m_name_to_test_inputs_value_map.swap(temp);
        }
        {
            std::unordered_map<std::string, const void *> temp;
            m_name_to_test_outputs_value_map.swap(temp);
        }
    }
}

FbsModel::~FbsModel()
{
    if (m_data && m_auto_free) {
        heap_caps_free(const_cast<uint8_t *>(m_data));
        m_data = nullptr;
    }
}

void FbsModel::print()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    int layer_num = 0;
    for (auto node_iter = fbs_model->graph()->node()->begin(); node_iter != fbs_model->graph()->node()->end();
         node_iter++) {
        const FlatBuffers::Dl::Node *node = *node_iter;
        std::string quant_type;
        this->get_operation_attribute(node->name()->str(), "quant_type", quant_type);
        ESP_LOGI(TAG, "----------------------------------%d----------------------------------", layer_num++);
        ESP_LOGI(TAG,
                 "name: %s, op type:%s, quant type:%s",
                 node->name()->c_str(),
                 node->op_type()->c_str(),
                 quant_type.c_str());

        // inputs
        for (auto iter = node->input()->begin(); iter != node->input()->end(); iter++) {
            std::string name = (*iter)->str();
            if (is_parameter(name)) {
                ESP_LOGI(TAG,
                         "parameter: %s, dtype:%s, exponents:%d, shape:%s",
                         name.c_str(),
                         dl::dtype_to_string(get_tensor_dtype(name)),
                         get_tensor_exponents(name)[0],
                         dl::vector_to_string(get_tensor_shape(name)).c_str());
            } else {
                ESP_LOGI(TAG,
                         "input: %s, dtype:%s, exponent:%d, shape:%s",
                         name.c_str(),
                         dl::dtype_to_string(get_value_info_dtype(name)),
                         get_value_info_exponent(name),
                         dl::vector_to_string(get_value_info_shape(name)).c_str());
            }
        }

        // outputs
        for (auto iter = node->output()->begin(); iter != node->output()->end(); iter++) {
            ESP_LOGI(TAG,
                     "output: %s, dtype:%s",
                     (*iter)->c_str(),
                     dl::dtype_to_string(get_value_info_dtype((*iter)->str())));
        }

        ESP_LOGI(TAG, "----------------------------------------------------------------------");
    }
}

std::vector<std::string> FbsModel::topological_sort()
{
    std::vector<std::string> nodes;
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;

    for (auto node_iter = fbs_model->graph()->node()->begin(); node_iter != fbs_model->graph()->node()->end();
         node_iter++) {
        nodes.push_back((*node_iter)->name()->str());
    }
    return nodes;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name, std::string attribute_name, int &ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->i() != nullptr) {
                    ret_value = static_cast<int>((*attribute_iter)->i()->i());
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name, std::string attribute_name, float &ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->f() != nullptr) {
                    ret_value = (*attribute_iter)->f()->f();
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            std::vector<int> &ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->ints() != nullptr) {
                    ret_value.reserve((*attribute_iter)->ints()->size());
                    for (auto iter = (*attribute_iter)->ints()->begin(); iter != (*attribute_iter)->ints()->end();
                         iter++) {
                        ret_value.push_back(static_cast<int>(*iter));
                    }
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            std::vector<float> &ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    ret_value.clear();
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->floats() != nullptr) {
                    ret_value.reserve((*attribute_iter)->floats()->size());
                    for (auto iter = (*attribute_iter)->floats()->begin(); iter != (*attribute_iter)->floats()->end();
                         iter++) {
                        ret_value.push_back(*iter);
                    }
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name, std::string attribute_name, std::string &ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->s() != nullptr) {
                    ret_value.assign(reinterpret_cast<const char *>((*attribute_iter)->s()->Data()),
                                     (*attribute_iter)->s()->size());
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            dl::activation_type_t &ret_value)
{
    esp_err_t ret = ESP_FAIL;
    ret_value = dl::Linear;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->s() != nullptr) {
                    std::string str_value(reinterpret_cast<const char *>((*attribute_iter)->s()->Data()),
                                          (*attribute_iter)->s()->size());
                    if (str_value == "None" || str_value == "Linear") {
                        ret_value = dl::Linear;
                    } else if (str_value == "Relu") {
                        ret_value = dl::ReLU;
                    } else if (str_value == "LeakyRelu") {
                        ret_value = dl::LeakyReLU;
                    } else if (str_value == "PRelu") {
                        ret_value = dl::PReLU;
                    } else {
                        ESP_LOGE(TAG, "The activation type(%s) is not support now", str_value.c_str());
                    }
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            dl::quant_type_t &ret_value)
{
    esp_err_t ret = ESP_FAIL;
    ret_value = dl::QUANT_TYPE_FLOAT32;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->s() != nullptr) {
                    std::string str_value(reinterpret_cast<const char *>((*attribute_iter)->s()->Data()),
                                          (*attribute_iter)->s()->size());
                    if (str_value == "S8" || str_value == "SYMM_S8") {
                        ret_value = dl::QUANT_TYPE_SYMM_8BIT;
                    } else if (str_value == "S16" || str_value == "SYMM_S16") {
                        ret_value = dl::QUANT_TYPE_SYMM_16BIT;
                    } else if (str_value == "S32" || str_value == "SYMM_S32") {
                        ret_value = dl::QUANT_TYPE_SYMM_32BIT;
                    } else if (str_value == "None" || str_value == "F32") {
                        ret_value = dl::QUANT_TYPE_FLOAT32;
                    } else {
                        ESP_LOGE(TAG, "The quant type(%s) is not support now", str_value.c_str());
                    }
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            dl::resize_mode_t &ret_value)
{
    esp_err_t ret = ESP_FAIL;
    ret_value = dl::RESIZE_NEAREST;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                if ((*attribute_iter)->s() != nullptr) {
                    std::string str_value(reinterpret_cast<const char *>((*attribute_iter)->s()->Data()),
                                          (*attribute_iter)->s()->size());
                    if (str_value == "nearest") {
                        ret_value = dl::RESIZE_NEAREST;
                    } else if (str_value == "linear") {
                        ret_value = dl::RESIZE_LINEAR;
                    } else if (str_value == "cubic") {
                        ret_value = dl::RESIZE_CUBIC;
                    } else {
                        ESP_LOGE(TAG, "The resize mode(%s) is not support now", str_value.c_str());
                    }
                    ret = ESP_OK;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_attribute(std::string node_name,
                                            std::string attribute_name,
                                            dl::TensorBase *&ret_value)
{
    esp_err_t ret = ESP_FAIL;

    if (node_name.empty() || attribute_name.empty() || m_name_to_node_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    if (ret_value) {
        ESP_LOGW(TAG, "(%s)The TensorBase pointer is not null, it may cause a memory leak.", __FUNCTION__);
    }

    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto attribute_iter = pCurNode->attribute()->begin(); attribute_iter != pCurNode->attribute()->end();
             attribute_iter++) {
            if (attribute_name == (*attribute_iter)->name()->str()) {
                const FlatBuffers::Dl::Tensor *pTensor = (*attribute_iter)->t();
                if (pTensor) {
                    std::vector<int> shape = fbs::get_tensor_shape(pTensor);
                    std::vector<int> exponents = fbs::get_tensor_exponents(pTensor);
                    const void *pRawData = fbs::get_tensor_raw_data(pTensor);
                    dl::dtype_t dtype = fbs::get_tensor_dtype(pTensor);
                    ret_value = create_tensor(
                        shape, pRawData, exponents, dtype, m_param_copy, MALLOC_CAP_SPIRAM, attribute_name.c_str());
                    ret = ret_value ? ESP_OK : ESP_FAIL;
                }
                break;
            }
        }
    }

    return ret;
}

esp_err_t FbsModel::get_operation_input_shape(std::string node_name, int index, std::vector<int> &ret_value)
{
    esp_err_t ret = ESP_FAIL;
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        if (index < pCurNode->input()->size()) {
            std::string tensor_name = pCurNode->input()->Get(index)->str();
            if (is_parameter(tensor_name)) {
                ret_value = get_tensor_shape(tensor_name);
            } else {
                ret_value = get_value_info_shape(tensor_name);
            }
            ret = ESP_OK;
        }
    }
    return ret;
}

esp_err_t FbsModel::get_operation_output_shape(std::string node_name, int index, std::vector<int> &ret_value)
{
    esp_err_t ret = ESP_FAIL;
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        if (index < pCurNode->output()->size()) {
            std::string tensor_name = pCurNode->output()->Get(index)->str();
            ret_value = get_value_info_shape(tensor_name);
            ret = ESP_OK;
        }
    }
    return ret;
}

std::string FbsModel::get_operation_type(std::string node_name)
{
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        return pCurNode->op_type()->str();
    } else {
        ESP_LOGE(TAG, "Unknown node name: %s", node_name.c_str());
        return "";
    }
}

esp_err_t FbsModel::get_operation_inputs_and_outputs(std::string node_name,
                                                     std::vector<std::string> &inputs,
                                                     std::vector<std::string> &outputs)
{
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        inputs.clear();
        outputs.clear();
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        for (auto iter = pCurNode->input()->begin(); iter != pCurNode->input()->end(); iter++) {
            inputs.push_back((*iter)->str());
        }
        for (auto iter = pCurNode->output()->begin(); iter != pCurNode->output()->end(); iter++) {
            outputs.push_back((*iter)->str());
        }
        return ESP_OK;
    } else {
        ESP_LOGE(TAG, "Unknown node name: %s", node_name.c_str());
    }
    return ESP_FAIL;
}

dl::TensorBase *FbsModel::get_operation_parameter(std::string node_name, int index, uint32_t caps)
{
    auto name2node_iter = m_name_to_node_map.find(node_name);
    if (name2node_iter != m_name_to_node_map.end()) {
        const FlatBuffers::Dl::Node *pCurNode = (const FlatBuffers::Dl::Node *)name2node_iter->second;
        if (index < pCurNode->input()->size()) {
            std::string tensor_name = pCurNode->input()->Get(index)->str();
            return this->get_parameter(tensor_name, caps);
        }
    }

    return nullptr;
}

dl::TensorBase *FbsModel::get_parameter(std::string tensor_name, uint32_t caps)
{
    if (!this->is_parameter(tensor_name)) {
        return nullptr;
    }

    std::vector<int> shape = this->get_tensor_shape(tensor_name);
    std::vector<int> exponents = this->get_tensor_exponents(tensor_name);
    const void *pRawData = this->get_tensor_raw_data(tensor_name);
    dl::dtype_t dtype = this->get_tensor_dtype(tensor_name);
    if (exponents.size() >= 1 && (shape.empty() || shape[0] > 0) && pRawData != nullptr &&
        dtype != dl::DATA_TYPE_UNDEFINED) {
        return create_tensor(shape, pRawData, exponents, dtype, m_param_copy, caps, tensor_name.c_str());
    }

    ESP_LOGE(TAG,
             "(%s)Unsupported exponent size, data type or abnormal shape for tensor %s.",
             __FUNCTION__,
             tensor_name.c_str());
    return nullptr;
}

esp_err_t FbsModel::get_operation_lut_name(std::string node_name, std::string &lut_name, std::string attribute_name)
{
    lut_name.clear();
    esp_err_t ret = this->get_operation_attribute(node_name, attribute_name, lut_name);
    if (ret != ESP_OK || lut_name.empty()) {
        return ESP_ERR_NOT_FOUND;
    }
    if (!this->is_parameter(lut_name)) {
        ESP_LOGE(TAG, "LUT initializer %s referenced by node %s was not found.", lut_name.c_str(), node_name.c_str());
        lut_name.clear();
        return ESP_ERR_NOT_FOUND;
    }
    return ESP_OK;
}

dl::TensorBase *FbsModel::get_operation_lut(std::string node_name, uint32_t caps, std::string attribute_name)
{
    std::string lut_name;
    if (this->get_operation_lut_name(node_name, lut_name, attribute_name) == ESP_OK) {
        return this->get_parameter(lut_name, caps);
    }
    return nullptr;
}

bool FbsModel::is_parameter(std::string name)
{
    auto name2initial_iter = m_name_to_initial_tensor_map.find(name);
    if (name2initial_iter != m_name_to_initial_tensor_map.end()) {
        return true;
    }

    return false;
}

const void *FbsModel::get_tensor_raw_data(std::string tensor_name)
{
    const void *ret = nullptr;

    if (tensor_name.empty() || m_name_to_initial_tensor_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2initial_iter = m_name_to_initial_tensor_map.find(tensor_name);
    if (name2initial_iter != m_name_to_initial_tensor_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2initial_iter->second;
        ret = fbs::get_tensor_raw_data(pTensor);
    }
    return ret;
}

dl::dtype_t FbsModel::get_tensor_dtype(std::string tensor_name)
{
    dl::dtype_t ret = dl::DATA_TYPE_UNDEFINED;

    if (tensor_name.empty() || m_name_to_initial_tensor_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2initial_iter = m_name_to_initial_tensor_map.find(tensor_name);
    if (name2initial_iter != m_name_to_initial_tensor_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2initial_iter->second;
        ret = fbs::get_tensor_dtype(pTensor);
    }
    return ret;
}

std::vector<int> FbsModel::get_tensor_shape(std::string tensor_name)
{
    std::vector<int> shape;
    if (tensor_name.empty() || m_name_to_initial_tensor_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return shape;
    }

    auto name2initial_iter = m_name_to_initial_tensor_map.find(tensor_name);
    if (name2initial_iter != m_name_to_initial_tensor_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2initial_iter->second;
        shape = fbs::get_tensor_shape(pTensor);
    }
    return shape;
}

std::vector<int> FbsModel::get_tensor_exponents(std::string tensor_name)
{
    std::vector<int> exponents;
    if (tensor_name.empty() || m_name_to_initial_tensor_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return exponents;
    }

    auto name2initial_iter = m_name_to_initial_tensor_map.find(tensor_name);
    if (name2initial_iter != m_name_to_initial_tensor_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2initial_iter->second;
        exponents = fbs::get_tensor_exponents(pTensor);
    }

    return exponents;
}

dl::dtype_t FbsModel::get_value_info_dtype(std::string var_name)
{
    dl::dtype_t ret = dl::DATA_TYPE_UNDEFINED;

    if (var_name.empty() || m_name_to_value_info_map.empty()) {
        ESP_LOGW(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2valueInfo_iter = m_name_to_value_info_map.find(var_name);
    if (name2valueInfo_iter != m_name_to_value_info_map.end()) {
        const FlatBuffers::Dl::ValueInfo *pInfo = (const FlatBuffers::Dl::ValueInfo *)name2valueInfo_iter->second;
        if (pInfo->value_info_type()->value_type() == FlatBuffers::Dl::TypeInfoValue_tensor_type) {
            const FlatBuffers::Dl::TensorTypeAndShape *shape_info =
                static_cast<const FlatBuffers::Dl::TensorTypeAndShape *>(pInfo->value_info_type()->value());
            ret = fbs_dtype_to_dl_dtype(shape_info->elem_type());
        } else {
            ESP_LOGE(TAG, "The value info type(%d) don't be support now", pInfo->value_info_type()->value_type());
        }
    }

    return ret;
}

std::vector<int> FbsModel::get_value_info_shape(std::string var_name)
{
    std::vector<int> shape;

    if (var_name.empty() || m_name_to_value_info_map.empty()) {
        ESP_LOGW(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return shape;
    }

    auto name2valueInfo_iter = m_name_to_value_info_map.find(var_name);
    if (name2valueInfo_iter != m_name_to_value_info_map.end()) {
        const FlatBuffers::Dl::ValueInfo *pInfo = (const FlatBuffers::Dl::ValueInfo *)name2valueInfo_iter->second;
        if (pInfo->value_info_type()->value_type() == FlatBuffers::Dl::TypeInfoValue_tensor_type) {
            const FlatBuffers::Dl::TensorTypeAndShape *shape_info =
                static_cast<const FlatBuffers::Dl::TensorTypeAndShape *>(pInfo->value_info_type()->value());
            int size = shape_info->shape()->dim()->size();
            shape.reserve(size);
            for (int i = 0; i < size; i++) {
                if (shape_info->shape()->dim()->Get(i)->value()->dim_type() ==
                    FlatBuffers::Dl::DimensionValueType_VALUE) {
                    shape.emplace_back(static_cast<int>(shape_info->shape()->dim()->Get(i)->value()->dim_value()));
                } else {
                    ESP_LOGE(TAG, "The dim type don't be support now");
                }
            }
        } else {
            ESP_LOGE(TAG, "The value info type(%d) don't be support now", pInfo->value_info_type()->value_type());
        }
    }
    return shape;
}

int FbsModel::get_value_info_exponent(std::string var_name)
{
    int exponent = 0;
    if (var_name.empty() || m_name_to_value_info_map.empty()) {
        ESP_LOGW(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return exponent;
    }

    auto name2valueInfo_iter = m_name_to_value_info_map.find(var_name);
    if (name2valueInfo_iter != m_name_to_value_info_map.end()) {
        const FlatBuffers::Dl::ValueInfo *pInfo = (const FlatBuffers::Dl::ValueInfo *)name2valueInfo_iter->second;
        if (pInfo != nullptr) {
            if (pInfo->exponents()->size() == 1) {
                exponent = static_cast<int>(pInfo->exponents()->Get(0));
            } else if (pInfo->exponents()->size() > 1) {
                exponent = static_cast<int>(pInfo->exponents()->Get(0));
                ESP_LOGW(TAG,
                         "The exponents size(%lu) of ValueInfo(%s) is not equal to 1",
                         pInfo->exponents()->size(),
                         var_name.c_str());
            } else {
                ESP_LOGE(TAG, "The exponents size of ValueInfo(%s) is 0", var_name.c_str());
            }
        }
    }
    return exponent;
}

const void *FbsModel::get_test_input_tensor_raw_data(std::string tensor_name)
{
    const void *ret = nullptr;

    if (tensor_name.empty() || m_name_to_test_inputs_value_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2test_input_iter = m_name_to_test_inputs_value_map.find(tensor_name);
    if (name2test_input_iter != m_name_to_test_inputs_value_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2test_input_iter->second;
        if (pTensor->raw_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->raw_data()->Data());
        } else {
            ESP_LOGW(TAG, "The tensor(%s) raw data is nullptr", tensor_name.c_str());
        }
    }
    return ret;
}

dl::TensorBase *FbsModel::get_test_input_tensor(std::string tensor_name)
{
    const void *pRawData = nullptr;
    dl::dtype_t dtype;
    std::vector<int> shape;
    std::vector<int> exponents;
    dl::TensorBase *tensor = nullptr;

    if (tensor_name.empty() || m_name_to_test_inputs_value_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return tensor;
    }

    auto name2test_input_iter = m_name_to_test_inputs_value_map.find(tensor_name);
    if (name2test_input_iter != m_name_to_test_inputs_value_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2test_input_iter->second;
        if (pTensor == nullptr) {
            ESP_LOGE(TAG, "The test input tensor(%s) is nullptr", tensor_name.c_str());
            return tensor;
        }
        // raw data
        if (pTensor->raw_data() != nullptr) {
            pRawData = reinterpret_cast<const void *>(pTensor->raw_data()->Data());
        } else {
            ESP_LOGW(TAG, "The tensor(%s) raw data is nullptr", tensor_name.c_str());
        }
        // shape
        for (auto iter = pTensor->dims()->begin(); iter != pTensor->dims()->end(); iter++) {
            shape.push_back(static_cast<int>(*iter));
        }

        // data type
        dtype = fbs_dtype_to_dl_dtype(pTensor->data_type());

        // exponent
        exponents.reserve(pTensor->exponents()->size());
        for (auto iter = pTensor->exponents()->begin(); iter != pTensor->exponents()->end(); iter++) {
            exponents.push_back(static_cast<int>(*iter));
        }
        tensor = create_tensor(shape, pRawData, exponents, dtype, false, MALLOC_CAP_DEFAULT, tensor_name.c_str());
    }
    return tensor;
}

const void *FbsModel::get_test_output_tensor_raw_data(std::string tensor_name)
{
    const void *ret = nullptr;

    if (tensor_name.empty() || m_name_to_test_outputs_value_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return ret;
    }

    auto name2test_output_iter = m_name_to_test_outputs_value_map.find(tensor_name);
    if (name2test_output_iter != m_name_to_test_outputs_value_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2test_output_iter->second;
        if (pTensor->raw_data() != nullptr) {
            ret = reinterpret_cast<const void *>(pTensor->raw_data()->Data());
        } else {
            ESP_LOGW(TAG, "The tensor(%s) raw data is nullptr", tensor_name.c_str());
        }
    }
    return ret;
}

dl::TensorBase *FbsModel::get_test_output_tensor(std::string tensor_name)
{
    const void *pRawData = nullptr;
    dl::dtype_t dtype;
    std::vector<int> shape;
    std::vector<int> exponents;
    dl::TensorBase *tensor = nullptr;

    if (tensor_name.empty() || m_name_to_test_outputs_value_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return tensor;
    }

    auto name2test_output_iter = m_name_to_test_outputs_value_map.find(tensor_name);
    if (name2test_output_iter != m_name_to_test_outputs_value_map.end()) {
        const FlatBuffers::Dl::Tensor *pTensor = (const FlatBuffers::Dl::Tensor *)name2test_output_iter->second;
        if (pTensor == nullptr) {
            ESP_LOGE(TAG, "The test output tensor(%s) is nullptr", tensor_name.c_str());
            return tensor;
        }
        // raw data
        if (pTensor->raw_data() != nullptr) {
            pRawData = reinterpret_cast<const void *>(pTensor->raw_data()->Data());
        } else {
            ESP_LOGW(TAG, "The tensor(%s) raw data is nullptr", tensor_name.c_str());
        }
        // shape
        for (auto iter = pTensor->dims()->begin(); iter != pTensor->dims()->end(); iter++) {
            shape.push_back(static_cast<int>(*iter));
        }

        // data type
        dtype = fbs_dtype_to_dl_dtype(pTensor->data_type());

        // exponent
        exponents.reserve(pTensor->exponents()->size());
        for (auto iter = pTensor->exponents()->begin(); iter != pTensor->exponents()->end(); iter++) {
            exponents.push_back(static_cast<int>(*iter));
        }
        tensor = create_tensor(shape, pRawData, exponents, dtype, false, MALLOC_CAP_DEFAULT, tensor_name.c_str());
    }
    return tensor;
}

std::vector<std::string> FbsModel::get_test_outputs_name()
{
    std::vector<std::string> test_outputs_name;

    if (m_name_to_test_outputs_value_map.empty()) {
        ESP_LOGE(TAG, "(%s)The initial parameter is error.", __FUNCTION__);
        return test_outputs_name;
    }

    for (auto iter = m_name_to_test_outputs_value_map.begin(); iter != m_name_to_test_outputs_value_map.end(); iter++) {
        test_outputs_name.push_back(iter->first);
    }
    return test_outputs_name;
}

std::vector<std::string> FbsModel::get_graph_inputs()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    std::vector<std::string> inputs;
    if (!fbs_model || !fbs_model->graph()) {
        return inputs;
    }

    for (auto iter = fbs_model->graph()->input()->begin(); iter != fbs_model->graph()->input()->end(); iter++) {
        inputs.push_back((*iter)->name()->str());
    }
    return inputs;
}

std::vector<std::string> FbsModel::get_graph_outputs()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    std::vector<std::string> outputs;
    if (!fbs_model || !fbs_model->graph()) {
        return outputs;
    }

    for (auto iter = fbs_model->graph()->output()->begin(); iter != fbs_model->graph()->output()->end(); iter++) {
        outputs.push_back((*iter)->name()->str());
    }
    return outputs;
}

std::string FbsModel::get_model_name()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    std::string name = "";
    if (fbs_model) {
        const ::flatbuffers::String *str = fbs_model->graph()->name();
        if (str) {
            name = str->str();
        }
    }

    return name;
}

int64_t FbsModel::get_model_version()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    if (fbs_model) {
        return fbs_model->model_version();
    }

    return 0;
}

std::string FbsModel::get_model_doc_string()
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    std::string doc_string = "";
    if (fbs_model) {
        const ::flatbuffers::String *doc = fbs_model->doc_string();
        if (doc) {
            doc_string = doc->str();
        }
    }

    return doc_string;
}

std::string FbsModel::get_model_metadata_prop(const std::string &key)
{
    const FlatBuffers::Dl::Model *fbs_model = (const FlatBuffers::Dl::Model *)m_model;
    std::string value = "";
    if (fbs_model && fbs_model->metadata_props()) {
        for (auto metadata_prop_iter = fbs_model->metadata_props()->begin();
             metadata_prop_iter != fbs_model->metadata_props()->end();
             metadata_prop_iter++) {
            if ((*metadata_prop_iter)->key()->str() == key) {
                value = metadata_prop_iter->value()->str();
                break;
            }
        }
    }

    return value;
}

void FbsModel::get_model_size(size_t *internal_size, size_t *psram_size, size_t *psram_rodata_size, size_t *flash_size)
{
    *internal_size = 0;
    *psram_size = 0;
    *psram_rodata_size = 0;
    *flash_size = 0;
    void *data = (void *)(m_data);
    dl::memory_addr_type_t mem_type = dl::tool::memory_addr_type(data);
    switch (m_location) {
    case MODEL_LOCATION_IN_FLASH_RODATA:
    case MODEL_LOCATION_IN_FLASH_PARTITION:
        *flash_size = m_size;
        if (m_encrypt) {
            if (mem_type == dl::MEMORY_ADDR_PSRAM) {
                *psram_size = m_size;
            } else {
                *internal_size = m_size;
            }
        }
        if (m_rodata_move) {
            *psram_rodata_size = m_size;
        }
        break;
    case MODEL_LOCATION_IN_SDCARD:
        if (mem_type == dl::MEMORY_ADDR_PSRAM) {
            *psram_size = m_size;
        } else {
            *internal_size = m_size;
        }
        break;
    }
}

} // namespace fbs
