#include <stdint.h>

#include "dl_memory_manager_greedy.hpp"
#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "fbs_model.hpp"
#include <format>

static const char *TAG = "dl::Model";

namespace dl {

Model::Model(const char *rodata_address_or_partition_label_or_path,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             const uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    m_internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m_model_context = new ModelContext();
    if (this->load(rodata_address_or_partition_label_or_path, location, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    m_internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(const char *rodata_address_or_partition_label_or_path,
             int model_index,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             const uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    m_internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m_model_context = new ModelContext();
    if (this->load(rodata_address_or_partition_label_or_path, location, model_index, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    m_internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(const char *rodata_address_or_partition_label_or_path,
             const char *model_name,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             const uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    m_internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m_model_context = new ModelContext();
    if (this->load(rodata_address_or_partition_label_or_path, location, model_name, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    m_internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(fbs::FbsModel *fbs_model, int max_internal_size, memory_manager_t mm_type)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    m_internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m_model_context = new ModelContext();
    if (this->load(fbs_model) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    m_internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::~Model()
{
    // If fbs_loader is NULL, this means fbs_model is created outside this class. So don't delete it.
    if (m_fbs_loader) {
        delete m_fbs_loader;

        if (m_fbs_model) {
            delete m_fbs_model;
        }
    }

    if (m_model_context) {
        delete m_model_context;
    }
    if (!m_execution_plan.empty()) {
        for (int i = 0; i < m_execution_plan.size(); i++) {
            delete m_execution_plan[i];
        }
    }
}

esp_err_t Model::load(const char *name, fbs::model_location_type_t location, const uint8_t *key, bool param_copy)
{
    m_fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(m_fbs_loader->load(key, param_copy));
}

esp_err_t Model::load(
    const char *name, fbs::model_location_type_t location, int model_index, const uint8_t *key, bool param_copy)
{
    m_fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(m_fbs_loader->load(model_index, key, param_copy));
}

esp_err_t Model::load(
    const char *name, fbs::model_location_type_t location, const char *model_name, const uint8_t *key, bool param_copy)
{
    m_fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(m_fbs_loader->load(model_name, key, param_copy));
}

esp_err_t Model::load(fbs::FbsModel *fbs_model)
{
    esp_err_t ret = ESP_OK;
    if (!fbs_model) {
        ESP_LOGE(TAG, "Fail to load model");
        ret = ESP_FAIL;
        return ret;
    }
    m_fbs_model = fbs_model; // fbs_model is created by fbs_loader, so we don't need to delete it.
    m_fbs_model->load_map();
    m_name = m_fbs_model->get_model_name();
    m_version = m_fbs_model->get_model_version();
    m_doc_string = m_fbs_model->get_model_doc_string();

    // Construct the execution plan.
    m_execution_plan.clear();
    dl::module::ModuleCreator *module_creator = dl::module::ModuleCreator::get_instance();
    m_model_context->clear();
    std::vector<std::string> op_inputs;
    std::vector<std::string> op_outputs;

    std::vector<std::string> sorted_nodes = m_fbs_model->topological_sort();
    for (int i = 0; i < sorted_nodes.size(); i++) {
        std::string node_name = sorted_nodes[i];

        // Create and add module
        std::string op_type = m_fbs_model->get_operation_type(node_name);
        if (op_type.empty()) {
            ESP_LOGE(TAG, "Can not find the operation %s", node_name.c_str());
            ret = ESP_FAIL;
            break;
        }
        dl::module::Module *module = module_creator->create(m_fbs_model, op_type, node_name);
        if (!module) {
            ESP_LOGE(TAG, "Do not support %s, please implement and register it first.", op_type.c_str());
            ret = ESP_FAIL;
            break;
        }
        m_execution_plan.push_back(module);

        // Add inputs and outputs
        m_fbs_model->get_operation_inputs_and_outputs(node_name, op_inputs, op_outputs);
        int index = 0;
        for (int j = 0; j < op_inputs.size(); j++) {
            bool is_parameter = m_fbs_model->is_parameter(op_inputs[j]);
            if (is_parameter || op_inputs[j].empty()) {
                index =
                    m_model_context->add_tensor(op_inputs[j], true, m_fbs_model->get_operation_parameter(node_name, j));
            } else {
                index = m_model_context->add_tensor(op_inputs[j], false, nullptr);
            }
            module->m_inputs_index.push_back(index); // assign input index of module
        }

        for (int j = 0; j < op_outputs.size(); j++) {
            index = m_model_context->add_tensor(op_outputs[j], false, nullptr);
            module->m_outputs_index.push_back(index); // assign output index of
        }
    }

    return ret;
}

void Model::build(size_t max_internal_size, memory_manager_t mm_type, bool preload)
{
    // If memory manager has been created, delete it and reset all modules
    m_fbs_model->load_map();
    MemoryManagerBase *memory_manager = nullptr;

    if (mm_type == MEMORY_MANAGER_GREEDY) {
        memory_manager = new MemoryManagerGreedy(max_internal_size);
    } else {
        ESP_LOGW(TAG, "Memory manager(%d) is not supported yet. Use MemoryManagerGreedy instead.", mm_type);
        memory_manager = new MemoryManagerGreedy(max_internal_size);
    }
    memory_manager->alloc(m_fbs_model, m_execution_plan, m_model_context);

    // get the TensorBase* of inputs and outputs
    std::vector<std::string> inputs_tmp = m_fbs_model->get_graph_inputs();
    std::vector<std::string> outputs_tmp = m_fbs_model->get_graph_outputs();
    m_inputs.clear();
    m_outputs.clear();
    for (int i = 0; i < inputs_tmp.size(); i++) {
        TensorBase *input_tensor = this->get_intermediate(inputs_tmp[i]);
        m_inputs.emplace(inputs_tmp[i], input_tensor);
    }
    for (int i = 0; i < outputs_tmp.size(); i++) {
        TensorBase *output_tensor = this->get_intermediate(outputs_tmp[i]);
        m_outputs.emplace(outputs_tmp[i], output_tensor);
    }

    m_fbs_model->clear_map();
    delete memory_manager;
}

void Model::run(runtime_mode_t mode)
{
    // execute each module.
    for (int i = 0; i < m_execution_plan.size(); i++) {
        dl::module::Module *module = m_execution_plan[i];
        if (module) {
            module->forward(m_model_context, mode);
        } else {
            break;
        }
    }
}

void Model::run(TensorBase *input, runtime_mode_t mode)
{
    if (m_inputs.size() != 1) {
        ESP_LOGW(TAG, "The inputs of model is not just one! This API will assign data to first input");
    }

    TensorBase *model_input = m_inputs.begin()->second;
    if (!model_input->assign(input)) {
        ESP_LOGE(TAG, "Assign input failed");
        return;
    }

    // execute each module.
    this->run(mode);
}

void Model::run(std::map<std::string, TensorBase *> &user_inputs,
                runtime_mode_t mode,
                std::map<std::string, TensorBase *> user_outputs)
{
    if (user_inputs.size() != m_inputs.size()) {
        ESP_LOGE(TAG,
                 "The size of user_inputs(%d) don't equal with the size of model inputs(%d).",
                 user_inputs.size(),
                 m_inputs.size());
        return;
    }

    for (auto user_inputs_iter = user_inputs.begin(); user_inputs_iter != user_inputs.end(); user_inputs_iter++) {
        std::string user_input_name = user_inputs_iter->first;
        TensorBase *user_input_tensor = user_inputs_iter->second;
        auto graph_input_iter = m_inputs.find(user_input_name);
        if (graph_input_iter == m_inputs.end()) {
            ESP_LOGE(TAG, "The input name(%s) isn't graph input.", user_input_name.c_str());
            return;
        }
        TensorBase *graph_input_tensor = graph_input_iter->second;
        if (!graph_input_tensor->assign(user_input_tensor)) {
            ESP_LOGE(TAG, "Assign input failed");
            return;
        }
    }

    // execute each module.
    for (int i = 0; i < m_execution_plan.size(); i++) {
        dl::module::Module *module = m_execution_plan[i];
        if (module) {
            module->forward(m_model_context, mode);
            // get the intermediate tensor for debug.
            if (!user_outputs.empty()) {
                for (auto user_outputs_iter = user_outputs.begin(); user_outputs_iter != user_outputs.end();
                     user_outputs_iter++) {
                    int user_tensor_index =
                        m_model_context->get_tensor_index(const_cast<std::string &>(user_outputs_iter->first));
                    if (user_tensor_index >= 0) {
                        std::vector<int> outputs_index = module->get_outputs_index();
                        for (int i = 0; i < outputs_index.size(); i++) {
                            if (user_tensor_index == outputs_index[i]) {
                                user_outputs_iter->second->assign(m_model_context->m_variables[user_tensor_index]);
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            break;
        }
    }
    return;
}

std::map<std::string, TensorBase *> &Model::get_inputs()
{
    return m_inputs;
}

TensorBase *Model::get_input()
{
    assert(m_inputs.size() == 1);
    return m_inputs.begin()->second;
}

TensorBase *Model::get_input(const std::string &name)
{
    if (name.empty()) {
        return get_input();
    }
    auto it = m_inputs.find(name);
    if (it == m_inputs.end()) {
        ESP_LOGE(TAG, "%s not found in inputs.", name.c_str());
        return nullptr;
    }
    return it->second;
}

TensorBase *Model::get_intermediate(const std::string &name)
{
    if (name.empty()) {
        ESP_LOGE(TAG, "Invalid name.");
        return nullptr;
    }
    return m_model_context->get_tensor(name);
}

std::map<std::string, TensorBase *> &Model::get_outputs()
{
    return m_outputs;
}

TensorBase *Model::get_output()
{
    assert(m_outputs.size() == 1);
    return m_outputs.begin()->second;
}

TensorBase *Model::get_output(const std::string &name)
{
    if (name.empty()) {
        return get_output();
    }
    auto it = m_outputs.find(name);
    if (it == m_outputs.end()) {
        ESP_LOGE(TAG, "%s not found in outputs.", name.c_str());
        return nullptr;
    }
    return it->second;
}

std::string Model::get_metadata_prop(const std::string &key)
{
    if (!m_fbs_model) {
        return "";
    }
    return m_fbs_model->get_model_metadata_prop(key);
}

void Model::print()
{
    if (!m_execution_plan.empty()) {
        for (int i = 0; i < m_execution_plan.size(); i++) {
            if (m_execution_plan[i]) {
                ESP_LOGI(TAG, "------------------------------- %d -------------------------------", i);
                if (m_execution_plan[i]) {
                    m_execution_plan[i]->print();
                } else {
                    break;
                }
            }
        }
        ESP_LOGI(TAG, "-------------------------------------------------------------\n");
    }
}

void Model::minimize()
{
    ESP_LOGW(TAG,
             "Minimize() will delete variables not used in model inference, which will make it impossible to test "
             "or debug the model.");

    m_internal_size += heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m_model_context->minimize();
    m_fbs_model->clear_map();
    m_internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    dl::module::ModuleCreator::get_instance()->clear();
}

esp_err_t Model::test()
{
    printf("\n");
    std::vector<TensorBase *> test_tensors_cache;
    m_fbs_model->load_map();
    std::map<std::string, TensorBase *> &graph_inputs = get_inputs();
    for (auto graph_inputs_iter = graph_inputs.begin(); graph_inputs_iter != graph_inputs.end(); graph_inputs_iter++) {
        std::string input_name = graph_inputs_iter->first;
        TensorBase *test_input = m_fbs_model->get_test_input_tensor(input_name);
        if (!test_input) {
            ESP_LOGE(TAG,
                     "Model input %s doesn't have a corresponding test input. Please enable export_test_values option "
                     "in esp-ppq when export espdl model.",
                     input_name.c_str());
            return ESP_FAIL;
        }
        if (!graph_inputs_iter->second->assign(test_input)) {
            ESP_LOGE(TAG, "Assign input failed");
            delete test_input;
            m_fbs_model->clear_map();
            return ESP_FAIL;
        }
        test_tensors_cache.emplace_back(test_input);
    }

    std::vector<std::string> test_outputs_name = m_fbs_model->get_test_outputs_name();
    std::vector<int> test_outputs_index;
    assert(test_outputs_name.size() > 0);
    for (const auto &name : test_outputs_name) {
        int index = m_model_context->get_tensor_index(name);
        if (index == -1) {
            ESP_LOGE(TAG, "There's no intermediate result or output named %s in model.", name.c_str());
            return ESP_FAIL;
        }
        test_outputs_index.emplace_back(index);
    }
    for (int i = 0; i < m_execution_plan.size(); i++) {
        dl::module::Module *module = m_execution_plan[i];
        module->forward(m_model_context, RUNTIME_MODE_SINGLE_CORE);
        std::vector<int> module_outputs_index = module->get_outputs_index();
        for (int index : module_outputs_index) {
            auto iter = std::find(test_outputs_index.begin(), test_outputs_index.end(), index);
            if (iter != test_outputs_index.end()) {
                size_t iter_index = std::distance(test_outputs_index.begin(), iter);
                std::string output_name = test_outputs_name[iter_index];
                ESP_LOGI(TAG, "Testing output %s.", output_name.c_str());
                dl::TensorBase *output = m_model_context->m_variables[index];
                dl::TensorBase *output_gt = m_fbs_model->get_test_output_tensor(output_name);
                assert(output);
                assert(output_gt);
                if (output->get_dtype() == DATA_TYPE_INT16 || output->get_dtype() == DATA_TYPE_UINT16) {
                    // The int16 quantization cannot be fully aligned, and there may be rounding errors of +-1.
                    if (!output->equal(output_gt, 1 + 1e-5, true)) {
                        ESP_LOGE(TAG, "Test output %s does not match\n", output_name.c_str());
                        delete output_gt;
                        m_fbs_model->clear_map();
                        return ESP_FAIL;
                    }
                } else {
                    if (!output->equal(output_gt, 1e-5, true)) {
                        ESP_LOGE(TAG, "Test output %s does not match\n", output_name.c_str());
                        delete output_gt;
                        m_fbs_model->clear_map();
                        return ESP_FAIL;
                    }
                }
                test_tensors_cache.emplace_back(output_gt);
            }
        }
    }

    for (auto &test_tensor : test_tensors_cache) {
        delete test_tensor;
    }
    m_fbs_model->clear_map();
    ESP_LOGI(TAG, "Test Pass!");
    return ESP_OK;
}

std::map<std::string, mem_info_t> Model::get_memory_info()
{
    std::map<std::string, mem_info_t> info;

    size_t psram_rodata_size;

    m_fbs_model->get_model_size(
        &info["fbs_model"].internal, &info["fbs_model"].psram, &psram_rodata_size, &info["fbs_model"].flash);
    info["fbs_model"].psram += psram_rodata_size;

    m_model_context->get_variable_memory_size(info["variable"]);
    m_model_context->get_parameter_memory_size(info["parameter"], false);
    m_model_context->get_parameter_memory_size(info["parameter_copy"], true);

    info["total"].psram = m_psram_size + psram_rodata_size;
    info["total"].internal = m_internal_size;
    info["total"].flash = info["fbs_model"].flash;

    if (!m_fbs_model->m_param_copy && std::max(info["parameter_copy"].psram, info["parameter_copy"].internal)) {
        info["total"] += info["parameter_copy"];
    }

    info["others"] = info["total"] - info["fbs_model"] - info["parameter_copy"] - info["variable"];

    if (info["fbs_model"].flash > 0 && info["parameter"].flash == 0) {
        info["parameter"].flash = std::max(info["parameter"].internal, info["parameter"].psram) +
            std::max(info["parameter_copy"].internal, info["parameter_copy"].psram);
    }

    return info;
}

std::map<std::string, module_info> Model::get_module_info()
{
    std::map<std::string, module_info> module_info;
    std::vector<std::string> sorted_nodes = m_fbs_model->topological_sort();
    assert(sorted_nodes.size() == m_execution_plan.size());
    DL_LOG_LATENCY_INIT();
    uint32_t total_latency = 0;
    m_fbs_model->load_map();
    for (int i = 0; i < sorted_nodes.size(); i++) {
        std::string module_name = sorted_nodes[i];
        std::string module_type = m_fbs_model->get_operation_type(module_name);
        DL_LOG_LATENCY_START();
        m_execution_plan[i]->forward(m_model_context, RUNTIME_MODE_SINGLE_CORE);
        DL_LOG_LATENCY_END();
        uint32_t module_latency = DL_LOG_LATENCY_GET();
        total_latency += module_latency;
        module_info[module_name] = {module_type, module_latency};
    }
    m_fbs_model->clear_map();
    module_info["total"] = {"", total_latency};
    return module_info;
}

static std::string gen_sep_str(std::initializer_list<size_t> width_list)
{
    std::string sep = "+-";
    int i = 0;
    for (auto width : width_list) {
        sep.append(width, '-');
        if (i != width_list.size() - 1) {
            sep += "-+-";
        } else {
            sep += "-+";
        }
        i++;
    }
    return sep;
};

static void print_table_name(const std::string &table_name, const std::string &sep)
{
    int n_space = (sep.size() - 2 - table_name.size());
    int n_front_space = n_space / 2;
    int n_back_space = n_space - n_space / 2;
    ESP_LOGI(TAG, "%s", sep.c_str());
    ESP_LOGI(TAG,
             "|%s%s%s|",
             std::string(n_front_space, ' ').c_str(),
             table_name.c_str(),
             std::string(n_back_space, ' ').c_str());
    ESP_LOGI(TAG, "%s", sep.c_str());
}

static void print_memory_info(const std::map<std::string, mem_info_t> &info)
{
    std::string table_name = "memory summary";
    std::vector<std::string> row_headers = {"fbs_model", "parameter", "parameter_copy", "variable", "others", "total"};
    std::vector<std::string> col_headers = {"", "internal RAM", "PSRAM", "FLASH"};

    // get_col_width
    size_t col0_width = strlen("parameter_copy");
    std::string sub_prefix = "└── ";
    auto get_fmt_size = [&sub_prefix](size_t size, bool sub_header) -> std::string {
        std::string fmt_size = std::format("{:<.2f}KB", size / 1024.f);
        if (sub_header) {
            fmt_size = (fmt_size == "0.00KB") ? "" : sub_prefix + fmt_size;
        } else if (fmt_size == "0.00KB") {
            fmt_size = "─";
        }
        return fmt_size;
    };
    size_t col1_width = std::max(std::max(get_fmt_size(info.at("total").internal, false).size(),
                                          get_fmt_size(info.at("parameter").internal, true).size()),
                                 col_headers[1].size());
    size_t col2_width = std::max(std::max(get_fmt_size(info.at("total").psram, false).size(),
                                          get_fmt_size(info.at("parameter").psram, true).size()),
                                 col_headers[2].size());
    size_t col3_width = std::max(std::max(get_fmt_size(info.at("total").flash, false).size(),
                                          get_fmt_size(info.at("parameter").flash, true).size()),
                                 col_headers[3].size());

    std::string sep = gen_sep_str({col0_width, col1_width, col2_width, col3_width});

    // table name
    print_table_name(table_name, sep);
    // col_headers
    ESP_LOGI(TAG,
             "| %-*s | %-*s | %-*s | %-*s |",
             col0_width,
             col_headers[0].c_str(),
             col1_width,
             col_headers[1].c_str(),
             col2_width,
             col_headers[2].c_str(),
             col3_width,
             col_headers[3].c_str());
    ESP_LOGI(TAG, "%s", sep.c_str());
    // body
    for (int i = 0; i < row_headers.size(); i++) {
        std::string row_header = row_headers[i];
        bool sub_header = (row_header == "parameter");
        std::string row = std::format("| {:<{}} | {:<{}} | {:<{}} | {:<{}} |",
                                      sub_header ? (sub_prefix + row_header) : row_header,
                                      col0_width,
                                      get_fmt_size(info.at(row_header).internal, sub_header),
                                      col1_width,
                                      get_fmt_size(info.at(row_header).psram, sub_header),
                                      col2_width,
                                      get_fmt_size(info.at(row_header).flash, sub_header),
                                      col3_width);
        ESP_LOGI(TAG, "%s", row.c_str());
        if (i == row_headers.size() - 1 || row_headers[i + 1] != "parameter") {
            ESP_LOGI(TAG, "%s", sep.c_str());
        }
    }
}

void Model::print_module_info(const std::map<std::string, module_info> &info, bool sort_module_by_latency)
{
    std::string table_name = "module summary";
    std::vector<std::string> col_headers = {"name", "type", "latency"};
    size_t col0_width = col_headers[0].size();
    size_t col1_width = col_headers[1].size();
    for (const auto &module_info : info) {
        col0_width = std::max(col0_width, module_info.first.size());
        col1_width = std::max(col1_width, module_info.second.type.size());
    }
    char latency_str[16];
#if DL_LOG_LATENCY_UNIT
    snprintf(latency_str, sizeof(latency_str), "%ldcycle", info.at("total").latency);
#else
    snprintf(latency_str, sizeof(latency_str), "%ldus", info.at("total").latency);
#endif
    size_t col2_width = std::max(col_headers[2].size(), strlen(latency_str));
    std::string sep = gen_sep_str({col0_width, col1_width, col2_width});

    // table name
    print_table_name(table_name, sep);
    // col_headers
    ESP_LOGI(TAG,
             "| %-*s | %-*s | %-*s |",
             col0_width,
             col_headers[0].c_str(),
             col1_width,
             col_headers[1].c_str(),
             col2_width,
             col_headers[2].c_str());
    ESP_LOGI(TAG, "%s", sep.c_str());
    // body
    if (sort_module_by_latency) {
        std::vector<std::pair<std::string, module_info>> info_vec(info.begin(), info.end());
        std::sort(info_vec.begin(), info_vec.end(), [](const auto &a, const auto &b) {
            module_info info_a = std::get<1>(a);
            module_info info_b = std::get<1>(b);
            return info_a.latency > info_b.latency;
        });
        for (const auto &info_pair : info_vec) {
            std::string name = std::get<0>(info_pair);
            std::string type = std::get<1>(info_pair).type;
            uint32_t latency = std::get<1>(info_pair).latency;
#if DL_LOG_LATENCY_UNIT
            snprintf(latency_str, sizeof(latency_str), "%ldcycle", latency);
#else
            snprintf(latency_str, sizeof(latency_str), "%ldus", latency);
#endif
            ESP_LOGI(TAG,
                     "| %-*s | %-*s | %-*s |",
                     col0_width,
                     name.c_str(),
                     col1_width,
                     type.c_str(),
                     col2_width,
                     latency_str);
            ESP_LOGI(TAG, "%s", sep.c_str());
        }
    } else {
        std::vector<std::string> sorted_nodes = m_fbs_model->topological_sort();
        sorted_nodes.emplace_back("total");
        for (const auto &key : sorted_nodes) {
#if DL_LOG_LATENCY_UNIT
            snprintf(latency_str, sizeof(latency_str), "%ldcycle", info.at(key).latency);
#else
            snprintf(latency_str, sizeof(latency_str), "%ldus", info.at(key).latency);
#endif
            ESP_LOGI(TAG,
                     "| %-*s | %-*s | %-*s |",
                     col0_width,
                     key.c_str(),
                     col1_width,
                     info.at(key).type.c_str(),
                     col2_width,
                     latency_str);
            ESP_LOGI(TAG, "%s", sep.c_str());
        }
    }
}

void Model::profile_memory()
{
    printf("\n");
    if (m_doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", m_name.c_str(), m_version);
    } else {
        ESP_LOGI(TAG, "model:%s, version:%lld, description:%s", m_name.c_str(), m_version, m_doc_string.c_str());
    }
    auto info = get_memory_info();

    if (m_fbs_loader) {
        ESP_LOGI(TAG, "%s", m_fbs_loader->get_model_location_string());
    }
    print_memory_info(info);
    printf("\n");
}

void Model::profile_module(bool sort_module_by_latency)
{
    printf("\n");
    if (m_doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", m_name.c_str(), m_version);
    } else {
        ESP_LOGI(TAG, "model:%s, version:%lld, description:%s", m_name.c_str(), m_version, m_doc_string.c_str());
    }
    auto info = get_module_info();
    print_module_info(info);
    printf("\n");
}

void Model::profile(bool sort_module_by_latency)
{
    printf("\n");
    if (m_doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", m_name.c_str(), m_version);
    } else {
        ESP_LOGI(TAG, "model:%s, version:%lld, description:%s", m_name.c_str(), m_version, m_doc_string.c_str());
    }
    if (m_fbs_loader) {
        ESP_LOGI(TAG, "%s", m_fbs_loader->get_model_location_string());
    }
    auto mem_info = get_memory_info();
    print_memory_info(mem_info);
    printf("\n");
    auto module_info = get_module_info();
    print_module_info(module_info, sort_module_by_latency);
    printf("\n");
}

} // namespace dl
