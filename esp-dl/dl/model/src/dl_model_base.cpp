#include <stdint.h>

#include "dl_memory_manager_greedy.hpp"
#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "fbs_model.hpp"

static const char *TAG = "dl::Model";

namespace dl {

Model::Model(const char *name,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (this->load(name, location, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(const char *name,
             int model_index,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (this->load(name, location, model_index, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(const char *name,
             const char *model_name,
             fbs::model_location_type_t location,
             int max_internal_size,
             memory_manager_t mm_type,
             uint8_t *key,
             bool param_copy)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (this->load(name, location, model_name, key, param_copy) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::Model(fbs::FbsModel *fbs_model, int max_internal_size, memory_manager_t mm_type)
{
    dl::module::ModuleCreator::get_instance()->register_dl_modules();
    internal_size = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (this->load(fbs_model) == ESP_OK) {
        this->build(max_internal_size, mm_type);
    }
    internal_size -= heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    psram_size -= heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

Model::~Model()
{
    // If fbs_loader is NULL, this means fbs_model is created outside this class. So don't delete it.
    if (fbs_loader) {
        delete fbs_loader;

        if (fbs_model) {
            delete fbs_model;
        }
    }

    if (memory_manager) {
        delete memory_manager;
    }
    if (!execution_plan.empty()) {
        for (int i = 0; i < execution_plan.size(); i++) {
            delete execution_plan[i];
        }
    }
}

esp_err_t Model::load(const char *name, fbs::model_location_type_t location, uint8_t *key, bool param_copy)
{
    fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(fbs_loader->load(key, param_copy));
}

esp_err_t Model::load(
    const char *name, fbs::model_location_type_t location, int model_index, uint8_t *key, bool param_copy)
{
    fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(fbs_loader->load(model_index, key, param_copy));
}

esp_err_t Model::load(
    const char *name, fbs::model_location_type_t location, const char *model_name, uint8_t *key, bool param_copy)
{
    fbs_loader = new fbs::FbsLoader(name, location);
    return this->load(fbs_loader->load(model_name, key, param_copy));
}

esp_err_t Model::load(fbs::FbsModel *fbs_model)
{
    esp_err_t ret = ESP_OK;
    if (!fbs_model) {
        ESP_LOGE(TAG, "Fail to load model");
        ret = ESP_FAIL;
        return ret;
    }
    this->fbs_model = fbs_model; // fbs_model is created by fbs_loader, so we don't need to delete it.
    fbs_model->load_map();
    this->name = fbs_model->get_model_name();
    this->version = fbs_model->get_model_version();
    this->doc_string = fbs_model->get_model_doc_string();

    // Construct the execution plan.
    execution_plan.clear();
    dl::module::ModuleCreator *module_creator = dl::module::ModuleCreator::get_instance();

    std::vector<std::string> sorted_nodes = fbs_model->topological_sort();
    for (int i = 0; i < sorted_nodes.size(); i++) {
        std::string node_name = sorted_nodes[i];
        std::string op_type = fbs_model->get_operation_type(node_name);
        if (op_type.empty()) {
            ESP_LOGE(TAG, "Can not find the operation %s", node_name.c_str());
            ret = ESP_FAIL;
            break;
        }
        dl::module::Module *module = module_creator->create(fbs_model, op_type, node_name);
        if (!module) {
            ESP_LOGE(TAG, "Do not support %s, please implement and register it first.", op_type.c_str());
            ret = ESP_FAIL;
            break;
        }
        execution_plan.push_back(module);
    }

    this->memory_manager = nullptr;
    return ret;
}

void Model::build(size_t max_internal_size, memory_manager_t mm_type, bool preload)
{
    // If memory manager has been created, delete it and reset all modules
    this->fbs_model->load_map();
    if (this->memory_manager) {
        delete this->memory_manager;
        for (int i = 0; i < execution_plan.size(); i++) {
            dl::module::Module *module = execution_plan[i];
            if (module) {
                module->reset();
            }
        }
    }

    if (mm_type == MEMORY_MANAGER_GREEDY) {
        this->memory_manager = new dl::memory::MemoryManagerGreedy(max_internal_size);
    }
    this->memory_manager->alloc(this->fbs_model, this->execution_plan);

    // get the TensorBase* of inputs and outputs
    std::vector<std::string> inputs_tmp = fbs_model->get_graph_inputs();
    std::vector<std::string> outputs_tmp = fbs_model->get_graph_outputs();
    this->inputs.clear();
    this->outputs.clear();
    for (int i = 0; i < inputs_tmp.size(); i++) {
        TensorBase *input_tensor = this->get_intermediate(inputs_tmp[i]);
        this->inputs.emplace(inputs_tmp[i], input_tensor);
    }
    for (int i = 0; i < outputs_tmp.size(); i++) {
        TensorBase *output_tensor = this->get_intermediate(outputs_tmp[i]);
        this->outputs.emplace(outputs_tmp[i], output_tensor);
    }

    this->fbs_model->clear_map();
}

void Model::run(runtime_mode_t mode)
{
    // execute each module.
    for (int i = 0; i < execution_plan.size(); i++) {
        dl::module::Module *module = execution_plan[i];
        if (module) {
            module->forward(this->memory_manager->tensors, mode);
        } else {
            break;
        }
    }
}

void Model::run(TensorBase *input, runtime_mode_t mode)
{
    if (this->inputs.size() != 1) {
        ESP_LOGW(TAG, "The inputs of model is not just one! This API will assign data to first input");
    }

    TensorBase *model_input = this->inputs.begin()->second;
    if (!model_input->assign(input)) {
        ESP_LOGE(TAG, "Assign input failed");
        return;
    }

    // execute each module.
    for (int i = 0; i < execution_plan.size(); i++) {
        dl::module::Module *module = execution_plan[i];
        if (module) {
            module->forward(this->memory_manager->tensors, mode);
        } else {
            break;
        }
    }
}

void Model::run(std::map<std::string, TensorBase *> &user_inputs,
                runtime_mode_t mode,
                std::map<std::string, TensorBase *> user_outputs)
{
    if (user_inputs.size() != this->inputs.size()) {
        ESP_LOGE(TAG,
                 "The size of user_inputs(%d) don't equal with the size of model inputs(%d).",
                 user_inputs.size(),
                 this->inputs.size());
        return;
    }

    for (auto user_inputs_iter = user_inputs.begin(); user_inputs_iter != user_inputs.end(); user_inputs_iter++) {
        std::string user_input_name = user_inputs_iter->first;
        TensorBase *user_input_tensor = user_inputs_iter->second;
        auto graph_input_iter = this->inputs.find(user_input_name);
        if (graph_input_iter == this->inputs.end()) {
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
    for (int i = 0; i < execution_plan.size(); i++) {
        dl::module::Module *module = execution_plan[i];
        if (module) {
            module->forward(this->memory_manager->tensors, mode);
            // get the intermediate tensor for debug.
            if (!user_outputs.empty()) {
                for (auto user_outputs_iter = user_outputs.begin(); user_outputs_iter != user_outputs.end();
                     user_outputs_iter++) {
                    int user_tensor_index =
                        this->memory_manager->get_tensor_index(const_cast<std::string &>(user_outputs_iter->first));
                    if (user_tensor_index >= 0) {
                        std::vector<int> outputs_index = module->get_outputs_index();
                        for (int i = 0; i < outputs_index.size(); i++) {
                            if (user_tensor_index == outputs_index[i]) {
                                user_outputs_iter->second->assign(this->memory_manager->tensors[user_tensor_index]);
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
    return this->inputs;
}

TensorBase *Model::get_intermediate(const std::string &name)
{
    if (name.empty()) {
        ESP_LOGE(TAG, "Invalid name.");
        return nullptr;
    }
    return this->memory_manager->get_tensor(name);
}

std::map<std::string, TensorBase *> &Model::get_outputs()
{
    return this->outputs;
}

void Model::print()
{
    if (!execution_plan.empty()) {
        for (int i = 0; i < execution_plan.size(); i++) {
            if (execution_plan[i]) {
                ESP_LOGI(TAG, "------------------------------- %d -------------------------------", i);
                if (execution_plan[i]) {
                    execution_plan[i]->print();
                } else {
                    break;
                }
            }
        }
        ESP_LOGI(TAG, "-------------------------------------------------------------\n");
    }
}

esp_err_t Model::test()
{
    printf("\n");
    fbs_model->load_map();
    std::map<std::string, TensorBase *> graph_inputs = get_inputs();
    for (auto graph_inputs_iter = graph_inputs.begin(); graph_inputs_iter != graph_inputs.end(); graph_inputs_iter++) {
        std::string input_name = graph_inputs_iter->first;
        TensorBase *test_input = fbs_model->get_test_input_tensor(input_name);
        if (!test_input) {
            ESP_LOGE(TAG,
                     "Model input %s doesn't have a corresponding test input. Please enable export_test_values option "
                     "in esp-ppq when export espdl model.",
                     input_name.c_str());
            return ESP_FAIL;
        }
        if (!graph_inputs_iter->second->assign(test_input)) {
            ESP_LOGE(TAG, "Assign input failed");
            fbs_model->clear_map();
            return ESP_FAIL;
        }
    }

    std::vector<std::string> test_outputs_name = fbs_model->get_test_outputs_name();
    std::vector<int> test_outputs_index;
    assert(test_outputs_name.size() > 0);
    for (const auto &name : test_outputs_name) {
        int index = memory_manager->get_tensor_index(name);
        if (index == -1) {
            ESP_LOGE(TAG, "There's no intermediate result or output named %s in model.", name.c_str());
            return ESP_FAIL;
        }
        test_outputs_index.emplace_back(index);
    }
    for (int i = 0; i < execution_plan.size(); i++) {
        dl::module::Module *module = execution_plan[i];
        module->forward(memory_manager->tensors);
        std::vector<int> module_outputs_index = module->get_outputs_index();
        for (int index : module_outputs_index) {
            auto iter = std::find(test_outputs_index.begin(), test_outputs_index.end(), index);
            if (iter != test_outputs_index.end()) {
                size_t iter_index = std::distance(test_outputs_index.begin(), iter);
                std::string output_name = test_outputs_name[iter_index];
                ESP_LOGI(TAG, "Testing output %s.", output_name.c_str());
                dl::TensorBase *output = memory_manager->tensors[index];
                dl::TensorBase *output_gt = fbs_model->get_test_output_tensor(output_name);
                assert(output);
                assert(output_gt);
                if (output->get_dtype() == DATA_TYPE_INT16 || output->get_dtype() == DATA_TYPE_UINT16) {
                    // The int16 quantization cannot be fully aligned, and there may be rounding errors of +-1.
                    if (!output->equal(output_gt, 1 + 1e-5, true)) {
                        ESP_LOGE(TAG, "Test output %s does not match\n", output_name.c_str());
                        fbs_model->clear_map();
                        return ESP_FAIL;
                    }
                } else {
                    if (!output->equal(output_gt, 1e-5, true)) {
                        ESP_LOGE(TAG, "Test output %s does not match\n", output_name.c_str());
                        fbs_model->clear_map();
                        return ESP_FAIL;
                    }
                }
            }
        }
    }

    fbs_model->clear_map();
    ESP_LOGI(TAG, "Test Pass!");
    return ESP_OK;
}

std::map<std::string, mem_info> Model::get_memory_info()
{
    std::map<std::string, mem_info> info;
    std::vector<std::string> sorted_nodes = fbs_model->topological_sort();
    assert(sorted_nodes.size() == execution_plan.size());

    size_t psram_rodata_size;
    fbs_model->get_model_size(
        &info["fbs_model"].internal, &info["fbs_model"].psram, &psram_rodata_size, &info["fbs_model"].flash);
    info["fbs_model"].psram += psram_rodata_size;

    mem_info param_in_fbs, param_out_fbs;
    for (int i = 0; i < sorted_nodes.size(); i++) {
        execution_plan[i]->get_param_memory_size(&param_in_fbs, &param_out_fbs, fbs_model);
        info["param_in_fbs"] += param_in_fbs;
        info["param_out_fbs"] += param_out_fbs;
    }
    if (info["fbs_model"].flash > 0) {
        info["param_in_fbs"].flash =
            std::max(std::max(info["param_in_fbs"].internal, info["param_in_fbs"].psram), info["param_in_fbs"].flash) +
            std::max(info["param_out_fbs"].internal, info["param_out_fbs"].psram);
    }

    info["mem_manager"].internal = memory_manager->get_internal_size();
    info["mem_manager"].psram = memory_manager->get_psram_size();
    info["mem_manager"].flash = 0;

    info["total"].internal = internal_size;
    info["total"].psram = psram_size;
    info["total"].psram += psram_rodata_size;
    info["total"].flash = info["fbs_model"].flash;

    info["others"].internal = info["total"].internal - info["mem_manager"].internal - info["fbs_model"].internal -
        info["param_out_fbs"].internal;
    info["others"].psram =
        info["total"].psram - info["mem_manager"].psram - info["fbs_model"].psram - info["param_out_fbs"].psram;
    info["others"].flash = 0;
    return info;
}

std::map<std::string, module_info> Model::get_module_info()
{
    std::map<std::string, module_info> module_info;
    std::vector<std::string> sorted_nodes = fbs_model->topological_sort();
    assert(sorted_nodes.size() == execution_plan.size());
    DL_LOG_LATENCY_INIT();
    uint32_t total_latency = 0;
    fbs_model->load_map();
    for (int i = 0; i < sorted_nodes.size(); i++) {
        std::string module_name = sorted_nodes[i];
        std::string module_type = fbs_model->get_operation_type(module_name);
        DL_LOG_LATENCY_START();
        execution_plan[i]->forward(memory_manager->tensors, RUNTIME_MODE_SINGLE_CORE);
        DL_LOG_LATENCY_END();
        uint32_t module_latency = DL_LOG_LATENCY_GET();
        total_latency += module_latency;
        module_info[module_name] = {module_type, module_latency};
    }
    fbs_model->clear_map();
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

static void print_memory_info(const std::map<std::string, mem_info> &info)
{
    std::string table_name = "memory summary";
    std::vector<std::string> header = {"", "internal RAM", "PSRAM", "FLASH"};
    auto it = std::max_element(
        info.begin(), info.end(), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); });
    size_t col0_width = it->first.size();
    char internal_str[16];
    snprintf(internal_str, sizeof(internal_str), "%.2fKB", info.at("total").internal / 1024.f);
    char psram_str[16];
    snprintf(psram_str, sizeof(psram_str), "%.2fKB", info.at("total").psram / 1024.f);
    char flash_str[16];
    snprintf(flash_str, sizeof(flash_str), "%.2fKB", info.at("total").flash / 1024.f);
    size_t col1_width = std::max(header[1].size(), strlen(internal_str));
    size_t col2_width = std::max(header[2].size(), strlen(psram_str));
    size_t col3_width = std::max(header[3].size(), strlen(flash_str));
    std::string sep = gen_sep_str({col0_width, col1_width, col2_width, col3_width});

    // table name
    print_table_name(table_name, sep);
    // header
    ESP_LOGI(TAG,
             "| %-*s | %-*s | %-*s | %-*s |",
             col0_width,
             header[0].c_str(),
             col1_width,
             header[1].c_str(),
             col2_width,
             header[2].c_str(),
             col3_width,
             header[3].c_str());
    ESP_LOGI(TAG, "%s", sep.c_str());
    // body
    std::vector<std::string> keys = {"fbs_model", "param_in_fbs", "param_out_fbs", "mem_manager", "others", "total"};
    for (const auto &key : keys) {
        snprintf(internal_str, sizeof(internal_str), "%.2fKB", info.at(key).internal / 1024.f);
        snprintf(psram_str, sizeof(psram_str), "%.2fKB", info.at(key).psram / 1024.f);
        snprintf(flash_str, sizeof(flash_str), "%.2fKB", info.at(key).flash / 1024.f);
        ESP_LOGI(TAG,
                 "| %-*s | %-*s | %-*s | %-*s |",
                 col0_width,
                 key.c_str(),
                 col1_width,
                 internal_str,
                 col2_width,
                 psram_str,
                 col3_width,
                 flash_str);
        ESP_LOGI(TAG, "%s", sep.c_str());
    }
}

void Model::print_module_info(const std::map<std::string, module_info> &info, bool sort_module_by_latency)
{
    std::string table_name = "module summary";
    std::vector<std::string> header = {"name", "type", "latency"};
    size_t col0_width = header[0].size();
    size_t col1_width = header[1].size();
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
    size_t col2_width = std::max(header[2].size(), strlen(latency_str));
    std::string sep = gen_sep_str({col0_width, col1_width, col2_width});

    // table name
    print_table_name(table_name, sep);
    // header
    ESP_LOGI(TAG,
             "| %-*s | %-*s | %-*s |",
             col0_width,
             header[0].c_str(),
             col1_width,
             header[1].c_str(),
             col2_width,
             header[2].c_str());
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
        std::vector<std::string> sorted_nodes = fbs_model->topological_sort();
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
    if (this->doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", this->name.c_str(), this->version);
    } else {
        ESP_LOGI(
            TAG, "model:%s, version:%lld, description:%s", this->name.c_str(), this->version, this->doc_string.c_str());
    }
    auto info = get_memory_info();
    print_memory_info(info);
    printf("\n");
}

void Model::profile_module(bool sort_module_by_latency)
{
    printf("\n");
    if (this->doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", this->name.c_str(), this->version);
    } else {
        ESP_LOGI(
            TAG, "model:%s, version:%lld, description:%s", this->name.c_str(), this->version, this->doc_string.c_str());
    }
    auto info = get_module_info();
    print_module_info(info);
    printf("\n");
}

void Model::profile(bool sort_module_by_latency)
{
    printf("\n");
    if (this->doc_string.empty()) {
        ESP_LOGI(TAG, "model:%s, version:%lld", this->name.c_str(), this->version);
    } else {
        ESP_LOGI(
            TAG, "model:%s, version:%lld, description:%s", this->name.c_str(), this->version, this->doc_string.c_str());
    }
    auto mem_info = get_memory_info();
    print_memory_info(mem_info);
    printf("\n");
    auto module_info = get_module_info();
    print_module_info(module_info, sort_module_by_latency);
    printf("\n");
}

} // namespace dl
