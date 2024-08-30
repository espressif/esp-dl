#pragma once
#include "dl_module_conv2d.hpp"
#include "dl_module_mul.hpp"
#include "dl_module_add.hpp"
#include "dl_module_resize2d.hpp"
#include "dl_module_global_avg_pool2d.hpp"
#include "dl_module_avg_pool2d.hpp"
#include "dl_module_concat.hpp"
#include "dl_module_sigmoid.hpp"
#include "dl_module_gemm.hpp"
#include "dl_module_requantize_linear.hpp"
#include "dl_module_prelu.hpp"
#include "fbs_loader.hpp"
#include <functional>
#include <iostream>
#include <map>
namespace dl {
namespace module {
class ModuleCreator {
public:
    using Creator = std::function<Module *(fbs::FbsModel *, std::string)>;

    static ModuleCreator *get_instance()
    {
        // This is thread safe for C++11, please refer to `Meyers' implementation of the Singleton pattern`
        static ModuleCreator instance;
        return &instance;
    }

    void register_module(const std::string &op_type, Creator creator) { ModuleCreator::creators[op_type] = creator; }

    Module *create(fbs::FbsModel *fbs_model, const std::string &op_type, const std::string name)
    {
        this->register_dl_modules();

        if (creators.find(op_type) != creators.end()) {
            return creators[op_type](fbs_model, name);
        }
        return nullptr;
    }

    void register_dl_modules()
    {
        if (creators.empty()) {
            this->register_module("Conv", Conv2D::deserialize);
            this->register_module("Mul", Mul2D::deserialize);
            this->register_module("Add", Add2D::deserialize);
            this->register_module("Resize", Resize2D::deserialize);
            this->register_module("GlobalAveragePool", GlobalAveragePool2D::deserialize);
            this->register_module("AveragePool", AveragePool2D::deserialize);
            this->register_module("Concat", Concat::deserialize);
            this->register_module("Sigmoid", Sigmoid::deserialize);
            this->register_module("Gemm", Gemm::deserialize);
            this->register_module("QuantizeLinear", RequantizeLinear::deserialize);
            this->register_module("DequantizeLinear", RequantizeLinear::deserialize);
            this->register_module("RequantizeLinear", RequantizeLinear::deserialize);
            this->register_module("PRelu", PRelu::deserialize);
        }
    }

    void print()
    {
        if (!creators.empty()) {
            for (auto it = creators.begin(); it != creators.end(); ++it) {
                printf("%s", (*it).first.c_str());
            }
        } else {
            printf("Create empty module\n");
        }
    }

private:
    ModuleCreator() {}
    ~ModuleCreator() {}
    ModuleCreator(const ModuleCreator &) = delete;
    ModuleCreator &operator=(const ModuleCreator &) = delete;
    std::map<std::string, Creator> creators;
};

} // namespace module
} // namespace dl
