#pragma once
#include "dl_module_add.hpp"
#include "dl_module_avg_pool2d.hpp"
#include "dl_module_clip.hpp"
#include "dl_module_concat.hpp"
#include "dl_module_conv2d.hpp"
#include "dl_module_exp.hpp"
#include "dl_module_flatten.hpp"
#include "dl_module_gemm.hpp"
#include "dl_module_global_avg_pool2d.hpp"
#include "dl_module_hardsigmoid.hpp"
#include "dl_module_hardswish.hpp"
#include "dl_module_leakyrelu.hpp"
#include "dl_module_log.hpp"
#include "dl_module_lut.hpp"
#include "dl_module_mul.hpp"
#include "dl_module_prelu.hpp"
#include "dl_module_relu.hpp"
#include "dl_module_requantize_linear.hpp"
#include "dl_module_reshape.hpp"
#include "dl_module_resize2d.hpp"
#include "dl_module_sigmoid.hpp"
#include "dl_module_sqrt.hpp"
#include "dl_module_squeeze.hpp"
#include "dl_module_tanh.hpp"
#include "dl_module_transpose.hpp"
#include "dl_module_unsqueeze.hpp"
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
            this->register_module("Tanh", Tanh::deserialize);
            this->register_module("Relu", Relu::deserialize);
            this->register_module("LeakyRelu", LeakyRelu::deserialize);
            this->register_module("HardSigmoid", HardSigmoid::deserialize);
            this->register_module("HardSwish", HardSwish::deserialize);
            this->register_module("Gelu", LUT::deserialize);
            this->register_module("Elu", LUT::deserialize);
            this->register_module("LUT", LUT::deserialize);
            this->register_module("Gemm", Gemm::deserialize);
            this->register_module("QuantizeLinear", RequantizeLinear::deserialize);
            this->register_module("DequantizeLinear", RequantizeLinear::deserialize);
            this->register_module("RequantizeLinear", RequantizeLinear::deserialize);
            this->register_module("PRelu", PRelu::deserialize);
            this->register_module("Clip", Clip::deserialize);
            this->register_module("Flatten", Flatten::deserialize);
            this->register_module("Reshape", Reshape::deserialize);
            this->register_module("Transpose", Transpose::deserialize);
            this->register_module("Exp", Exp::deserialize);
            this->register_module("Log", Log::deserialize);
            this->register_module("Sqrt", Sqrt::deserialize);
            this->register_module("Squeeze", Squeeze::deserialize);
            this->register_module("Unsqueeze", Unsqueeze::deserialize);
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
