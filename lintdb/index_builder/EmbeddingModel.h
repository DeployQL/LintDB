#pragma once

#include <vector>
#include <string>
#include <memory>
#include "lintdb/index_builder/Tokenizer.h"

namespace Ort {
    class Env;
    class Session;
    class AllocatorWithDefaultOptions;
    class Value;

}

namespace lintdb {
    struct ModelInput {
        InputIds input_ids;
        AttentionMask attention_mask;        
    };

    
    std::string print_shape(const std::vector<std::int64_t>& shape);

    Ort::Value vec_to_tensor(const std::vector<int32_t>& vec, const std::vector<int64_t>& shape);

    struct EmbeddingModel {
        EmbeddingModel(const std::string& path);
        std::vector<float> encode(ModelInput& inputs) const;

        size_t get_dims() const;

        ~EmbeddingModel();

        private:
            std::unique_ptr<Ort::Env> env;
            std::unique_ptr<Ort::Session> session;

            std::vector<std::string> input_names;
            std::vector<std::string> output_names;
    };
}