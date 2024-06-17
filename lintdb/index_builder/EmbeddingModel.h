#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "lintdb/index_builder/Tokenizer.h"

namespace Ort {
class Env;
class Session;
class AllocatorWithDefaultOptions;
class Value;

} // namespace Ort

namespace lintdb {
struct ModelInput {
    InputIds input_ids;
    AttentionMask attention_mask;
};

/**
 * BatchOutput is a helper struct that allows us to extract the output of a
 * batched model.
 *
 * Retrieving embeddings from this struct removes padding.
 */
struct BatchOutput {
    BatchOutput() = default;

    BatchOutput(
            const std::vector<float>& data,
            const std::vector<int64_t>& shape,
            const std::vector<size_t>& dims)
            : data(data), shape(shape), dims(dims) {}

    std::vector<float> get(int i) const {
        std::vector<float> output;

        auto starting_point =
                i * shape[1] * shape[2]; // i * max_len * embedding_dimension
        // dims[i] is the original length of the sequence. shape[2] is the
        // embedding dimension. this gives us the N token embeddings without
        // padding.
        output.insert(
                output.end(),
                data.begin() + starting_point,
                data.begin() + starting_point + dims[i] * shape[2]);
        return output;
    }

   private:
    std::vector<float> data;    /// [batch_size, max_len, embedding_dimension].
                                /// data that needs to be trimmed.
    std::vector<int64_t> shape; /// the shape of the data.
    std::vector<size_t> dims;   /// the original dimensions of each input.
};

std::string print_shape(const std::vector<int64_t>& shape);

Ort::Value vec_to_tensor(
        const std::vector<int32_t>& vec,
        const std::vector<int64_t>& shape);

struct EmbeddingModel {
    EmbeddingModel(const std::string& path);
    std::vector<float> encode(const ModelInput& inputs) const;
    BatchOutput encode(const std::vector<ModelInput>& inputs) const;

    size_t get_dims() const;

    ~EmbeddingModel();

   private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};
} // namespace lintdb