#include "lintdb/index_builder/Tokenizer.h"
#include "lintdb/index_builder/EmbeddingModel.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>

TEST(TokenizerTest, TestTokenizers) {
    std::string path = "colbert_tokenizer.json";
    lintdb::Tokenizer tokenizer(path, 512);
    std::string text = "hello, world!";
    auto ids = tokenizer.encode(text);
    auto decoded = tokenizer.decode(ids);
    ASSERT_EQ(text, decoded);
}

TEST(EmbeddingModelTest, TestModel) {
    std::string path = "model.onnx";
    lintdb::EmbeddingModel model(path);

    std::string tok_path = "colbert_tokenizer.json";
    lintdb::Tokenizer tokenizer(tok_path, 512);
    std::string text = "hello, world!";
    auto ids = tokenizer.encode(text);

    lintdb::ModelInput input;
    input.input_ids = ids;

    std::vector<int32_t> attn;
    for(auto id: ids) {
        if(id == 0) {
            attn.push_back(0);
        } else {
            attn.push_back(1);
        }
    }
    input.attention_mask = attn;

    auto output = model.encode(input);
    ASSERT_EQ(output.size(), input.attention_mask.size()*128);
}