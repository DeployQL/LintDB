#include "lintdb/index_builder/Tokenizer.h"
#include "lintdb/index_builder/EmbeddingModel.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>
//
//TEST(TokenizerTest, TestTokenizers) {
//    std::string path = "colbert_tokenizer.json";
//    lintdb::Tokenizer tokenizer(path, 512);
//    std::string text = "hello, world!";
//    auto ids = tokenizer.encode(text);
//    auto decoded = tokenizer.decode(ids);
//    ASSERT_EQ("[CLS] [unused1] hello, world! [SEP]", decoded);
//
//    auto query_ids = tokenizer.encode(text, true);
//    auto query_decoded = tokenizer.decode(query_ids);
//    ASSERT_EQ("[CLS] [unused0] hello world [SEP]", query_decoded);
//}
//
//TEST(EmbeddingModelTest, TestModel) {
//    std::string path = "model.onnx";
//    lintdb::EmbeddingModel model(path);
//
//    std::string tok_path = "colbert_tokenizer.json";
//    lintdb::Tokenizer tokenizer(tok_path, 512);
//    std::string text = "hello, world!";
//    auto ids = tokenizer.encode(text);
//
//    lintdb::ModelInput input;
//    input.input_ids = ids;
//
//    std::vector<int32_t> attn;
//    for(auto id: ids) {
//        if(id == 0) {
//            attn.push_back(0);
//        } else {
//            attn.push_back(1);
//        }
//    }
//    input.attention_mask = attn;
//
//    auto output = model.encode(input);
//    ASSERT_EQ(output.size(), input.attention_mask.size()*128);
//}
//
//TEST(EmbeddingModelTest, TestBatch) {
//    std::string path = "model.onnx";
//    lintdb::EmbeddingModel model(path);
//
//    std::string tok_path = "colbert_tokenizer.json";
//    lintdb::Tokenizer tokenizer(tok_path, 512);
//    std::string text = "hello, world!";
//    auto ids = tokenizer.encode(text);
//
//    lintdb::ModelInput input;
//    input.input_ids = ids;
//
//    std::vector<int32_t> attn;
//    for(auto id: ids) {
//        if(id == 0) {
//            attn.push_back(0);
//        } else {
//            attn.push_back(1);
//        }
//    }
//    input.attention_mask = attn;
//
//    std::vector<lintdb::ModelInput> batch = {input, input, input};
//
//    auto batch_output = model.encode(batch);
//
//    for(int i=0; i < batch.size(); i++) {
//        auto output = batch_output.get(i);
//        ASSERT_EQ(output.size(), batch[i].attention_mask.size()*128);
//    }
//}