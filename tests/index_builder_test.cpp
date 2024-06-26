#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "lintdb/index_builder/EmbeddingModel.h"
#include "lintdb/index_builder/Tokenizer.h"

 TEST(TokenizerTest, TestTokenizers) {
    std::string path = "../assets/colbert_tokenizer.json";
    lintdb::Tokenizer tokenizer(path, 512);
    std::string text = "hello, world!";
    auto ids = tokenizer.encode(text);
    auto decoded = tokenizer.decode(ids);
    ASSERT_EQ("[CLS] [unused1] hello, world! [SEP]", decoded);

    auto query_ids = tokenizer.encode(text, true);
    auto query_decoded = tokenizer.decode(query_ids);
    ASSERT_EQ("[CLS] [unused0] hello world [SEP]", query_decoded);
}

 TEST(EmbeddingModelTest, TestModel) {
    std::string path = "../assets/model.onnx";
    lintdb::EmbeddingModel model(path);

    std::string tok_path = "../assets/colbert_tokenizer.json";
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

 TEST(EmbeddingModelTest, TestBatch) {
    std::string path = "../assets/model.onnx";
    lintdb::EmbeddingModel model(path);

    std::string tok_path = "../assets/colbert_tokenizer.json";
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

    std::vector<lintdb::ModelInput> batch = {input, input, input};

    auto batch_output = model.encode(batch);

    for(int i=0; i < batch.size(); i++) {
        auto output = batch_output.get(i);
        ASSERT_EQ(output.size(), batch[i].attention_mask.size()*128);
    }
}

 TEST(EmbeddingModelTest, TestBatchIsSameAsSingle) {
    std::string path = "../assets/model.onnx";
    lintdb::EmbeddingModel model(path);

    std::string tok_path = "../assets/colbert_tokenizer.json";
    lintdb::Tokenizer tokenizer(tok_path, 512);
    std::string text = "hello, world!";

    auto create_input = [&](std::string text) {
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

        return input;
    };

    auto input = create_input(text);
    auto input_two = create_input("this is a longer sentence in the batch!");
    auto input_three = create_input("asldfhk alsdhkfl asfjksdlkjf lsjkdfljlasdfj aslkf hasdlfkhlasjdf aslfk aslkd fh");

    std::vector<lintdb::ModelInput> batch = {input, input_two, input_three};

    auto batch_output = model.encode(batch);

    for(int i=0; i < batch.size(); i++) {
            auto output = batch_output.get(i);

            auto single = model.encode(batch[i]);
            ASSERT_EQ(output.size(), single.size());
            ASSERT_EQ(output.size(), batch[i].input_ids.size() * 128);
            for(int j=0; j < output.size(); j++) {
                ASSERT_NE(output[j], 0);
                ASSERT_EQ(output[j], single[j]);
            }
    }
}
