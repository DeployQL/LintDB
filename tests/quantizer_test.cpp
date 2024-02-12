
#include <gtest/gtest.h>
#include "lintdb/quantizer.h"
#include "lintdb/ResidualQuantizer.h"
#include "lintdb/EmbeddingBlock.h"
#include <iostream>

// Demonstrate some basic assertions.
TEST(QuantizerTest, InitializesCorrectly) {
  std::vector<size_t> nbits = { 32 }; // nbits is how many bits to encode the centroids in.
  lintdb::ResidualQuantizer quantizer(128, nbits);
  EXPECT_EQ(quantizer.dimensions(), 128);
  EXPECT_EQ(quantizer.code_size(), 4);
  EXPECT_EQ(quantizer.is_trained(), false);
}

TEST(QuantizerTest, EncodesCorrectly) {
    std::vector<size_t> nbits = {32};
    const size_t dims = 128;
    const size_t num_data = 1000;
    lintdb::ResidualQuantizer quantizer(dims, nbits);

    // initialize a few fake embeddings
    std::vector<float> embeddings(num_data * 128);
    for (size_t i = 0; i < num_data * dims; i++) {
        embeddings[i] = i;
    }
    
    quantizer.train(num_data, embeddings.data());

    std::vector<uint8_t> codes;
    codes.resize(num_data * quantizer.code_size());

    std::vector<float> ee(num_data * 128);
    for (size_t i = 0; i < num_data * dims; i++) {
        ee[i] = i;
    }

    quantizer.encode(num_data, ee, codes);
}