
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>

// Demonstrate some basic assertions.
TEST(IndexTest, InitializesCorrectly) {
    std::vector<size_t> nbits = { 32 }; // nbits is how many bits to encode the centroids in.
    lintdb::ResidualQuantizer quantizer(128, nbits);
    lintdb::IndexIVF index(&quantizer, 128); // number of centroids

    EXPECT_EQ(index.nlist, 128);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.is_trained, false);
    EXPECT_EQ(quantizer.is_trained(), false);
}

TEST(IndexTest, TrainsCorrectly) {
    size_t dim = 128;
    size_t num_docs = 5;
    size_t num_tokens = 10;

    size_t num_bits = 2;

    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    std::vector<size_t> nbits = { 32 }; // nbits is how many bits to encode the centroids in.
    lintdb::ResidualQuantizer quantizer(128, nbits);
    lintdb::IndexIVF index(&quantizer, 128); // number of centroids

    index.
}