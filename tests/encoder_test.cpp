#include <gtest/gtest.h>
#define private public
#include <faiss/utils/random.h>
#include <filesystem>
#include <gsl/span>
#include <iostream>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/Passages.h"
#include "lintdb/index.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/quantizers/io.h"
#include "lintdb/quantizers/Quantizer.h"

TEST(EncoderTest, ResidualsAreEncodedCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t binarize_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");

    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }

    auto qc = lintdb::QuantizerConfig{2, 128, 2};
    std::shared_ptr<lintdb::Quantizer> quantizer = create_quantizer(lintdb::IndexEncoding::BINARIZER, qc);
    lintdb::DefaultEncoder encoder(128, quantizer);
    encoder.train(buf.data(), num_docs * num_tokens, dim, 2, 2);

    std::vector<float> fake_doc(dim * num_tokens, 2);

    auto pass = lintdb::EmbeddingPassage(fake_doc.data(), num_tokens, dim, 1);
    auto encoded_doc = encoder.encode_vectors(pass);

    EXPECT_EQ(encoded_doc->residuals.size(), num_tokens * dim / (8 / binarize_bits)); // 100 tokens * 128 dim / 4 dimensions per byte = 3200 bytes.
}

TEST(EncoderTest, NoCompressionWorksCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t binarize_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");

    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }

    lintdb::DefaultEncoder encoder(2, 2, 2, 128, 0, lintdb::IndexEncoding::NONE);
    encoder.train(buf.data(), num_docs * num_tokens, dim);

    std::vector<float> fake_doc(dim * num_tokens, 2);

    auto pass = lintdb::EmbeddingPassage(fake_doc.data(), num_tokens, dim, 1);
    auto encoded_doc = encoder.encode_vectors(pass);

    // our encoded residuals should be the same size they were before encoding, taking the float size into account.
    EXPECT_EQ(encoded_doc->residuals.size(), num_tokens * dim * sizeof(float));

    auto decoded_doc = encoder.decode_vectors(encoded_doc->codes, encoded_doc->residuals, num_tokens);
    
    // our decoded doc gives us back the same token embedding size.
    EXPECT_EQ(decoded_doc.size(), num_tokens*dim);
    
    for (auto i=0; i<num_tokens*dim; i++) {
        EXPECT_EQ(decoded_doc[i], fake_doc[i]);
    }
}