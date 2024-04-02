#include <gtest/gtest.h>

#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/invlists/RocksdbList.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include <cblas.h>
#include <filesystem>
#include <gsl/span>

TEST(IndexTest, ResidualsAreEncodedCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t binarize_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
     // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, binarize_bits);

    index.train(num_docs * num_tokens, buf);

    std::vector<float> fake_doc(dim * num_tokens, 2);

    auto pass = lintdb::RawPassage(fake_doc.data(), num_tokens, dim, 1, "something" );

    // auto doc = index.encode_vectors(pass);
    // EXPECT_EQ(doc->residuals.size(), num_tokens * binarize_bits); // 100 tokens * 2 bits == 200 bytes.

    // auto code_span = gsl::span(doc->codes.data(), doc->codes.size());
    // auto residual_span = gsl::span(doc->residuals.data(), doc->residuals.size());
    // auto decoded = index.decode_vectors(code_span, residual_span, num_tokens, dim);

    // for(size_t i=0; i<num_tokens; i++) {
    //     for(size_t j=0; j<dim; j++) {
    //         EXPECT_EQ(decoded[i*dim + j], fake_doc[i*dim + j]);
    //     }
    // }
}