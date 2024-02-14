
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include "lintdb/plaid.h"
#include <filesystem>
#include <faiss/utils/hamming.h>
#include <iostream>
#include <arrayfire.h>

TEST(PlaidTests, ReadsCodesCorrectly) {
    // query scores reflect the max scores per centroid for all query terms we've seen. we'll use 5 centroids.
    std::vector<float> query_scores = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    // let's imagine that all token codes represent centroid 2.
    std::vector<code_t> doc_codes = { 2, 2, 2, 2};

    // now let's score documents.
    auto array_query_scores = af::array(5, query_scores.data());
    auto score = lintdb::score_documents_by_codes(
        array_query_scores, 
        doc_codes.data(), 
        4 // num tokens. we'er saying there are 4 tokens.
    );
    // the score is 3, because we're summing the max scores for each centroid.
    // because each token represents centroid 2, we take the score of 3.0 once.
    // we shouldn't duplicate the score for a given centroid.
    EXPECT_FLOAT_EQ(score, 3.0);

}