
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

TEST(PlaidTests, ReadsCodesCorrectly) {
    // query scores reflect the max scores per centroid for all query terms we've seen. we'll use 5 centroids.
    std::vector<float> query_scores = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    // let's imagine that all token codes represent centroid 2.
    std::vector<code_t> doc_codes = { 2, 2, 2, 2};

    // now let's score documents.
    auto score = lintdb::score_documents_by_codes(
        query_scores, 
        doc_codes.data(), 
        4, // num tokens. we'er saying there are 4 tokens.
        0.0
    );
    // the score is 3, because we're summing the max scores for each centroid.
    // because each token represents centroid 2, we take the score of 3.0 once.
    // we shouldn't duplicate the score for a given centroid.
    EXPECT_FLOAT_EQ(score, 3.0);

}

TEST(PlaidTests, MaxCentroidScoresIsCorrect) {
    // we'll generate 10 centroids, 5 unique.
    std::vector<idx_t> coarse_idx = { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
    // we'll generate 10 random distances, 5 unique
    std::vector<float> distances = { 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    // we'll generate 5 random distances.
    std::vector<float> max_scores = lintdb::max_score_by_centroid(
        coarse_idx, 
        distances, 
        2, // k_per_token
        5, // num_tokens
        5 // num_centroids
    );

    // we expect the max scores to be the same as the distances.
    for (size_t i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(max_scores[i], distances[i]);
    }
}