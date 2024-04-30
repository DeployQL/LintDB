
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include "lintdb/retriever/plaid.h"
#include <filesystem>
#include <faiss/utils/hamming.h>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "lintdb/util.h"

TEST(PlaidTests, ReadsCodesCorrectly) {
    // query scores reflect the max scores per centroid for all query terms we've seen. we'll use 5 centroids.
    std::vector<float> query_scores = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    // let's imagine that all token codes represent centroid 2.
    std::vector<code_t> doc_codes = { 2, 2, 2, 2};

    // now let's score documents.
    auto score = lintdb::score_documents_by_codes(
        query_scores, 
        doc_codes, 
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

TEST(PlaidTests, CodesSameAsColBERT) {
    // centroid scores is (n_centroids, nquery_vectors)
    auto colbert = [](std::vector<code_t> codes, std::vector<float> centroid_scores, size_t nquery_vectors) {
        std::vector<float> per_doc_approx_scores(nquery_vectors, -9999);

        std::unordered_set<int> seen_codes;
         for (int j = 0; j < codes.size(); j++) {
            auto code = codes[j];
            if (seen_codes.find(code) == seen_codes.end()) {
                for (int k = 0; k < nquery_vectors; k++) {
                    per_doc_approx_scores[k] =
                        std::max(per_doc_approx_scores[k],
                                 centroid_scores
                                     [code * nquery_vectors + k]);
                }
                seen_codes.insert(code);
            }
        }
        float score = 0;
        for (int k = 0; k < nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }

        return score;
    };

    // we have a document where each token has a code.
    std::vector<code_t> doc_codes = { 0, 1, 2, 3, 4, 4, 0, 1, 4, 3, 2 };
    // we have a list of centroid scores for each token. this should be (n_centroids, nquery_vectors)
    std::vector<float> centroid_scores(5 * 5, 0.0);
    std::vector<idx_t> coarse_idx(5*5, 0);
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            centroid_scores[i * 5 + j] = (rand() / RAND_MAX + 1.);
            coarse_idx[i * 5 + j] = i;
        }
    }

    // for our code, let's get the max per centroid.
    auto max_scores = lintdb::max_score_by_centroid(
        coarse_idx, 
        centroid_scores, 
        5, // k_per_token. multiplied with num_tokens to get the full length of centroid_scores.
        5, // num_tokens
        5 // num_centroids
    );
    auto actual = lintdb::score_documents_by_codes(
        max_scores,
        doc_codes,
        0.0
    );
    auto expected = colbert(doc_codes, centroid_scores, 5);

    EXPECT_FLOAT_EQ(actual, expected);
}

TEST(PlaidTests, CodeScoresSameAsColBERT) {
    // centroid scores is (n_centroids, nquery_vectors)
    auto colbert = [](std::vector<code_t> codes, std::vector<float> centroid_scores, size_t nquery_vectors) {
        std::vector<float> per_doc_approx_scores(nquery_vectors, -9999);

        std::unordered_set<int> seen_codes;
         for (int j = 0; j < codes.size(); j++) {
            auto code = codes[j];
            if (seen_codes.find(code) == seen_codes.end()) {
                for (int k = 0; k < nquery_vectors; k++) {
                    per_doc_approx_scores[k] =
                        std::max(per_doc_approx_scores[k],
                                 centroid_scores
                                     [code * nquery_vectors + k]);
                }
                seen_codes.insert(code);
            }
        }
        float score = 0;
        for (int k = 0; k < nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }

        return score;
    };

    // we have a document where each token has a code.
    std::vector<code_t> doc_codes = { 0, 1, 2, 3, 4, 4, 0, 1, 4, 3, 2 };
    // we have a list of centroid scores for each token. this should be (n_centroids, nquery_vectors)
    std::vector<float> centroid_scores(100 * 5, 0.0);
    std::vector<idx_t> coarse_idx(100*5, 0);
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 100; j++) {
            centroid_scores[i * 100 + j] = (rand() / RAND_MAX + 1.);
            coarse_idx[i * 100 + j] = i;
        }
    }

    // for our code, let's get the max per centroid.
    auto actual = lintdb::colbert_centroid_score(
        doc_codes,
        centroid_scores,
        5, // nquery_vectors
        100, // n_centroids
        -1
    );
    auto expected = colbert(doc_codes, centroid_scores, 5);

    EXPECT_FLOAT_EQ(actual, expected);
}

TEST(PlaidTests, ResidualScoresTheSame) {
    // we bastardize the colbert code to be more amenable to our testing.
    // we only test for one document.
    // we're looking for the max scores per query vector.
    auto colbert = [](std::vector<float> scores, size_t doc_length, size_t nquery_vectors) {
        auto scores_offset = scores.begin();
        std::vector<float> max_scores(nquery_vectors);
         for (int j = 0; j < doc_length; j++) {
            std::transform(max_scores.begin(),
                           max_scores.begin() + nquery_vectors,
                           scores_offset, max_scores.begin(),
                           [](float a, float b) { return std::max(a, b); });
            scores_offset += nquery_vectors;
        }

        auto sum = 0.;
        for(auto score: max_scores) {
            sum += score;
        }

        return sum;
    };

    auto matmul_transposedB = [](const std::vector<float>& A, int m, int k,
                                const std::vector<float>& B, int n, int l) {
        std::vector<float> C(m * n, 0.0);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int p = 0; p < k; ++p) {
                    C[i * n + j] += A[i * k + p] * B[j * l + p];
                }
            }
        }

        return C;
    };

    // imagine we calculated our scores, creating an n_doc_token x n_query_token matrix.
    size_t num_query_tokens = 5;
    size_t dim = 128;
    std::vector<float> query_vectors(num_query_tokens * dim, 0.0);
    for (size_t i = 0; i < num_query_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            query_vectors[i * dim + j] = (rand() / RAND_MAX + 1.);
        }
    }

    size_t num_doc_tokens = 10;
    std::vector<float> doc_vectors(num_doc_tokens * dim, 0.0);
    for (size_t i = 0; i < num_doc_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            doc_vectors[i * dim + j] = (rand() / RAND_MAX + 1.);
        }
    }

    lintdb::normalize_vector(query_vectors.data(), num_query_tokens, dim);
    lintdb::normalize_vector(doc_vectors.data(), num_doc_tokens, dim);

    auto scores = matmul_transposedB(doc_vectors, num_doc_tokens, dim, query_vectors, num_query_tokens, dim);

    auto colbert_score = colbert(scores, num_doc_tokens, num_query_tokens);
    // auto colbert_score_blas = colbert(output())

    auto actual = lintdb::score_document_by_residuals(query_vectors, num_query_tokens, doc_vectors.data(), num_doc_tokens, dim, false);

    EXPECT_FLOAT_EQ(actual, colbert_score);
}