#include "lintdb/plaid.h"
#include <faiss/utils/hamming.h>
#include <glog/logging.h>
#include <iostream>
#include <unordered_set>
#include "lintdb/api.h"

namespace lintdb {
// returns a list of scores for each centroid.
float score_documents_by_codes(
        const gsl::span<float>
                max_scores_by_centroid, // the max score per centroid across the
                                        // query tokens.
        const code_t* doc_codes,
        size_t num_tokens,
        const float centroid_score_threshold) {
    // Initialize a vector to store the approximate scores for each document
    float doc_score = 0;
    std::unordered_set<code_t> unique_codes;
    // Iterate over each token code.
    for (size_t i = 0; i < num_tokens; i++) {
        auto index = doc_codes[i];
        if (unique_codes.find(index) != unique_codes.end()) {
            continue;
        }

        // we get the centroid score from query_scores. this is the max score
        // found in the query for that centroid.
        doc_score += max_scores_by_centroid[index];

        unique_codes.insert(index);
    }

    return doc_score;
}

std::vector<float> max_score_by_centroid(
        gsl::span<idx_t> coarse_idx,
        gsl::span<float> distances,
        size_t k_per_token,
        size_t num_tokens,
        size_t num_centroids) {
    std::vector<float> max_scores(num_centroids, 0);
    assert(coarse_idx.size() == num_tokens * k_per_token);
    assert(distances.size() == num_tokens * k_per_token);

    for (size_t i = 0; i < num_tokens * k_per_token; i++) {
        auto idx = coarse_idx[i];
        auto dist = distances[i];

        if (dist > max_scores[idx]) {
            max_scores[idx] = dist;
        }
    }
    return max_scores;
}

float score_document_by_residuals(
        const gsl::span<float>
                query_vectors, // size: (num_query_tokens, num_dim)
        const size_t num_query_tokens,
        const float* doc_residuals, // size: (num_doc_tokens, num_dim)
        const size_t num_doc_tokens,
        const size_t dim) {
    // use BLAS functions to matmul doc residuals with the transposed query
    // vectors. we'll use the sum of the max scores for each centroid.
    int m = num_query_tokens; // rows of op(A) and of matrix C.
    int n = num_doc_tokens;   // columns of matrix op(B) and of matrix C.
    int k = dim; // the number of columns of op(A) and rows of op(B).

    std::vector<float> output(m * n, 0);
    // gives us a num_query_tokens x num_doc_tokens matrix.
    cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            m, // 3
            n, // 3
            k, // 2
            1.0,
            query_vectors.data(), // m x k
            k, // leading dimension is the length of the first dimension
               // (columns)
            doc_residuals, // should be k x n after transpose
            k,             // this is the leading dimension of B, not op(b)
            0.000,
            output.data(), // m x n
            n);

    // find the max score for each doc_token.
    std::vector<float> max_scores(n, 0);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            auto score = output[i * n + j];
            if (score > max_scores[j]) {
                max_scores[j] = score;
            }
        }
    }

    float maxsim = 0;
    for (size_t i = 0; i < n; i++) {
        maxsim += max_scores[i];
    }

    return maxsim;
}
} // namespace lintdb