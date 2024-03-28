#include "lintdb/plaid.h"
#include <faiss/utils/hamming.h>
#include <glog/logging.h>
#include <iostream>
#include <unordered_set>
#include "lintdb/api.h"
#include "lintdb/util.h"
#include <numeric>

namespace lintdb {

float score_documents_by_codes(
        const gsl::span<float>
                max_scores_by_centroid, // the max score per centroid across the
                                        // query tokens.
        const std::vector<code_t>& doc_codes,
        const float centroid_score_threshold,
        const idx_t expected_id) {
    // Initialize a vector to store the approximate scores for each document
    float doc_score = 0;
    std::unordered_set<code_t> unique_codes;
    // Iterate over each token code.
    for (auto index : doc_codes) {
        assert(index < max_scores_by_centroid.size() && "index out of bounds");
        if ((unique_codes.find(index) != unique_codes.end()) ||
            (max_scores_by_centroid[index] < centroid_score_threshold)) {
            continue;
        }

        // we get the centroid score from query_scores. this is the max score
        // found in the query for that centroid.
        doc_score += max_scores_by_centroid[index];
        unique_codes.insert(index);
    }

    return doc_score;
}

float colbert_centroid_score(
        std::vector<code_t>& doc_codes,
        std::vector<float>& centroid_scores,
        size_t nquery_vectors,
        size_t n_centroids,
        const idx_t expected_id) {
    std::vector<float> per_doc_approx_scores(nquery_vectors, -9999);

    std::unordered_set<int> seen_codes;
    for (int j = 0; j < doc_codes.size(); j++) {
        auto code = doc_codes[j];
        if (seen_codes.find(code) == seen_codes.end()) {
            for (int k = 0; k < nquery_vectors; k++) {
                per_doc_approx_scores[k] =
                    std::max(per_doc_approx_scores[k], centroid_scores[k * n_centroids + code]);
                                // centroid_scores
                                //     [code * nquery_vectors + k]);
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
}

// below, we are summing up for every centroid. this ignores per word
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
        const gsl::span<const float>
                query_vectors, // size: (num_query_tokens, num_dim)
        const size_t num_query_tokens,
        float* doc_residuals, // size: (num_doc_tokens, num_dim)
        const size_t num_doc_tokens,
        const size_t dim,
        bool normalize) {
    // use BLAS functions to matmul doc residuals with the transposed query
    // vectors. we'll use the sum of the max scores for each centroid.
    int m = num_doc_tokens; // rows of op(A) and of matrix C.
    int n = num_query_tokens;   // columns of matrix op(B) and of matrix C.
    int k = dim; // the number of columns of op(A) and rows of op(B).

    if (normalize) {
        normalize_vector(doc_residuals, num_doc_tokens, dim);
    }

    std::vector<float> output(m * n, 0);
    // gives us a num_doc_tokens x num_query_tokens matrix.
    cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            m, // 8
            n, // 4
            k, // 128
            1.0,
            doc_residuals, // m x k
            k, // leading dimension is the length of the first dimension
               // (columns)
            query_vectors.data(), // should be k x n after transpose
            k,             // this is the leading dimension of B, not op(b)
            0.000,
            output.data(), // m x n
            n);

    // find the max score for each doc_token.
    std::vector<float> max_scores(n, 0);
    for (size_t i = 0; i < m; i++) { // per num_doc_tokens
        for (size_t j = 0; j < n; j++) { // per num_query_tokens
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