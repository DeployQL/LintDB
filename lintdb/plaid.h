#ifndef LINTDB_PLAID_H
#define LINTDB_PLAID_H

#include <cblas.h>
#include <algorithm>
#include <bitset>
#include <gsl/span>
#include <vector>
#include "lintdb/api.h"

namespace lintdb {
/**
 * score_documents_by_codes aggregates a document score based on each token's
 * code and how well it matches the query.
 *
 * We return the list of scores for each centroid.
 */
float score_documents_by_codes(
        const gsl::span<float>
                max_scores_by_centroid, // the max score per centroid across the
                                        // query tokens.
        const std::vector<code_t>& doc_codes,
        const float centroid_score_threshold,
        const idx_t expected_id = -1);

std::vector<float> max_score_by_centroid(
        gsl::span<idx_t> coarse_idx,
        gsl::span<float> distances,
        size_t k_per_token,
        size_t num_tokens,
        size_t num_centroids);

float colbert_centroid_score(
        std::vector<code_t>& doc_codes, 
        std::vector<float>& centroid_scores,
        size_t nquery_vectors,
        size_t n_centroids,
        const idx_t expected_id);

float score_document_by_residuals(
        const gsl::span<float>
                query_vectors, // size: (num_query_tokens, num_dim)
        const size_t num_query_tokens,
        float* doc_residuals, // size: (num_doc_tokens, num_dim)
        const size_t num_doc_tokens,
        const size_t dim,
        bool normalize=true);

} // namespace lintdb

#endif