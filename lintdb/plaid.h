#ifndef LINTDB_PLAID_H
#define LINTDB_PLAID_H

#include <vector>
#include <bitset>
#include <algorithm>
#include "lintdb/api.h"
#include <cblas.h>
#include <gsl/span>

namespace lintdb {
     /**
     * score_documents_by_codes aggregates a document score based on each token's
     * code and how well it matches the query.
     * 
     * We return the list of scores for each centroid.
    */
    float score_documents_by_codes(
        const gsl::span<float> max_scores_by_centroid, // the max score per centroid across the query tokens.
        const code_t* doc_codes,
        size_t num_tokens,
        const float centroid_score_threshold
    );

    std::vector<float> max_score_by_centroid(
        gsl::span<idx_t> coarse_idx,
        gsl::span<float> distances,
        size_t k_per_token,
        size_t num_tokens,
        size_t num_centroids
    );

    float score_document_by_residuals(
        const gsl::span<float> query_vectors, // size: (num_query_tokens, num_dim)
        const size_t num_query_tokens,
        const float* doc_residuals, // size: (num_doc_tokens, num_dim)
        const size_t num_doc_tokens,
        const size_t dim
    );
}

#endif