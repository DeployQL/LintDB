#ifndef LINTDB_RETRIEVER_RETRIEVER_H
#define LINTDB_RETRIEVER_RETRIEVER_H

#include <gsl/span>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/SearchResult.h"

namespace lintdb {
struct RetrieverOptions {
    const size_t total_centroids_to_calculate;
    const size_t num_second_pass;
    const idx_t expected_id;
    const float centroid_threshold;
    const size_t k_top_centroids;
    const size_t n_probe;
    const size_t num_docs_to_score = 4000;
    const size_t out_second_stage = 1000;
    const size_t nearest_tokens_to_fetch = 100;
};

struct Retriever {
   public:
    Retriever() = default;
    virtual ~Retriever() = default;

    virtual std::vector<SearchResult> retrieve(
            const idx_t tenant,
            const gsl::span<const float> query_data,
            const size_t n, // num tokens
            const size_t k, // num to return
            const RetrieverOptions& opts) = 0;
};
} // namespace lintdb

#endif