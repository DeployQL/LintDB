#ifndef LINTDB_RETRIEVER_RETRIEVER_H
#define LINTDB_RETRIEVER_RETRIEVER_H

#include <vector>
#include "lintdb/api.h"
#include "lintdb/SearchResult.h"
#include <gsl/span>

namespace lintdb {
    struct RetrieverOptions {
        const size_t total_centroids_to_calculate;
        const size_t num_second_pass;
        const idx_t expected_id;
    };
    
    struct Retriever {
    public:
        Retriever() = default;
        virtual ~Retriever() = default;

        virtual std::vector<SearchResult> retrieve(
            const idx_t tenant, 
            const std::vector<idx_t>& pid_list, 
            const std::vector<float>& reordered_distances,
            const gsl::span<const float> query_data,
            const size_t n, // num tokens
            const size_t k, // num to return
            const RetrieverOptions& opts
        ) = 0;
    };
}

#endif