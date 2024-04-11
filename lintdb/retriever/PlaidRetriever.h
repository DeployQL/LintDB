#ifndef LINTDB_RETRIEVER_PLAID_RETRIEVER_H
#define LINTDB_RETRIEVER_PLAID_RETRIEVER_H

#include <vector>
#include "lintdb/SearchResult.h"
#include "lintdb/Encoder.h"
#include "lintdb/invlists/InvertedList.h"
#include <stddef.h>
#include <gsl/span>

namespace lintdb {
    struct PlaidOptions {
        const size_t total_centroids_to_calculate;
        const size_t num_second_pass;
        const idx_t expected_id;
        
    };
    
    /**
     * PlaidRetriever implements the Plaid Engine from: https://arxiv.org/pdf/2205.09707.pdf
     * 
     * This is a two-pass retrieval engine that uses a combination of centroid scores and residual scores.
     * 
     * Implementation Note: Retrievers depend on both the encoder and the forward index in order to 
     * get codes and residuals. There's probably a missing abstraction.
    */
    struct PlaidRetriever {
        public:
        PlaidRetriever(std::shared_ptr<ForwardIndex> index, std::shared_ptr<Encoder> encoder);
        std::vector<SearchResult> retrieve(
            const idx_t tenant, 
            const std::vector<idx_t>& pid_list, 
            const std::vector<float>& reordered_distances,
            const gsl::span<const float> query_data,
            const size_t n, // num tokens
            const size_t k, // num to return
            const PlaidOptions& opts
        );

        private:
        std::shared_ptr<ForwardIndex> index_;
        std::shared_ptr<Encoder> encoder_;

        std::vector<std::pair<float, idx_t>> rank_phase_one(
             const std::vector<std::unique_ptr<DocumentCodes>>&,
            const std::vector<float>& reordered_distances,
            const size_t n,
            const PlaidOptions& opts
        );

        std::vector<std::pair<float, idx_t>> rank_phase_two(
            const std::vector<idx_t>& top_25_ids,
            const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
            const std::vector<std::unique_ptr<DocumentResiduals>>& doc_residuals,
            const std::unordered_map<idx_t, size_t>& pid_to_index,
            const gsl::span<const float> query_data,
            const size_t n,
            const PlaidOptions& opts
        );
    };
}

#endif