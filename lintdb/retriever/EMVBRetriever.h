#ifndef LINTDB_RETRIEVER_PLAID_RETRIEVER_H
#define LINTDB_RETRIEVER_PLAID_RETRIEVER_H

#include <vector>
#include "lintdb/SearchResult.h"
#include "lintdb/Encoder.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/retriever/Retriever.h"
#include <cstddef>
#include <gsl/span>

namespace faiss {
struct ProductQuantizer;
}

namespace lintdb {    
    /**
     * DocCandidate keeps track of the emvb score, doc id, and index position in doc codes.
     * 
     * It's only used for bookkeeping in the retriever.
    */
    template<typename ScoreType>
    struct DocCandidate {
        ScoreType score;
        idx_t doc_id;
        size_t index_position;
    };

    /**
     * EMVBRetriever implements the optimizations in: https://arxiv.org/pdf/2404.02805.pdf
     * 
    */
    struct EMVBRetriever: public Retriever {
        public:
        EMVBRetriever(
            std::shared_ptr<InvertedList> inverted_list,
            std::shared_ptr<ForwardIndex> index, 
            std::shared_ptr<Encoder> encoder,
            size_t num_subquantizers = 16
        );

        std::vector<SearchResult> retrieve(
            const idx_t tenant, 
            const gsl::span<const float> query_data,
            const size_t n, // num tokens
            const size_t k, // num to return
            const RetrieverOptions& opts
        );

        private:
        std::shared_ptr<InvertedList> inverted_list_;
        std::shared_ptr<ForwardIndex> index_;
        std::shared_ptr<Encoder> encoder_;

        size_t num_subquantizers;

        // compute_hit_frequency
        std::vector<idx_t> top_passages(
            const idx_t tenant, 
            const gsl::span<const float> query_data, 
            const size_t n, // num query tokens
            const RetrieverOptions& opts,
            std::vector<float>& query_scores,
            std::vector<uint32_t>& bitvectors
        );

        //
        std::vector<DocCandidate<size_t>> rank_phase_one(
             const std::vector<std::unique_ptr<DocumentCodes>>&,
            const std::vector<float>& reordered_distances,
            const std::vector<uint32_t>& bitvectors,
            const size_t n,
            const RetrieverOptions& opts
        );

        std::vector<DocCandidate<float>> rank_by_centroids(
            const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
            const std::vector<DocCandidate<size_t>> candidates,
            const std::vector<float>& reordered_distances,
            const size_t n,
            const RetrieverOptions& opts
        );

        std::vector<DocCandidate<float>> rank_phase_two(
            const std::vector<DocCandidate<float>>& candidates,
            const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
            const std::vector<std::unique_ptr<DocumentResiduals>>& doc_residuals,
            const std::vector<float>& distances,
            const gsl::span<const float> query_data,
            const size_t n,
            const size_t num_to_return,
            const RetrieverOptions& opts
        );

        std::vector<std::pair<float, idx_t>> get_top_centroids( 
            const std::vector<idx_t>& coarse_idx,
            const std::vector<float>& distances, 
            const size_t n,
            const size_t total_centroids_to_calculate,
            const size_t k_top_centroids,
            const size_t n_probe) const;

        /**
         * lookup_pids accesses the inverted list for a given ivf_id and returns the passage ids.
         * 
        */
        std::vector<idx_t> lookup_pids(const uint64_t tenant, const idx_t ivf_id) const;

    };
}

#endif