#ifndef LINTDB_RETRIEVER_PLAID_RETRIEVER_H
#define LINTDB_RETRIEVER_PLAID_RETRIEVER_H

#include <cstddef>
#include <gsl/span>
#include <tuple>
#include <vector>
#include "lintdb/Encoder.h"
#include "lintdb/SearchResult.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/retrievers/Retriever.h"
#include "lintdb/scoring/plaid.h"

namespace lintdb {
/**
 * PlaidRetriever implements the Plaid Engine from:
 * https://arxiv.org/pdf/2205.09707.pdf
 *
 * This is a two-pass retrieval engine that uses a combination of centroid
 * scores and residual scores.
 *
 * Implementation Note: Retrievers depend on both the encoder and the forward
 * index in order to get codes and residuals. There's probably a missing
 * abstraction.
 */
struct PlaidRetriever : public Retriever {
   public:
    PlaidRetriever(
            std::shared_ptr<InvertedList> inverted_list,
            std::shared_ptr<ForwardIndex> index,
            std::shared_ptr<Encoder> encoder);

    std::vector<SearchResult> retrieve(
            const idx_t tenant,
            const gsl::span<const float> query_data,
            const size_t n, // num tokens
            const size_t k, // num to return
            const RetrieverOptions& opts);

   private:
    std::shared_ptr<InvertedList> inverted_list_;
    std::shared_ptr<ForwardIndex> index_;
    std::shared_ptr<Encoder> encoder_;

    std::vector<idx_t> top_passages(
            const idx_t tenant,
            const gsl::span<const float> query_data,
            const size_t n,
            const RetrieverOptions& opts,
            std::vector<float>& distances);

    std::vector<std::pair<float, idx_t>> rank_phase_one(
            const std::vector<std::unique_ptr<DocumentCodes>>&,
            const std::vector<float>& reordered_distances,
            const size_t n,
            const RetrieverOptions& opts);

    std::vector<std::tuple<float, idx_t, DocumentScore>> rank_phase_two(
            const std::vector<idx_t>& top_25_ids,
            const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
            const std::vector<std::unique_ptr<DocumentResiduals>>&
                    doc_residuals,
            const std::unordered_map<idx_t, size_t>& pid_to_index,
            const gsl::span<const float> query_data,
            const size_t n,
            const RetrieverOptions& opts);

    std::vector<std::pair<float, idx_t>> get_top_centroids(
            const std::vector<idx_t>& coarse_idx,
            const std::vector<float>& distances,
            const size_t n,
            const size_t total_centroids_to_calculate,
            const size_t k_top_centroids,
            const size_t n_probe) const;

    /**
     * lookup_pids accesses the inverted list for a given ivf_id and returns the
     * passage ids.
     *
     */
    std::vector<idx_t> lookup_pids(const uint64_t tenant, const idx_t ivf_id)
            const;
};
} // namespace lintdb

#endif