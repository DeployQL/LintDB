#pragma once

#include <gsl/span>
#include <map>
#include <memory>
#include <vector>
#include "lintdb/Encoder.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/quantizers/InvertedListScanner.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/retrievers/Retriever.h"

namespace lintdb {

/**
 * XTRRetriever implements XTR from google.
 *
 * Unlike ColBERT, this method doesn't retrieve all token codes for the
 * documents. Instead, it relies on the tokens retrieved from the inverted list.
 * Depending on search parameters, we may or may not retrieve a score for each
 * query token, and this method imputes the missing scores.
 *
 * https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
 */
class XTRRetriever : public Retriever {
   public:
    XTRRetriever(
            std::shared_ptr<InvertedList> inverted_list,
            std::shared_ptr<ForwardIndex> index,
            std::shared_ptr<Encoder> encoder,
            std::shared_ptr<ProductEncoder> product_encoder);

    std::vector<SearchResult> retrieve(
            idx_t tenant,
            gsl::span<const float> query_data,
            size_t n, // num tokens
            size_t k, // num to return
            const RetrieverOptions& opts) override;

    static void get_document_scores(
            const size_t n,
            vector<ScoredPartialDocumentCodes>& all_doc_codes,
            map<idx_t, std::vector<float>>& document_scores,
            vector<float>& lowest_query_scores);

    static void impute_missing_scores(
            const size_t n,
            map<idx_t, std::vector<float>>& document_scores,
            const std::vector<float>& lowest_query_scores);

    std::vector<ScoredPartialDocumentCodes> get_tokens(
            const idx_t tenant,
            const std::vector<QueryTokenCentroidScore>& token_centroid_scores,
            const gsl::span<const float> query_data,
            const size_t n,
            const size_t num_tokens_to_return = 100);

   private:
    std::shared_ptr<InvertedList> inverted_list_;
    std::shared_ptr<ForwardIndex> index_;
    std::shared_ptr<Encoder> encoder_;
    std::shared_ptr<ProductEncoder> product_encoder_;

    std::vector<QueryTokenCentroidScore> filter_top_centroids_per_query_token(
            const std::vector<idx_t>& coarse_idx,
            const std::vector<float>& distances,
            const size_t n, // num_tokens
            const size_t total_centroids_to_calculate,
            const size_t k_top_centroids) const;
};
} // namespace lintdb