#pragma once

#include "lintdb/retriever/Retriever.h"
#include <memory>
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/Encoder.h"
#include <gsl/span>
#include "lintdb/quantizers/ProductEncoder.h"

namespace lintdb {

/**
 * XTRRetriever implements XTR from google.
 *
 * Unlike ColBERT, this method doesn't retrieve all token codes for the documents.
 * Instead, it relies on the tokens retrieved from the inverted list. Depending
 * on search parameters, we may or may not retrieve a score for each query token,
 * and this method imputes the missing scores.
 *
 * https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
 */
    class XTRRetriever: public Retriever {
       public:
        XTRRetriever(
                std::shared_ptr<InvertedList> inverted_list,
                std::shared_ptr<ForwardIndex> index,
                std::shared_ptr<Encoder> encoder,
                std::shared_ptr<ProductEncoder> product_encoder
                );


        std::vector<SearchResult> retrieve(
            idx_t tenant,
            gsl::span<const float> query_data,
            size_t n, // num tokens
            size_t k, // num to return
            const RetrieverOptions& opts
        ) = 0;

        private:
        std::shared_ptr<InvertedList> inverted_list_;
        std::shared_ptr<ForwardIndex> index_;
        std::shared_ptr<Encoder> encoder_;
        std::shared_ptr<ProductEncoder> product_encoder_;


        std::vector<std::pair<float, idx_t>> get_top_centroids(
                const std::vector<idx_t>& coarse_idx,
                const std::vector<float>& distances,
                size_t n, // num_tokens
                size_t total_centroids_to_calculate,
                size_t k_top_centroids,
                size_t n_probe) const;
    };
}