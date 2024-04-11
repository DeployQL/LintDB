#include <omp.h>
#include "lintdb/retriever/PlaidRetriever.h"
#include "lintdb/retriever/Retriever.h"
#include "lintdb/plaid.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/invlists/EncodedDocument.h"
#include <glog/logging.h>

namespace lintdb {
    PlaidRetriever::PlaidRetriever(
        std::shared_ptr<ForwardIndex> index,
        std::shared_ptr<Encoder> encoder
    ) : index_(index), encoder_(encoder) {

    }

    std::vector<std::pair<float, idx_t>> PlaidRetriever::rank_phase_one(
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<float>& reordered_distances,
        const size_t n,
        const RetrieverOptions& opts
    ) {
        /**
         * score by passage codes
         */

        std::vector<std::pair<float, idx_t>> pid_scores(doc_codes.size());

        #pragma omp for
        for (int i = 0; i < doc_codes.size(); i++) {
            auto codes = doc_codes[i]->codes;

            float score = colbert_centroid_score(
                codes,
                reordered_distances,
                n,
                opts.total_centroids_to_calculate,
                doc_codes[i]->id
            );
            pid_scores[i] = std::pair<float, idx_t>(score, doc_codes[i]->id);
        }


        VLOG(10) << "number of passages to evaluate: " << pid_scores.size();
        assert(pid_scores.size() == doc_codes.size());
        // according to the paper, we take the top 25%.
        std::sort(
                pid_scores.begin(),
                pid_scores.end(),
                std::greater<std::pair<float, idx_t>>());

        return pid_scores;
    }

     std::vector<std::pair<float, idx_t>> PlaidRetriever::rank_phase_two(
        const std::vector<idx_t>& top_25_ids,
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<std::unique_ptr<DocumentResiduals>>& doc_residuals,
        const std::unordered_map<idx_t, size_t>& pid_to_index,
        const gsl::span<const float> query_data,
        const size_t n,
        const RetrieverOptions& opts
    ) {

        std::vector<std::pair<float, idx_t>> actual_scores(top_25_ids.size());
#pragma omp for
    for (int i = 0; i < top_25_ids.size(); i++) {
        auto residuals = doc_residuals[i]->residuals;

        auto id = doc_residuals[i]->id;
        auto codes = doc_codes[pid_to_index.at(id)]->codes;

        std::vector<float> decompressed = encoder_->decode_vectors(
            gsl::span<code_t>(codes),
            gsl::span<residual_t>(residuals),
            doc_residuals[i]->num_tokens
        );

        const auto data_span = gsl::span(query_data.data(), n * encoder_->dim);
        float score = score_document_by_residuals(
            data_span,
            n,
            decompressed.data(),
            doc_residuals[i]->num_tokens,
            encoder_->dim,
            true);

        actual_scores[i] = std::pair<float, idx_t>(score, top_25_ids[i]);
    }
    // according to the paper, we take the top 25%.
    std::sort(
        actual_scores.begin(),
        actual_scores.end(),
        std::greater<std::pair<float, idx_t>>()
    );

    return actual_scores;
    }

    std::vector<SearchResult> PlaidRetriever::retrieve(
        const idx_t tenant, 
        const std::vector<idx_t>& pid_list, 
        const std::vector<float>& reordered_distances,
        const gsl::span<const float> query_data,
        const size_t n, // num tokens
        const size_t k, // num to return
        const RetrieverOptions& opts
        ) {

    auto doc_codes = index_->get_codes(tenant, pid_list);
    
    // create a mapping from pid to the index. we'll need this to hydrate
    // residuals.
    std::unordered_map<idx_t, size_t> pid_to_index;
    for (size_t i = 0; i < doc_codes.size(); i++) {
        auto id = doc_codes[i]->id;
        pid_to_index[id] = i;
    }

    auto pid_scores = rank_phase_one(doc_codes, reordered_distances, n, opts);

    // colBERT has a ndocs param which limits the number of documents to score.
    size_t cutoff = pid_scores.size();
    if (opts.num_second_pass != 0 ) {
        cutoff = opts.num_second_pass;
    }
    auto num_rerank = std::max(size_t(1), cutoff / 4);
    num_rerank = std::min(num_rerank, pid_scores.size());

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                pid_scores.begin(),
                pid_scores.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != pid_scores.end()) {
            auto pos = it - pid_scores.begin();
            LOG(INFO) << "found expected id in pid code scores at position: " << pos << " score: " << it->first;
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank << ". expected id is not being reranked";
            }
        }
    }

    VLOG(10) << "num to rerank: " << num_rerank;
    std::vector<std::pair<float, idx_t>> top_25_scores(
            pid_scores.begin(), pid_scores.begin() + num_rerank);

    /**
     * score by passage residuals
     */
    std::vector<idx_t> top_25_ids;
    std::transform(
            top_25_scores.begin(),
            top_25_scores.end(),
            std::back_inserter(top_25_ids),
            [](std::pair<float, idx_t> p) { return p.second; });
    auto doc_residuals = index_->get_residuals(tenant, top_25_ids);

    auto actual_scores = rank_phase_two(top_25_ids, doc_codes, doc_residuals, pid_to_index, query_data, n, opts);

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                actual_scores.begin(),
                actual_scores.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != actual_scores.end()) {
            auto pos = it - actual_scores.begin();
            LOG(INFO) << "expected id found in residual scores: " << pos << " with score: " << it->first;
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank << ". expected id has been dropped";
            }
        }
    }

    size_t num_to_return = std::min<size_t>(actual_scores.size(), k);
    std::vector<std::pair<float, idx_t>> top_k_scores(
            actual_scores.begin(), actual_scores.begin() + num_to_return);

    std::vector<SearchResult> results;
    std::transform(
            top_k_scores.begin(),
            top_k_scores.end(),
            std::back_inserter(results),
            [](std::pair<float, idx_t> p) {
                return SearchResult{p.second, p.first};
            });

    return results;

    }
}