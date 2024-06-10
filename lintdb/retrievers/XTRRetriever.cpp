#include "lintdb/retrievers/XTRRetriever.h"
#include <glog/logging.h>

#include <glog/log_severity.h>
#include <utility>
#include "lintdb/quantizers/InvertedListScanner.h"

namespace lintdb {
    XTRRetriever::XTRRetriever(
        std::shared_ptr<InvertedList> inverted_list,
        std::shared_ptr<ForwardIndex> index, 
        std::shared_ptr<Encoder> encoder,
        std::shared_ptr<ProductEncoder> product_encoder
    ) : Retriever(), inverted_list_(std::move(inverted_list)), index_(std::move(index)), encoder_(std::move(encoder)), product_encoder_(std::move(product_encoder)) {}

    std::vector<SearchResult> XTRRetriever::retrieve(
        const idx_t tenant, 
        const gsl::span<const float> query_data,
        const size_t n, // num tokens
        const size_t k, // num to return
        const RetrieverOptions& opts
    ) {
    std::vector<idx_t> coarse_idx(n * opts.total_centroids_to_calculate);
    std::vector<float> distances(n * opts.total_centroids_to_calculate);
    encoder_->search(
            query_data.data(),
            n,
            coarse_idx,
            distances,
            opts.total_centroids_to_calculate,
            opts.centroid_threshold);

    // returns k top centroids per query token.
    auto top_centroids = filter_top_centroids_per_query_token(
            coarse_idx,
            distances,
            n,
            opts.total_centroids_to_calculate,
            opts.k_top_centroids);

    InvertedListScanner scanner(product_encoder_, query_data.data(), n);

    // step 1: get the top token neighbors.
    auto all_doc_codes = get_document_codes(
            tenant,
            top_centroids,
            query_data,
            n
    );

    // for each doc partial result, we need to assemble the top scores per query token.
    std::map<idx_t, std::vector<float>> document_scores;
    std::vector<float> lowest_query_scores;
    get_document_scores(n, all_doc_codes, document_scores, lowest_query_scores);

    // step 2: impute missing query token scores.
    // for each missing query score in the documents, impute the missing score.
    impute_missing_scores(n, document_scores, lowest_query_scores);

    // step 3: aggregate scores
    std::vector<SearchResult> results;
    for (const auto&[doc_id, scores]: document_scores) {
        float score = 0;
        for (const auto& s: scores) {
            score += s;
        }

        results.emplace_back(SearchResult{doc_id, score});
    }

    std::sort(results.begin(), results.end(), std::greater<>());

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                results.begin(),
                results.end(),
                [opts](SearchResult p) {
                    return p.id == opts.expected_id;
                });
        if (it != results.end()) {
            auto pos = it - results.begin();
            LOG(INFO) << "found expected id in pid code scores at position: "
                      << pos << " score: " << it->score;
            if (pos > k) {
                LOG(INFO) << "num to return: " << k
                          << ". expected id is not being returned";
            }
        }
    }

    return results;
}

std::vector<ScoredPartialDocumentCodes> XTRRetriever::get_document_codes(
        const idx_t tenant,
        const std::vector<QueryTokenCentroidScore>& token_centroid_scores,
        const gsl::span<const float> query_data,
        const size_t n) {
    InvertedListScanner scanner(product_encoder_, query_data.data(), n);

    std::vector<ScoredPartialDocumentCodes> all_doc_codes;
#pragma omp parallel for
    for(const auto centroid_score : token_centroid_scores) {
        auto centroid_idx = centroid_score.centroid_id;
        // get the query tokens that want to search this centroid_score and their distance to the centroid_score.

        std::unique_ptr<Iterator> it = inverted_list_->get_iterator(tenant, centroid_idx);
        std::vector<ScoredPartialDocumentCodes> scored = scanner.scan(
                centroid_idx, std::move(it), {centroid_score});

#pragma omp critical
    {
        all_doc_codes.insert(all_doc_codes.end(), scored.begin(), scored.end());

    }
    }

    return all_doc_codes;
}

void XTRRetriever::impute_missing_scores(
        const size_t n,
        std::map<idx_t, std::vector<float>>& document_scores,
        const std::vector<float>& lowest_query_scores) {
    for (auto&[doc_id, scores]: document_scores) {
        for (size_t i = 0; i < n; i++) {
            if (scores[i] == numeric_limits<float>::lowest()) {
                if (lowest_query_scores[i] == numeric_limits<float>::max()) {
                    scores[i] = 0;
                } else {
                    scores[i] = lowest_query_scores[i];
                }
            }
        }
    }
}

void XTRRetriever::get_document_scores(
        const size_t n,
        std::vector<ScoredPartialDocumentCodes>& all_doc_codes,
        std::map<idx_t, std::vector<float>>& document_scores,
        std::vector<float>& lowest_query_scores) {
    lowest_query_scores.resize(n, numeric_limits<float>::max());

    for (const auto& doc_code : all_doc_codes) {
        if (document_scores.find(doc_code.doc_id) == document_scores.end()) {
            document_scores[doc_code.doc_id] = std::vector<float>(n, std::numeric_limits<float>::lowest());
        }

        // we want to keep around the highest score per query token for document scores
        // while finding the lowest score per query token to impute missing scores.
        for(const auto&[key, value]: doc_code.query_token_scores) {
            if (value > document_scores[doc_code.doc_id][key]) {
                document_scores[doc_code.doc_id][key] = value;
            }
            if (value < lowest_query_scores[key]) {
                lowest_query_scores[key] = value;
            }
        }
    }
}

std::vector<QueryTokenCentroidScore> XTRRetriever::
        filter_top_centroids_per_query_token(
        const std::vector<idx_t>& coarse_idx,
        const std::vector<float>& distances,
                const size_t n, // num_tokens
                const size_t total_centroids_to_calculate,
                const size_t k_top_centroids) const {
    // we're finding the highest centroid scores per query token.
    std::vector<QueryTokenCentroidScore> results;

    size_t max_centroids_to_return = std::min(k_top_centroids, total_centroids_to_calculate);
    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < max_centroids_to_return; j++) {
            auto centroid_of_interest =
                    coarse_idx[i * total_centroids_to_calculate + j];

            auto distance = distances[i * total_centroids_to_calculate + j];
            results.push_back({i, centroid_of_interest, distance});
        }
    }

    return results;
}

}