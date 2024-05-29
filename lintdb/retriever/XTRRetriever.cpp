#include "lintdb/retriever/XTRRetriever.h"
#include <glog/logging.h>

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
//    std::vector<float> reordered_distances(n * opts.total_centroids_to_calculate);
    // this method is no longer spitting out data.
    // top_centroids shows zero centroids found.
    encoder_->search(
            query_data.data(),
            n,
            coarse_idx,
            distances,
            opts.total_centroids_to_calculate,
            opts.centroid_threshold);

    auto top_centroids = get_top_centroids(
            coarse_idx,
            distances,
            n,
            opts.total_centroids_to_calculate,
            opts.k_top_centroids,
            opts.n_probe);

    InvertedListScanner scanner(product_encoder_, query_data.data(), n);

    std::vector<ScoredPartialDocumentCodes> all_doc_codes;
    for(const auto centroid: top_centroids) {
        auto centroid_idx = centroid.second;

        // get the query tokens that want to search this centroid.
        std::vector<idx_t> query_tokens;
        for (size_t i = 0; i < n; i++) {
            // only consider the top k centroids for each query token.
            for (size_t j = 0; j < opts.k_top_centroids; j++) {
                if (coarse_idx[i * opts.total_centroids_to_calculate + j] == centroid_idx) {
                    query_tokens.push_back(i);
                }
            }
        }

        std::unique_ptr<Iterator> it = inverted_list_->get_iterator(tenant, centroid_idx);
        std::vector<ScoredPartialDocumentCodes> scored = scanner.scan(centroid_idx, std::move(it), query_tokens);

        all_doc_codes.insert(all_doc_codes.end(), scored.begin(), scored.end());
    }

    // step 1: get the top token neighbors.
    // for each doc partial result, we need to assemble the top scores per query token.
    std::map<idx_t, std::vector<float>> document_scores;
    std::vector<float> lowest_query_scores(n, std::numeric_limits<float>::max());
    for (const auto& doc_code : all_doc_codes) {
        if (document_scores.find(doc_code.doc_id) == document_scores.end()) {
            document_scores[doc_code.doc_id] = std::vector<float>(n, std::numeric_limits<float>::lowest());
        }

        // only save the max score per query token.
        for(const auto&[key, value]: doc_code.query_token_scores) {
            if (value > document_scores[doc_code.doc_id][key]) {
                document_scores[doc_code.doc_id][key] = value;
            }
            if (value < lowest_query_scores[key]) {
                lowest_query_scores[key] = value;
            }
        }
    }

    // step 2: impute missing query token scores.
    // for each missing query score in the documents, impute the missing score.
    for (auto&[doc_id, scores]: document_scores) {
        for (size_t i = 0; i < n; i++) {
            if (scores[i] == std::numeric_limits<float>::lowest()) {
                scores[i] = lowest_query_scores[i];
            }
        }
    }

    // step 3: aggregate scores
    std::vector<SearchResult> results;
    for (const auto&[doc_id, scores]: document_scores) {
        float score = 0;
        for (const auto& s: scores) {
            score += s;
        }

        results.emplace_back(SearchResult{doc_id, score/n});
    }

    std::sort(results.begin(), results.end(), std::greater<>());

    return results;
}

std::vector<std::pair<float, idx_t>> XTRRetriever::get_top_centroids(
        const std::vector<idx_t>& coarse_idx,
        const std::vector<float>& distances,
        const size_t n, // num_tokens
        const size_t total_centroids_to_calculate,
        const size_t k_top_centroids,
        const size_t n_probe) const {
    // we're finding the highest centroid scores per centroid.
    std::vector<float> high_scores(encoder_->get_num_centroids(), 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k_top_centroids; j++) {
            auto centroid_of_interest =
                    coarse_idx[i * total_centroids_to_calculate + j];
            // Note: including the centroid score threshold is not part of the
            // original colBERT model.
            // distances[i*total_centroids_to_calculate+j] >
            // centroid_score_threshold &&

            if (distances[i * total_centroids_to_calculate + j] >
                high_scores[centroid_of_interest]) {
                high_scores[centroid_of_interest] =
                        distances[i * total_centroids_to_calculate + j];
            }
        }
    }

    // lets prepare a min heap comparator.
    auto comparator = [](std::pair<float, idx_t> p1,
                         std::pair<float, idx_t> p2) {
        return p1.first > p2.first;
    };

    std::vector<std::pair<float, idx_t>> centroid_scores;
    centroid_scores.reserve(n_probe);
    for (int i = 0; i < high_scores.size(); i++) {
        auto key = i;
        auto score = high_scores[i];
        // Note(MB): removing the filtering by score enables searching with
        // exact copies.
        if (score > 0) {
            if (centroid_scores.size() < n_probe) {
                centroid_scores.emplace_back(score, key);

                if (centroid_scores.size() == n_probe) {
                    std::make_heap(
                            centroid_scores.begin(),
                            centroid_scores.end(),
                            comparator);
                }
            } else if (score > centroid_scores.front().first) {
                std::pop_heap(
                        centroid_scores.begin(),
                        centroid_scores.end(),
                        comparator);
                centroid_scores.front() = std::pair<float, idx_t>(score, key);
                std::push_heap(
                        centroid_scores.begin(),
                        centroid_scores.end(),
                        comparator);
            }
        }
    }

    if (centroid_scores.size() < n_probe) {
        std::sort(
                centroid_scores.begin(),
                centroid_scores.end(),
                std::greater<std::pair<float, idx_t>>());
    } else {
        std::sort_heap(
                centroid_scores.begin(), centroid_scores.end(), comparator);
    }

    centroid_scores.resize(n_probe);

    VLOG(1) << "num centroids: " << centroid_scores.size();
    for (auto p : centroid_scores) {
        VLOG(1) << "centroid: " << p.second << " score: " << p.first;
    }

    return centroid_scores;
}
}