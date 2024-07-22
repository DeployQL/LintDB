#include "lintdb/retrievers/PlaidRetriever.h"
#include <glog/logging.h>
#include <omp.h>
#include <tuple>
#include <algorithm>
#include <vector>
#include "lintdb/SearchOptions.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/retrievers/Retriever.h"
#include "lintdb/scoring/plaid.h"

#ifndef LINTDB_CHUNK_SIZE
#define LINTDB_CHUNK_SIZE 1000
#endif

namespace lintdb {
PlaidRetriever::PlaidRetriever(
        std::shared_ptr<InvertedList> inverted_list,
        std::shared_ptr<ForwardIndex> index,
        std::shared_ptr<Encoder> encoder)
        : inverted_list_(inverted_list), index_(index), encoder_(encoder) {}

std::vector<idx_t> PlaidRetriever::top_passages(
        const idx_t tenant,
        const gsl::span<const float> query_data,
        const size_t n,
        const RetrieverOptions& opts,
        std::vector<float>& reordered_distances) {
    std::vector<idx_t> coarse_idx(n * opts.total_centroids_to_calculate);
    std::vector<float> distances(n * opts.total_centroids_to_calculate);

    encoder_->search_quantizer(
            query_data.data(),
            n,
            coarse_idx,
            distances,
            opts.total_centroids_to_calculate,
            opts.centroid_threshold);

    // this feels like duplicating the matmul done inside of the above call,
    // but for now, we'll reorder the distances into a matrix.
    // then, we'll find unique top centroids to search.

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < opts.total_centroids_to_calculate; j++) {
            auto current_code =
                    coarse_idx[i * opts.total_centroids_to_calculate + j];
            float dis = distances[i * opts.total_centroids_to_calculate + j];
            reordered_distances
                    [i * opts.total_centroids_to_calculate + current_code] =
                            dis;
        }
    }

    auto centroid_scores = get_top_centroids(
            coarse_idx,
            distances,
            n,
            opts.total_centroids_to_calculate,
            opts.k_top_centroids,
            opts.n_probe);

    auto num_centroids_to_eval =
            std::min<size_t>(opts.n_probe, centroid_scores.size());

    /**
     * Get passage ids
     */
    std::unordered_set<idx_t> global_pids;

#pragma omp parallel
    {
        std::vector<idx_t> local_pids;
#pragma omp for schedule(static, LINTDB_CHUNK_SIZE) nowait
        for (size_t i = 0; i < num_centroids_to_eval; i++) {
            auto idx = centroid_scores[i].second;
            if (idx == -1) {
                continue;
            }
            local_pids = lookup_pids(tenant, idx);
            VLOG(100) << "centroid: " << idx
                      << " number of local pids: " << local_pids.size();
#pragma omp critical
            { global_pids.insert(local_pids.begin(), local_pids.end()); }
        }

    } // end parallel

    if (global_pids.size() == 0) {
        return std::vector<idx_t>();
    }

    auto pid_list = std::vector<idx_t>(global_pids.begin(), global_pids.end());

    return pid_list;
}

std::vector<std::pair<float, idx_t>> PlaidRetriever::rank_phase_one(
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<float>& reordered_distances,
        const size_t n,
        const RetrieverOptions& opts) {
    /**
     * score by passage codes
     */

    std::vector<std::pair<float, idx_t>> pid_scores(doc_codes.size());

#pragma omp for schedule(static, LINTDB_CHUNK_SIZE)
    for (int i = 0; i < doc_codes.size(); i++) {
        auto codes = doc_codes[i]->codes;

        float score = colbert_centroid_score(
                codes,
                reordered_distances,
                n,
                opts.total_centroids_to_calculate,
                doc_codes[i]->id);
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

std::vector<std::tuple<float, idx_t, DocumentScore>> PlaidRetriever::
        rank_phase_two(
                const std::vector<idx_t>& top_25_ids,
                const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
                const std::vector<std::unique_ptr<DocumentResiduals>>&
                        doc_residuals,
                const std::unordered_map<idx_t, size_t>& pid_to_index,
                const gsl::span<const float> query_data,
                const size_t n,
                const RetrieverOptions& opts) {
    std::vector<std::tuple<float, idx_t, DocumentScore>> actual_scores(
            top_25_ids.size());
#pragma omp for schedule(dynamic, LINTDB_CHUNK_SIZE)
    for (int i = 0; i < top_25_ids.size(); i++) {
        auto residuals = doc_residuals[i]->residuals;

        auto id = doc_residuals[i]->id;
        auto codes = doc_codes[pid_to_index.at(id)]->codes;
        std::vector<float> decompressed = encoder_->decode_vectors(
                gsl::span<code_t>(codes),
                gsl::span<residual_t>(residuals),
                doc_residuals[i]->num_tokens);
        const auto data_span =
                gsl::span(query_data.data(), n * encoder_->get_dim());
        DocumentScore score = score_document_by_residuals(
                data_span,
                n,
                decompressed.data(),
                doc_residuals[i]->num_tokens,
                encoder_->get_dim(),
                doc_residuals[i]->id,
                true);
        actual_scores[i] = std::tuple<float, idx_t, DocumentScore>(
                score.score, top_25_ids[i], score);
    }
    auto comparator = [](std::tuple<float, idx_t, DocumentScore> p1,
                         std::tuple<float, idx_t, DocumentScore> p2) {
        return std::get<0>(p1) > std::get<0>(p2);
    };
    std::sort(actual_scores.begin(), actual_scores.end(), comparator);

    return actual_scores;
}

std::vector<SearchResult> PlaidRetriever::retrieve(
        const idx_t tenant,
        const gsl::span<const float> query_data,
        const size_t n, // num tokens
        const size_t k, // num to return
        const RetrieverOptions& opts) {
    std::vector<float> distances(n * opts.total_centroids_to_calculate);
    auto pid_list = top_passages(tenant, query_data, n, opts, distances);

    auto doc_codes = index_->get_codes(tenant, pid_list);
    // create a mapping from pid to the index. we'll need this to hydrate
    // residuals.
    std::unordered_map<idx_t, size_t> pid_to_index;
    for (size_t i = 0; i < doc_codes.size(); i++) {
        auto id = doc_codes[i]->id;
        pid_to_index[id] = i;
    }

    auto pid_scores = rank_phase_one(doc_codes, distances, n, opts);
    // colBERT has a ndocs param which limits the number of documents to score.
    size_t cutoff = pid_scores.size();
    if (opts.num_second_pass != 0) {
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
            LOG(INFO) << "found expected id in pid code scores at position: "
                      << pos << " score: " << it->first;
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank
                          << ". expected id is not being reranked";
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
    auto actual_scores = rank_phase_two(
            top_25_ids,
            doc_codes,
            doc_residuals,
            pid_to_index,
            query_data,
            n,
            opts);

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                actual_scores.begin(),
                actual_scores.end(),
                [opts](std::tuple<float, idx_t, DocumentScore> p) {
                    return std::get<1>(p) == opts.expected_id;
                });
        if (it != actual_scores.end()) {
            auto pos = it - actual_scores.begin();
            LOG(INFO) << "expected id found in residual scores: " << pos
                      << " with score: " << std::get<0>(*it);
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank
                          << ". expected id has been dropped";
            }
        }
    }

    size_t num_to_return = std::min<size_t>(actual_scores.size(), k);
    std::vector<std::tuple<float, idx_t, DocumentScore>> top_k_scores(
            actual_scores.begin(), actual_scores.begin() + num_to_return);

    std::vector<SearchResult> results;
    std::transform(
            top_k_scores.begin(),
            top_k_scores.end(),
            std::back_inserter(results),
            [](std::tuple<float, idx_t, DocumentScore> p) {
                const auto [score, pid, doc_score] = p;
                SearchResult res;
                res.id = pid;
                res.score = score;
                res.token_scores = doc_score.tokens;
                return res;
            });
    return results;
}

std::vector<idx_t> PlaidRetriever::lookup_pids(
        const uint64_t tenant,
        const idx_t idx) const {
    auto it = inverted_list_->get_iterator(tenant, idx);
    std::vector<idx_t> local_pids;
    for (; it->has_next(); it->next()) {
        auto k = it->get_key();

        local_pids.push_back(k.doc_id);
    }

    return local_pids;
}

std::vector<std::pair<float, idx_t>> PlaidRetriever::get_top_centroids(
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
        if (score > 0) {
            if (centroid_scores.size() < n_probe) {
                centroid_scores.push_back(std::pair<float, idx_t>(score, key));

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
                std::greater<>());
    } else {
        std::sort_heap(
                centroid_scores.begin(), centroid_scores.end(), comparator);
    }

    VLOG(1) << "num centroids: " << centroid_scores.size();
    for (auto p : centroid_scores) {
        VLOG(1) << "centroid: " << p.second << " score: " << p.first;
    }

    return centroid_scores;
}
} // namespace lintdb