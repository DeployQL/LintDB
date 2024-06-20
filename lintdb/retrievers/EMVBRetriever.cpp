#include "lintdb/retrievers/EMVBRetriever.h"
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <glog/logging.h>
#include <omp.h>
#include <numeric>
#include <unordered_set>
#include "lintdb/SearchOptions.h"
#include "lintdb/assert.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/retrievers/Retriever.h"
#include "lintdb/retrievers/emvb.h"
#include "lintdb/retrievers/emvb_util.h"
#include "lintdb/retrievers/plaid.h"
#include "lintdb/utils/math.h"

namespace lintdb {

EMVBRetriever::EMVBRetriever(
        std::shared_ptr<InvertedList> inverted_list,
        std::shared_ptr<ForwardIndex> index,
        std::shared_ptr<Encoder> encoder,
        size_t num_subquantizers)
        : inverted_list_(inverted_list),
          index_(index),
          encoder_(encoder),
          num_subquantizers(num_subquantizers) {
    LINTDB_THROW_IF_NOT_MSG(
            encoder_->get_quantizer()->get_type() ==
                    QuantizerType::PRODUCT_ENCODER,
            "encoder must be a product encoder");
}

// called find_candidate_docs
std::vector<idx_t> EMVBRetriever::top_passages(
        const idx_t tenant,
        const gsl::span<const float> query_data,
        const size_t n, // num query tokens
        const RetrieverOptions& opts,
        std::vector<float>& query_scores,
        std::vector<uint32_t>& bitvectors) {
    size_t m = n;
    size_t nn = encoder_->nlist;
    size_t k = encoder_->dim;
    float alpha = 1.0;
    float beta = 0.0;

    size_t lda = size_t(encoder_->dim);
    size_t ldb = size_t(encoder_->dim);
    size_t out_dim = size_t(encoder_->nlist);
    // we need to treat this as operating in column major format.
    // we want doc_res x query_vectors^T = C, but have row major data.
    // because of that, we want to calculate query_vectors x doc_res = C^T
    MlasGemm("T",
           "N",
           &nn,
           &m,
           &k,
           &alpha,
           encoder_->get_centroids(), // size: (nlist x dim)
           &lda,
           query_data.data(), // size: (num_query_tok x dim)
           &ldb,
           &beta,
           query_scores.data(), // size: (num_query_tok x nlist)
           &out_dim);

    std::vector<size_t> start_sorted(n * encoder_->get_num_centroids());
    size_t offset = 0;

    // std::vector<size_t> closest_centroid_ids;
    // closest_centroid_ids.reserve(opts.n_probe * n);
    std::unordered_set<size_t> closest_centroid_ids(opts.n_probe * n);

    for (size_t i = 0; i < n; i++) {
        size_t new_offset = filter_query_scores(
                query_scores,
                encoder_->get_num_centroids(),
                i,
                opts.centroid_threshold,
                start_sorted,
                offset);

        assign_bitvector_32(start_sorted, offset, new_offset, i, bitvectors);

        if (new_offset - offset >= opts.n_probe) {
            std::vector<float> candidate_centroids_scores;
            for (size_t j = 0; j < new_offset - offset; j++) {
                auto centroid_id = start_sorted[offset + j];
                auto score = query_scores[i * encoder_->nlist + centroid_id];
                candidate_centroids_scores.push_back(score);
            }

            size_t current_n_probe = opts.n_probe;
            while (current_n_probe > 0) {
                size_t temp_argmax = std::distance(
                        candidate_centroids_scores.begin(),
                        std::max_element(
                                candidate_centroids_scores.begin(),
                                candidate_centroids_scores.end()));
                size_t argmax = start_sorted[temp_argmax];
                closest_centroid_ids.insert(argmax);

                // in the first iteration, save the centroid assigments

                candidate_centroids_scores[temp_argmax] = -1;
                current_n_probe--;
            }
        } else {
            auto start =
                    query_scores.begin() + i * encoder_->get_num_centroids();
            auto end = query_scores.begin() +
                    (i + 1) * encoder_->get_num_centroids();
            size_t argmax = std::distance(start, std::max_element(start, end));
            closest_centroid_ids.insert(argmax);
        }
        offset = new_offset;
    }

    std::unordered_set<idx_t> global_pids;

    std::vector<idx_t> closest_centroid_list(
            closest_centroid_ids.begin(), closest_centroid_ids.end());

#pragma omp parallel
    {
        std::vector<idx_t> local_pids;
#pragma omp for nowait
        for (const auto& idx : closest_centroid_list) {
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

std::vector<DocCandidate<size_t>> EMVBRetriever::rank_phase_one(
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<float>& distances,
        const std::vector<uint32_t>& bitvectors,
        const size_t n,
        const RetrieverOptions& opts) {
    std::vector<DocCandidate<size_t>> pid_scores(doc_codes.size());

#pragma omp parallel for
    for (int i = 0; i < doc_codes.size(); i++) {
        auto codes = doc_codes[i]->codes;

        uint32_t mask = 0;
        std::vector<size_t> centroid_ids(codes.size());
        for (size_t j = 0; j < codes.size(); j++) {
            centroid_ids[j] = codes[j];
            mask |= bitvectors[codes[j]];
        }
        size_t score = popcount(mask);
        pid_scores[i] =
                DocCandidate<size_t>{score, doc_codes[i]->id, size_t(i)};
    }

    VLOG(10) << "number of passages to evaluate: " << pid_scores.size();
    assert(pid_scores.size() == doc_codes.size());

    auto comparator = [](DocCandidate<size_t> p1, DocCandidate<size_t> p2) {
        return p1.score > p2.score;
    };

    std::sort(pid_scores.begin(), pid_scores.end(), comparator);

    return std::vector<DocCandidate<size_t>>(
            pid_scores.begin(), pid_scores.begin() + opts.num_docs_to_score);
}

std::vector<DocCandidate<float>> EMVBRetriever::rank_by_centroids(
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<DocCandidate<size_t>> candidates,
        const std::vector<float>& distances,
        const size_t n,
        const RetrieverOptions& opts) {
    std::vector<DocCandidate<float>> pid_scores(doc_codes.size());

#pragma omp parallel for if (candidates.size() > 10000)
    for (int i = 0; i < candidates.size(); i++) {
        auto index = candidates[i].index_position;
        auto codes = doc_codes[index]->codes;

        std::vector<float> scores = compute_ip_with_centroids(
                codes, distances, opts.total_centroids_to_calculate, n);
        float score = compute_score_by_column_reduction(
                scores, doc_codes[index]->num_tokens, n);

        pid_scores[i] = DocCandidate<float>{score, doc_codes[index]->id, index};
    }

    VLOG(10) << "number of passages to evaluate: " << pid_scores.size();
    assert(pid_scores.size() == doc_codes.size());

    auto comparator = [](DocCandidate<float> p1, DocCandidate<float> p2) {
        return p1.score > p2.score;
    };

    // according to the paper, we take the top 25%.
    std::sort(pid_scores.begin(), pid_scores.end(), comparator);

    size_t cutoff = pid_scores.size();
    if (opts.num_second_pass != 0) {
        cutoff = opts.num_second_pass;
    }

    auto num_rerank = std::min(cutoff, pid_scores.size());

    return std::vector<DocCandidate<float>>(
            pid_scores.begin(), pid_scores.begin() + num_rerank);
}

std::vector<DocCandidate<float>> EMVBRetriever::rank_phase_two(
        const std::vector<DocCandidate<float>>& candidates,
        // doc codes are for ALL potential documents. it's important to lookup
        // the candidate index position in candidates.
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        // doc residuals are only looked up for the existing candidates, and
        // because of this, won't match the candidate index position.
        const std::vector<std::unique_ptr<DocumentResiduals>>& doc_residuals,
        const std::vector<float>&
                distances, // shape: (num_centroids x num_query_tok)
        const gsl::span<const float> query_data,
        const size_t num_query_tokens,
        const size_t num_to_return,
        const RetrieverOptions& opts) {
    // auto dsub = encoder_->get_dim() / num_subquantizers;
    auto ksub = 1 << encoder_->get_nbits();

    Quantizer* quantizer = encoder_->get_quantizer();
    auto product_quantizer = dynamic_cast<ProductEncoder*>(quantizer);
    // I would like to use the following at some point, but distance computers
    // aren't thread safe. It also won't precompute all of the tables without
    // extending it. The benefit is that we can take advantage of faiss'
    // optimizations. auto computer =
    // product_quantizer->pq->get_FlatCodesDistanceComputer();
    // computer->set_query(query_data.data());

    std::vector<float> distance_table(
            num_query_tokens * ksub * this->num_subquantizers);
    product_quantizer->pq->pq.compute_inner_prod_tables(
            num_query_tokens, query_data.data(), distance_table.data());

    std::vector<DocCandidate<float>> final_scores;
    final_scores.reserve(num_to_return);
    // lets prepare a min heap comparator.
    auto comparator = [](DocCandidate<float> p1, DocCandidate<float> p2) {
        return p1.score > p2.score;
    };
#pragma omp parallel for
    for (size_t doc = 0; doc < candidates.size(); doc++) {
        std::vector<float> maxes(num_query_tokens, 0);

        auto candidate = candidates[doc];
        auto doc_tokens = doc_residuals.at(doc)->num_tokens;

        for (size_t i = 0; i < num_query_tokens; i++) {
            std::vector<float> current_distance(num_query_tokens * doc_tokens);
            for (size_t j = 0; j < doc_tokens; j++) {
                current_distance[i * doc_tokens + j] =
                        distances[j * num_query_tokens + i];
            }

            auto filtered_centroids = filter_centroids_in_scoring(
                    opts.centroid_threshold, distances.data(), doc_tokens);

            for (size_t idx = 0; idx < filtered_centroids.size(); idx++) {
                auto distance = compute_distances_one_qt_one_doc(
                        i,
                        idx,
                        distance_table,
                        ksub,
                        num_subquantizers,
                        num_query_tokens,
                        doc_residuals.at(doc)->residuals);

                current_distance[i * doc_tokens + idx] += distance;
            }

            maxes[i] = *std::max_element(
                    current_distance.begin(), current_distance.end());
        }
        float score = std::accumulate(maxes.begin(), maxes.end(), 0.0);

#pragma omp critical
        {
            if (final_scores.size() < num_to_return) {
                final_scores.push_back(DocCandidate<float>{
                        score, candidate.doc_id, candidate.index_position});

                if (final_scores.size() == num_to_return) {
                    std::make_heap(
                            final_scores.begin(),
                            final_scores.end(),
                            comparator);
                }
            } else if (score > final_scores.front().score) {
                if (opts.expected_id == final_scores.front().doc_id) {
                    LOG(INFO)
                            << "expected id being dropped from phase two results. score: "
                            << score;
                }

                std::pop_heap(
                        final_scores.begin(), final_scores.end(), comparator);
                final_scores.front() = DocCandidate<float>{
                        score, candidate.doc_id, candidate.index_position};
                std::push_heap(
                        final_scores.begin(), final_scores.end(), comparator);
            }
        } // end omp critical
    }

    if (final_scores.size() < num_to_return) {
        std::sort(final_scores.begin(), final_scores.end(), comparator);
    } else {
        std::sort_heap(final_scores.begin(), final_scores.end(), comparator);
    }

    return final_scores;
}

std::vector<SearchResult> EMVBRetriever::retrieve(
        const idx_t tenant,
        const gsl::span<const float> query_data,
        const size_t n, // num tokens
        const size_t k, // num to return
        const RetrieverOptions& opts) {
    std::vector<float> distances(n * opts.total_centroids_to_calculate);
    std::vector<uint32_t> bitvectors(n * opts.total_centroids_to_calculate);
    auto pid_list =
            top_passages(tenant, query_data, n, opts, distances, bitvectors);

    auto doc_codes = index_->get_codes(tenant, pid_list);

    // emvb uses a bitvector to decide what documents to include first.
    auto one_candidates =
            rank_phase_one(doc_codes, distances, bitvectors, n, opts);

    auto candidates_centroid_ranked =
            rank_by_centroids(doc_codes, one_candidates, distances, n, opts);

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                candidates_centroid_ranked.begin(),
                candidates_centroid_ranked.end(),
                [opts](DocCandidate<float> p) {
                    return p.doc_id == opts.expected_id;
                });
        if (it != candidates_centroid_ranked.end()) {
            auto pos = it - candidates_centroid_ranked.begin();
            LOG(INFO) << "found expected id in pid code scores at position: "
                      << pos << " score: " << it->score;
        }
    }

    VLOG(10) << "num to rerank: " << candidates_centroid_ranked.size();

    /**
     * score by passage residuals
     */
    std::vector<idx_t> top_25_ids;
    std::transform(
            candidates_centroid_ranked.begin(),
            candidates_centroid_ranked.end(),
            std::back_inserter(top_25_ids),
            [](DocCandidate<float> p) { return p.doc_id; });
    auto doc_residuals = index_->get_residuals(tenant, top_25_ids);

    auto actual_scores = rank_phase_two(
            candidates_centroid_ranked,
            doc_codes,
            doc_residuals,
            distances,
            query_data,
            n,
            k,
            opts);

    std::vector<SearchResult> results;
    std::transform(
            actual_scores.begin(),
            actual_scores.end(),
            std::back_inserter(results),
            [](DocCandidate<float> p) {
                return SearchResult{p.doc_id, p.score};
            });

    return results;
}

std::vector<idx_t> EMVBRetriever::lookup_pids(
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

std::vector<std::pair<float, idx_t>> EMVBRetriever::get_top_centroids(
        const std::vector<idx_t>& coarse_idx,
        const std::vector<float>& distances,
        const size_t n, // num_tokens
        const size_t total_centroids_to_calculate,
        const size_t k_top_centroids,
        const size_t n_probe) const {
    // we're finding the highest centroid scores per centroid.
    std::vector<float> high_scores(encoder_->nlist, 0);
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
                std::greater<std::pair<float, idx_t>>());
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