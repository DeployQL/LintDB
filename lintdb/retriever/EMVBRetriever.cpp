#include <omp.h>
#include "lintdb/retriever/EMVBRetriever.h"
#include "lintdb/retriever/Retriever.h"
#include "lintdb/retriever/plaid.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/invlists/EncodedDocument.h"
#include <glog/logging.h>
#include "lintdb/retriever/emvb.h"
#include "lintdb/retriever/emvb_util.h"
#include <unordered_set>

namespace lintdb {
    EMVBRetriever::EMVBRetriever(
        std::shared_ptr<InvertedList> inverted_list,
        std::shared_ptr<ForwardIndex> index,
        std::shared_ptr<Encoder> encoder,
        size_t num_subquantizers
    ) : inverted_list_(inverted_list), index_(index), encoder_(encoder) {
        pq = std::make_unique<faiss::ProductQuantizer>(
            encoder_->get_dim(),
            num_subquantizers,
            encoder_->get_nbits()
        );
    }

    // called find_candidate_docs
    std::vector<idx_t> EMVBRetriever::top_passages(
        const idx_t tenant, 
        const gsl::span<const float> query_data, 
        const size_t n, // num query tokens
        const RetrieverOptions& opts,
        std::vector<float>& query_scores,
        std::vector<uint32_t>& bitvectors
        ) {
        
        cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                n,
                encoder_->nlist,
                encoder_->dim,
                1.0,
                query_data.data(), // size: (num_query_tok x dim)
                encoder_->dim,
                encoder_->get_centroids(), // size: (nlist x dim)
                encoder_->dim,
                0.0,
                query_scores.data(), // size: (num_query_tok x nlist)
                encoder_->nlist);


        std::vector<size_t> start_sorted(n*encoder_->get_num_centroids());
        size_t offset = 0;

        // std::vector<size_t> closest_centroid_ids;
        // closest_centroid_ids.reserve(opts.n_probe * n);
        std::unordered_set<size_t> closest_centroid_ids(opts.n_probe * n);

        for( size_t i =0; i < n; i++) {
            size_t new_offset = filter_query_scores(
                query_scores,
                encoder_->get_num_centroids(),
                i,
                opts.centroid_threshold,
                start_sorted,
                offset
            );

            assign_bitvector_32(
                start_sorted, offset, new_offset,  i, bitvectors
            );

            if(new_offset - offset >= opts.n_probe) {
                std::vector<float> candidate_centroids_scores;
                for (size_t j=0; j < new_offset - offset; j++) {
                    auto centroid_id = start_sorted[offset+j];
                    auto score = query_scores[i*encoder_->nlist+centroid_id];
                    candidate_centroids_scores.push_back(score);
                }

                size_t current_n_probe = opts.n_probe;
                while (current_n_probe > 0) {
                    size_t temp_argmax = std::distance(candidate_centroids_scores.begin(), std::max_element(candidate_centroids_scores.begin(), candidate_centroids_scores.end()));
                    size_t argmax = start_sorted[temp_argmax];
                    closest_centroid_ids.insert(argmax);

                    // in the first iteration, save the centroid assigments

                    candidate_centroids_scores[temp_argmax] = -1;
                    current_n_probe--;
                }
            } else {
                auto start = query_scores.begin() + i * encoder_->get_num_centroids();
                auto end = query_scores.begin() + (i + 1) * encoder_->get_num_centroids();
                size_t argmax = std::distance(start, std::max_element(start, end));
                closest_centroid_ids.insert(argmax);
            }            
            offset = new_offset;
        }
        
        std::unordered_set<idx_t> global_pids;

    #pragma omp parallel
        {
            std::vector<idx_t> local_pids;
            #pragma omp for nowait
            for (auto idx = closest_centroid_ids.begin(); idx != closest_centroid_ids.end(); idx++) {
                local_pids = lookup_pids(tenant, *idx);
                VLOG(100) << "centroid: " << *idx << " number of local pids: " << local_pids.size();
                #pragma omp critical
                {
                    global_pids.insert(local_pids.begin(), local_pids.end());
                }
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
        const RetrieverOptions& opts
    ) {
        std::vector<DocCandidate<size_t>> pid_scores(doc_codes.size());

        #pragma omp for
        for (int i = 0; i < doc_codes.size(); i++) {
            auto codes = doc_codes[i]->codes;

            uint32_t mask = 0;
            std::vector<size_t> centroid_ids(codes.size());
            for (size_t j = 0; j < codes.size(); j++) {
                centroid_ids[j] = codes[j];
                mask |= bitvectors[codes[j]];
            }
            size_t score = popcount(mask);
            pid_scores[i] = DocCandidate<size_t>{score, doc_codes[i]->id, size_t(i)};
        }


        VLOG(10) << "number of passages to evaluate: " << pid_scores.size();
        assert(pid_scores.size() == doc_codes.size());

        std::sort(
                pid_scores.begin(),
                pid_scores.end(),
                std::greater<DocCandidate<size_t>>());

        return std::vector<DocCandidate<size_t>>(pid_scores.begin(), pid_scores.begin() + opts.num_docs_to_score);
    }

    std::vector<DocCandidate<float>> EMVBRetriever::rank_by_centroids(
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        const std::vector<DocCandidate<size_t>> candidates,
        const std::vector<float>& distances,
        const size_t n,
        const RetrieverOptions& opts
    ) {
        std::vector<DocCandidate<float>> pid_scores(doc_codes.size());

        #pragma omp for simd schedule(static, 16)
        for (int i = 0; i < candidates.size(); i++) {
            auto index = candidates[i].index_position;
            auto codes = doc_codes[index]->codes;

            std::vector<float> scores = compute_ip_with_centroids(
                codes,
                distances,
                opts.total_centroids_to_calculate,
                n
            );
            float score = compute_score_by_column_reduction(
                scores,
                doc_codes[index]->num_tokens,
                n
            );

            pid_scores[i] = DocCandidate<float>{score, doc_codes[index]->id, index};
        }


        VLOG(10) << "number of passages to evaluate: " << pid_scores.size();
        assert(pid_scores.size() == doc_codes.size());

        // according to the paper, we take the top 25%.
        std::sort(
                pid_scores.begin(),
                pid_scores.end(),
                std::greater<std::pair<size_t, idx_t>>());


        size_t cutoff = pid_scores.size();
        if (opts.num_second_pass != 0 ) {
            cutoff = opts.num_second_pass;
        }

        auto num_rerank = std::min(cutoff, pid_scores.size());

        return std::vector<DocCandidate<float>>(pid_scores.begin(), pid_scores.begin() + num_rerank);
    }


     std::vector<DocCandidate<float>> EMVBRetriever::rank_phase_two(
        const std::vector<DocCandidate<float>>& candidates,
        // doc codes are for ALL potential documents. it's important to lookup the candidate index position in candidates.
        const std::vector<std::unique_ptr<DocumentCodes>>& doc_codes,
        // doc residuals are only looked up for the existing candidates, and because of this, won't match the candidate index position.
        const std::vector<std::unique_ptr<DocumentResiduals>>& doc_residuals,
        const gsl::span<const float> query_data,
        const size_t n,
        const RetrieverOptions& opts
    ) {
        auto dsub = encoder_->get_dim() / n;
        auto ksub = 1 << encoder_->get_nbits();
        auto M = candidates.size(); // ProductQuantizer has a variable M that EMVB specifies as the number of residuals. 
        std::vector<float> distance_table(n * ksub * M);
        pq->compute_inner_prod_tables(n, query_data.data(), distance_table.data());
    }

    std::vector<SearchResult> EMVBRetriever::retrieve(
        const idx_t tenant, 
        const gsl::span<const float> query_data,
        const size_t n, // num tokens
        const size_t k, // num to return
        const RetrieverOptions& opts
        ) {
            

    std::vector<float> distances(n*opts.total_centroids_to_calculate);
    std::vector<uint32_t> bitvectors(n*opts.total_centroids_to_calculate);
    auto pid_list = top_passages(tenant, query_data, n, opts, distances, bitvectors);

    auto doc_codes = index_->get_codes(tenant, pid_list);

    // emvb uses a bitvector to decide what documents to include first.
    auto one_candidates = rank_phase_one(doc_codes, distances, bitvectors, n, opts);

    auto candidates_centroid_ranked = rank_by_centroids(doc_codes, one_candidates, distances, n, opts);

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                candidates_centroid_ranked.begin(),
                candidates_centroid_ranked.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != candidates_centroid_ranked.end()) {
            auto pos = it - candidates_centroid_ranked.begin();
            LOG(INFO) << "found expected id in pid code scores at position: " << pos << " score: " << it->score;
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
            [](std::pair<float, idx_t> p) { return p.second; });
    auto doc_residuals = index_->get_residuals(tenant, top_25_ids);

    auto actual_scores = rank_phase_two(candidates_centroid_ranked, doc_codes, doc_residuals, query_data, n, opts);

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                actual_scores.begin(),
                actual_scores.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != actual_scores.end()) {
            auto pos = it - actual_scores.begin();
            LOG(INFO) << "expected id found in residual scores: " << pos << " with score: " << it->score;
            if (pos > k) {
                LOG(INFO) << "top 25 cutoff: " << k << ". expected id has been dropped";
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

std::vector<idx_t> EMVBRetriever::lookup_pids(const uint64_t tenant, const idx_t idx) const {
    auto it = inverted_list_->get_iterator(tenant, idx);
    std::vector<idx_t> local_pids;
    for (; it->has_next(); it->next()) {
        auto k = it->get_key();
        
        local_pids.push_back(k.id);
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
            auto centroid_of_interest = coarse_idx[i*total_centroids_to_calculate+j];

            // Note: including the centroid score threshold is not part of the original colBERT model.
            // distances[i*total_centroids_to_calculate+j] > centroid_score_threshold && 
                
            if (distances[i*total_centroids_to_calculate+j] > high_scores[centroid_of_interest]) {
                high_scores[centroid_of_interest] = distances[i*total_centroids_to_calculate+j];
            }
        }
    }

    // lets prepare a min heap comparator.
    auto comparator = [](std::pair<float, idx_t> p1, std::pair<float, idx_t> p2) {
        return p1.first > p2.first;
    };

    std::vector<std::pair<float, idx_t>> centroid_scores;
    centroid_scores.reserve(n_probe);
    for(int i=0; i < high_scores.size(); i++) {
        auto key = i;
        auto score = high_scores[i];
        // Note(MB): removing the filtering by score enables searching with exact copies.
        if (score > 0){
            if (centroid_scores.size() < n_probe) {
                centroid_scores.push_back(std::pair<float, idx_t>(score, key));

                if (centroid_scores.size() == n_probe) {
                    std::make_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
                }
            } else if (score > centroid_scores.front().first) {
                std::pop_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
                centroid_scores.front() = std::pair<float, idx_t>(score, key);
                std::push_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
            }
        }
    }

    if(centroid_scores.size() < n_probe) {
        std::sort(
                centroid_scores.begin(),
                centroid_scores.end(),
                std::greater<std::pair<float, idx_t>>()
        );
    } else {
        std::sort_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
    }

    VLOG(1) << "num centroids: " << centroid_scores.size();
    for(auto p : centroid_scores) {
        VLOG(1) << "centroid: " << p.second << " score: " << p.first;
    }

    return centroid_scores;
}
}