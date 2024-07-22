#include "KnnNearestCentroids.h"
#include "lintdb/quantizers/impl/kmeans.h"
#include <glog/logging.h>

namespace lintdb {
    void KnnNearestCentroids::calculate(
            std::vector<float>& query,
            const size_t num_query_tokens,
            const std::shared_ptr<ICoarseQuantizer> quantizer,
            const size_t num_calculated_centroids) {
        this->num_centroids = quantizer->num_centroids();
        this->total_centroids_to_calculate = num_calculated_centroids;
        this->query = query;
        this->num_query_tokens = num_query_tokens;

        distances.resize(num_query_tokens * total_centroids_to_calculate);
        coarse_idx.resize(num_query_tokens * total_centroids_to_calculate);
        reordered_distances.resize(num_query_tokens * total_centroids_to_calculate);

        quantizer->search(
                num_query_tokens,
                query.data(),
                total_centroids_to_calculate,
                distances.data(),
                coarse_idx.data()
        );

        // We use this for ColBERT scoring.
        for (int i = 0; i <  num_centroids; i++) {
            for (int j = 0; j < total_centroids_to_calculate; j++) {
                auto current_code =
                        coarse_idx[i * total_centroids_to_calculate + j];
                float dis = distances[i * total_centroids_to_calculate + j];
                reordered_distances
                [i * total_centroids_to_calculate + current_code] =
                        dis;
            }
        }

    }

    std::vector<std::pair<float, idx_t>> KnnNearestCentroids::get_top_centroids(
            const size_t k_top_centroids,
            const size_t n_probe) const {

        if (top_centroids.size() == k_top_centroids) {
            return top_centroids;
        }
        // we're finding the highest centroid scores per centroid.
        std::vector<float> high_scores(num_centroids, 0);
        for (size_t i = 0; i < num_centroids; i++) {
            for (size_t j = 0; j < k_top_centroids; j++) {
                auto centroid_of_interest =
                        coarse_idx[i * num_centroids + j];
                // Note: including the centroid score threshold is not part of the
                // original colBERT model.
                // distances[i*total_centroids_to_calculate+j] >
                // centroid_score_threshold &&

                if (distances[i * num_centroids + j] >
                    high_scores[centroid_of_interest]) {
                    high_scores[centroid_of_interest] =
                            distances[i * num_centroids + j];
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
            if (score >= 0) {
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
} // lintdb