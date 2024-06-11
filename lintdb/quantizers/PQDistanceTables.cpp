#include "PQDistanceTables.h"
#include <faiss/IndexPQ.h>
#include "lintdb/assert.h"
#include <faiss/utils/distances.h>
#include <faiss/impl/code_distance/code_distance.h>
#include <faiss/impl/ProductQuantizer.h>
#include <glog/logging.h>

namespace lintdb {
PQDistanceTables::PQDistanceTables(
            const float* query_data,
            size_t num_tokens,
            size_t dim,
            const std::shared_ptr<faiss::IndexPQ> ipq,
            bool is_ip
        ): ipq(ipq), is_ip(is_ip), dim(dim) {

    // right now, we only support IP.
    LINTDB_THROW_IF_NOT(ipq->metric_type == faiss::METRIC_INNER_PRODUCT);

    for (size_t i = 0; i < num_tokens; i++) {
        std::vector<float> distance_table(ipq->pq.M * ipq->pq.ksub);
        ipq->pq.compute_inner_prod_table(query_data + i * dim, distance_table.data());
        distance_tables.push_back(distance_table);
    }

}

std::vector<float> PQDistanceTables::calculate_query_distances(
        const std::vector<idx_t>& query_tokens_to_score,
        const std::vector<float>& precomputed_distances,
        const std::vector<uint8_t>& codes) {
    std::vector<float> results(precomputed_distances);
//    for(auto pc : precomputed_distances) {
//        LOG(INFO) << "precomputed distance: " << pc;
//    }

    for(int j = 0; j < query_tokens_to_score.size(); j++) {
        auto query_token_id = query_tokens_to_score[j];
        auto sim_table = distance_tables[query_token_id];
        float score = faiss::distance_single_code<faiss::PQDecoderGeneric>(
                ipq->pq,
                sim_table.data(),
                codes.data()
        );
//        LOG(INFO) << "score: " << score;
        results[j] += score;

    }
    return results;
}

} // namespace lintdb