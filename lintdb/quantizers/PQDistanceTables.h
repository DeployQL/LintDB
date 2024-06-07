#ifndef LINTDB_PQDISTANCETABLES_H
#define LINTDB_PQDISTANCETABLES_H

#include <cstddef>
#include <list>
#include <vector>
#include <memory>
#include "lintdb/api.h"

namespace faiss {
struct IndexPQ;
}

namespace lintdb {

/**
 * PQDistanceTables calculates scores for a given query token and doc token.
 *
 * This class holds all of the compute logic and returns the pieces of the calculation
 * to InvertedListScanner.
 */
class PQDistanceTables {
   public:
    PQDistanceTables(const float* query_data, size_t num_tokens, size_t dim, std::shared_ptr<faiss::IndexPQ> ipq, bool is_ip = true);

    /**
     * precompute_list_tables precomputes the distance to the list's centroid
     * using the quantizer. We store the initial distance to each query token.
     */
//    std::vector<float> precompute_list_tables(const std::vector<idx_t>& query_token_ids);

    std::vector<float> calculate_query_distances(const std::vector<idx_t>& query_tokens_to_score, const std::vector<float>& precomputed_distances, const uint8_t* codes);


   private:
    std::vector<std::vector<float>> distance_tables;
    std::shared_ptr<faiss::IndexPQ> ipq;
    bool is_ip;
    size_t dim;

};

} // namespace lintdb

#endif // LINTDB_PQDISTANCETABLES_H
