#include "InvertedListScanner.h"
#include <faiss/IndexPQ.h>
#include <faiss/utils/distances.h>
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace lintdb {
InvertedListScanner::InvertedListScanner(
        std::shared_ptr<ProductEncoder>& quantizer,
        const float* query_data,
        size_t num_tokens)
        : quantizer(quantizer), code_size(quantizer->code_size()) {
    distance_tables = quantizer->get_distance_tables(query_data, num_tokens);
}

std::vector<ScoredPartialDocumentCodes> InvertedListScanner::scan(
        const idx_t key,
        const std::unique_ptr<Iterator> list_iterator,
        const std::vector<QueryTokenCentroidScore>& query_tokens_to_score) {
    std::vector<idx_t> query_token_ids;
    query_token_ids.reserve(query_tokens_to_score.size());
    for (const auto& q : query_tokens_to_score) {
        query_token_ids.push_back(q.query_token);
    }

    std::vector<float> precomputed_distances;
    precomputed_distances.reserve(query_tokens_to_score.size());
    for (const auto& q : query_tokens_to_score) {
        precomputed_distances.push_back(q.distance);
    }

    std::vector<ScoredPartialDocumentCodes> results;
    for (; list_iterator->has_next(); list_iterator->next()) {
        auto partial_codes = list_iterator->get_value();
        size_t num_tokens = partial_codes.partial_residuals.size() / code_size;
        if (num_tokens != 1) {
            LOG(WARNING)
                    << "Codes found in inverted index are the wrong size. residual size: "
                    << partial_codes.partial_residuals.size()
                    << " code size: " << code_size;
        }

        ScoredPartialDocumentCodes doc_results;
        auto token_key = list_iterator->get_key();
        doc_results.doc_id = token_key.doc_id;
        doc_results.doc_token_id = token_key.token_id;

        auto scores = distance_tables->calculate_query_distances(
                query_token_ids,
                precomputed_distances,
                partial_codes.partial_residuals);

        for (idx_t i = 0; i < scores.size(); i++) {
            const auto query_token_id = query_token_ids[i];
            doc_results.query_token_id = query_token_id;
            doc_results.score = scores[i];
        }

        results.push_back(doc_results);
    }
    return results;
}

} // namespace lintdb