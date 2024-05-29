//
// Created by matt on 5/25/24.
//

#include "InvertedListScanner.h"
#include <faiss/IndexPQ.h>
#include <glog/logging.h>
#include <memory>

namespace lintdb {
InvertedListScanner::InvertedListScanner(
        std::shared_ptr<ProductEncoder>& quantizer,
        const float* query_data,
        size_t num_tokens):
    code_size(quantizer->code_size()) {
    distance_tables = std::make_unique<PQDistanceTables>(query_data, num_tokens, quantizer->dim, quantizer->pq, true);
}

std::vector<ScoredPartialDocumentCodes> InvertedListScanner::scan(idx_t key, std::unique_ptr<Iterator> list_iterator, std::vector<idx_t>& query_tokens_to_score) {
    auto distance_to_query_tokens = distance_tables->precompute_list_tables(key);

    std::vector<ScoredPartialDocumentCodes> results;
    for (; list_iterator->has_next(); list_iterator->next()) {
        auto partial_codes = list_iterator->get_value();
        size_t num_tokens = partial_codes.partial_residuals.size() / code_size;
        if (num_tokens != 1) {
            LOG(WARNING) << "Codes found in inverted index are the wrong size. residual size: " << partial_codes.partial_residuals.size() << " code size: " << code_size;
        }

        ScoredPartialDocumentCodes doc_results;
        auto token_key = list_iterator->get_token_key();
        doc_results.doc_id = token_key.doc_id;
        doc_results.token_id = token_key.token_id;

        auto scores = distance_tables->calculate_query_distances(
                query_tokens_to_score,
                distance_to_query_tokens,
                partial_codes.partial_residuals.data()
        );

        for(idx_t i=0; i < scores.size(); i++) {
            const auto query_token_id = query_tokens_to_score[i];
            doc_results.query_token_scores.insert({query_token_id, scores[i]});
        }

        results.push_back(doc_results);
    }
    return results;

}

} // namespace lintdb