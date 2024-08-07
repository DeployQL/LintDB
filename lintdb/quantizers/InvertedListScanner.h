#ifndef LINTDB_INVERTEDLISTSCANNER_H
#define LINTDB_INVERTEDLISTSCANNER_H

#include <map>
#include <memory>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/PQDistanceTables.h"
#include "lintdb/quantizers/Quantizer.h"
#include "ProductEncoder.h"

namespace lintdb {

/**
 * ScoredPartialDocumentCodes holds per-token scores to help calculate
 * sum-of-max scores.
 *
 * Each token in a document is scored across the query tokens, and we want to
 * keep the max score per query token.
 */
struct ScoredPartialDocumentCodes {
    idx_t doc_id;
    idx_t doc_token_id;
    idx_t query_token_id;
    float score;

    ScoredPartialDocumentCodes() = default;

    ScoredPartialDocumentCodes(
            idx_t doc_id,
            idx_t doc_token_id,
            idx_t query_token_id,
            float score)
            : doc_id(doc_id),
              doc_token_id(doc_token_id),
              query_token_id(query_token_id),
              score(score) {}
};

/**
 * QueryTokenCentroidScore holds the distance between a query token and a
 centroid.
 *
 * This is passed to scan to help calculate the score of a token.

 */
struct QueryTokenCentroidScore {
    idx_t query_token;
    idx_t centroid_id;
    float distance;
};

/**
 * InvertedListScanner helps us scan through an inverted list and score the
 * results.
 *
 * The score is going to be a calculation between the stored codes, the
 * centroid, and the query.
 */
class InvertedListScanner {
   public:
    InvertedListScanner(
            std::shared_ptr<ProductEncoder>& quantizer,
            const float* query_data,
            size_t num_tokens);

    std::vector<ScoredPartialDocumentCodes> scan(
            idx_t key,
            const std::unique_ptr<Iterator> list_iterator,
            const std::vector<QueryTokenCentroidScore>& query_tokens_to_score);

   private:
    std::unique_ptr<PQDistanceTables> distance_tables;
    std::shared_ptr<ProductEncoder> quantizer;
    size_t code_size;
};

} // namespace lintdb

#endif // LINTDB_INVERTEDLISTSCANNER_H
