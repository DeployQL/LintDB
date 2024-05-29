#ifndef LINTDB_INVERTEDLISTSCANNER_H
#define LINTDB_INVERTEDLISTSCANNER_H

#include <memory>
#include <vector>
#include "ProductEncoder.h"
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/PQDistanceTables.h"
#include "lintdb/quantizers/Quantizer.h"
#include <map>

namespace lintdb {

/**
 * ScoredPartialDocumentCodes holds per-token scores to help calculate sum-of-max scores.
 *
 * Each token in a document is scored across the query tokens, and we want to keep
 * the max score per query token.
 */
struct ScoredPartialDocumentCodes {
    idx_t doc_id;
    idx_t token_id;
    std::map<idx_t, float> query_token_scores; // score per token stored in the inverted list
};

/**
 * InvertedListScanner helps us scan through an inverted list and score the results.
 *
 * The score is going to be a calculation between the stored codes, the centroid, and the query.
 */
class InvertedListScanner {
   public:
    InvertedListScanner(
            std::shared_ptr<ProductEncoder>& quantizer,
            const float* query_data,
            size_t num_tokens
    );

    std::vector<ScoredPartialDocumentCodes> scan(idx_t key, std::unique_ptr<Iterator> list_iterator, std::vector<idx_t>& query_tokens_to_score);

   private:
        std::unique_ptr<PQDistanceTables> distance_tables;
        size_t code_size;
        std::vector<idx_t> query_tokens_to_score;

};

} // namespace lintdb

#endif // LINTDB_INVERTEDLISTSCANNER_H
