#include "lintdb/plaid.h"
#include <faiss/utils/hamming.h>
#include <iostream>
#include "lintdb/api.h"
#include <unordered_set>

namespace lintdb {
    // returns a list of scores for each centroid.
    float score_documents_by_codes(
        const af::array& query_scores, // query scores are the max centroid scores per query token.
        const code_t* doc_codes,
        size_t num_tokens
    ) {
        // Initialize a vector to store the approximate scores for each document
        float doc_score = 0;
        std::unordered_set<code_t> unique_codes;
        // Iterate over each token code.
        for (size_t i = 0; i < num_tokens; i++) {
            auto index = doc_codes[i];
            if (unique_codes.find(index) != unique_codes.end()) {
                continue;
            }

            // we get the centroid score from query_scores. this is the max score found in the query for that centroid.
            doc_score += query_scores(index).scalar<float>();

            unique_codes.insert(index);
        }

        return doc_score;
    }
}