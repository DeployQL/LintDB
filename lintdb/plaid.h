#ifndef LINTDB_PLAID_H
#define LINTDB_PLAID_H

#include <vector>
#include <bitset>
#include <algorithm>
#include "lintdb/api.h"
#include <arrayfire.h>

namespace lintdb {
    /**
     * score_documents_by_codes aggregates a document score based on each token's
     * code and how well it matches the query.
     * 
     * We return the list of scores for each centroid.
    */
    float score_documents_by_codes(
        const af::array& query_scores, // the max score pe
        const code_t* doc_codes,
        size_t num_tokens
    );
}

#endif