#ifndef LINTDB_SEARCH_RESULT_H
#define LINTDB_SEARCH_RESULT_H

#include "lintdb/api.h"

namespace lintdb {

/**
 * SearchResult is a simple struct to hold the results of a search.
 *
 */
struct SearchResult {
    idx_t id;
    float score;
};
} // namespace lintdb

#endif