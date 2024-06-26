#ifndef LINTDB_SEARCH_RESULT_H
#define LINTDB_SEARCH_RESULT_H

#include <map>
#include <string>
#include <vector>
#include "lintdb/api.h"

namespace lintdb {

/**
 * SearchResult is a simple struct to hold the results of a search.
 *
 */
struct SearchResult {
    idx_t id;    /// the document's id.
    float score; /// the final score as determined by the database.
    std::map<std::string, std::string>
            metadata; /// Optionally, metadata that was indexed for the
                      /// document.
    std::vector<float> token_scores; /// Document token scores.

    SearchResult() = default;

    bool operator<(const SearchResult& other) const {
        return score < other.score;
    }
    bool operator>(const SearchResult& other) const {
        return score > other.score;
    }
};

} // namespace lintdb

#endif