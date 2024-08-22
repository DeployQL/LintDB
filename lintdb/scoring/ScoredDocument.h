#pragma once
#include <memory>
#include <vector>
#include "lintdb/query/DocValue.h"

namespace lintdb {
struct ScoredDocument {
    double score = 0;
    idx_t doc_id = -1;
    std::vector<lintdb::DocValue>
            values; /// ScoredDocument takes ownership of the values, because
    /// we assume we are iterating over a DocIterator and the values are only
    /// valid for the duration of the iteration.

    ScoredDocument() = default;

    ScoredDocument(float score, idx_t doc_id, std::vector<lintdb::DocValue> values)
            : score(score), doc_id(doc_id), values(std::move(values)) {}

    bool operator<(const ScoredDocument& other) const {
        return score < other.score;
    }

    bool operator>(const ScoredDocument& other) const {
        return score > other.score;
    }
};
}
