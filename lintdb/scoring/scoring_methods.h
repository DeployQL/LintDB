#pragma once

#include "lintdb/schema/DataTypes.h"
#include "lintdb/query/DocValue.h"
#include "lintdb/query/KnnNearestCentroids.h"
#include "lintdb/scoring/plaid.h"
#include <vector>
#include <string>
#include <memory>

namespace lintdb {

typedef double score_t;
typedef score_t (*UnaryScoringMethodFunction)(const std::vector<DocValue>& values);
typedef score_t (*NaryScoringMethodFunction)(const std::vector<score_t>& values);
typedef score_t (*EmbeddingScoringMethodFunction)(const std::vector<DocValue>& values, std::shared_ptr<KnnNearestCentroids> knn);

score_t score_one(const std::vector<DocValue>& values);

score_t plaid_similarity(const std::vector<DocValue>& values, std::shared_ptr<KnnNearestCentroids> knn);


enum class UnaryScoringMethod {
    ONE = 0,
};

score_t score(const UnaryScoringMethod method, const std::vector<DocValue>& values);

enum class EmbeddingScoringMethod {
    PLAID = 0,
    COLBERT = 1
};

score_t score_embeddings(const EmbeddingScoringMethod method, const std::vector<DocValue>& values, std::shared_ptr<KnnNearestCentroids> knn);

score_t sum(const std::vector<score_t>& values);

score_t reduce(const std::vector<score_t>& values);

score_t max(const std::vector<score_t>& values);

enum class NaryScoringMethod {
    SUM = 0,
    REDUCE = 1,
    MAX = 2,
};
score_t score(const NaryScoringMethod method, const std::vector<score_t>& values);

}