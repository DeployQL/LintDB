#pragma once

#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/query/DocValue.h"
#include "lintdb/schema/FieldMapper.h"
#include "lintdb/scoring/Scorer.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/SearchResult.h"
#include "Query.h"
#include "QueryContext.h"
#include "lintdb/scoring/ScoredDocument.h"

namespace lintdb {
/**
 * QueryExecutor helps manage the execution of queries.
 *
 * The basic flow of retrieval is:
 * 1. Optimize the query.
 * 2. Translate the query into a series of document iterators.
 * 3. Scan those iterators to retrieve the right documents.
 * 4. Score the documents.
 *
 */
class QueryExecutor {
   public:
    QueryExecutor(Scorer& ranker);

    std::vector<ScoredDocument> execute(
            QueryContext& context,
            const Query& query,
            const size_t num_results,
            const SearchOptions& opts);

   private:
    Scorer& ranker;
};

} // namespace lintdb
