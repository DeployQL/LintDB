#include "QueryExecutor.h"
#include "DocIterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/DocEncoder.h"
#include <queue>
#include <vector>
#include <unordered_map>
#include "lintdb/query/KnnNearestCentroids.h"
#include "DocValue.h"

namespace lintdb {
    QueryExecutor::QueryExecutor(Scorer& scorer):
            scorer(scorer)  {}

    std::vector<SearchResult> QueryExecutor::execute(
            QueryContext &context,
            const Query &query,
            const size_t num_results,
            const SearchOptions& opts) {
        std::unique_ptr<DocIterator> doc_it = query.root->process(context, opts);

        std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<>> results;
        for(;doc_it->is_valid(); doc_it->advance()) {
            LOG(INFO) << "doc id: " << doc_it->doc_id();
            SearchResult result;
            result.id = doc_it->doc_id();

            std::vector<DocValue> dvs = doc_it->fields();

            float score = scorer.score(doc_it->doc_id(), dvs);
            result.score = score;

            results.emplace(result);
        }

        std::vector<SearchResult> top_results;
        while(!results.empty() && top_results.size() < num_results) {
            top_results.push_back(results.top());
            results.pop();
        }

        return top_results;
    }
} // lintdb