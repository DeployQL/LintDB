#include "QueryExecutor.h"
#include "DocIterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/DocEncoder.h"
#include <queue>
#include <vector>
#include <unordered_map>
#include "lintdb/query/KnnNearestCentroids.h"
#include "DocValue.h"
#include "decode.h"

namespace lintdb {
    QueryExecutor::QueryExecutor(Scorer& retriever, Scorer& ranker):
            retriever(retriever), ranker(ranker) {}

    std::vector<ScoredDocument> QueryExecutor::execute(
            QueryContext &context,
            const Query &query,
            const size_t num_results,
            const SearchOptions& opts) {
        std::unique_ptr<DocIterator> doc_it = query.root->process(context, opts);

        std::priority_queue<ScoredDocument, std::vector<ScoredDocument>, std::less<>> results;
        for(;doc_it->is_valid(); doc_it->advance()) {
            if (opts.expected_id != -1 && doc_it->doc_id() == opts.expected_id) {
                LOG(INFO) << "expected document found";
            }
            std::vector<DocValue> dvs = doc_it->fields();
            // optionally decode quantized tensors from fields.
            for (auto & dv : dvs) {
                // ColBERT is a special case where we don't have a value to decode.
                if(dv.unread_value) {
                    continue;
                }
                dv = decode_vectors(context, dv);
            }

            ScoredDocument scored = retriever.score(context, doc_it->doc_id(), dvs);
            if (opts.expected_id != -1 && doc_it->doc_id() == opts.expected_id) {
                LOG(INFO) << "\tscore: " << scored.score;
            }
            results.push(scored);
        }

        std::vector<ScoredDocument> top_results;
        while(!results.empty() && top_results.size() < opts.num_second_pass) {
            top_results.push_back(results.top());
            results.pop();
        }

        std::vector<ScoredDocument> top_results_ranked(top_results.size());
        for (size_t i = 0; i < top_results.size(); i++) {
            top_results_ranked[i] = ranker.score(context, top_results[i].doc_id, top_results[i].values);
        }

        std::sort(top_results_ranked.begin(), top_results_ranked.end(), [](const ScoredDocument& a, const ScoredDocument& b) {
            return a.score > b.score;
        });

        // return num_results from top_results_ranked
        std::vector<ScoredDocument> final_results;
        for (size_t i = 0; i < num_results && i < top_results_ranked.size(); i++) {
            final_results.push_back(top_results_ranked[i]);
        }

        return final_results;
    }

} // lintdb