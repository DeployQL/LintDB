#include "QueryExecutor.h"
#include <vector>
#include "decode.h"
#include "DocIterator.h"
#include "DocValue.h"
#include "lintdb/query/KnnNearestCentroids.h"
#include "lintdb/scoring/ContextCollector.h"
#include "lintdb/scoring/ScoredDocument.h"

namespace lintdb {
QueryExecutor::QueryExecutor(Scorer& ranker)
        : ranker(ranker) {}

std::vector<ScoredDocument> QueryExecutor::execute(
        QueryContext& context,
        const Query& query,
        const size_t num_results,
        const SearchOptions& opts) {
    std::unique_ptr<DocIterator> doc_it = query.root->process(context, opts);

    std::vector<std::pair<idx_t, std::vector<DocValue>>> documents;
    for(; doc_it->is_valid(); doc_it->advance()) {
        std::vector<DocValue> dvs = doc_it->fields();

        documents.emplace_back(doc_it->doc_id(), dvs);
    }

    std::vector<ScoredDocument> results(documents.size());
#pragma omp parallel for if(documents.size() > 100)
    for(int i = 0; i < documents.size(); i++) {
        auto doc = documents[i];
//        for (auto& dv : doc.second) {
//            // ColBERT is a special case where we don't have a value to decode.
//            if (dv.unread_value) {
//                continue;
//            }
//            dv = decode_vectors(context, dv);
//        }
        ScoredDocument scored = doc_it->score(doc.second);
        scored.doc_id = doc.first;

        if (opts.expected_id != -1 && doc.first == opts.expected_id) {
            LOG(INFO) << "\tscore: " << scored.score;
        }

        results[i] = scored;
    } // end for

    std::sort(results.begin(), results.end(), std::greater<>());

    size_t num_to_rank = std::min(results.size(), opts.num_second_pass);

    std::vector<ScoredDocument> top_results_ranked(num_to_rank);
    for (size_t i = 0; i < num_to_rank; i++) {
        top_results_ranked[i] = ranker.score(
                context, results[i].doc_id, results[i].values);
    }

    std::sort(
            top_results_ranked.begin(),
            top_results_ranked.end(),
            std::greater<>());

    // return num_results from top_results_ranked
    std::vector<ScoredDocument> final_results;
    for (size_t i = 0; i < num_results && i < top_results_ranked.size(); i++) {
        final_results.push_back(top_results_ranked[i]);
    }

    return final_results;
}

} // namespace lintdb