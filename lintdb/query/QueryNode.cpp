#include "QueryNode.h"
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/KeyBuilder.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/query/KnnNearestCentroids.h"
#include "lintdb/query/QueryContext.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/scoring/ContextCollector.h"

namespace lintdb {
std::unique_ptr<DocIterator> TermQueryNode::process(
        QueryContext& context,
        const SearchOptions& opts) {
    uint8_t field_id = context.getFieldMapper()->getFieldID(this->value.name);
    std::string prefix = create_index_prefix(
            context.getTenant(),
            field_id,
            this->value.data_type,
            this->value.value);
    std::unique_ptr<Iterator> it = context.getIndex()->get_iterator(prefix);
    return std::make_unique<TermIterator>(
            std::move(it), this->value.data_type, score_method);
}

std::unique_ptr<DocIterator> VectorQueryNode::process(
        QueryContext& context,
        const SearchOptions& opts) {
    uint8_t field_id = context.getFieldMapper()->getFieldID(this->value.name);
    DataType field_type = context.getFieldMapper()->getDataType(field_id);

    std::shared_ptr<KnnNearestCentroids> nearest_centroids =
            context.getOrCreateNearestCentroids(this->value.name);
    size_t num_centroids =
            context.getCoarseQuantizer(this->value.name)->num_centroids();
    if (!nearest_centroids->is_valid()) {
        Tensor query = std::get<Tensor>(this->value.value);

        size_t num_tensors = this->value.num_tensors;
        nearest_centroids->calculate(
                query,
                num_tensors,
                context.getCoarseQuantizer(this->value.name),
                num_centroids);
    }

    size_t max_centroids = std::min(opts.k_top_centroids, num_centroids);

    std::vector<std::pair<float, idx_t>> top_centroids =
            nearest_centroids->get_top_centroids(max_centroids, opts.n_probe);

    std::vector<std::unique_ptr<DocIterator>> iterators;

    std::vector<idx_t> invalid_centroids;
    std::vector<idx_t> valid_centroids;
    for (const auto& centroid : top_centroids) {
        std::string prefix = create_index_prefix(
                context.getTenant(),
                field_id,
                DataType::QUANTIZED_TENSOR,
                centroid.second);
        std::unique_ptr<Iterator> it = context.getIndex()->get_iterator(prefix);
        if (!it->is_valid()) {
            invalid_centroids.push_back(centroid.second);
            continue;
        }
        valid_centroids.push_back(centroid.second);

        auto field_types = context.getFieldMapper()->getFieldTypes(field_id);

        auto doc_it = std::make_unique<TermIterator>(
                std::move(it), DataType::QUANTIZED_TENSOR, UnaryScoringMethod::ONE, true);
        iterators.push_back(std::move(doc_it));
    }
    VLOG(5) << "Invalid centroids: " << invalid_centroids.size() << " out of "
            << top_centroids.size();
    // get unqiue centroids
    std::sort(invalid_centroids.begin(), invalid_centroids.end());
    invalid_centroids.erase(
            std::unique(invalid_centroids.begin(), invalid_centroids.end()),
            invalid_centroids.end());

    for (const auto& invalid_centroid : invalid_centroids) {
        VLOG(5) << "Invalid centroid: " << invalid_centroid;
    }
    for (const auto& valid_centroid : valid_centroids) {
        VLOG(5) << "Valid centroid: " << valid_centroid;
    }

    // TODO (mbarta): We can improve how we encapsulate the context collection for scoring.
    // TODO (mbarta): We can validate that the fields require context by checking the field.
    ContextCollector context_collector;
    if (score_method == EmbeddingScoringMethod::PLAID || score_method == EmbeddingScoringMethod::COLBERT) {
       context_collector.add_field(context, this->value.name);
    }

    return std::make_unique<ANNIterator>(std::move(iterators), std::move(context_collector), std::move(nearest_centroids), score_method);
}

std::unique_ptr<DocIterator> AndQueryNode::process(
        QueryContext& context,
        const SearchOptions& opts) {
    std::vector<std::unique_ptr<DocIterator>> iterators;
    for (const auto& child : children_) {
        iterators.emplace_back(child->process(context, opts));
    }
    return std::make_unique<AndIterator>(std::move(iterators), score_method);
}

std::unique_ptr<DocIterator> OrQueryNode::process(
        QueryContext& context,
        const SearchOptions& opts) {
    std::vector<std::unique_ptr<DocIterator>> iterators;
    for (const auto& child : children_) {
        iterators.emplace_back(child->process(context, opts));
    }
    return std::make_unique<OrIterator>(std::move(iterators), score_method);
}
} // namespace lintdb
