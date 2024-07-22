#include "QueryNode.h"
#include <limits>
#include <vector>
#include <utility>
#include <memory>
#include "lintdb/query/QueryContext.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/query/KnnNearestCentroids.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/invlists/KeyBuilder.h"

namespace lintdb {
    std::unique_ptr<DocIterator> TermQueryNode::process(QueryContext& context, const SearchOptions& opts) {
        uint8_t field_id = context.getFieldMapper()->getFieldID(this->field);
        std::string prefix = create_index_prefix(context.getTenant(), field_id, this->value.data_type, this->value.value);
        std::unique_ptr<Iterator> it = context.getIndex()->get_iterator(prefix);
        return std::make_unique<TermIterator>(std::move(it), this->value.data_type);
    }

    std::unique_ptr<DocIterator> VectorQueryNode::process(QueryContext& context, const SearchOptions& opts) {
        uint8_t field_id = context.getFieldMapper()->getFieldID(this->field);
        DataType field_type = context.getFieldMapper()->getDataType(field_id);

        std::shared_ptr<KnnNearestCentroids> nearest_centroids = context.getOrCreateNearestCentroids(this->field);
        if (!nearest_centroids->is_valid()) {
            Tensor query = std::get<Tensor>(this->value.value);
            size_t num_centroids = context.getCoarseQuantizer(this->field)->num_centroids();
            size_t num_tensors = this->value.num_tensors;
            nearest_centroids->calculate(
                    query,
                    num_tensors,
                    context.getCoarseQuantizer(this->field),
                    num_centroids
                    );
        }

        std::vector<std::pair<float, idx_t>> top_centroids = nearest_centroids->get_top_centroids(opts.k_top_centroids, opts.n_probe);

        std::vector<std::unique_ptr<DocIterator>> iterators;

        for(const auto& centroid : top_centroids) {
            std::string prefix = create_index_prefix(context.getTenant(), field_id, DataType::QUANTIZED_TENSOR, centroid.second);
            std::unique_ptr<Iterator> it = context.getIndex()->get_iterator(prefix);
            if(!it->is_valid()) {
                LOG(WARNING) << "iterator is not valid for field: " << this->field << " and centroid: " << centroid.second;
                continue;
            }

            auto field_types = context.getFieldMapper()->getFieldTypes(field_id);

            auto doc_it = std::make_unique<TermIterator>(std::move(it), DataType::QUANTIZED_TENSOR, true);
            iterators.push_back(std::move(doc_it));
        }
        return std::make_unique<ANNIterator>(std::move(iterators));
    }

    std::unique_ptr<DocIterator> AndQueryNode::process(QueryContext& context, const SearchOptions& opts) {
        std::vector<std::unique_ptr<DocIterator>> iterators;
        for(const auto& child : children_) {
            iterators.emplace_back(child->process(context, opts));
        }
        return std::make_unique<AndIterator>(std::move(iterators));
    }
}

