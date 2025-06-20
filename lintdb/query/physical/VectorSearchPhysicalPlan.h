#pragma once

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <faiss/utils/distances.h>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/datasources/DataSource.h"
#include "lintdb/datasources/serialization.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/invlists/KeyBuilder.h"

namespace lintdb {
namespace query {

/**
 * @brief Physical vector search plan that performs nearest neighbor search
 */
class VectorSearchPhysicalPlan : public PhysicalPlan {
public:
    VectorSearchPhysicalPlan(
        const std::string& column_name,
        std::shared_ptr<DataSource> data_source,
        uint64_t tenant_id,
        const std::vector<float>& query_vector,
        size_t k,
        const std::string& distance_metric = "L2")
        : data_source_(data_source),
          tenant_id_(tenant_id),
          column_name_(column_name),
          query_vector_(query_vector),
          k_(k),
          distance_metric_(distance_metric) {
    }

    std::shared_ptr<Schema> schema() const override { return get_child(0)->schema(); }

    std::vector<std::shared_ptr<arrow::RecordBatch>> execute() override {
        std::vector<std::shared_ptr<arrow::RecordBatch>> result;

        auto schema = this->schema();

        // Get the field definition for the column we're searching
        const Field& field = schema->get_field(column_name_);

        std::string prefix = create_index_prefix(tenant_id_, field.field_id, field.data_type, 0);
        auto iter = data_source_->scan_prefix(prefix);
        
        // Priority queue to maintain top-k results
        // For L2 distance, we want smaller distances (max heap)
        // For inner product, we want larger values (min heap)
        bool is_similarity = (distance_metric_ == "cosine" || distance_metric_ == "ip");
        using Result = std::pair<float, idx_t>;
        auto compare = [is_similarity](const Result& a, const Result& b) {
            return is_similarity ? a.first > b.first : a.first < b.first;
        };
        std::priority_queue<Result, std::vector<Result>, decltype(compare)> top_k(compare);

        // Process each vector in the iterator
        while (iter->is_valid()) {
            auto key = iter->key();
            auto value = iter->value();
            
            // Decode the value using our serialization system
            auto decoded_value = deserialize_field_value(field, value);
            
            // Get the vector from the decoded value
            if (const auto* tensor = std::get_if<Tensor>(&decoded_value)) {
                // Compute distance based on metric
                float distance;
                if (distance_metric_ == "L2") {
                    distance = faiss::fvec_L2sqr(query_vector_.data(), tensor->data(), tensor->size());
                } else if (distance_metric_ == "cosine") {
                    // For cosine similarity, we compute inner product of normalized vectors
                    float ip = faiss::fvec_inner_product(query_vector_.data(), tensor->data(), tensor->size());
                    float norm1 = faiss::fvec_norm_L2sqr(query_vector_.data(), query_vector_.size());
                    float norm2 = faiss::fvec_norm_L2sqr(tensor->data(), tensor->size());
                    distance = ip / (std::sqrt(norm1) * std::sqrt(norm2));
                } else if (distance_metric_ == "ip") {
                    distance = faiss::fvec_inner_product(query_vector_.data(), tensor->data(), tensor->size());
                } else {
                    throw std::runtime_error("Unsupported distance metric: " + distance_metric_);
                }

                // Add to top-k results
                if (top_k.size() < k_ || compare({distance, std::stoll(key)}, top_k.top())) {
                    top_k.push({distance, std::stoll(key)});
                    if (top_k.size() > k_) {
                        top_k.pop();
                    }
                }
            }
            
            iter->advance();
        }

        // Create Arrow arrays for results
        arrow::Int64Builder id_builder;
        arrow::FloatBuilder distance_builder;
        
        // Convert priority queue to sorted vector
        std::vector<Result> sorted_results;
        while (!top_k.empty()) {
            sorted_results.push_back(top_k.top());
            top_k.pop();
        }
        std::reverse(sorted_results.begin(), sorted_results.end());

        // Build arrays
        for (const auto& [distance, id] : sorted_results) {
            auto status = id_builder.Append(id);
            if (!status.ok()) {
                throw std::runtime_error("Failed to append ID: " + status.ToString());
            }
            status = distance_builder.Append(distance);
            if (!status.ok()) {
                throw std::runtime_error("Failed to append distance: " + status.ToString());
            }
        }

        // Create arrays
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> distance_array;
        auto status = id_builder.Finish(&id_array);
        if (!status.ok()) {
            throw std::runtime_error("Failed to finish ID array: " + status.ToString());
        }
        status = distance_builder.Finish(&distance_array);
        if (!status.ok()) {
            throw std::runtime_error("Failed to finish distance array: " + status.ToString());
        }

        // Create schema for results
        auto result_schema = arrow::schema({
            arrow::field("id", arrow::int64()),
            arrow::field("distance", arrow::float32())
        });

        // Create record batch
        auto batch = arrow::RecordBatch::Make(
            result_schema,
            sorted_results.size(),
            {id_array, distance_array}
        );
        
        result.push_back(batch);
        return result;
    }

    std::vector<std::shared_ptr<PhysicalPlan>> children() const override {
        return children_;
    }

    std::string to_string() const override {
        return "PhysicalVectorSearch(k=" + std::to_string(k_) + 
               ", metric=" + distance_metric_ + ")";
    }

private:
    std::shared_ptr<DataSource> data_source_;
    uint64_t tenant_id_;
    std::string column_name_;
    std::vector<float> query_vector_;
    size_t k_;
    std::string distance_metric_;
    std::vector<std::shared_ptr<PhysicalPlan>> children_;
};

} // namespace query
} // namespace lintdb 