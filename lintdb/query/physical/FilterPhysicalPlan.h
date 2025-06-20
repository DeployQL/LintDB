#pragma once

#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/query/LogicalExpr.h"

namespace lintdb {
namespace query {

/**
 * @brief Physical filter plan that applies a predicate
 */
class FilterPhysicalPlan : public PhysicalPlan {
public:
    FilterPhysicalPlan(
        std::shared_ptr<PhysicalPlan> child,
        std::shared_ptr<LogicalExpr> filter)
        : filter_(std::move(filter)) {
        add_child(std::move(child));
    }

    std::shared_ptr<Schema> schema() const override { return get_child(0)->schema(); }

    std::vector<std::shared_ptr<arrow::RecordBatch>> execute() override {
        std::vector<std::shared_ptr<arrow::RecordBatch>> result;
        auto input_batches = get_child(0)->execute();
        
        for (const auto& batch : input_batches) {
            // TODO: Implement filter logic
            // 1. Evaluate the filter expression on the input batch
            // 2. Create a new RecordBatch with only the rows that match
            
            result.push_back(batch);  // Placeholder
        }
        
        return result;
    }

    std::vector<std::shared_ptr<PhysicalPlan>> children() const override {
        return children_;
    }

    std::string to_string() const override {
        return "PhysicalFilter(" + filter_->to_string() + ")";
    }

private:
    std::shared_ptr<LogicalExpr> filter_;
};

} // namespace query
} // namespace lintdb 