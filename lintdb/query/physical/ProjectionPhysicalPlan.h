#pragma once

#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/query/LogicalExpr.h"

namespace lintdb {
namespace query {

/**
 * @brief Physical projection plan that selects specific columns
 */
class ProjectionPhysicalPlan : public PhysicalPlan {
public:
    ProjectionPhysicalPlan(
        std::shared_ptr<PhysicalPlan> child,
        const std::vector<std::shared_ptr<LogicalExpr>>& projections)
        : projections_(projections) {
        add_child(std::move(child));
    }

    std::shared_ptr<Schema> schema() const override {
        // TODO: Implement schema derivation from projections
        return get_child(0)->schema();
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> execute() override {
        std::vector<std::shared_ptr<arrow::RecordBatch>> result;
        auto input_batches = get_child(0)->execute();
        
        for (const auto& batch : input_batches) {
            // TODO: Implement projection logic
            // For each projection expression:
            // 1. Evaluate the expression on the input batch
            // 2. Create a new RecordBatch with the projected columns
            
            result.push_back(batch);  // Placeholder
        }
        
        return result;
    }

    std::vector<std::shared_ptr<PhysicalPlan>> children() const override {
        return children_;
    }

    std::string to_string() const override {
        std::string result = "PhysicalProject(";
        for (size_t i = 0; i < projections_.size(); ++i) {
            if (i > 0) result += ", ";
            result += projections_[i]->to_string();
        }
        result += ")";
        return result;
    }

private:
    std::vector<std::shared_ptr<LogicalExpr>> projections_;
};

} // namespace query
} // namespace lintdb 