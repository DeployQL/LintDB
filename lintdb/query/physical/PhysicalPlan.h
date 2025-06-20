#pragma once

#include <memory>
#include <string>
#include <vector>
#include <arrow/record_batch.h>
#include "lintdb/schema/Schema.h"

namespace lintdb {
namespace query {

/**
 * @brief Base interface for all physical plans
 */
class PhysicalPlan {
public:
    virtual ~PhysicalPlan() = default;

    /**
     * @brief Get the schema of the plan
     * @return The schema
     */
    virtual std::shared_ptr<Schema> schema() const = 0;

    /**
     * @brief Execute the plan and return a sequence of record batches
     * @return Sequence of record batches
     */
    virtual std::vector<std::shared_ptr<arrow::RecordBatch>> execute() = 0;

    /**
     * @brief Get the child plans of this plan
     * @return Vector of child plans
     */
    virtual std::vector<std::shared_ptr<PhysicalPlan>> children() const = 0;

    /**
     * @brief Get a string representation of this plan
     * @return The string representation
     */
    virtual std::string to_string() const = 0;

protected:
    /**
     * @brief Add a child plan
     * @param child The child plan to add
     */
    void add_child(std::shared_ptr<PhysicalPlan> child) {
        children_.push_back(std::move(child));
    }

    /**
     * @brief Get the number of children
     * @return Number of children
     */
    size_t num_children() const {
        return children_.size();
    }

    /**
     * @brief Get a specific child by index
     * @param index The index of the child to get
     * @return The child plan at the specified index
     * @throws std::out_of_range if index is invalid
     */
    std::shared_ptr<PhysicalPlan> get_child(size_t index) const {
        if (index >= children_.size()) {
            throw std::out_of_range("Child index out of range");
        }
        return children_[index];
    }

    /**
     * @brief Clear all children
     */
    void clear_children() {
        children_.clear();
    }

    std::vector<std::shared_ptr<PhysicalPlan>> children_;
};

} // namespace query
} // namespace lintdb 