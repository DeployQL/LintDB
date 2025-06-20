#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lintdb/query/LogicalExpr.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/datasources/DataSource.h"

namespace lintdb {
namespace query {

/**
 * @brief Interface for all logical plans
 */
class LogicalPlan {
public:
    virtual ~LogicalPlan() = default;

    /**
     * @brief Get the schema of the plan
     * @return The schema
     */
    virtual std::shared_ptr<Schema> schema() const = 0;

    /**
     * @brief Get the child plans of this plan
     * @return Vector of child plans
     */
    virtual std::vector<std::shared_ptr<LogicalPlan>> children() const = 0;

    /**
     * @brief Get a string representation of this plan
     * @return The string representation
     */
    virtual std::string to_string() const = 0;

    /**
     * @brief Get the data source associated with this plan
     * @return The data source
     */
    virtual const std::shared_ptr<DataSource>& data_source() const = 0;

    /**
     * @brief Get the tenant ID associated with this plan
     * @return The tenant ID
     */
    virtual uint64_t tenant_id() const = 0;
};

/**
 * @brief Abstract base class implementing common functionality for logical plans
 */
class AbstractLogicalPlan : public LogicalPlan {
public:
    AbstractLogicalPlan(std::shared_ptr<DataSource> data_source, uint64_t tenant_id)
        : data_source_(std::move(data_source)), tenant_id_(tenant_id) {}

    const std::shared_ptr<DataSource>& data_source() const override { return data_source_; }
    uint64_t tenant_id() const override { return tenant_id_; }

protected:
    std::shared_ptr<DataSource> data_source_;
    uint64_t tenant_id_;
};

/**
 * @brief Scan plan that reads data from a data source
 */
class ScanPlan : public AbstractLogicalPlan {
public:
    ScanPlan(std::shared_ptr<DataSource> data_source, uint64_t tenant_id, std::shared_ptr<Schema> schema)
        : AbstractLogicalPlan(std::move(data_source), tenant_id), schema_(std::move(schema)) {}

    std::shared_ptr<Schema> schema() const override { return schema_; }
    std::vector<std::shared_ptr<LogicalPlan>> children() const override { return {}; }
    std::string to_string() const override { return "Scan"; }

private:
    std::shared_ptr<Schema> schema_;
};

/**
 * @brief Projection plan that selects specific columns
 */
class ProjectionPlan : public AbstractLogicalPlan {
public:
    ProjectionPlan(
        std::shared_ptr<DataSource> data_source,
        uint64_t tenant_id,
        std::shared_ptr<LogicalPlan> child,
        const std::vector<std::shared_ptr<LogicalExpr>>& projections)
        : AbstractLogicalPlan(std::move(data_source), tenant_id),
          child_(std::move(child)),
          projections_(projections) {}

    std::shared_ptr<Schema> schema() const override {
        // TODO: Implement schema derivation from projections
        return child_->schema();
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {child_};
    }

    std::string to_string() const override {
        std::string result = "Project(";
        for (size_t i = 0; i < projections_.size(); ++i) {
            if (i > 0) result += ", ";
            result += projections_[i]->to_string();
        }
        result += ")";
        return result;
    }

    const std::vector<std::shared_ptr<LogicalExpr>>& projections() const {
        return projections_;
    }

private:
    std::shared_ptr<LogicalPlan> child_;
    std::vector<std::shared_ptr<LogicalExpr>> projections_;
};

/**
 * @brief Filter plan that applies a predicate
 */
class FilterPlan : public AbstractLogicalPlan {
public:
    FilterPlan(
        std::shared_ptr<DataSource> data_source,
        uint64_t tenant_id,
        std::shared_ptr<LogicalPlan> child,
        std::shared_ptr<LogicalExpr> filter)
        : AbstractLogicalPlan(std::move(data_source), tenant_id),
          child_(std::move(child)),
          filter_(std::move(filter)) {}

    std::shared_ptr<Schema> schema() const override { return child_->schema(); }
    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {child_};
    }

    std::string to_string() const override {
        return "Filter(" + filter_->to_string() + ")";
    }

    const std::shared_ptr<LogicalExpr>& filter() const { return filter_; }

private:
    std::shared_ptr<LogicalPlan> child_;
    std::shared_ptr<LogicalExpr> filter_;
};

/**
 * @brief Vector search plan that performs nearest neighbor search
 */
class VectorSearchPlan : public AbstractLogicalPlan {
public:
    VectorSearchPlan(
        const std::string& column_name,
        std::shared_ptr<DataSource> data_source,
        uint64_t tenant_id,
        std::shared_ptr<LogicalPlan> child,
        const std::vector<float>& query_vector,
        size_t k,
        const std::string& distance_metric = "L2")
        : AbstractLogicalPlan(std::move(data_source), tenant_id),
          column_name_(column_name),
          child_(std::move(child)),
          query_vector_(query_vector),
          k_(k),
          distance_metric_(distance_metric) {}

    std::shared_ptr<Schema> schema() const override { return child_->schema(); }
    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {child_};
    }

    std::string to_string() const override {
        return "VectorSearch(k=" + std::to_string(k_) + 
               ", metric=" + distance_metric_ + ")";
    }

    const std::string& column_name() const { return column_name_; }
    const std::vector<float>& query_vector() const { return query_vector_; }
    size_t k() const { return k_; }
    const std::string& distance_metric() const { return distance_metric_; }

private:
    std::string column_name_;
    std::shared_ptr<LogicalPlan> child_;
    std::vector<float> query_vector_;
    size_t k_;
    std::string distance_metric_;
};

} // namespace query
} // namespace lintdb 