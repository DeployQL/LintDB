#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lintdb/query/LogicalExpr.h"
#include "lintdb/query/LogicalPlan.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/datasources/DataSource.h"

namespace lintdb::query {

class DataFrame {
public:
    virtual ~DataFrame() = default;

    /**
     * Project specific columns from the DataFrame.
     * @param exprs The expressions to project
     * @return A new DataFrame with the projected columns
     */
    virtual std::shared_ptr<DataFrame> project(const std::vector<std::shared_ptr<LogicalExpr>>& exprs) = 0;

    /**
     * Filter rows from the DataFrame.
     * @param expr The filter expression
     * @return A new DataFrame with filtered rows
     */
    virtual std::shared_ptr<DataFrame> filter(const std::shared_ptr<LogicalExpr>& expr) = 0;

    /**
     * Perform vector search on the DataFrame.
     * @param query_vector The query vector
     * @param k The number of results to return
     * @param distance_metric The distance metric to use (default: L2)
     * @return A new DataFrame with search results
     */
    virtual std::shared_ptr<DataFrame> vector_search(
        const std::vector<float>& query_vector,
        size_t k,
        const std::string& distance_metric = "L2") = 0;

    /**
     * Get the schema of the DataFrame.
     * @return The schema
     */
    virtual std::shared_ptr<lintdb::Schema> schema() const = 0;

    /**
     * Get the logical plan of the DataFrame.
     * @return The logical plan
     */
    virtual std::shared_ptr<LogicalPlan> logical_plan() const = 0;

    /**
     * Set the tenant ID for this DataFrame.
     * @param tenant_id The tenant ID to set
     * @return A new DataFrame with the updated tenant ID
     */
    virtual std::shared_ptr<DataFrame> tenant(uint64_t tenant_id) = 0;

    /**
     * Create a new DataFrame with the given schema and data source.
     * @param schema The schema
     * @param data_source The data source
     * @return A new DataFrame
     */
    static std::shared_ptr<DataFrame> create(
        std::shared_ptr<lintdb::Schema> schema,
        std::shared_ptr<lintdb::DataSource> data_source);
};

class DataFrameImpl : public DataFrame {
public:
    DataFrameImpl(
        std::shared_ptr<LogicalPlan> plan,
        std::shared_ptr<lintdb::Schema> schema)
        : plan_(plan), schema_(schema) {}

    std::shared_ptr<DataFrame> project(const std::vector<std::shared_ptr<LogicalExpr>>& exprs) override {
        auto new_plan = std::make_shared<ProjectionPlan>(
            plan_->data_source(),
            plan_->tenant_id(),
            plan_,
            exprs);
        return std::make_shared<DataFrameImpl>(new_plan, schema_);
    }

    std::shared_ptr<DataFrame> filter(const std::shared_ptr<LogicalExpr>& expr) override {
        auto new_plan = std::make_shared<FilterPlan>(
            plan_->data_source(),
            plan_->tenant_id(),
            plan_,
            expr);
        return std::make_shared<DataFrameImpl>(new_plan, schema_);
    }

    std::shared_ptr<DataFrame> vector_search(
        const std::vector<float>& query_vector,
        size_t k,
        const std::string& distance_metric = "L2") override {
        auto new_plan = std::make_shared<VectorSearchPlan>(
            plan_->data_source(),
            plan_->tenant_id(),
            plan_,
            query_vector,
            k,
            distance_metric);
        return std::make_shared<DataFrameImpl>(new_plan, schema_);
    }

    std::shared_ptr<lintdb::Schema> schema() const override {
        return schema_;
    }

    std::shared_ptr<LogicalPlan> logical_plan() const override {
        return plan_;
    }

    std::shared_ptr<DataFrame> tenant(uint64_t tenant_id) override {
        // Create a new scan plan with the updated tenant ID
        auto new_plan = std::make_shared<ScanPlan>(
            plan_->data_source(),
            tenant_id,
            schema_);
        return std::make_shared<DataFrameImpl>(new_plan, schema_);
    }

private:
    std::shared_ptr<LogicalPlan> plan_;
    std::shared_ptr<lintdb::Schema> schema_;
};

inline std::shared_ptr<DataFrame> DataFrame::create(
    std::shared_ptr<lintdb::Schema> schema,
    std::shared_ptr<lintdb::DataSource> data_source) {
    // Initialize with tenant_id 0, can be updated later with tenant()
    auto plan = std::make_shared<ScanPlan>(data_source, 0, schema);
    return std::make_shared<DataFrameImpl>(plan, schema);
}

} // namespace lintdb::query