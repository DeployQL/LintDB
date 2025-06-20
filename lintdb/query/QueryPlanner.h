#pragma once

#include <memory>
#include "lintdb/query/LogicalPlan.h"
#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/query/PhysicalExpr.h"

namespace lintdb::query {

/**
 * @brief Query planner that converts logical plans to physical plans
 */
class QueryPlanner {
public:
    QueryPlanner() = default;
    ~QueryPlanner() = default;

    /**
     * @brief Create a physical plan from a logical plan
     * @param logical The logical plan to convert
     * @return A unique pointer to the physical plan
     */
    std::unique_ptr<PhysicalPlan> create_physical_plan(const std::shared_ptr<LogicalPlan>& logical);

private:
    /**
     * @brief Create a physical plan for a scan operation
     * @param scan The logical scan plan
     * @return A unique pointer to the physical scan plan
     */
    std::unique_ptr<PhysicalPlan> plan_scan(const ScanPlan& scan);

    /**
     * @brief Create a physical plan for a filter operation
     * @param filt The logical filter plan
     * @return A unique pointer to the physical filter plan
     */
    std::unique_ptr<PhysicalPlan> plan_filter(const FilterPlan& filt);

    /**
     * @brief Create a physical plan for a projection operation
     * @param proj The logical projection plan
     * @return A unique pointer to the physical projection plan
     */
    std::unique_ptr<PhysicalPlan> plan_project(const ProjectionPlan& proj);

    /**
     * @brief Create a physical plan for a vector search operation
     * @param vs The logical vector search plan
     * @return A unique pointer to the physical vector search plan
     */
    std::unique_ptr<PhysicalPlan> plan_vector_search(const VectorSearchPlan& vs);

    std::unique_ptr<PhysicalExpr> create_physical_expr(const std::shared_ptr<LogicalExpr>& expr, const LogicalPlan& plan);
};

} // namespace lintdb::query 