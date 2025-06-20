#include "lintdb/query/QueryPlanner.h"
#include "lintdb/query/LogicalExpr.h"
#include "lintdb/query/PhysicalExpr.h"
#include "lintdb/query/physical/ScanPhysicalPlan.h"
#include "lintdb/query/physical/FilterPhysicalPlan.h"
#include "lintdb/query/physical/ProjectionPhysicalPlan.h"
#include "lintdb/query/physical/VectorSearchPhysicalPlan.h"

namespace lintdb::query {

std::unique_ptr<PhysicalPlan> QueryPlanner::create_physical_plan(const std::shared_ptr<LogicalPlan>& logical) {
    if (auto scan = dynamic_cast<ScanPlan*>(logical.get())) {
        return plan_scan(*scan);
    }
    else if (auto filt = dynamic_cast<FilterPlan*>(logical.get())) {
        return plan_filter(*filt);
    }
    else if (auto proj = dynamic_cast<ProjectionPlan*>(logical.get())) {
        return plan_project(*proj);
    }
    else if (auto vs = dynamic_cast<VectorSearchPlan*>(logical.get())) {
        return plan_vector_search(*vs);
    }
    else {
        throw std::runtime_error("Unsupported logical plan node");
    }
}

std::unique_ptr<PhysicalPlan> QueryPlanner::plan_scan(const ScanPlan& scan) {
    return std::make_unique<ScanPhysicalPlan>(scan.data_source(), scan.tenant_id(), scan.schema());
}

std::unique_ptr<PhysicalPlan> QueryPlanner::plan_filter(const FilterPlan& filt) {
    auto children = filt.children();
    if (children.empty()) {
        throw std::runtime_error("Filter plan must have a child plan");
    }
    auto child_phys = create_physical_plan(children[0]);
    
    return std::make_unique<FilterPhysicalPlan>(
        std::move(child_phys),
        filt.filter()
    );
}

std::unique_ptr<PhysicalPlan> QueryPlanner::plan_project(const ProjectionPlan& proj) {
    auto children = proj.children();
    if (children.empty()) {
        throw std::runtime_error("Projection plan must have a child plan");
    }
    auto child_phys = create_physical_plan(children[0]);
    
    return std::make_unique<ProjectionPhysicalPlan>(
        std::move(child_phys),
        proj.projections()
    );
}

std::unique_ptr<PhysicalPlan> QueryPlanner::plan_vector_search(const VectorSearchPlan& vs) {
    return std::make_unique<VectorSearchPhysicalPlan>(
        vs.column_name(),
        vs.data_source(),
        vs.tenant_id(),
        vs.query_vector(),
        vs.k(),
        vs.distance_metric()
    );
}

std::unique_ptr<PhysicalExpr> QueryPlanner::create_physical_expr(const std::shared_ptr<LogicalExpr>& expr, const LogicalPlan& plan) {
    if (!expr) {
        return nullptr;
    }

    switch (expr->type()) {
        case LogicalExpr::Type::LITERAL: {
            auto lit = std::static_pointer_cast<Literal>(expr);
            return std::make_unique<LiteralPhysicalExpr>(lit->to_string());
        }
        case LogicalExpr::Type::BINARY: {
            auto bin = std::static_pointer_cast<BinaryExpr>(expr);
            auto left = create_physical_expr(bin->left(), plan);
            auto right = create_physical_expr(bin->right(), plan);
            return std::make_unique<BinaryPhysicalExpr>(
                std::move(left),
                std::move(right),
                bin->op()
            );
        }
        case LogicalExpr::Type::COLUMN: {
            auto col = std::static_pointer_cast<Column>(expr);
            auto schema = plan.schema();
            auto field_idx = schema->get_field_index(col->to_string());
            return std::make_unique<ColumnPhysicalExpr>(field_idx);
        }
        default:
            throw std::runtime_error("Unsupported logical expression type");
    }
}

} // namespace lintdb::query 