#pragma once

#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/datasources/DataSource.h"

namespace lintdb {
namespace query {

/**
 * @brief Physical scan plan that reads data from a data source
 */
class ScanPhysicalPlan : public PhysicalPlan {
public:
    explicit ScanPhysicalPlan(
        std::shared_ptr<DataSource> data_source,
        uint64_t tenant_id, 
        std::shared_ptr<Schema> schema)
        : data_source_(std::move(data_source)), tenant_id_(tenant_id), schema_(std::move(schema)) {}

    std::shared_ptr<Schema> schema() const override { return schema_; }

    std::vector<std::shared_ptr<arrow::RecordBatch>> execute() override {
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        auto iterator = data_source_->scan(tenant_id_);
        
        // Process data in batches
        const size_t BATCH_SIZE = 1024;  // Configurable batch size
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.resize(schema_->fields.size());
        
        // TODO: Implement conversion from key-value pairs to Arrow arrays
        // This will depend on your data format and serialization scheme
        
        return batches;
    }

    std::vector<std::shared_ptr<PhysicalPlan>> children() const override {
        return children_;
    }

    std::string to_string() const override {
        return "PhysicalScan";
    }

private:
    std::shared_ptr<DataSource> data_source_;
    uint64_t tenant_id_;
    std::shared_ptr<Schema> schema_;
};

} // namespace query
} // namespace lintdb 