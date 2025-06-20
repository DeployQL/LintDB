#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/builder.h>
#include "lintdb/query/physical/PhysicalPlan.h"
#include "lintdb/query/physical/ScanPhysicalPlan.h"
#include "lintdb/query/physical/FilterPhysicalPlan.h"
#include "lintdb/query/physical/ProjectionPhysicalPlan.h"
#include "lintdb/query/physical/VectorSearchPhysicalPlan.h"
#include "lintdb/query/LogicalExpr.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/datasources/DataSource.h"
#include "lintdb/datasources/DataSourceIterator.h"

using namespace lintdb;
using namespace lintdb::query;

class MockDataSource : public DataSource {
public:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

    Schema schema() const override {
        return Schema({});
    }

    std::unique_ptr<DataSourceIterator> scan(uint64_t tenant_id) override {
        return std::make_unique<MockDataSourceIterator>(batches);
    }

    std::unique_ptr<KeyValueIterator> scan_prefix(const std::string& prefix) override {
        return nullptr;
    }

    std::string get(const std::string& key) override {
        return "";
    }
};

class MockDataSourceIterator : public DataSourceIterator {
public:
    explicit MockDataSourceIterator(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches)
        : batches_(batches), current_(0) {}

    bool has_next() override {
        return current_ < batches_.size();
    }

    std::shared_ptr<arrow::RecordBatch> next() override {
        if (!has_next()) {
            return nullptr;
        }
        return batches_[current_++];
    }

private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t current_;
};

class PhysicalPlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple schema with two columns: id (int64) and value (string)
        std::vector<Field> fields = {
            Field("id", DataType::INT64, {}, {}),
            Field("value", DataType::STRING, {}, {})
        };
        schema_ = std::make_shared<Schema>(fields);

        // Create a mock data source with some test data
        data_source_ = std::make_shared<MockDataSource>();
        
        // Create a record batch with test data
        arrow::Int64Builder id_builder;
        arrow::StringBuilder value_builder;
        
        ASSERT_TRUE(id_builder.AppendValues({1, 2, 3, 4, 5}).ok());
        ASSERT_TRUE(value_builder.AppendValues({"a", "b", "c", "d", "e"}).ok());
        
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> value_array;
        
        ASSERT_TRUE(id_builder.Finish(&id_array).ok());
        ASSERT_TRUE(value_builder.Finish(&value_array).ok());
        
        auto batch = arrow::RecordBatch::Make(
            schema_->to_arrow_schema(),
            5,
            {id_array, value_array}
        );
        
        data_source_->batches.push_back(batch);
    }

    std::shared_ptr<Schema> schema_;
    std::shared_ptr<MockDataSource> data_source_;
};

TEST_F(PhysicalPlanTest, TestScanPhysicalPlan) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    
    // Test schema
    EXPECT_EQ(scan_plan->schema(), schema_);
    
    // Test execution
    auto batches = scan_plan->execute();
    ASSERT_EQ(batches.size(), 1);
    
    auto batch = batches[0];
    EXPECT_EQ(batch->num_rows(), 5);
    EXPECT_EQ(batch->num_columns(), 2);
    
    // Verify data
    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    
    EXPECT_EQ(id_array->Value(0), 1);
    EXPECT_EQ(value_array->GetString(0), "a");
    EXPECT_EQ(id_array->Value(4), 5);
    EXPECT_EQ(value_array->GetString(4), "e");
}

TEST_F(PhysicalPlanTest, TestFilterPhysicalPlan) {
    // Create a scan plan
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    
    // Create a filter expression: id > 3
    auto id_col = std::make_shared<Column>(0, "id");
    auto lit = std::make_shared<Literal>(3);
    auto filter_expr = std::make_shared<BinaryExpr>(id_col, lit, BinaryExpr::Op::GT);
    
    // Create a filter plan
    auto filter_plan = std::make_shared<FilterPhysicalPlan>(scan_plan, filter_expr);
    
    // Test execution
    auto batches = filter_plan->execute();
    ASSERT_EQ(batches.size(), 1);
    
    auto batch = batches[0];
    EXPECT_EQ(batch->num_rows(), 2);  // Only rows with id > 3
    
    // Verify filtered data
    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    
    EXPECT_EQ(id_array->Value(0), 4);
    EXPECT_EQ(value_array->GetString(0), "d");
    EXPECT_EQ(id_array->Value(1), 5);
    EXPECT_EQ(value_array->GetString(1), "e");
}

TEST_F(PhysicalPlanTest, TestProjectionPhysicalPlan) {
    // Create a scan plan
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    
    // Create projection expressions: [id, value]
    std::vector<std::shared_ptr<LogicalExpr>> projections = {
        std::make_shared<Column>(0, "id"),
        std::make_shared<Column>(1, "value")
    };
    
    // Create a projection plan
    auto projection_plan = std::make_shared<ProjectionPhysicalPlan>(scan_plan, projections);
    
    // Test execution
    auto batches = projection_plan->execute();
    ASSERT_EQ(batches.size(), 1);
    
    auto batch = batches[0];
    EXPECT_EQ(batch->num_rows(), 5);
    EXPECT_EQ(batch->num_columns(), 2);
    
    // Verify projected data
    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
    
    EXPECT_EQ(id_array->Value(0), 1);
    EXPECT_EQ(value_array->GetString(0), "a");
    EXPECT_EQ(id_array->Value(4), 5);
    EXPECT_EQ(value_array->GetString(4), "e");
}

TEST_F(PhysicalPlanTest, TestVectorSearchPhysicalPlan) {
    // Create a scan plan
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    
    // Create a query vector
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    
    // Create a vector search plan
    auto vector_search_plan = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,  // k=2
        "L2"  // L2 distance
    );
    
    // Add the scan plan as a child
    vector_search_plan->add_child(scan_plan);
    
    // Test execution
    auto batches = vector_search_plan->execute();
    // Note: The actual results will depend on the vector search implementation
    // This test just verifies that the plan executes without errors
    EXPECT_NO_THROW(vector_search_plan->execute());
}

TEST_F(PhysicalPlanTest, TestPlanToString) {
    // Create a scan plan
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    EXPECT_EQ(scan_plan->to_string(), "PhysicalScan");
    
    // Create a filter plan
    auto filter_expr = std::make_shared<BinaryExpr>(
        std::make_shared<Column>(0, "id"),
        std::make_shared<Literal>(3),
        BinaryExpr::Op::GT
    );
    auto filter_plan = std::make_shared<FilterPhysicalPlan>(scan_plan, filter_expr);
    EXPECT_EQ(filter_plan->to_string(), "PhysicalFilter(id > 3)");
    
    // Create a projection plan
    std::vector<std::shared_ptr<LogicalExpr>> projections = {
        std::make_shared<Column>(0, "id"),
        std::make_shared<Column>(1, "value")
    };
    auto projection_plan = std::make_shared<ProjectionPhysicalPlan>(scan_plan, projections);
    EXPECT_EQ(projection_plan->to_string(), "PhysicalProject(id, value)");
    
    // Create a vector search plan
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    auto vector_search_plan = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,
        "L2"
    );
    EXPECT_EQ(vector_search_plan->to_string(), "PhysicalVectorSearch(k=2, metric=L2)");
}

TEST_F(PhysicalPlanTest, TestPlanChildren) {
    // Create a scan plan
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    EXPECT_EQ(scan_plan->children().size(), 0);
    
    // Create a filter plan with scan as child
    auto filter_expr = std::make_shared<BinaryExpr>(
        std::make_shared<Column>(0, "id"),
        std::make_shared<Literal>(3),
        BinaryExpr::Op::GT
    );
    auto filter_plan = std::make_shared<FilterPhysicalPlan>(scan_plan, filter_expr);
    EXPECT_EQ(filter_plan->children().size(), 1);
    EXPECT_EQ(filter_plan->children()[0], scan_plan);
    
    // Create a projection plan with filter as child
    std::vector<std::shared_ptr<LogicalExpr>> projections = {
        std::make_shared<Column>(0, "id"),
        std::make_shared<Column>(1, "value")
    };
    auto projection_plan = std::make_shared<ProjectionPhysicalPlan>(filter_plan, projections);
    EXPECT_EQ(projection_plan->children().size(), 1);
    EXPECT_EQ(projection_plan->children()[0], filter_plan);
} 