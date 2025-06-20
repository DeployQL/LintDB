#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/builder.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/array_binary.h>
#include "lintdb/query/physical/VectorSearchPhysicalPlan.h"
#include "lintdb/query/physical/ScanPhysicalPlan.h"
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

class VectorSearchPhysicalPlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a schema with id, vector, and metadata columns
        std::vector<Field> fields = {
            Field("id", DataType::INT64, {}, {}),
            Field("vector", DataType::FLOAT32, {FieldType::FIXED_SIZE_LIST}, {{"size", "3"}}),
            Field("metadata", DataType::STRING, {}, {})
        };
        schema_ = std::make_shared<Schema>(fields);

        // Create a mock data source with test data
        data_source_ = std::make_shared<MockDataSource>();
        
        // Create test data
        arrow::Int64Builder id_builder;
        arrow::StringBuilder metadata_builder;
        
        // Create vector data
        std::vector<float> vectors = {
            1.0f, 2.0f, 3.0f,  // Vector 1
            2.0f, 3.0f, 4.0f,  // Vector 2
            3.0f, 4.0f, 5.0f,  // Vector 3
            4.0f, 5.0f, 6.0f,  // Vector 4
            5.0f, 6.0f, 7.0f   // Vector 5
        };
        
        // Create fixed-size list array for vectors
        auto vector_type = arrow::fixed_size_list(arrow::float32(), 3);
        arrow::FixedSizeListBuilder vector_builder(
            arrow::default_memory_pool(),
            std::make_shared<arrow::FloatBuilder>(),
            vector_type
        );
        
        // Append vectors
        for (size_t i = 0; i < 5; ++i) {
            ASSERT_TRUE(vector_builder.Append().ok());
            auto value_builder = std::static_pointer_cast<arrow::FloatBuilder>(vector_builder.value_builder());
            ASSERT_TRUE(value_builder->AppendValues(vectors.data() + i * 3, 3).ok());
        }
        
        // Append IDs and metadata
        ASSERT_TRUE(id_builder.AppendValues({1, 2, 3, 4, 5}).ok());
        ASSERT_TRUE(metadata_builder.AppendValues({"a", "b", "c", "d", "e"}).ok());
        
        // Finish building arrays
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> vector_array;
        std::shared_ptr<arrow::Array> metadata_array;
        
        ASSERT_TRUE(id_builder.Finish(&id_array).ok());
        ASSERT_TRUE(vector_builder.Finish(&vector_array).ok());
        ASSERT_TRUE(metadata_builder.Finish(&metadata_array).ok());
        
        // Create record batch
        auto batch = arrow::RecordBatch::Make(
            schema_->to_arrow_schema(),
            5,
            {id_array, vector_array, metadata_array}
        );
        
        data_source_->batches.push_back(batch);
    }

    std::shared_ptr<Schema> schema_;
    std::shared_ptr<MockDataSource> data_source_;
};

TEST_F(VectorSearchPhysicalPlanTest, TestBasicVectorSearch) {
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
    ASSERT_EQ(batches.size(), 1);
    
    auto batch = batches[0];
    EXPECT_EQ(batch->num_rows(), 2);  // Should return top 2 results
    EXPECT_EQ(batch->num_columns(), 3);  // id, vector, metadata
    
    // Verify the results are ordered by distance
    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto vector_array = std::static_pointer_cast<arrow::FixedSizeListArray>(batch->column(1));
    auto metadata_array = std::static_pointer_cast<arrow::StringArray>(batch->column(2));
    
    // First result should be closest to query vector
    EXPECT_EQ(id_array->Value(0), 1);  // First vector is closest
    EXPECT_EQ(metadata_array->GetString(0), "a");
    
    // Second result should be second closest
    EXPECT_EQ(id_array->Value(1), 2);  // Second vector is second closest
    EXPECT_EQ(metadata_array->GetString(1), "b");
}

TEST_F(VectorSearchPhysicalPlanTest, TestDifferentKValues) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    
    // Test with k=1
    auto plan_k1 = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        1,
        "L2"
    );
    plan_k1->add_child(scan_plan);
    auto batches_k1 = plan_k1->execute();
    ASSERT_EQ(batches_k1.size(), 1);
    EXPECT_EQ(batches_k1[0]->num_rows(), 1);
    
    // Test with k=3
    auto plan_k3 = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        3,
        "L2"
    );
    plan_k3->add_child(scan_plan);
    auto batches_k3 = plan_k3->execute();
    ASSERT_EQ(batches_k3.size(), 1);
    EXPECT_EQ(batches_k3[0]->num_rows(), 3);
    
    // Test with k=5 (all vectors)
    auto plan_k5 = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        5,
        "L2"
    );
    plan_k5->add_child(scan_plan);
    auto batches_k5 = plan_k5->execute();
    ASSERT_EQ(batches_k5.size(), 1);
    EXPECT_EQ(batches_k5[0]->num_rows(), 5);
}

TEST_F(VectorSearchPhysicalPlanTest, TestDifferentDistanceMetrics) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    
    // Test L2 distance
    auto plan_l2 = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,
        "L2"
    );
    plan_l2->add_child(scan_plan);
    auto batches_l2 = plan_l2->execute();
    ASSERT_EQ(batches_l2.size(), 1);
    
    // Test cosine distance
    auto plan_cosine = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,
        "cosine"
    );
    plan_cosine->add_child(scan_plan);
    auto batches_cosine = plan_cosine->execute();
    ASSERT_EQ(batches_cosine.size(), 1);
    
    // Results might be different for different metrics
    auto id_array_l2 = std::static_pointer_cast<arrow::Int64Array>(batches_l2[0]->column(0));
    auto id_array_cosine = std::static_pointer_cast<arrow::Int64Array>(batches_cosine[0]->column(0));
    
    // Note: We don't assert specific values here since the ordering might be different
    // for different distance metrics
    EXPECT_EQ(id_array_l2->length(), 2);
    EXPECT_EQ(id_array_cosine->length(), 2);
}

TEST_F(VectorSearchPhysicalPlanTest, TestEmptyResults) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    
    // Test with k=0
    auto plan_k0 = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        0,
        "L2"
    );
    plan_k0->add_child(scan_plan);
    auto batches_k0 = plan_k0->execute();
    ASSERT_EQ(batches_k0.size(), 1);
    EXPECT_EQ(batches_k0[0]->num_rows(), 0);
}

TEST_F(VectorSearchPhysicalPlanTest, TestInvalidInputs) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    
    // Test with empty query vector
    std::vector<float> empty_vector;
    auto plan_empty = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        empty_vector,
        2,
        "L2"
    );
    plan_empty->add_child(scan_plan);
    EXPECT_THROW(plan_empty->execute(), std::runtime_error);
    
    // Test with invalid distance metric
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    auto plan_invalid_metric = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,
        "invalid_metric"
    );
    plan_invalid_metric->add_child(scan_plan);
    EXPECT_THROW(plan_invalid_metric->execute(), std::runtime_error);
}

TEST_F(VectorSearchPhysicalPlanTest, TestSchemaConsistency) {
    auto scan_plan = std::make_shared<ScanPhysicalPlan>(data_source_, 1, schema_);
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};
    
    auto vector_search_plan = std::make_shared<VectorSearchPhysicalPlan>(
        data_source_,
        1,
        query_vector,
        2,
        "L2"
    );
    vector_search_plan->add_child(scan_plan);
    
    // Verify that the schema is preserved
    EXPECT_EQ(vector_search_plan->schema(), schema_);
    
    // Verify that the output batch has the same schema
    auto batches = vector_search_plan->execute();
    ASSERT_EQ(batches.size(), 1);
    EXPECT_TRUE(batches[0]->schema()->Equals(schema_->to_arrow_schema()));
} 