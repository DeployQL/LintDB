#include <gtest/gtest.h>
#include "lintdb/query/DataFrame.h"
#include "lintdb/query/LogicalExpr.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/DataTypes.h"

using namespace lintdb;
using namespace lintdb::query;

class DataFrameTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple schema for testing
        schema_ = std::make_shared<Schema>();
        
        // Create field parameters
        FieldParameters params;
        
        // Add fields with proper construction
        Field id_field("id", DataType::INTEGER, {FieldType::Stored}, params);
        Field name_field("name", DataType::TEXT, {FieldType::Stored}, params);
        Field age_field("age", DataType::INTEGER, {FieldType::Stored}, params);
        Field vector_field("vector", DataType::FLOAT, {FieldType::Stored}, params);
        
        schema_->add_field(id_field);
        schema_->add_field(name_field);
        schema_->add_field(age_field);
        schema_->add_field(vector_field);
    }

    std::shared_ptr<Schema> schema_;
};

TEST_F(DataFrameTest, TestProjection) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Create projection expressions
    auto id_col = std::make_shared<Column>("id");
    auto name_col = std::make_shared<Column>("name");

    // Apply projection
    auto projected_df = df->project({id_col, name_col});
    auto result_plan = projected_df->logical_plan();

    // Verify projection was added to plan
    EXPECT_EQ(result_plan->projections().size(), 2);
    EXPECT_EQ(result_plan->projections()[0]->to_string(), "id");
    EXPECT_EQ(result_plan->projections()[1]->to_string(), "name");
}

TEST_F(DataFrameTest, TestFilter) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Create filter expression: age > 30
    auto age_col = std::make_shared<Column>("age");
    auto literal = std::make_shared<Literal>("30");
    auto filter_expr = std::make_shared<BinaryExpr>(age_col, BinaryExpr::Op::GT, literal);

    // Apply filter
    auto filtered_df = df->filter(filter_expr);
    auto result_plan = filtered_df->logical_plan();

    // Verify filter was added to plan
    EXPECT_NE(result_plan->filter(), nullptr);
    EXPECT_EQ(result_plan->filter()->to_string(), "age > 30");
}

TEST_F(DataFrameTest, TestVectorSearch) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Create query vector
    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    size_t k = 5;

    // Apply vector search
    auto search_df = df->vector_search(query_vector, k, "L2");
    auto result_plan = search_df->logical_plan();

    // Verify vector search was added to plan
    EXPECT_EQ(result_plan->query_vector(), query_vector);
    EXPECT_EQ(result_plan->k(), k);
    EXPECT_EQ(result_plan->distance_metric(), "L2");
}

TEST_F(DataFrameTest, TestComplexQuery) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Create a complex query: project id and name, filter age > 30, then do vector search
    auto id_col = std::make_shared<Column>("id");
    auto name_col = std::make_shared<Column>("name");
    auto age_col = std::make_shared<Column>("age");
    auto literal = std::make_shared<Literal>("30");
    auto filter_expr = std::make_shared<BinaryExpr>(age_col, BinaryExpr::Op::GT, literal);

    // Apply operations in sequence
    auto projected_df = df->project({id_col, name_col});
    auto filtered_df = projected_df->filter(filter_expr);
    
    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    auto search_df = filtered_df->vector_search(query_vector, 10, "L2");
    auto result_plan = search_df->logical_plan();

    // Verify all operations were added to plan
    EXPECT_EQ(result_plan->projections().size(), 2);
    EXPECT_NE(result_plan->filter(), nullptr);
    EXPECT_EQ(result_plan->query_vector(), query_vector);
    EXPECT_EQ(result_plan->k(), 10);
    EXPECT_EQ(result_plan->distance_metric(), "L2");
}

TEST_F(DataFrameTest, TestLogicalPlan) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Test that logical plan is accessible and matches the input plan
    EXPECT_EQ(df->logical_plan(), plan);
}

TEST_F(DataFrameTest, TestSchema) {
    auto plan = std::make_shared<LogicalPlan>();
    auto df = std::make_shared<DataFrameImpl>(plan, schema_);

    // Test that schema is accessible and matches the input schema
    EXPECT_EQ(df->schema(), schema_);
} 