
#include <json/json.h>
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/Document.h"
#include "lintdb/schema/DocProcessor.h"
#include "lintdb/quantizers/Quantizer.h"

#include "mocks.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace lintdb;
using ::testing::_;
using ::testing::Return;

// Helper function to create a sample schema
Schema createSampleSchema() {
    Schema schema;
    Field field1 = {"intField", DataType::INTEGER, {FieldType::Stored}, {0, "", QuantizerType::NONE}};
    Field field2 = {"floatField", DataType::FLOAT, {FieldType::Stored}, {0, "", QuantizerType::NONE}};
    Field field3 = {"tensorField", DataType::TENSOR, {FieldType::Stored}, {2, "", QuantizerType::PRODUCT_ENCODER}};
    schema.fields.push_back(field1);
    schema.fields.push_back(field2);
    schema.fields.push_back(field3);
    return schema;
}

// Helper function to create a sample document
Document createSampleDocument() {
    Document document(1, {
            FieldValue("intField", 42),
            FieldValue("floatField", 3.14f),
            FieldValue("tensorField", Tensor{0.1f, 0.2f})
    });
    return document;
}


TEST(DocumentProcessor, ProcessDocumentWithValidFields) {
    std::unique_ptr<MockIndexWriter> mockIndexWriter = std::make_unique<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    std::shared_ptr<lintdb::FieldMapper> fieldMapper = std::make_shared<lintdb::FieldMapper>();
    lintdb::Schema schema;
    lintdb::Field field1 = {"field1", lintdb::DataType::INTEGER, {lintdb::FieldType::Stored}, {0, "", lintdb::QuantizerType::NONE}};
    schema.fields.push_back(field1);

    fieldMapper->addSchema(schema);

    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::ICoarseQuantizer>> coarseQuantizerMap;

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, std::move(mockIndexWriter));

    lintdb::Document document(1, {{lintdb::FieldValue("field1", 10)}});

    // quantizer does not get called for non-tensor fields
    EXPECT_CALL(*mockQuantizer, sa_encode(_, _, _)).Times(0);

    processor.processDocument(1, document);
}

TEST(DocumentProcessor, ProcessDocumentWithInvalidField) {
    std::unique_ptr<MockIndexWriter> mockIndexWriter = std::make_unique<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    std::shared_ptr<lintdb::FieldMapper> fieldMapper = std::make_shared<lintdb::FieldMapper>();
    lintdb::Schema schema;
    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::ICoarseQuantizer>> coarseQuantizerMap;

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, std::move(mockIndexWriter));

    lintdb::Document document(1, {{lintdb::FieldValue("invalid_field", 10)}});

    EXPECT_CALL(*mockQuantizer, sa_encode(_, _, _)).Times(0);

    EXPECT_THROW(processor.processDocument(1, document), std::invalid_argument);
}

TEST(DocumentProcessor, ProcessDocumentWithTensorField) {
    std::unique_ptr<MockIndexWriter> mockIndexWriter = std::make_unique<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    auto mockCoarseQuantizer = std::make_shared<MockCoarseQuantizer>();

    std::shared_ptr<lintdb::FieldMapper> fieldMapper = std::make_shared<lintdb::FieldMapper>();
    lintdb::Schema schema;
    lintdb::Field field1 = {"field1", lintdb::DataType::TENSOR, {lintdb::FieldType::Indexed}, {3, "", lintdb::QuantizerType::PRODUCT_ENCODER}};
    schema.fields.push_back(field1);

    fieldMapper->addSchema(schema);

    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::ICoarseQuantizer>> coarseQuantizerMap = {{"field1", mockCoarseQuantizer}};

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, std::move(mockIndexWriter));

    lintdb::Document document(0, {{lintdb::FieldValue("field1", lintdb::Tensor{1.0f, 2.0f, 3.0f})}});

    EXPECT_CALL(*mockQuantizer, code_size()).WillRepeatedly(Return(3));
    EXPECT_CALL(*mockCoarseQuantizer, is_trained()).WillRepeatedly(Return(true));

    EXPECT_CALL(*mockQuantizer, sa_encode(_, _ , _)).Times(1);

    processor.processDocument(1, document);
}