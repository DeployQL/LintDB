#include <gtest/gtest.h>
#include <json/json.h>
#include "Schema.h"
#include "Document.h"
#include "DocumentProcessor.h"
#include "InvertedIndexDocument.h"
#include "ForwardIndexDocument.h"
#include "mocks.h"

// Helper function to create a sample schema
Schema createSampleSchema() {
    Schema schema;
    Field field1 = {"intField", DataType::INTEGER, FieldType::Stored, {0, "", QuantizationType::NONE}};
    Field field2 = {"floatField", DataType::FLOAT, FieldType::Stored, {0, "", QuantizationType::NONE}};
    Field field3 = {"tensorField", DataType::TENSOR, FieldType::Stored, {2, "", QuantizationType::PQ}};
    Field field4 = {"tensorArrayField", DataType::TENSOR_ARRAY, FieldType::Stored, {2, "", QuantizationType::VQ}};
    schema.fields.push_back(field1);
    schema.fields.push_back(field2);
    schema.fields.push_back(field3);
    schema.fields.push_back(field4);
    return schema;
}

// Helper function to create a sample document
Document createSampleDocument() {
    Document document;
    document.fields["intField"] = FieldValue(42);
    document.fields["floatField"] = FieldValue(3.14f);
    document.fields["tensorField"] = FieldValue(Tensor{0.1f, 0.2f});
    document.fields["tensorArrayField"] = FieldValue(TensorArray{{0.1f, 0.2f}, {0.3f, 0.4f}});
    return document;
}


TEST(DocumentProcessor, ProcessDocumentWithValidFields) {
    auto mockIndexWriter = std::make_shared<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    lintdb::FieldMapper fieldMapper;
    lintdb::Schema schema;
    lintdb::Field field1 = {"field1", lintdb::DataType::INTEGER, lintdb::FieldType::Stored, {0, "", lintdb::QuantizationType::NONE}};
    schema.fields.push_back(field1);

    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::CoarseQuantizer>> coarseQuantizerMap;

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, mockIndexWriter);

    lintdb::Document document;
    document.fields = {{"field1", lintdb::FieldValue(10)}};

    EXPECT_CALL(*mockIndexWriter, addDocument(_)).Times(1);
    // quantizer does not get called for non-tensor fields
    EXPECT_CALL(*mockQuantizer, quantize(_)).Times(0);

    processor.processDocument(1, document);
}

TEST(DocumentProcessor, ProcessDocumentWithInvalidField) {
    auto mockIndexWriter = std::make_shared<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    lintdb::FieldMapper fieldMapper;
    lintdb::Schema schema;
    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::CoarseQuantizer>> coarseQuantizerMap;

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, mockIndexWriter);

    lintdb::Document document;
    document.fields = {{"invalid_field", lintdb::FieldValue(10)}};

    EXPECT_CALL(*mockIndexWriter, addDocument(_)).Times(0);
    EXPECT_CALL(*mockQuantizer, quantize(_)).Times(0);

    EXPECT_THROW(processor.processDocument(1, document), std::invalid_argument);
}

TEST(DocumentProcessor, ProcessDocumentWithTensorField) {
    auto mockIndexWriter = std::make_shared<MockIndexWriter>();
    auto mockQuantizer = std::make_shared<MockQuantizer>();
    lintdb::FieldMapper fieldMapper;
    lintdb::Schema schema;
    lintdb::Field field1 = {"field1", lintdb::DataType::TENSOR, lintdb::FieldType::Indexed, {3, "", lintdb::QuantizationType::PQ}};
    schema.fields.push_back(field1);

    std::unordered_map<std::string, std::shared_ptr<lintdb::Quantizer>> quantizerMap = {{"field1", mockQuantizer}};
    std::unordered_map<std::string, std::shared_ptr<lintdb::CoarseQuantizer>> coarseQuantizerMap;

    lintdb::DocumentProcessor processor(schema, quantizerMap, coarseQuantizerMap, fieldMapper, mockIndexWriter);

    lintdb::Document document;
    document.fields = {{"field1", lintdb::FieldValue(lintdb::Tensor{1.0f, 2.0f, 3.0f})}};

    EXPECT_CALL(*mockIndexWriter, addDocument(_)).Times(1);
    EXPECT_CALL(*mockQuantizer, quantize(_)).Times(1);

    processor.processDocument(1, document);
}