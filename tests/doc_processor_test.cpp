#include <gtest/gtest.h>
#include <json/json.h>
#include "Schema.h"
#include "Document.h"
#include "DocumentProcessor.h"
#include "InvertedIndexDocument.h"
#include "ForwardIndexDocument.h"

// Helper function to create a sample schema
Schema createSampleSchema() {
    Schema schema;
    Field field1 = {"intField", DataType::INTEGER, FieldType::Stored, {0, "", QuantizationType::NONE}};
    Field field2 = {"floatField", DataType::FLOAT, FieldType::Stored, {0, "", QuantizationType::NONE}};
    Field field3 = {"tensorField", DataType::TENSOR, FieldType::Stored, {128, "", QuantizationType::PQ}};
    Field field4 = {"tensorArrayField", DataType::TENSOR_ARRAY, FieldType::Stored, {128, "", QuantizationType::VQ}};
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
    document.fields["tensorField"] = FieldValue(Tensor{0.1f, 0.2f, 0.3f});
    document.fields["tensorArrayField"] = FieldValue(TensorArray{{0.1f, 0.2f}, {0.3f, 0.4f}});
    return document;
}

TEST(DocumentProcessorTests, EncodeDecodeDocument) {
    // Create a schema and a document processor
    Schema schema = createSampleSchema();
    DocumentProcessor processor(schema);

    // Create a sample document
    Document document = createSampleDocument();

    // Process the document
    EncodedDocument encodedDoc = processor.processDocument(document);

    // Verify encoded document
    ASSERT_EQ(encodedDoc.encoded_fields.size(), 4);
    EXPECT_NE(encodedDoc.encoded_fields.find("intField"), encodedDoc.encoded_fields.end());
    EXPECT_NE(encodedDoc.encoded_fields.find("floatField"), encodedDoc.encoded_fields.end());
    EXPECT_NE(encodedDoc.encoded_fields.find("tensorField"), encodedDoc.encoded_fields.end());
    EXPECT_NE(encodedDoc.encoded_fields.find("tensorArrayField"), encodedDoc.encoded_fields.end());

    // Create index documents
    InvertedIndexDocument invertedDoc;
    ForwardIndexDocument forwardDoc;
    processor.serializeEncodedDocument(encodedDoc, invertedDoc, forwardDoc);

    // Verify forward index document
    ASSERT_EQ(forwardDoc.stored_fields.size(), 4);
    EXPECT_NE(forwardDoc.stored_fields.find("intField"), forwardDoc.stored_fields.end());
    EXPECT_NE(forwardDoc.stored_fields.find("floatField"), forwardDoc.stored_fields.end());
    EXPECT_NE(forwardDoc.stored_fields.find("tensorField"), forwardDoc.stored_fields.end());
    EXPECT_NE(forwardDoc.stored_fields.find("tensorArrayField"), forwardDoc.stored_fields.end());
}
