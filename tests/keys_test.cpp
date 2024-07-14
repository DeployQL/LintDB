#include <gtest/gtest.h>
#include "lintdb/invlists/KeyBuilder.h"
#include "lintdb/schema/DataTypes.h"
#include <string>
#include <chrono>

class KeySerializationTests : public ::testing::Test {
protected:
    lintdb::KeyBuilder builder;
};

TEST_F(KeySerializationTests, SerializeAndDeserializeInvertedIndexKey_IntegerType) {
    std::string expectedKey = builder.add(static_cast<uint64_t>(1)) // tenant
            .add(static_cast<uint8_t>(2)) // field
            .add(lintdb::DataType::INTEGER) // field_type
            .add(static_cast<idx_t>(3)) // inverted_list
            .add(static_cast<idx_t>(4)) // doc_id
            .build();
    lintdb::InvertedIndexKey key(expectedKey);
    ASSERT_EQ(key.field(), uint8_t(2));
    ASSERT_EQ(key.doc_id(), 4);
}

TEST_F(KeySerializationTests, SerializeAndDeserializeContextKey) {
    std::string expectedKey = builder.add(static_cast<uint64_t>(1)) // tenant
            .add(static_cast<uint8_t>(2)) // field
            .add(static_cast<idx_t>(3)) // doc_id
            .build();
    lintdb::ContextKey key(expectedKey);
    ASSERT_EQ(key.doc_id(), 3);
}

TEST_F(KeySerializationTests, SerializeAndDeserializeForwardIndexKey) {
    std::string expectedKey = builder.add(static_cast<uint64_t>(1)) // tenant
            .add(static_cast<idx_t>(2)) // doc_id
            .build();
    lintdb::ForwardIndexKey key(expectedKey);
    ASSERT_EQ(key.doc_id(), 2);
}