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
    idx_t actual = std::get<idx_t>(key.field_value());
    ASSERT_EQ(actual, 3);
    ASSERT_EQ(key.doc_id(), 4);
}

TEST_F(KeySerializationTests, SerializeAndDeserializeInvertedIndexKey_StringType) {
    std::string expectedKey = lintdb::create_index_id(1, 2, lintdb::DataType::TEXT, "some value", 123);
    lintdb::InvertedIndexKey key(expectedKey);
    ASSERT_EQ(key.field(), uint8_t(2));
    auto actual = std::get<std::string>(key.field_value());
    ASSERT_EQ(actual, "some value");
    ASSERT_EQ(key.doc_id(), 123);
}

TEST_F(KeySerializationTests, SerializeAndDeserializeInvertedIndexKey_DateType) {
    lintdb::DateTime now = std::chrono::time_point_cast<lintdb::Duration>(std::chrono::system_clock::now());
    std::string expectedKey = lintdb::create_index_id(1, 2, lintdb::DataType::DATETIME, lintdb::DateTime(now), 123);
    lintdb::InvertedIndexKey key(expectedKey);
    ASSERT_EQ(key.field(), uint8_t(2));

    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, lintdb::DateTime>) {
            // Handle DateTime
            std::cout << "DateTime with ms: " << arg.time_since_epoch().count() << std::endl;
        } else {
            // Handle other types
            std::cout << "Not a DateTime" << std::endl;
        }
    }, key.field_value());

    auto actual = std::get<lintdb::DateTime>(key.field_value());
    ASSERT_EQ(actual, now);
    ASSERT_EQ(key.doc_id(), 123);
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