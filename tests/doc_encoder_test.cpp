#include <gtest/gtest.h>
#include "lintdb/schema/DocEncoder.h"
#include "bitsery/bitsery.h"
#include "bitsery/adapter/buffer.h"
#include "lintdb/schema/DocEncoder.h"
#include "lintdb/schema/ProcessedData.h"

TEST(DocEncoder, EncodeInvertedDataForTensorDataType) {
    lintdb::DocEncoder encoder;
    lintdb::ProcessedData data;
    data.value.data_type = lintdb::DataType::TENSOR;
    data.value.num_tensors = 2;
    data.centroid_ids = {1, 2};
    data.tenant = 0;
    data.field = 1;
    data.doc_id = 1;
    data.value.value = lintdb::Tensor{1.0f, 2.0f, 3.0f, 4.0f};

    auto result = encoder.encode_inverted_data(data, 2);

    EXPECT_EQ(result.size(), 2);
}

TEST(DocEncoder, EncodeInvertedDataForNonTensorDataType) {
    lintdb::DocEncoder encoder;
    lintdb::ProcessedData data;
    data.value.data_type = lintdb::DataType::INTEGER;
    data.value.value = 10;
    data.tenant = 0;
    data.field = 1;
    data.doc_id = 1;

    auto result = encoder.encode_inverted_data(data, 2);

    EXPECT_EQ(result.size(), 1);
}

TEST(DocEncoder, EncodeInvertedMappingData) {
    lintdb::DocEncoder encoder;
    lintdb::ProcessedData data;
    data.tenant = 0;
    data.field = 1;
    data.doc_id = 1;
    data.centroid_ids = {1, 2, 3};

    auto result = encoder.encode_inverted_mapping_data(data);

    EXPECT_EQ(result.size(), 1);
}


TEST(DocEncoder, EncodeContextData) {
    lintdb::DocEncoder encoder;
    lintdb::ProcessedData data;
    data.tenant = 0;
    data.field = 1;
    data.doc_id = 1;
    data.value.value = "context";

    auto result = encoder.encode_context_data(data);

    EXPECT_FALSE(result.key.empty());
    EXPECT_FALSE(result.value.empty());
}