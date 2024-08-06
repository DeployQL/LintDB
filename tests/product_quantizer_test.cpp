#include <gtest/gtest.h>
#include "lintdb/quantizers/impl/product_quantizer.h"
#include "lintdb/quantizers/impl/kmeans.h"

using namespace lintdb;
// Fixture for ProductQuantizer tests
class ProductQuantizerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize with dimensionality, number of sub-quantizers, and number of centroids per sub-quantizer
        pq_euclidean = new ProductQuantizer(4, 2, 2, Metric::EUCLIDEAN);
        pq_inner_product = new ProductQuantizer(4, 2, 2, Metric::INNER_PRODUCT);
    }

    void TearDown() override {
        delete pq_euclidean;
        delete pq_inner_product;
    }

    ProductQuantizer* pq_euclidean;
    ProductQuantizer* pq_inner_product;
};

TEST_F(ProductQuantizerTest, TestFitEuclidean) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    ASSERT_NO_THROW(pq_euclidean->fit(data));
}

TEST_F(ProductQuantizerTest, TestFitInnerProduct) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    ASSERT_NO_THROW(pq_inner_product->fit(data));
}

TEST_F(ProductQuantizerTest, TestQuantizeEuclidean) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_euclidean->fit(data);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> codes = pq_euclidean->quantize(vec);

    ASSERT_EQ(codes.size(), pq_euclidean->get_M());
}

TEST_F(ProductQuantizerTest, TestQuantizeInnerProduct) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_inner_product->fit(data);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> codes = pq_inner_product->quantize(vec);

    ASSERT_EQ(codes.size(), pq_inner_product->get_M());
}

TEST_F(ProductQuantizerTest, TestDecodeEuclidean) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_euclidean->fit(data);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> codes = pq_euclidean->quantize(vec);
    std::vector<float> decoded_vec = pq_euclidean->decode(codes);

    ASSERT_EQ(decoded_vec.size(), vec.size());
}

TEST_F(ProductQuantizerTest, TestDecodeInnerProduct) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_inner_product->fit(data);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> codes = pq_inner_product->quantize(vec);
    std::vector<float> decoded_vec = pq_inner_product->decode(codes);

    ASSERT_EQ(decoded_vec.size(), vec.size());
}

TEST_F(ProductQuantizerTest, TestComputeCodesEuclidean) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_euclidean->fit(data);

    std::vector<std::vector<uint8_t>> codes = pq_euclidean->compute_codes(data);

    ASSERT_EQ(codes.size(), data.size());
    for (const auto& code : codes) {
        ASSERT_EQ(code.size(), pq_euclidean->get_M());
    }
}

TEST_F(ProductQuantizerTest, TestComputeCodesInnerProduct) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq_inner_product->fit(data);

    std::vector<std::vector<uint8_t>> codes = pq_inner_product->compute_codes(data);

    ASSERT_EQ(codes.size(), data.size());
    for (const auto& code : codes) {
        ASSERT_EQ(code.size(), pq_inner_product->get_M());
    }
}
