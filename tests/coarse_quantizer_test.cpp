#include <gtest/gtest.h>
#include "lintdb/quantizers/CoarseQuantizer.h"
#include <iostream>
#include "lintdb/version.h"

using namespace lintdb;
// Fixture for CoarseQuantizer tests
class CoarseQuantizerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cq = new CoarseQuantizer(4);
    }

    void TearDown() override {
        delete cq;
    }

    CoarseQuantizer* cq;
};

TEST_F(CoarseQuantizerTest, TestTrain) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };
    size_t n = data.size();
    size_t dim = data[0].size();
    std::vector<float> flat_data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
    }

    ASSERT_NO_THROW(cq->train(n, flat_data.data(), 3));
}

TEST_F(CoarseQuantizerTest, TestAssign) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };
    size_t n = data.size();
    size_t dim = data[0].size();
    std::vector<float> flat_data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
    }

    cq->train(n, flat_data.data(), 3);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<idx_t> codes(1);
    cq->assign(1, vec.data(), codes.data());

    ASSERT_EQ(codes.size(), 1);
    ASSERT_GE(codes[0], 0);
    ASSERT_LT(codes[0], 3);
}

TEST_F(CoarseQuantizerTest, TestDecode) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };
    size_t n = data.size();
    size_t dim = data[0].size();
    std::vector<float> flat_data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
    }

    cq->train(n, flat_data.data(), 3);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<idx_t> codes(1);
    cq->assign(1, vec.data(), codes.data());

    std::vector<float> decoded_vec(dim);
    cq->sa_decode(1, codes.data(), decoded_vec.data());

    ASSERT_EQ(decoded_vec.size(), vec.size());
}

TEST_F(CoarseQuantizerTest, TestSerializeDeserialize) {
    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };
    size_t n = data.size();
    size_t dim = data[0].size();
    std::vector<float> flat_data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
    }

    cq->train(n, flat_data.data(),3);
    cq->save("coarse_quantizer.dat");
    lintdb::Version version;
    auto cq_loaded = CoarseQuantizer::deserialize("coarse_quantizer.dat", version);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<idx_t> codes(1);
    std::cout << "cq_loaded->assign" << std::endl;
    cq_loaded->assign(1, vec.data(), codes.data());
    std::cout << "cq_loaded->assign done" << std::endl;

    ASSERT_EQ(codes.size(), 1);
}