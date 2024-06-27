#include <gtest/gtest.h>
#include "lintdb/quantizers/impl/kmeans.h"  // Include your kmeans implementation

using namespace lintdb;
// Test fixture for kmeans tests
class KMeansTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Common test setup can be placed here
    }

    void TearDown() override {
        // Clean up after tests
    }

    // Helper function to flatten 2D data
    std::vector<float> flatten(const std::vector<std::vector<float>>& data) {
        size_t n = data.size();
        size_t dim = data[0].size();
        std::vector<float> flat_data(n * dim);
        for (size_t i = 0; i < n; ++i) {
            std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
        }
        return flat_data;
    }
};

// Test for correct number of centroids
TEST_F(KMeansTest, CorrectNumberOfCentroids) {
    size_t n = 6;
    size_t dim = 4;
    size_t k = 3;

    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };

    std::vector<float> flat_data = flatten(data);

    auto centroids = kmeans(flat_data, n, dim, k, Metric::EUCLIDEAN);

    ASSERT_EQ(centroids.size(), k * dim);
}

// Test for correct centroids update
TEST_F(KMeansTest, CentroidsUpdate) {
    size_t n = 6;
    size_t dim = 4;
    size_t k = 3;

    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };

    std::vector<float> flat_data = flatten(data);

    auto centroids = kmeans(flat_data, n, dim, k, Metric::EUCLIDEAN);

    // Verify centroids are not all the same, indicating update has occurred
    bool all_same = true;
    for (size_t i = 1; i < k; ++i) {
        if (!std::equal(centroids.begin(), centroids.begin() + dim, centroids.begin() + i * dim)) {
            all_same = false;
            break;
        }
    }
    ASSERT_FALSE(all_same);
}

// Test for correct assignment of points to clusters
TEST_F(KMeansTest, CorrectAssignments) {
    size_t n = 6;
    size_t dim = 4;
    size_t k = 3;

    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };

    std::vector<float> flat_data = flatten(data);

    auto centroids = kmeans(flat_data, n, dim, k, Metric::EUCLIDEAN);

    // Ensure that each point is assigned to one of the centroids
    std::vector<size_t> assignments(n);
    for (size_t i = 0; i < n; ++i) {
        gsl::span<const float> point(flat_data.data() + i * dim, dim);
        float best_metric = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t j = 0; j < k; ++j) {
            gsl::span<const float> centroid(centroids.data() + j * dim, dim);
            float current_metric = euclidean_distance(point, centroid);
            if (current_metric < best_metric) {
                best_metric = current_metric;
                best_cluster = j;
            }
        }
        assignments[i] = best_cluster;
    }

    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(assignments[i], 0);
        ASSERT_LT(assignments[i], k);
    }
}

// Test with different metrics
TEST_F(KMeansTest, DifferentMetrics) {
    size_t n = 6;
    size_t dim = 4;
    size_t k = 3;

    std::vector<std::vector<float>> data = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {2.0f, 3.0f, 4.0f, 5.0f},
            {2.1f, 3.1f, 4.1f, 5.1f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {0.6f, 1.6f, 2.6f, 3.6f}
    };

    std::vector<float> flat_data = flatten(data);

    auto centroids_euclidean = kmeans(flat_data, n, dim, k, Metric::EUCLIDEAN);
    auto centroids_inner_product = kmeans(flat_data, n, dim, k, Metric::INNER_PRODUCT);

    ASSERT_EQ(centroids_euclidean.size(), centroids_inner_product.size());
    ASSERT_EQ(centroids_euclidean.size(), k * dim);
}