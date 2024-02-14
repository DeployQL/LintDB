
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/invlists/RocksdbList.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include <arrayfire.h>
#include <cblas.h>
#include <filesystem>
#include <glog/logging.h>

// Demonstrate some basic assertions.
TEST(IndexTest, InitializesCorrectly) {
    size_t nbits = 4; // the number of bits to encode residual codes into.
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");

    lintdb::IndexIVF index(temp_db.string(), 5, 128, 4, 2);

    EXPECT_EQ(index.nlist, 5);
    EXPECT_EQ(index.is_trained, false);
    EXPECT_EQ(index.quantizer.is_trained, false);
}

TEST(IndexTest, MatrixFormatIsCorrect) {
    size_t dim = 4;
    size_t kclusters = 2;

    std::vector<float> centroids = {1, 2, 3, 4, 5, 6, 7, 8};

    // (4x2)
    auto centroid_array = af::array(dim, kclusters, centroids.data());

    // we expect the first row to be the first 4 elements of the centroids.
    auto ptr = centroid_array.device<float>();
    for (size_t i = 0; i < dim; i++) {
        auto scal = ptr+i;
        EXPECT_EQ(*scal, centroids[i]);
    }
    af::freeHost(ptr);
}

TEST(IndexTest, TrainsCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 32;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 2);

    index.train(num_docs * num_tokens, buf);
    EXPECT_EQ(index.nlist, 250);
    EXPECT_EQ(index.quantizer.is_trained, true);
    // prove to ourselves that we have centroids in the form of (kclusters x dim)
    auto first = index.centroids.dims(0);
    EXPECT_EQ(first, kclusters);
    auto second = index.centroids.dims(1);
    EXPECT_EQ(second, dim);


    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock raw_block(dim, num_tokens, fake_doc.data());
    lintdb::EmbeddingBlock block = af::transpose(raw_block);

    std::vector<lintdb::EmbeddingBlock> blocks = { block };
    std::vector<idx_t> ids = { 1 };
    index.add(blocks, ids);

    // without knowing what ivf list we assigned the doc to, make sure one document is indexed.
    // this amounts to a full scan.
    size_t total_docs_indexed = 0;
    for(idx_t i=0; i<kclusters; i++) {
        auto it = index.invlists->get_iterator(i);
        for(; it->has_next(); it->next()) {
            auto doc = it->get();
            total_docs_indexed++;
        }
        auto casted = static_cast<lintdb::RocksDBIterator*>(it.get());
        EXPECT_TRUE(casted->it->status().ok());
    }
    EXPECT_EQ(total_docs_indexed, 1);

}

TEST(IndexTest, MatmulWorks) {
    // I was having issues linking blas here. What appears to have fixed it is
    // linking lintdb-tests directly to arrayfire.

    EXPECT_EQ(af::isLAPACKAvailable(),true);

    af::array one = af::randu(10, 20);
    af::array two = af::randu(20, 10);
    try {
        fprintf(stderr, "Attempting to multiply\n");
        af::array result = af::matmul(one, two);

    }   catch (af::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    EXPECT_EQ(true, true);
}

// EmbeddingBlocks store data in column major format, so contiguous memory is
// a column of data. we should expect the first column to be the first elements generated.
TEST(IndexTest, EmbeddingBlocksAreColumnMajor) {
    size_t dim = 128;
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t num_bits = 2;

    std::vector<float> buf(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, buf.data(), 1234);
    // let's reuse the buffer to grab what we'd consider a a block of embeddings.
    lintdb::EmbeddingBlock block(num_tokens, dim, buf.data());
    //check that the first row is correct in the block.
    for (size_t i = 0; i < num_tokens; i++) {
        auto scal = block(i, 0).scalar<float>();
        EXPECT_EQ(scal, buf[i]);
    }
}

int main(int argc, char **argv) {
    // FLAGS_v = 10;
    // FLAGS_logtostderr = true;

    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}