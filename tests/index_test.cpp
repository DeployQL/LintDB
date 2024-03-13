#include <gtest/gtest.h>
// TODO(mbarta): we introspect the invlists during tests. We can fix this with better abstractions.
#define private public
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/invlists/RocksdbList.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include <cblas.h>
#include <filesystem>
#include <glog/logging.h>
#include <gsl/span>
#include "lintdb/util.h"

// Demonstrate some basic assertions.
TEST(IndexTest, InitializesCorrectly) {
    size_t nbits = 4; // the number of bits to encode residual codes into.
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");

    lintdb::IndexIVF index(temp_db.string(), 5, 128, 2);

    EXPECT_EQ(index.nlist, 5);
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

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, false, true);

    index.train(num_docs * num_tokens, buf);
    EXPECT_EQ(index.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock block(fake_doc.data(), num_tokens, dim);

    lintdb::RawPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage> docs = { doc };
    index.add(docs);

    // without knowing what ivf list we assigned the doc to, make sure one document is indexed.
    // this amounts to a full scan.
    for(idx_t i=0; i<kclusters; i++) {
        lintdb::Key start{0, i, 0, true};
        lintdb::Key end{0, i, std::numeric_limits<idx_t>::max(), false};
        std::string start_string = start.serialize();
        std::string end_string = end.serialize();
        lintdb::RocksDBInvertedList casted = static_cast<lintdb::RocksDBInvertedList&>(*index.index_);
        std::unique_ptr<lintdb::Iterator> it = casted.get_iterator(start_string, end_string);
        for(; it->has_next(); it->next()) {
            lintdb::Key key = it->get_key();
            auto id = key.id;
            EXPECT_EQ(id, idx_t(1));
        }
    }

}

// EmbeddingBlocks store data in column major format, so contiguous memory is
// a column of data. we should expect the first column to be the first elements generated.
TEST(IndexTest, EmbeddingBlocksAreRowMajor) {
    size_t dim = 128;
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t num_bits = 2;

    std::vector<float> buf(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, buf.data(), 1234);
    // let's reuse the buffer to grab what we'd consider a a block of embeddings.
    lintdb::EmbeddingBlock block{buf.data(), num_tokens, dim};
    //check that the first row is correct in the block.
    for (size_t i = 0; i < num_tokens; i++) {
        auto scal = block.embeddings[i];
        EXPECT_EQ(scal, buf[i]);
    }
}

TEST(IndexTest, SearchCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits);

    index.train(num_docs * num_tokens, buf);

    EXPECT_EQ(index.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens, 1);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::RawPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage> docs = { doc };
    index.add(docs);

    auto opts = lintdb::SearchOptions();
    opts.expected_id = 1;
    auto results = index.search(block, 10, 5, opts);

    EXPECT_GT(results.size(), 0);
    // we expect to get back the same document we added.
    auto actual = results[0].id;
    EXPECT_EQ(actual, 1);
}

TEST(IndexTest, LoadsCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 1;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("test_index");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }

    lintdb::IndexIVF* index = new lintdb::IndexIVF(temp_db.string(), kclusters, dim, centroid_bits);

    index->train(num_docs * num_tokens, buf);

    delete index;

    auto loaded_index = lintdb::IndexIVF(temp_db.string());

    std::vector<float> query(dim * num_tokens, 1);
    loaded_index.search(query.data(), num_tokens, dim, 10, 5);
}


int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}