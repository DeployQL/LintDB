#include "lintdb/index.h"
#include <faiss/utils/random.h>
#include <chrono>
#include <iostream>

int main() {
    auto index = lintdb::IndexIVF("/home/matt/deployql/LintDB/experiments/py_index_bench_colbert-lifestyle-2024-03-20");


    size_t dim = 128;
    size_t num_docs = 100;
    size_t num_tokens = 100;

    lintdb::SearchOptions opts;
    opts.centroid_score_threshold = 0;
    opts.k_top_centroids = 64;
    opts.num_second_pass = 1024;

    size_t num_bits = 2;

    std::vector<float> buf(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, buf.data(), 1234);
    auto block = lintdb::EmbeddingBlock(buf.data(), num_tokens, dim);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto results = index.search(0, block, 64, 100, opts);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "Elapsed time: " << elapsed_time_ms << "ms" << std::endl;
    std::cout << "Results: " << results.size() << std::endl;
}