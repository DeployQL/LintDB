#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include "lintdb/index.h"

static void BM_lintdb_search(benchmark::State& state) {
    // std::string path = "/mnt/data/py_index_bench_colbert-lifestyle-2024-04-16-pq";
    std::string path = "experiments/py_index_bench_colbert-lifestyle-2024-04-03";
    lintdb::IndexIVF index(path);
    for (auto _ : state) {
        state.PauseTiming();
        size_t dims = 128;
        size_t num_tokens = 32;
        std::vector<float> query(dims * num_tokens, 1);

        size_t n_probe = 32;
        size_t k = 100;
        lintdb::SearchOptions opts;
        
        state.ResumeTiming();
        index.search(0, query.data(), num_tokens, dims, n_probe, k, opts);

    }
}

BENCHMARK(BM_lintdb_search)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();