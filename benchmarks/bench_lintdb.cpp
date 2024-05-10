#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include "lintdb/index.h"
#include "lintdb/index_builder/EmbeddingModel.h"
#include "lintdb/index_builder/Tokenizer.h"

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

static void BM_lintdb_embed(benchmark::State& state) {
    std::string path = "assets/model.onnx";
    lintdb::EmbeddingModel model(path);

    std::string tok_path = "assets/colbert_tokenizer.json";
    lintdb::Tokenizer tokenizer(tok_path, 512);
    std::string text = "In order to effectively benchmark an ML model, it's crucial to meticulously design a comprehensive evaluation framework encompassing diverse datasets, validation techniques, and performance metrics, ensuring robustness and reliability in assessing its predictive capabilities across various real-world scenarios, thereby facilitating informed decisions regarding model selection, optimization strategies, and deployment considerations, ultimately advancing the state-of-the-art in machine learning research and applications.";

    for(auto _ : state) {
        auto ids = tokenizer.encode(text);

        lintdb::ModelInput input;
        input.input_ids = ids;

        std::vector<int32_t> attn;
        for(auto id: ids) {
            if(id == 0) {
                attn.push_back(0);
            } else {
                attn.push_back(1);
            }
        }
        input.attention_mask = attn;

        auto output = model.encode(input);
    }
}

BENCHMARK(BM_lintdb_search)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_lintdb_embed)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();