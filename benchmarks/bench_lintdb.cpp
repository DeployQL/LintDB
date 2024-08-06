#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include "lintdb/index.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/quantizers/Quantizer.h"

lintdb::Document create_document(size_t num_tokens, size_t dim){
    std::vector<float> vector;
    for (size_t j = 0; j < num_tokens; j++) {
        std::vector<float> data(dim, j);
        vector.insert(vector.end(), data.begin(), data.end());
    }
    lintdb::FieldValue fv("colbert", vector, num_tokens);
    std::vector<lintdb::FieldValue> fields = {fv};

    lintdb::Document doc(0, fields );
    return doc;
}

inline std::filesystem::path create_temporary_directory(
        unsigned long long max_tries = 1000) {
    auto tmp_dir = std::filesystem::temp_directory_path();
    unsigned long long i = 0;
    std::random_device dev;
    std::mt19937 prng(dev());
    std::uniform_int_distribution<uint64_t> rand(0);
    std::filesystem::path path;
    while (true) {
        std::stringstream ss;
        ss << std::hex << rand(prng);
        path = tmp_dir / ss.str();
        // true if the directory was created.
        if (std::filesystem::create_directory(path)) {
            break;
        }
        if (i == max_tries) {
            throw std::runtime_error("could not find non-existing directory");
        }
        i++;
    }
    return path;
}


static void BM_lintdb_add(benchmark::State& state) {
    lintdb::Schema schema;

    lintdb::Field colbert;
    colbert.name = "colbert";
    colbert.data_type = lintdb::DataType::TENSOR;
    colbert.field_types = {lintdb::FieldType::Colbert};
    lintdb::FieldParameters fp;
    fp.dimensions = 128;
    fp.num_centroids = 10;
    fp.num_iterations = 2;
    fp.quantization = lintdb::QuantizerType::BINARIZER;
    fp.nbits = 1;
    colbert.parameters = fp;

    schema.add_field(colbert);

    auto temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    std::vector<lintdb::Document> docs;
    for (size_t i = 0; i < 50; i++) {
        docs.push_back(create_document(120, 128));
    }
    index.train(docs);

    auto doc = create_document(120, 128);

    for(auto _ : state) {
        index.add(0, {doc});
    }

}

BENCHMARK(BM_lintdb_add)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();