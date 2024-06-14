#include "lintdb/index.h"
#include <glog/logging.h>
#include <json/reader.h>
#include <json/writer.h>
#include <omp.h>
#include <rocksdb/db.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <gsl/span>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "lintdb/assert.h"
#include "lintdb/api.h"
#include "lintdb/cf.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/retrievers/Retriever.h"
#include "lintdb/schema/util.h"
#include "lintdb/util.h"
#include "lintdb/version.h"
#include "lintdb/quantizers/io.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/retrievers/XTRRetriever.h"
#include "lintdb/invlists/RocksdbInvertedListV2.h"

namespace lintdb {
// env var to set the number of threads for processing.
const char* PROCESSING_THREADS = "LINTDB_NUM_THREADS";

std::ostream& operator<<(std::ostream& os, const Configuration& config) {
    os << "Configuration(" << config.nlist << ", " << config.nbits << ", "
       << config.niter << ", " << config.dim << ", " << config.num_subquantizers
       << ", " << serialize_encoding(config.quantizer_type) << ")";
    return os;
}

IndexIVF::IndexIVF(const std::string& path, bool read_only)
        : read_only(read_only), path(path) {
    LOG(INFO) << "loading LintDB from path: " << path;

    // set all of our individual attributes.
    this->config = this->read_metadata(path);

    initialize_inverted_list(config.lintdb_version);

    load_retrieval(path, config);
}

IndexIVF::IndexIVF(std::string path, Configuration& config)
        : config(config), read_only(false), path(path) {
    LINTDB_THROW_IF_NOT(config.nlist <= std::numeric_limits<code_t>::max());
    this->config = config;

    initialize_inverted_list(config.lintdb_version);
    initialize_retrieval(this->config.quantizer_type);
}

IndexIVF::IndexIVF(
        std::string path,
        size_t nlist,
        size_t dim,
        size_t binarize_nbits,
        size_t niter,
        size_t num_subquantizers,
        IndexEncoding quantizer_type,
        bool read_only)
        : read_only(read_only), path(path) {
        LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

        Configuration config;
        config.nlist = nlist;
        config.nbits = binarize_nbits;
        config.niter = niter;
        config.dim = dim;
        config.num_subquantizers = num_subquantizers;
        config.quantizer_type = quantizer_type;
        config.lintdb_version = LINTDB_VERSION;

        this->config = config;

        initialize_inverted_list(config.lintdb_version);
        initialize_retrieval(this->config.quantizer_type);
}

IndexIVF::IndexIVF(const IndexIVF& other, const std::string path) {
    // we'll leverage the loading methods and construct the index components
    // from files on disk.
    this->config = Configuration(other.config);
    this->read_only = false; // copying an existing index will always create a writeable blank index.

    this->path = path;

    this->initialize_inverted_list(this->config.lintdb_version);
    load_retrieval(other.path, other.config);

    this->save();
}

void IndexIVF::initialize_retrieval(IndexEncoding quantizer_type) {
    // set omp threads
    if(std::getenv(PROCESSING_THREADS) != nullptr) {
        omp_set_num_threads(std::stoi(std::getenv(PROCESSING_THREADS)));
    }
    switch (quantizer_type) {
        case IndexEncoding::NONE:
            this->quantizer = nullptr;
            this->encoder = std::make_unique<DefaultEncoder>(
                    this->config.dim,
                    quantizer);
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;

        case IndexEncoding::BINARIZER:
            this->quantizer = std::make_unique<Binarizer>(config.nbits, config.dim);
            this->encoder = std::make_unique<DefaultEncoder>(
                    this->config.dim,
                    quantizer);
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;

        case IndexEncoding::PRODUCT_QUANTIZER:
            this->quantizer = std::make_unique<ProductEncoder>(
                    config.dim, config.nbits, config.num_subquantizers);
            this->encoder = std::make_unique<DefaultEncoder>(
                    this->config.dim,
                    quantizer);
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;
        case IndexEncoding::XTR:
            this->quantizer = std::make_unique<ProductEncoder>(
                    config.dim, config.nbits, config.num_subquantizers);

            this->encoder = std::make_unique<DefaultEncoder>(
                    this->config.dim,
                    this->quantizer);

            this->retriever = std::make_unique<XTRRetriever>(
                    this->inverted_list_, this->index_, this->encoder, std::dynamic_pointer_cast<ProductEncoder>(this->quantizer));
            break;

        default:
            throw LintDBException("Quantizer type not valid.");
    }

    // this is just legacy behavior.
    if (this->config.nlist) {
        this->encoder->nlist = this->config.nlist;
        this->encoder->niter = this->config.niter;
    }
}

void IndexIVF::load_retrieval(std::string path, const Configuration& config) {
    QuantizerConfig qc {
            config.nbits,
            config.dim,
            config.num_subquantizers
    };
    this->quantizer = load_quantizer(path, config.quantizer_type, qc);

    auto ec = EncoderConfig{
            config.nlist,
            config.nbits,
            config.niter,
            config.dim,
            config.num_subquantizers,
            config.quantizer_type};
    this->encoder = DefaultEncoder::load(path, this->quantizer, ec);

    switch(config.quantizer_type) {
        case IndexEncoding::XTR:
            this->retriever = std::make_unique<XTRRetriever>(
                    this->inverted_list_, this->index_, this->encoder, std::dynamic_pointer_cast<ProductEncoder>(this->quantizer));
            break;
        case IndexEncoding::BINARIZER:
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;
        case IndexEncoding::PRODUCT_QUANTIZER:
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;
        case IndexEncoding::NONE:
            this->retriever = std::make_unique<PlaidRetriever>(
                    PlaidRetriever(this->inverted_list_, this->index_, this->encoder));
            break;
        default:
            throw LintDBException("Index Encoding not known.");
    }

    // this is just legacy behavior.
    if (this->config.nlist) {
        this->encoder->nlist = config.nlist;
        this->encoder->niter = config.niter;
    }
}

void IndexIVF::initialize_inverted_list(Version& version) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();

    rocksdb::DB* ptr;
    rocksdb::Status s;
    if (!read_only) {
        s = rocksdb::DB::Open(
                options, path, cfs, &(this->column_families), &ptr);
    } else {
        s = rocksdb::DB::OpenForReadOnly(
                options, path, cfs, &(this->column_families), &ptr);
    }
    if (!s.ok()) {
        LOG(ERROR) << s.ToString();
    }
    assert(s.ok());
    auto owned_ptr =
            std::shared_ptr<rocksdb::DB>(ptr);
    this->db = owned_ptr;

    this->index_ = std::make_shared<RocksdbForwardIndex>(owned_ptr, this->column_families, version);
    if (this->config.quantizer_type == IndexEncoding::XTR) {
        this->inverted_list_ = std::make_shared<RocksdbInvertedListV2>(owned_ptr, this->column_families, version);
    } else {
        this->inverted_list_ = std::make_shared<RocksdbInvertedList>(owned_ptr, this->column_families, version);
    }
}

void IndexIVF::train(size_t n, std::vector<float>& embeddings, size_t nlist, size_t niter) {
    this->train(embeddings.data(), n, config.dim, nlist, niter);
}

void IndexIVF::train(float* embeddings, size_t n, size_t dim, size_t nlist, size_t niter) {
    assert(config.nlist <= std::numeric_limits<code_t>::max() &&
           "nlist must be less than 32 bits.");

    if (nlist != 0) {
        this->config.nlist = nlist;
    }
    if (niter != 0) {
        this->config.niter = niter;
    }

    encoder->train(embeddings, n, dim, nlist, niter);
    this->save();
}

void IndexIVF::train(float* embeddings, int n, int dim, size_t nlist, size_t niter) {

    train(embeddings, static_cast<size_t>(n), static_cast<size_t>(dim), nlist, niter);
}

void IndexIVF::save() {
    this->encoder->save(this->path);
    save_quantizer(path, quantizer.get());
    this->write_metadata();
}

void IndexIVF::flush() {
    rocksdb::FlushOptions fo;
    this->db->Flush(fo, column_families);
}

std::vector<SearchResult> IndexIVF::search(
        const uint64_t tenant,
        const EmbeddingBlock& block,
        const size_t n_probe,
        const size_t k,
        const SearchOptions& opts) const {
    return search(
            tenant,
            block.embeddings.data(),
            block.num_tokens,
            block.dimensions,
            n_probe,
            k,
            opts);
}

std::vector<SearchResult> IndexIVF::search(
    const uint64_t tenant,
    const EmbeddingBlock& block,
    const size_t k,
    const SearchOptions& opts) const {
    return search(
        tenant,
        block.embeddings.data(),
        block.num_tokens,
        block.dimensions,
        opts.n_probe,
        k,
        opts);
}

std::vector<SearchResult> IndexIVF::search(
        const uint64_t tenant,
        const float* data,
        const int n,
        const int dim,
        const size_t k,
        const SearchOptions& opts) const {
    return search(
            tenant,
            data,
            n,
            dim,
            opts.n_probe,
            k,
            opts);
}

/**
 * Implementation note:
 *
 * when we look at what IVF lists to search, we have several parameters that
 * will influence this.
 * 1. k_top_centroids: responsible for how many centroids per token we include
 * in our search before sorting.
 * 2. n_probe: the number of lists we search on after sorting.
 */
std::vector<SearchResult> IndexIVF::search(
        const uint64_t tenant,
        const float* data,
        const int n,
        const int dim,
        const size_t n_probe,
        const size_t k,
        const SearchOptions& opts) const {
    // per block, run a matrix multiplication and find the nearest centroid.
    // block: (num_tokens x dimensions)
    // centroids: (nlist x dimensions)
    // result: (num_tokens x nlist)
    gsl::span<const float> query_span = gsl::span(data, n);
    RetrieverOptions plaid_options = RetrieverOptions{
            .total_centroids_to_calculate = config.nlist,
            .num_second_pass = opts.num_second_pass,
            .expected_id = opts.expected_id,
            .centroid_threshold = opts.centroid_score_threshold,
            .k_top_centroids = opts.k_top_centroids,
            .n_probe = n_probe,
            .nearest_tokens_to_fetch = opts.nearest_tokens_to_fetch,
    };

    auto results =
            this->retriever->retrieve(tenant, query_span, n, k, plaid_options);
    std::vector<idx_t> ids;
    for (auto& result : results) {
        ids.push_back(result.id);
    }
    if(config.lintdb_version.metadata_enabled) {
        auto metadata = this->index_->get_metadata(tenant, ids);

        for (size_t i = 0; i < results.size(); i++) {
            auto md= metadata[i]->metadata;
            for(auto& m : md) {
                results[i].metadata[m.first] = m.second;
            }
        }
    }

    return results;
}

void IndexIVF::set_centroids(float* data, int n, int dim) {
    encoder->set_centroids(data, n, dim);
}

void IndexIVF::set_weights(
        const std::vector<float> weights,
        const std::vector<float> cutoffs,
        const float avg_residual) {
    encoder->set_weights(weights, cutoffs, avg_residual);
}

void IndexIVF::add(const uint64_t tenant, const std::vector<EmbeddingPassage>& docs) {
    LINTDB_THROW_IF_NOT(this->encoder->is_trained);

    for (auto doc : docs) {
        add_single(tenant, doc);
    }
}

void IndexIVF::add_single(const uint64_t tenant, const EmbeddingPassage& doc) {
    auto encoded = encoder->encode_vectors(doc);
    inverted_list_->add(tenant, encoded.get());
    //
    index_->add(tenant, encoded.get(), config.quantizer_type != IndexEncoding::XTR);
}

void IndexIVF::remove(const uint64_t tenant, const std::vector<idx_t>& ids) {
    inverted_list_->remove(tenant, ids);
    index_->remove(tenant, ids);
}

void IndexIVF::update(
        const uint64_t tenant,
        const std::vector<EmbeddingPassage>& docs) {
    std::vector<idx_t> ids;
    for (auto doc : docs) {
        ids.push_back(doc.id);
    }
    remove(tenant, ids);
    add(tenant, docs);
}

void IndexIVF::merge(const std::string path) {
    Configuration incoming_config = read_metadata(path);

    LINTDB_THROW_IF_NOT(this->config == incoming_config);

    rocksdb::Options options;
    options.create_if_missing = false;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    std::vector<rocksdb::ColumnFamilyHandle*> other_cfs;

    rocksdb::DB* ptr;
    rocksdb::Status s =
            rocksdb::DB::OpenForReadOnly(options, path, cfs, &other_cfs, &ptr);
    assert(s.ok());

    inverted_list_->merge(ptr, other_cfs);
    index_->merge(ptr, other_cfs);

    for (auto cf : other_cfs) {
        db->DestroyColumnFamilyHandle(cf);
    }
}

void IndexIVF::write_metadata() {
    std::string out_path = path + "/" + METADATA_FILENAME;
    std::ofstream out(out_path);

    Json::Value metadata;
    // the below castings to Json types enables this to build on M1 Mac.
    metadata["nlist"] = Json::UInt64(this->config.nlist);
    metadata["nbits"] = Json::UInt64(this->config.nbits);
    metadata["dim"] = Json::UInt64(this->config.dim);
    metadata["niter"] = Json::UInt64(this->config.niter);
    metadata["num_subquantizers"] =
            Json::UInt64(this->config.num_subquantizers);

    auto quantizer_type = serialize_encoding(this->config.quantizer_type);
    metadata["quantizer_type"] = Json::String(quantizer_type);

    metadata["lintdb_version"] = Json::String(LINTDB_VERSION_STRING);

    Json::StyledWriter writer;
    out << writer.write(metadata);
    out.close();
}

Configuration IndexIVF::read_metadata(std::string path) {
    this->path = path;
    std::string in_path = path + "/" + METADATA_FILENAME;
    std::ifstream in(in_path);
    if (!in) {
        throw LintDBException("Could not read metadata from path: " + in_path);
    }

    Json::Reader reader;
    Json::Value metadata;
    reader.parse(in, metadata);

    Configuration config;

    config.nlist = metadata["nlist"].asUInt();
    config.nbits = metadata["nbits"].asUInt();
    config.dim = metadata["dim"].asUInt();
    config.niter = metadata["niter"].asUInt();
    config.num_subquantizers = metadata["num_subquantizers"].asUInt();

    auto quantizer_type = metadata["quantizer_type"].asString();
    config.quantizer_type = deserialize_encoding(quantizer_type);

    std::string version = metadata.get("lintdb_version", "0.0.0").asString();
    config.lintdb_version = Version(version);

    return config;
}

} // namespace lintdb