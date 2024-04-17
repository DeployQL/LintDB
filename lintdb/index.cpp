
#include "lintdb/index.h"
#include <glog/logging.h>
#include <omp.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "lintdb/assert.h"
#include "lintdb/cf.h"
#include "lintdb/invlists/RocksdbList.h"
#include "lintdb/schema/util.h"
#include "lintdb/retriever/Retriever.h"
#include <stdio.h>
#include <json/writer.h>
#include <json/json.h>
#include <json/reader.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <rocksdb/db.h>
#include <gsl/span>
#include "lintdb/util.h"


namespace lintdb {
  std::ostream& operator<<(std::ostream& os, const Configuration& config) {
        os << "Configuration(" << config.nlist << ", " << config.nbits << ", " << config.niter << ", " << config.dim << ", " << config.num_subquantizers << ", " << serialize_encoding(config.quantizer_type) << ")";
        return os;
    }

IndexIVF::IndexIVF(std::string path, bool read_only): read_only(read_only), path(path) {
    LOG(INFO) << "loading LintDB from path: " << path;

    // set all of our individual attributes.
    Configuration index_config = this->read_metadata(path);
    this->config = index_config;

    auto config = EncoderConfig{
        this->config.nlist, 
        this->config.nbits, 
        this->config.niter, 
        this->config.dim, 
        index_config.quantizer_type,
        this->config.num_subquantizers
    };
    this->encoder = DefaultEncoder::load(path, config);

    initialize_inverted_list();
}

IndexIVF::IndexIVF(std::string path, Configuration& config)
        : config(config), path(path) {
    IndexIVF(
        path,
        config.nlist,
        config.dim,
        config.nbits,
        config.niter,
        config.num_subquantizers,
        config.quantizer_type
    );
}

IndexIVF::IndexIVF(
        std::string path,
        size_t nlist,
        size_t dim,
        size_t binarize_nbits,
        size_t niter,
        size_t num_subquantizers,
        IndexEncoding quantizer_type,
        bool read_only
    ) : read_only(read_only), path(path) {
    LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

    Configuration config = Configuration{
        .nlist = nlist,
        .nbits = binarize_nbits,
        .niter = niter,
        .dim = dim,
        .num_subquantizers = num_subquantizers,
        .quantizer_type = quantizer_type
    };
    this->config = config;

    this->encoder = std::make_unique<DefaultEncoder>(
        nlist, binarize_nbits, niter, dim, num_subquantizers, quantizer_type
    );

    initialize_inverted_list();
}

IndexIVF::IndexIVF(const IndexIVF& other, const std::string path) {
    // we'll leverage the loading methods and construct the index components from files on disk.
    this->config = Configuration(other.config);
    this->read_only = false; // copying an index will always be writeable.

    this->path = path;

    auto config = EncoderConfig{
        this->config.nlist, 
        this->config.nbits, 
        this->config.niter, 
        this->config.dim, 
        this->config.quantizer_type, 
        this->config.num_subquantizers
    };
    this->encoder = DefaultEncoder::load(other.path, config);
    this->initialize_inverted_list();

    this->save();
}

void IndexIVF::initialize_inverted_list() {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    LOG(INFO) << "read only: " << read_only;

    if (!read_only) {
        rocksdb::OptimisticTransactionDB* ptr2;
        rocksdb::Status s = rocksdb::OptimisticTransactionDB::Open(
                options, path, cfs, &(this->column_families), &ptr2);
        if (!s.ok()) {
            LOG(ERROR) << s.ToString();
        }
        assert(s.ok());
        auto owned_ptr = std::shared_ptr<rocksdb::OptimisticTransactionDB>(ptr2);
        this->db = owned_ptr;
        auto index = std::make_shared<WritableRocksDBInvertedList>(WritableRocksDBInvertedList(owned_ptr, this->column_families));
        this->index_ = index;
        this->inverted_list_ = index;
    } else {
        rocksdb::DB* ptr;
        rocksdb::Status s = rocksdb::DB::OpenForReadOnly(
                options, path, cfs, &(this->column_families), &ptr);
        LOG(INFO) << s.ToString();
        assert(s.ok());
        auto owned_ptr = std::shared_ptr<rocksdb::DB>(ptr);
        this->db = owned_ptr;
        auto index = std::make_shared<ReadOnlyRocksDBInvertedList>(ReadOnlyRocksDBInvertedList(owned_ptr, this->column_families));
        this->index_ = index;
        this->inverted_list_ = index; 
    }

    this->retriever = std::make_unique<PlaidRetriever>(PlaidRetriever(this->inverted_list_, this->index_, this->encoder));

}

void IndexIVF::train(size_t n, std::vector<float>& embeddings) {
    this->train(embeddings.data(), n, config.dim);
}

void IndexIVF::train(float* embeddings, size_t n, size_t dim) {
    encoder->train(embeddings, n, dim);
    this->save();
}

void IndexIVF::train(float* embeddings, int n, int dim) {
    assert(config.nlist <= std::numeric_limits<code_t>::max() && "nlist must be less than 32 bits.");
    train(embeddings, static_cast<size_t>(n), static_cast<size_t>(dim));
}

void IndexIVF::save() {
    this->encoder->save(this->path);
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
    return search(tenant, block.embeddings.data(), block.num_tokens, block.dimensions, n_probe, k, opts);
}

/**
 * Implementation note: 
 * 
 * when we look at what IVF lists to search, we have several parameters that will influence this.
 * 1. k_top_centroids: responsible for how many centroids per token we include in our search before sorting.
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
    const float centroid_score_threshold = opts.centroid_score_threshold;
    const size_t total_centroids_to_calculate = config.nlist;
    const size_t k_top_centroids = opts.k_top_centroids;

    gsl::span<const float> query_span = gsl::span(data, n);
    RetrieverOptions plaid_options = RetrieverOptions{
        .total_centroids_to_calculate = config.nlist,
        .num_second_pass = opts.num_second_pass,
        .expected_id = opts.expected_id,
        .centroid_threshold = opts.centroid_score_threshold,
        .k_top_centroids = opts.k_top_centroids,
        .n_probe = n_probe
    };

    auto results = this->retriever->retrieve(
        tenant,
        query_span,
        n,
        k,
        plaid_options
    );


    return results;
}

void IndexIVF::set_centroids(float* data, int n, int dim) {
    encoder->set_centroids(data, n, dim);
}

void IndexIVF::set_weights(const std::vector<float> weights, const std::vector<float> cutoffs, const float avg_residual) {
    encoder->set_weights(weights, cutoffs, avg_residual);
}


void IndexIVF::add(const uint64_t tenant, const std::vector<RawPassage>& docs) {
    LINTDB_THROW_IF_NOT(this->encoder->is_trained);

    for (auto doc : docs) {
        add_single(tenant, doc);
    }
}

void IndexIVF::add_single(const uint64_t tenant, const RawPassage& doc) {
    auto encoded = encoder->encode_vectors(doc);

    index_->add(tenant, std::move(encoded));
}

void IndexIVF::remove(const uint64_t tenant, const std::vector<idx_t>& ids) {
    index_->remove(tenant, ids);
}

void IndexIVF::update(const uint64_t tenant, const std::vector<RawPassage>& docs) {
    std::vector<idx_t> ids;
    for (auto doc : docs) {
        ids.push_back(doc.id);
    }
    remove(tenant, ids);
    add(tenant, docs);
}

void IndexIVF::merge(const std::string path) {

    Configuration incoming_config = read_metadata(path);

    LOG(INFO) << "incoming config: " << incoming_config;
    LOG(INFO) << "current config: " << this->config;

    LINTDB_THROW_IF_NOT(this->config == incoming_config);

    rocksdb::Options options;
    options.create_if_missing = false;
    options.create_missing_column_families = false;

    auto cfs = create_column_families();
    std::vector<rocksdb::ColumnFamilyHandle*> other_cfs;

    rocksdb::DB* ptr;
    rocksdb::Status s = rocksdb::DB::OpenForReadOnly(
            options, path, cfs, &other_cfs, &ptr);
    assert(s.ok());
    std::shared_ptr<rocksdb::DB> owned_ptr = std::shared_ptr<rocksdb::DB>(ptr);

    index_->merge(owned_ptr, other_cfs);

    for (auto cf : other_cfs) {
            owned_ptr->DestroyColumnFamilyHandle(cf);
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
    metadata["num_subquantizers"] = Json::UInt64(this->config.num_subquantizers);

    auto quantizer_type = serialize_encoding(this->config.quantizer_type);
    metadata["quantizer_type"] = Json::String(quantizer_type);

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

    return config;
}
} // namespace lintdb