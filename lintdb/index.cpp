
#include "lintdb/index.h"
#include <glog/logging.h>
#include <omp.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "lintdb/assert.h"
#include "lintdb/cf.h"
#include "lintdb/invlists/RocksdbList.h"
#include "lintdb/plaid.h"
#include "lintdb/schema/util.h"
#include "lintdb/retriever/Retriever.h"
#include <stdio.h>
#include <json/writer.h>
#include <json/json.h>
#include <json/reader.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <rocksdb/db.h>
#include <gsl/span>

namespace lintdb {
IndexIVF::IndexIVF(std::string path, bool read_only): path(path), read_only(read_only) {
    LOG(INFO) << "loading LintDB from path: " << path;

    // set all of our individual attributes.
    Configuration index_config = this->read_metadata(path);
    this->nlist = index_config.nlist;
    this->nbits = index_config.nbits;
    this->dim = index_config.dim;
    this->niter = index_config.niter;
    this->use_compression = index_config.use_compression;
    this->read_only = read_only;

    auto config = EncoderConfig{
        nlist, nbits, niter, dim, use_compression
    };
    this->encoder = DefaultEncoder::load(path, config);

    initialize_inverted_list();

    this->retriever = std::make_unique<PlaidRetriever>(PlaidRetriever(this->index_, this->encoder));
}

IndexIVF::IndexIVF(std::string path, Configuration& config)
        : nlist(config.nlist), nbits(config.nbits), path(path) {
    IndexIVF(
        path,
        config.nlist,
        config.dim,
        config.nbits,
        config.niter,
        config.use_compression
    );
}

IndexIVF::IndexIVF(
        std::string path,
        size_t nlist,
        size_t dim,
        size_t binarize_nbits,
        size_t niter,
        bool use_compression,
        bool read_only
    ) : path(path), nlist(nlist), dim(dim), nbits(binarize_nbits), niter(niter), use_compression(use_compression), read_only(read_only){
    // for now, we can only handle 32 bit coarse codes.
    LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

    this->encoder = std::make_unique<DefaultEncoder>(
        nlist, nbits, niter, dim, use_compression
    );

    initialize_inverted_list();

    this->retriever = std::make_unique<PlaidRetriever>(PlaidRetriever(this->index_, this->encoder));
}

IndexIVF::IndexIVF(const IndexIVF& other, const std::string path) {
    // we'll leverage the loading methods and construct the index components from files on disk.
    Configuration index_config = this->read_metadata(other.path);
    this->nlist = index_config.nlist;
    this->nbits = index_config.nbits;
    this->dim = index_config.dim;
    this->niter = index_config.niter;
    this->use_compression = index_config.use_compression;
    this->read_only = other.read_only;

    this->path = path;

    auto config = EncoderConfig{
        nlist, nbits, niter, dim, use_compression
    };
    this->encoder = DefaultEncoder::load(other.path, config);

    initialize_inverted_list();

    this->retriever = std::make_unique<PlaidRetriever>(PlaidRetriever(this->index_, this->encoder));

    this->save();
}

void IndexIVF::initialize_inverted_list() {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
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
        this->index_ = std::make_unique<WritableRocksDBInvertedList>(WritableRocksDBInvertedList(owned_ptr, this->column_families));
    } else {
        rocksdb::DB* ptr;
        rocksdb::Status s = rocksdb::DB::OpenForReadOnly(
                options, path, cfs, &(this->column_families), &ptr);
        assert(s.ok());
        auto owned_ptr = std::shared_ptr<rocksdb::DB>(ptr);
        this->db = owned_ptr;
        this->index_ = std::make_unique<ReadOnlyRocksDBInvertedList>(ReadOnlyRocksDBInvertedList(owned_ptr, this->column_families));
    }
}

void IndexIVF::train(size_t n, std::vector<float>& embeddings) {
    this->train(embeddings.data(), n, dim);
}

void IndexIVF::train(float* embeddings, size_t n, size_t dim) {
    encoder->train(embeddings, n, dim);
    this->save();
}

void IndexIVF::train(float* embeddings, int n, int dim) {
    assert(nlist <= std::numeric_limits<code_t>::max() && "nlist must be less than 32 bits.");
    train(embeddings, static_cast<size_t>(n), static_cast<size_t>(dim));
}

void IndexIVF::save() {
    this->encoder->save(this->path);
    this->write_metadata();
}

std::vector<std::pair<float, idx_t>> IndexIVF::get_top_centroids(
    const std::vector<idx_t>& coarse_idx,
    const std::vector<float>& distances, 
    const size_t n, // num_tokens
    const size_t total_centroids_to_calculate,
    const float centroid_score_threshold,
    const size_t k_top_centroids,
    const size_t n_probe) const {

    
    // we're finding the highest centroid scores per centroid.
    std::vector<float> high_scores(nlist, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k_top_centroids; j++) {
            auto centroid_of_interest = coarse_idx[i*total_centroids_to_calculate+j];

            // Note: including the centroid score threshold is not part of the original colBERT model.
            // distances[i*total_centroids_to_calculate+j] > centroid_score_threshold && 
                
            if (distances[i*total_centroids_to_calculate+j] > high_scores[centroid_of_interest]) {
                high_scores[centroid_of_interest] = distances[i*total_centroids_to_calculate+j];
            }
        }
    }

    // lets prepare a min heap comparator.
    auto comparator = [](std::pair<float, idx_t> p1, std::pair<float, idx_t> p2) {
        return p1.first > p2.first;
    };

    std::vector<std::pair<float, idx_t>> centroid_scores;
    centroid_scores.reserve(n_probe);
    for(int i=0; i < high_scores.size(); i++) {
        auto key = i;
        auto score = high_scores[i];
        // Note(MB): removing the filtering by score enables searching with exact copies.
        if (score > 0){
            if (centroid_scores.size() < n_probe) {
                centroid_scores.push_back(std::pair<float, idx_t>(score, key));

                if (centroid_scores.size() == n_probe) {
                    std::make_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
                }
            } else if (score > centroid_scores.front().first) {
                std::pop_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
                centroid_scores.front() = std::pair<float, idx_t>(score, key);
                std::push_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
            }
        }
    }

    if(centroid_scores.size() < n_probe) {
        std::sort(
                centroid_scores.begin(),
                centroid_scores.end(),
                std::greater<std::pair<float, idx_t>>()
        );
    } else {
        std::sort_heap(centroid_scores.begin(), centroid_scores.end(), comparator);
    }

    VLOG(1) << "num centroids: " << centroid_scores.size();
    for(auto p : centroid_scores) {
        VLOG(1) << "centroid: " << p.second << " score: " << p.first;
    }

    return centroid_scores;
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
    return search(tenant, block.embeddings.data(), block.num_tokens, block.dimensions, n_probe, k);
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
    const size_t total_centroids_to_calculate = nlist;
    const size_t k_top_centroids = opts.k_top_centroids;
    const size_t num_second_pass = opts.num_second_pass;

    std::vector<idx_t> coarse_idx(n*total_centroids_to_calculate);
    std::vector<float> distances(n*total_centroids_to_calculate);
    encoder->search_quantizer(
        data,
        n,
        coarse_idx,
        distances,
        total_centroids_to_calculate,
        centroid_score_threshold
    );

    // // well, to get to the other side of this, we reorder the distances
    // // in order of the centroids.
    std::vector<float> reordered_distances(n*total_centroids_to_calculate);

    for(int i=0; i < n; i++) {
        for(int j=0; j < total_centroids_to_calculate; j++) {
            auto current_code = coarse_idx[i*total_centroids_to_calculate+j];
            float dis = distances[i*total_centroids_to_calculate+j];
            reordered_distances[i*total_centroids_to_calculate+current_code] = dis;
        }
    }

    auto centroid_scores = get_top_centroids(
        coarse_idx,
        distances,
        n,
        total_centroids_to_calculate,
        centroid_score_threshold,
        k_top_centroids,
        n_probe
    );

    auto num_centroids_to_eval = std::min<size_t>(n_probe, centroid_scores.size());

    if (opts.expected_id != -1) {
        LOG(INFO) << "expected id: " << opts.expected_id;
        LOG(INFO) << "centroid score size: " << centroid_scores.size();
        auto mapping = index_->get_mapping(tenant, opts.expected_id);
        std::unordered_set<idx_t> mapping_set(mapping.begin(), mapping.end());
        for(int i = 0; i < centroid_scores.size(); i++) {
            if (mapping_set.find(centroid_scores[i].second) != mapping_set.end()) {
                LOG(INFO) << "expected id found in centroid: " << centroid_scores[i].second;
                if (i > num_centroids_to_eval) {
                    LOG(INFO) << "this centroid is not being searched.";
                };
            };
        }
    }
    /**
     * Get passage ids
     */
    std::unordered_set<idx_t> global_pids;

#pragma omp parallel
    {
        std::vector<idx_t> local_pids;
        #pragma omp for nowait
        for (size_t i = 0; i < num_centroids_to_eval; i++) {
            auto idx = centroid_scores[i].second;
            if (idx == -1) {
                continue;
            }
            local_pids = lookup_pids(tenant, idx);
            VLOG(100) << "number of local pids: " << local_pids.size();
            #pragma omp critical
            {
                global_pids.insert(local_pids.begin(), local_pids.end());
            }
        }
        
    } // end parallel

    if (global_pids.size() == 0) {
        return std::vector<SearchResult>();
    }

    auto pid_list = std::vector<idx_t>(global_pids.begin(), global_pids.end());

    gsl::span<const float> query_span = gsl::span(data, n);
    RetrieverOptions plaid_options = RetrieverOptions{
        .num_second_pass = opts.num_second_pass,
        .total_centroids_to_calculate = nlist,
        .expected_id = opts.expected_id
    };

    auto results = this->retriever->retrieve(
        tenant,
        pid_list,
        reordered_distances,
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
    Configuration our_config = Configuration{
        nlist, nbits, niter, dim, use_compression};
    Configuration incoming_config = read_metadata(path);

    LINTDB_THROW_IF_NOT(our_config == incoming_config);

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

std::vector<idx_t> IndexIVF::lookup_pids(const uint64_t tenant, const idx_t idx) const {
    Key start_key{tenant, idx, 0, true};
    // instead of using the max key value, we use the next centroid idx so that we include all
    // document ids.
    Key end_key{tenant, idx+1, 0, true};
    const std::string start_string = start_key.serialize();
    const std::string end_string = end_key.serialize();

    // our iterator abstraction was not working. The upper bound key was
    // being destroyed during iteration, which causes rocksdb to throw
    // an error. for now, we use a raw rocksdb iteration.
    auto options = rocksdb::ReadOptions();
    rocksdb::Slice end_slice(end_string);
    options.iterate_upper_bound = &end_slice;
    auto it = std::unique_ptr<rocksdb::Iterator>(db->NewIterator(
            options, column_families[kIndexColumnIndex]));

    std::vector<idx_t> local_pids;
    rocksdb::Slice prefix(start_string);
    it->Seek(prefix);
    for (; it->Valid(); it->Next()) {
        auto k = it->key().ToString();
        auto key = Key::from_slice(k);
        
        local_pids.push_back(key.id);
    }

    return local_pids;
}

void IndexIVF::write_metadata() {
    std::string out_path = path + "/" + METADATA_FILENAME;
    std::ofstream out(out_path);

    Json::Value metadata;
    // the below castings to Json types enables this to build on M1 Mac.
    metadata["nlist"] = Json::UInt64(nlist);
    metadata["nbits"] = Json::UInt64(nbits);
    metadata["dim"] = Json::UInt64(dim);
    metadata["niter"] = Json::UInt64(niter);
    metadata["use_compression"] = use_compression;

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
    config.use_compression = metadata["use_compression"].asBool();

    return config;
}
} // namespace lintdb