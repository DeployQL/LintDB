
#include "lintdb/index.h"
#include <faiss/impl/FaissException.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/utils.h>
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
#include <stdio.h>
#include <json/writer.h>
#include <json/json.h>
#include <json/reader.h>

namespace lintdb {
IndexIVF::IndexIVF(std::string path) {
    faiss::Index* quantizer;
    faiss::Index* binarizer;

    LOG(INFO) << "loading LintDB from path: " << path;

    // set all of our individual attributes.
    this->read_metadata(path);
    auto config = EncoderConfig{
        nlist, nbits, niter, dim
    };
    this->encoder = DefaultEncoder::load(path, config);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    rocksdb::Status s = rocksdb::DB::Open(
            options, path, cfs, &(this->column_families), &(this->db));
    assert(s.ok());

    this->index_ = new RocksDBInvertedList(*this->db, this->column_families);
}

IndexIVF::IndexIVF(std::string path, size_t dim, Configuration& config)
        : nlist(config.nlist), nbits(config.nbits), path(path) {
    IndexIVF(
        path,
        config.nlist,
        dim,
        config.nbits,
        config.niter,
        config.use_ivf
    );
}

IndexIVF::IndexIVF(
        std::string path,
        size_t nlist,
        size_t dim,
        size_t binarize_nbits,
        size_t niter,
        bool use_ivf
        ) : path(path), nlist(nlist), dim(dim), nbits(binarize_nbits), niter(niter), use_ivf(use_ivf) {
    // for now, we can only handle 32 bit coarse codes.
    LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

    this->encoder = std::make_unique<DefaultEncoder>(
        path, nlist, nbits, niter, dim
    );

    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    rocksdb::Status s = rocksdb::DB::Open(
            options, path, cfs, &(this->column_families), &(this->db));
    assert(s.ok());

    this->index_ = new RocksDBInvertedList(*this->db, this->column_families);
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

std::vector<SearchResult> IndexIVF::search(
        EmbeddingBlock& block,
        size_t n_probe,
        size_t k,
        SearchOptions opts) const {
    return search(block.embeddings.data(), block.num_tokens, block.dimensions, n_probe, k);
}

/**
 * Implementation note: 
 * 
 * when we look at what IVF lists to search, we have several parameters that will influence this.
 * 1. k_top_centroids: responsible for how many centroids per token we include in our search before sorting.
 * 2. n_probe: the number of lists we search on after sorting.
*/
std::vector<SearchResult> IndexIVF::search(
    float* data,
    int n,
    int dim,
    size_t n_probe,
    size_t k,
    SearchOptions opts) const {
    if (this->use_ivf) {
        LINTDB_THROW_IF_NOT(this->encoder->is_trained);
    }


    // per block, run a matrix multiplication and find the nearest centroid.
    // block: (num_tokens x dimensions)
    // centroids: (nlist x dimensions)
    // result: (num_tokens x nlist)
    const float centroid_score_threshold = 0.45;
    const size_t total_centroids_to_calculate = nlist;
    const size_t k_top_centroids = 2;
    const size_t num_second_pass = 1024;

    std::vector<idx_t> coarse_idx(n*total_centroids_to_calculate);
    std::vector<float> distances(n*total_centroids_to_calculate);
    encoder->search(
        data,
        n,
        coarse_idx,
        distances,
        total_centroids_to_calculate,
        centroid_score_threshold
    );
    // get the centroids closest to each token.
    // std::vector<idx_t> coarse_idx(n*k_top_centroids);
    // std::vector<float> distances(n*k_top_centroids);
    std::vector<std::pair<float, idx_t>> centroid_scores;
    for(size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k_top_centroids; j++) {
            centroid_scores.push_back(
                std::pair<float, idx_t>(
                    // the index here is that we've calculated
                    // the distance across a total number of centroids,
                    // but we only want to take the k_top_centroids.
                    distances[i*total_centroids_to_calculate+j], 
                    coarse_idx[i*total_centroids_to_calculate+j]
                )
            );
        }
    }
    std::sort(
            centroid_scores.begin(),
            centroid_scores.end(),
            std::greater<std::pair<float, idx_t>>()
    );
    //define lambda to get unique pairs
    auto unique_pair = [](std::pair<float, idx_t> p1, std::pair<float, idx_t> p2) {
        return p1.second == p2.second;
    };
    centroid_scores.erase(
            std::unique(centroid_scores.begin(), centroid_scores.end(), unique_pair),
            centroid_scores.end()
    );
    auto num_centroids_to_eval = std::min<size_t>(n_probe, centroid_scores.size());

    if (opts.expected_id != -1) {
        auto mapping = index_->get_mapping(opts.expected_id);
        std::unordered_set<idx_t> mapping_set(mapping.begin(), mapping.end());
        for(int i = 0; i < centroid_scores.size(); i++) {
            if (mapping_set.find(centroid_scores[i].second) != mapping_set.end()) {
                LOG(INFO) << "expected id found in centroid: " << i;
                if (i > num_centroids_to_eval) {
                    LOG(INFO) << "expected id has been dropped from search.";
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
            VLOG(10) << "Including centroid: " << idx << " in search.";
            if (idx == -1) {
                continue;
            }
            Key start_key{kDefaultTenant, idx, 0, true};
            // instead of using the max key value, we use the next centroid idx so that we include all
            // document ids.
            Key end_key{kDefaultTenant, idx+1, 0, true};
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
            rocksdb::Slice prefix(start_string);
            it->Seek(prefix);
            for (; it->Valid(); it->Next()) {
                auto k = it->key().ToString();
                auto key = Key::from_slice(k);
                VLOG(100) << "in centroid: " << key.inverted_list_id << " with id: " << key.id;
                if (opts.expected_id != -1 && key.id == opts.expected_id) {
                    LOG(INFO) << "found expected id in centroid: " << key.inverted_list_id;
                }
                local_pids.push_back(key.id);
            }
        }
#pragma omp critical
        global_pids.insert(local_pids.begin(), local_pids.end());
    } // end parallel

    VLOG(100) << "found total num pids: " << global_pids.size();
    if (global_pids.size() == 0) {
        return std::vector<SearchResult>();
    }

    /**
     * score by passage codes
     */
    std::vector<idx_t> pid_list(global_pids.begin(), global_pids.end());
    auto doc_codes = index_->get_codes(pid_list);
    std::vector<float> max_scores_per_centroid = max_score_by_centroid(
            coarse_idx, distances, total_centroids_to_calculate, n, nlist);
    // create a mapping from pid to the index. we'll need this to hydrate
    // residuals.
    std::unordered_map<idx_t, size_t> pid_to_index;
    for (size_t i = 0; i < pid_list.size(); i++) {
        pid_to_index[doc_codes[i]->id] = i;
    }

    std::vector<std::pair<float, idx_t>> pid_scores(pid_list.size());

    #pragma omp for
    for (int i = 0; i < pid_list.size(); i++) {
        auto codes = doc_codes[i]->codes;

        float score = score_documents_by_codes(
                max_scores_per_centroid,
                codes, // we'll have num_token codes
                centroid_score_threshold);
        pid_scores[i] = std::pair<float, idx_t>(score, doc_codes[i]->id);
    }


    VLOG(100) << "got pid scores: " << pid_scores.size();
    assert(pid_scores.size() == pid_list.size());
    // according to the paper, we take the top 25%.
    std::sort(
            pid_scores.begin(),
            pid_scores.end(),
            std::greater<std::pair<float, idx_t>>());

    // colBERT has a ndocs param which limits the number of documents to score.
    size_t cutoff = pid_scores.size();
    if (num_second_pass != 0 ) {
        cutoff = num_second_pass;
    }
    auto num_rerank = std::max(size_t(1), cutoff / 4);
    num_rerank = std::min(num_rerank, pid_scores.size());

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                pid_scores.begin(),
                pid_scores.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != pid_scores.end()) {
            auto pos = it - pid_scores.begin();
            LOG(INFO) << "found expected id in pid code scores at position: " << pos;
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank << ". expected id has been dropped";
            }
        }
    }

    VLOG(10) << "num to rerank: " << num_rerank;
    std::vector<std::pair<float, idx_t>> top_25_scores(
            pid_scores.begin(), pid_scores.begin() + num_rerank);

    /**
     * score by passage residuals
     */
    std::vector<idx_t> top_25_ids;
    std::transform(
            top_25_scores.begin(),
            top_25_scores.end(),
            std::back_inserter(top_25_ids),
            [](std::pair<float, idx_t> p) { return p.second; });
    auto doc_residuals = index_->get_residuals(top_25_ids);

    std::vector<std::pair<float, idx_t>> actual_scores(top_25_ids.size());
#pragma omp for
    for (int i = 0; i < top_25_ids.size(); i++) {
        auto residuals = doc_residuals[i]->residuals;

        auto codes = doc_codes[pid_to_index[doc_residuals[i]->id]]->codes;
        std::vector<float> decompressed = encoder->decode_vectors(
            gsl::span<code_t>(codes),
            gsl::span<residual_t>(residuals),
            doc_residuals[i]->num_tokens,
            dim);

        const auto data_span = gsl::span(data, n * dim);
        float score = score_document_by_residuals(
            data_span,
            n,
            decompressed.data(),
            doc_residuals[i]->num_tokens,
            dim);

        actual_scores[i] = std::pair<float, idx_t>(score, top_25_ids[i]);
    }
    // according to the paper, we take the top 25%.
    std::sort(
        actual_scores.begin(),
        actual_scores.end(),
        std::greater<std::pair<float, idx_t>>()
    );

    if (opts.expected_id != -1) {
        auto it = std::find_if(
                pid_scores.begin(),
                pid_scores.end(),
                [opts](std::pair<float, idx_t> p) {
                    return p.second == opts.expected_id;
                });
        if (it != pid_scores.end()) {
            auto pos = it - pid_scores.begin();
            LOG(INFO) << "expected id found in residual scores: " << pos;
            if (pos > num_rerank) {
                LOG(INFO) << "top 25 cutoff: " << num_rerank << ". expected id has been dropped";
            }
        }
    }

    size_t num_to_return = std::min<size_t>(actual_scores.size(), k);
    std::vector<std::pair<float, idx_t>> top_k_scores(
            actual_scores.begin(), actual_scores.begin() + num_to_return);

    std::vector<SearchResult> results;
    std::transform(
            top_k_scores.begin(),
            top_k_scores.end(),
            std::back_inserter(results),
            [](std::pair<float, idx_t> p) {
                return SearchResult{p.second, p.first};
            });

    return results;
}

void IndexIVF::set_centroids(float* data, int n, int dim) {
    encoder->set_centroids(data, n, dim);
}

void IndexIVF::add(const std::vector<RawPassage>& docs) {
    LINTDB_THROW_IF_NOT(this->encoder->is_trained);

    for (auto doc : docs) {
        add_single(doc);
    }
}

void IndexIVF::add_single(const RawPassage& doc) {
    auto encoded = encoder->encode_vectors(doc);
    int i = 0;

    index_->add(std::move(encoded));
}

void IndexIVF::remove(const std::vector<idx_t>& ids) {
    index_->remove(ids);
}

void IndexIVF::update(const std::vector<RawPassage>& docs) {
    std::vector<idx_t> ids;
    for (auto doc : docs) {
        ids.push_back(doc.id);
    }
    remove(ids);
    add(docs);
}

void IndexIVF::write_metadata() {
    std::string out_path = path + "/" + METADATA_FILENAME;
    std::ofstream out(out_path);

    Json::Value metadata;
    metadata["nlist"] = nlist;
    metadata["nbits"] = nbits;
    metadata["dim"] = dim;
    metadata["niter"] = niter;
    metadata["use_ivf"] = use_ivf;

    Json::StyledWriter writer;
    out << writer.write(metadata);
    out.close();
}

void IndexIVF::read_metadata(std::string path) {
    this->path = path;
    std::string in_path = path + "/" + METADATA_FILENAME;
    std::ifstream in(in_path);
    if (!in) {
        throw LintDBException("Could not read metadata from path: " + in_path);
    }

    Json::Reader reader;
    Json::Value metadata;
    reader.parse(in, metadata);

    nlist = metadata["nlist"].asUInt();
    nbits = metadata["nbits"].asUInt();
    dim = metadata["dim"].asUInt();
    niter = metadata["niter"].asUInt();
    use_ivf = metadata["use_ivf"].asBool();
}
} // namespace lintdb