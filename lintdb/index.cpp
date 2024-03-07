
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

    if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
        fclose(file);
        quantizer = faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str());
    } else {
        throw LintDBException("Index not found at path: " + path);
    }

    if (FILE *file = fopen((path + "/" + BINARIZER_FILENAME).c_str(), "r")) {
        fclose(file);
        binarizer = faiss::read_index((path + "/" + BINARIZER_FILENAME).c_str());
    } else {
        throw LintDBException("Index not found at path: " + path);
    }
    // set all of our individual attributes.
    this->read_metadata(path);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    rocksdb::Status s = rocksdb::DB::Open(
            options, path, cfs, &(this->column_families), &(this->db));
    assert(s.ok());

    this->invlists = new RocksDBInvertedList(*this->db, this->column_families);
    this->quantizer = quantizer;
    VLOG(100) << "Quantizer loaded with " << quantizer->ntotal << " centroids.";
    this->binarizer = binarizer;


    this->is_trained = true;
}

IndexIVF::IndexIVF(std::string path, size_t dim, Configuration& config)
        : nlist(config.nlist), nbits(config.nbits), path(path) {
    this->is_trained = config.use_ivf;
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

    this->quantizer = new faiss::IndexFlatIP(dim);
    this->binarizer = new faiss::IndexLSH(dim, binarize_nbits);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    rocksdb::Status s = rocksdb::DB::Open(
            options, path, cfs, &(this->column_families), &(this->db));
    assert(s.ok());

    this->invlists = new RocksDBInvertedList(*this->db, this->column_families);
}

IndexIVF::IndexIVF(
        std::string path,
        faiss::Index* quantizer,
        faiss::Index* binarizer) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    rocksdb::Status s = rocksdb::DB::Open(
            options, path, cfs, &(this->column_families), &(this->db));
    assert(s.ok());

    this->invlists = new RocksDBInvertedList(*this->db, this->column_families);
    this->quantizer = quantizer;
    this->binarizer = binarizer;

    // TODO (mbarta): It would be great to not have to cast here.
    auto lsh = dynamic_cast<faiss::IndexLSH*>(this->binarizer);
    this->nbits = lsh->nbits;

    this->dim = this->quantizer->d;
    this->nlist = this->quantizer->ntotal;
    this->is_trained = true;
}

void IndexIVF::train(size_t n, std::vector<float>& embeddings) {
    this->train(embeddings.data(), n, dim);
}

void IndexIVF::train(float* embeddings, size_t n, size_t dim) {
    // TODO(mbarta): this can follow faiss' example and be stored inside its own
    // index.
    try {
        faiss::ClusteringParameters cp;
        cp.niter = this->niter;
        cp.nredo = 1;
        cp.seed = 1234;
        faiss::Clustering clus(dim, nlist, cp);

        LOG(INFO) << "Training index with " << n << " embeddings.";
        LOG(INFO) << "niter: " << cp.niter;
        LOG(INFO) << "nredo: " << cp.nredo;
        LOG(INFO) << "seed: " << cp.seed;

        faiss::IndexFlatIP assigner(dim);
        clus.train(n, embeddings, assigner);
        quantizer->add(nlist, clus.centroids.data());
        LOG(INFO) << "Quantizer has " << quantizer->ntotal << " centroids.";

        LOG(INFO) << "Training binarizer with " << n << " embeddings.";
        // train binarizer on residuals.
        std::vector<idx_t> assign(n);
        quantizer->assign(n, embeddings, assign.data());

        std::vector<float> residuals(n * dim);
        quantizer->compute_residual_n(
                n, embeddings, residuals.data(), assign.data());

        binarizer->train(n, residuals.data());

        this->save();
        LOG(INFO) << "Training complete.";
        this->is_trained = true;
    } catch (const faiss::FaissException& e) {
        LOG(ERROR) << "Faiss exception: " << e.what();
        throw LintDBException(e.what());
    }
}

void IndexIVF::train(float* embeddings, int n, int dim) {
    train(embeddings, static_cast<size_t>(n), static_cast<size_t>(dim));
}

void IndexIVF::save() {
    auto quantizer_path = path + "/"+ QUANTIZER_FILENAME;
    faiss::write_index(quantizer, quantizer_path.c_str());

    auto binarizer_path = path + "/" + BINARIZER_FILENAME;
    faiss::write_index(binarizer, binarizer_path.c_str());

    this->write_metadata();
}

std::vector<SearchResult> IndexIVF::search(
        EmbeddingBlock& block,
        size_t n_probe,
        size_t k) const {
    return search(block.embeddings.data(), block.num_tokens, block.dimensions, n_probe, k);
}

std::vector<SearchResult> IndexIVF::search(
    float* data,
    int n,
    int dim,
    size_t n_probe,
    size_t k) const {
    if (this->use_ivf) {
        LINTDB_THROW_IF_NOT(this->is_trained);
    }


    // per block, run a matrix multiplication and find the nearest centroid.
    // block: (num_tokens x dimensions)
    // centroids: (nlist x dimensions)
    // result: (num_tokens x nlist)
    const float centroid_score_threshold = 0.5;
    const size_t num_tokens = n;
    const size_t k_top_centroids = 1; // this can dictate whether we pass our tests or not!

    // get the centroids closest to each token.
    std::vector<idx_t> coarse_idx(n*k_top_centroids);
    std::vector<float> distances(n*k_top_centroids);
    // we get back the k top centroid matches per token.
    quantizer->search(
            n,
            data,
            k_top_centroids, // it will already give us the highest matching centroids per word.
            distances.data(),
            coarse_idx.data());

    std::vector<std::pair<float, idx_t>> centroid_scores;
    for(size_t i = 0; i < num_tokens; i++) {
        for (size_t j = 0; j < k_top_centroids; j++) {
            centroid_scores.push_back(
                std::pair<float, idx_t>(
                    distances[i*k_top_centroids+j], 
                    coarse_idx[i*k_top_centroids+j]
                )
            );
        }
    }
    std::sort(
            centroid_scores.begin(),
            centroid_scores.end(),
            std::greater<std::pair<float, idx_t>>()
    );
    
    /**
     * Get passage ids
     */
    std::unordered_set<idx_t> global_pids;

#pragma omp parallel
    {
        std::vector<idx_t> local_pids;
#pragma omp for nowait
        for (size_t i = 0; i < n_probe; i++) {
            auto idx = centroid_scores[i].second;
            VLOG(10) << "Including centroid: " << idx << " in search.";
            if (idx == -1) {
                continue;
            }
            Key start_key{kDefaultTenant, idx, 0, true};
            Key end_key{kDefaultTenant, idx, std::numeric_limits<idx_t>::max()};
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

                VLOG(100) << "found document: " << key.id << "in list: " << idx;
                local_pids.push_back(key.id);
            }
        }
#pragma omp critical
        global_pids.insert(local_pids.begin(), local_pids.end());
    } // end parallel

    VLOG(100) << "got global pids: " << global_pids.size();
    if (global_pids.size() == 0) {
        return std::vector<SearchResult>();
    }

    /**
     * score by passage codes
     */
    std::vector<idx_t> pid_list(global_pids.begin(), global_pids.end());
    auto doc_codes = invlists->get_codes(pid_list);
    std::vector<float> max_scores_per_centroid = max_score_by_centroid(
            coarse_idx, distances, k_top_centroids, n, nlist);
    std::vector<std::pair<float, idx_t>> pid_scores(pid_list.size());
    // create a mapping from pid to the index. we'll need this to hydrate
    // residuals.
    std::unordered_map<idx_t, size_t> pid_to_index;
    for (size_t i = 0; i < pid_list.size(); i++) {
        pid_to_index[doc_codes[i]->id] = i;
    }


    #pragma omp for
    for (int i = 0; i < pid_list.size(); i++) {
        auto codes = doc_codes[i]->codes;

        float score = score_documents_by_codes(
                max_scores_per_centroid,
                codes, // we'll have num_token codes
                centroid_score_threshold);
        // VLOG(100) << "approximate score for pid: " << pid_list[i]
        //           << " is: " << score;
        pid_scores[i] = std::pair<float, idx_t>(score, pid_list[i]);
    }


    VLOG(100) << "got pid scores: " << pid_scores.size();
    assert(pid_scores.size() == pid_list.size());
    // according to the paper, we take the top 25%.
    std::sort(
            pid_scores.begin(),
            pid_scores.end(),
            std::greater<std::pair<float, idx_t>>());
    auto top_25 = std::max(size_t(1), pid_scores.size() / 4);

    VLOG(100) << "top 25%: " << top_25;
    std::vector<std::pair<float, idx_t>> top_25_scores(
            pid_scores.begin(), pid_scores.begin() + top_25);

    /**
     * score by passage residuals
     */
    std::vector<idx_t> top_25_ids;
    std::transform(
            top_25_scores.begin(),
            top_25_scores.end(),
            std::back_inserter(top_25_ids),
            [](std::pair<float, idx_t> p) { return p.second; });
    auto doc_residuals = invlists->get_residuals(top_25_ids);

    std::vector<std::pair<float, idx_t>> actual_scores(top_25_ids.size());
#pragma omp for
    for (int i = 0; i < top_25_ids.size(); i++) {
        auto residuals = doc_residuals[i]->residuals;
        VLOG(100) << "got residuals: " << residuals.size();

        auto codes = doc_codes[pid_to_index[doc_residuals[i]->id]]->codes;
        std::vector<float> decompressed = decode_vectors(
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
        VLOG(100) << "actual score for pid: " << top_25_ids[i]
                  << " is: " << score;
        actual_scores[i] = std::pair<float, idx_t>(score, top_25_ids[i]);
    }
    VLOG(100) << "got actual scores: " << actual_scores.size();
    // according to the paper, we take the top 25%.
    std::sort(
        actual_scores.begin(),
        actual_scores.end(),
        std::greater<std::pair<float, idx_t>>()
    );

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

void IndexIVF::add(const std::vector<RawPassage>& docs) {
    LINTDB_THROW_IF_NOT(is_trained);

    for (auto doc : docs) {
        add_single(doc);
    }
}

void IndexIVF::add_single(const RawPassage& doc) {
    auto encoded = encode_vectors(doc);
    invlists->add(std::move(encoded));
}

void IndexIVF::remove(const std::vector<idx_t>& ids) {
    invlists->remove(ids);
}

void IndexIVF::update(const std::vector<RawPassage>& docs) {
    std::vector<idx_t> ids;
    for (auto doc : docs) {
        ids.push_back(doc.id);
    }
    remove(ids);
    add(docs);
}

// we need to encode the vectors into their code representations.
// coarse_idx is the inverse list we should assign to.
// NOTE: EmbeddingBlocks are column major and the quantizer expects row major.
std::unique_ptr<EncodedDocument> IndexIVF::encode_vectors(
        const RawPassage& doc) const {
    LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());
    auto num_tokens = doc.embedding_block.num_tokens;

    const float* data_ptr = doc.embedding_block.embeddings.data();

    // get the centroids closest to each token.
    std::vector<idx_t> coarse_idx(num_tokens);
    quantizer->assign(num_tokens, data_ptr, coarse_idx.data());
    assert(coarse_idx.size() == num_tokens);

    // compute residual
    std::vector<float> raw_residuals(num_tokens * dim);
    std::vector<residual_t> residual_codes(num_tokens * nbits);
    for (size_t i = 0; i < num_tokens; i++) {
        quantizer->compute_residual(
                data_ptr + i * dim,
                raw_residuals.data() + i * dim,
                coarse_idx[i]);
        binarizer->sa_encode(
                1,
                raw_residuals.data() + i * dim,
                residual_codes.data() + i * nbits);
    }
    std::vector<code_t> token_coarse_idx;
    std::transform(
            coarse_idx.begin(),
            coarse_idx.end(),
            std::back_inserter(token_coarse_idx),
            [](idx_t idx) { return static_cast<code_t>(idx); });
    
    return std::make_unique<EncodedDocument>(EncodedDocument(
            token_coarse_idx, residual_codes, num_tokens, doc.id, doc.doc_id));
}

std::vector<float> IndexIVF::decode_vectors(
        gsl::span<const code_t> codes,
        gsl::span<const residual_t> residuals,
        size_t num_tokens,
        size_t dim) const {
    std::vector<float> decoded_embeddings(dim * num_tokens);
    for (size_t i = 0; i < num_tokens; i++) {
        auto centroid_id = codes[i];

        // add the centroid to the decoded embedding.
        quantizer->reconstruct(
                centroid_id, decoded_embeddings.data() + i * dim);

        // decode the residual code from the binary code.
        std::vector<float> decoded_residuals(dim);
        binarizer->sa_decode(
                1,
                residuals.data() + i * binarizer->sa_code_size(),
                decoded_residuals.data());

        // add the residual to the decoded embedding.
        for (size_t j = 0; j < dim; j++) {
            decoded_embeddings[i * dim + j] += decoded_residuals[j];
        }
    }
    return decoded_embeddings;
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