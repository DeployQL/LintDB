
#include "lintdb/index.h"
#include "lintdb/assert.h"
#include <omp.h>
#include "lintdb/schema/util.h"
#include <vector>
#include <iostream>
#include "lintdb/invlists/RocksdbList.h"
#include <string>
#include "lintdb/plaid.h"
#include <fstream>
#include <faiss/index_io.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/utils.h>
#include <filesystem>
#include <glog/logging.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/slice_transform.h>
#include "lintdb/cf.h"
#include <faiss/utils/Heap.h>

namespace lintdb {
    IndexIVF::IndexIVF(
        std::string path,
        size_t nlist,
        size_t dim,
        size_t code_nbits,
        size_t binarize_nbits
        ):  path(path),
            nlist(nlist),
            quantizer(dim, 1, code_nbits),
            binarizer(dim, binarize_nbits),
            dim(dim){
            
            // for now, we can only handle 32 bit coarse codes.
            LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

            rocksdb::Options options;
            options.create_if_missing = true;
            options.create_missing_column_families = true;

            auto cfs = create_column_families();
            rocksdb::Status s = rocksdb::DB::Open(options, path, cfs, &(this->column_families), &(this->db));
            assert(s.ok());
            
            invlists = std::make_unique<RocksDBInvertedList>(RocksDBInvertedList(*this->db, this->column_families));
    }

    void IndexIVF::train(size_t n, std::vector<float>& embeddings) {
        // TODO(mabrta): this can follow faiss' example and be stored inside its own index.
        faiss::ClusteringParameters cp;
        faiss::Clustering clus(dim, nlist, cp);

        LOG(INFO) << "Training index with " << n << " embeddings.";

        //train coarse quantizer using pq as the index.
        clus.train(n, embeddings.data(), quantizer);

        LOG(INFO) << "Training binarizer with " << n << " embeddings.";
        //train binarizer on residuals.
        std::vector<idx_t> assign(n);
        quantizer.assign(n, embeddings.data(), assign.data());

        std::vector<float> residuals(n * dim);
        quantizer.compute_residual_n(n, embeddings.data(), residuals.data(), assign.data());

        binarizer.train(n, residuals.data());

        // auto col_major = row_to_column_major(quantizer.pq.centroids.data(), quantizer.pq.centroids.size(), nlist, dim);
        // the centroids are in row-major format on disk. The transpose of this data is the same as column wise.
        // i.e. we have (nlist x dim) centroids that will get stored as (dim x nlist) on disk.
        centroids = af::transpose(af::array(dim, nlist, quantizer.pq.centroids.data()));

        LOG(INFO) << "Training complete.";
    }

    void IndexIVF::save() {
        std::filesystem::path path(this->path);
        path.append(CENTROIDS_FILENAME);
        af::saveArray("centroids", centroids, path.c_str());
        auto quantizer_path = path.parent_path().append(QUANTIZER_FILENAME);
        faiss::write_index(&quantizer, quantizer_path.c_str());
        auto binarizer_path = path.parent_path().append(BINARIZER_FILENAME);
        faiss::write_index(&binarizer, binarizer_path.c_str());
    }

    // instead of what we do here, we need to call quantizer.search to get get the centroid indx_t's.
    // once we have that, faiss uses search_preassigned to do the search. preassigned seems in reference to
    // the pre-quantization, but i'm unsure still.
    // either case, we want to search each retrieved list for the best matches.
    SearchResults IndexIVF::search(EmbeddingBlock& block, size_t n_probe) const {
        std::vector<EmbeddingBlock> blocks = { block };
        
        for (auto block: blocks) {
            // per block, run a matrix multiplication and find the nearest centroid.
            // block: (num_tokens x dimensions)
            // centroids: (nlist x dimensions)
            // result: (num_tokens x nlist)
            auto centroid_t = af::transpose(centroids);
            af::array ntok_nlist = af::matmul(block, centroid_t);
            // this should then become (1 x nlist)
            auto nlist_maxes = af::max(ntok_nlist, 0 /*dimension to aggregate on*/);

            af::array sequential_indices = af::seq(nlist);
            af::array out_indices = af::array(nlist);
            af::array out_values = af::array(nlist);

            // we get the top k indices to search on.
            af::topk(out_values, out_indices, af::moddims(nlist_maxes, nlist), n_probe, 0);

            std::vector<idx_t> coarse_idx(n_probe);
            std::vector<float> coarse_dis(n_probe);
            for(size_t i = 0; i < n_probe; i++) {
                auto idx_float = out_indices(i, 0).scalar<float>();
                idx_t idx = static_cast<idx_t>(idx_float);
                coarse_idx[i] = idx;
                coarse_dis[i] = out_values(i, 0).scalar<float>();
            }

            // now, retrieve lists from the inverted index, run max sim against them,
            // and return the top n results.
#pragma omp parallel for
            for(auto idx : coarse_idx) {
                faiss::float_maxheap_array_t heap;
                auto results = search_single_list(idx, block, nlist_maxes);
            }
        }

        return SearchResults();
    }

    SearchResults IndexIVF::search_single_list(size_t idx, EmbeddingBlock& block, af::array& nlist_maxes) const {
        auto it = invlists->get_iterator(idx);
        while (it->has_next()) {
            it->next();
            auto doc = it->get();
            // PLAID filters documents before decompressing the codes.
            // it compares the codes (which are integers pointing to centroids)
            // to the query's score against those centroids. 
            float approx_score = score_documents_by_codes(
                nlist_maxes,
                doc.codes.data(),
                doc.num_tokens
            );

            // we need to decode the document and run a max sim against it.
            // we can use the quantizer to decode the document.
            auto decoded_docs = decode_vectors(doc);

            //
            auto final_approx_score = af::matmulNT(block, decoded_docs);

        }
    }   

    void IndexIVF::add(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> ids) {

        auto one = af::randu(10, 20);
        auto two = af::randu(20, 10);
        auto result = af::matmul(one, two);

        std::vector<idx_t> coarse_idx;
        for (auto block: blocks) {
            // per block, run a matrix multiplication and find the nearest centroid.
            // block: (num_tokens x dimensions)
            // centroids: (nlist x dimensions)
            // result: (num_tokens x nlist)

            auto query_transpose = af::transpose(centroids);
            af::array ntok_nlist = af::matmul(block, query_transpose);
            
            // still floats
            auto nlist_maxes = af::max(ntok_nlist, 0 /*dimension to aggregate on*/);

            af::array sequential_indices = af::seq(nlist);
            af::array out_indices(nlist);
            af::sort(out_indices, sequential_indices, nlist_maxes, 0, false);

            auto first_index = out_indices(0,0).scalar<float>();
            int index = static_cast<idx_t>(first_index);

            VLOG(1) << "Adding to list " << index;
            coarse_idx.push_back(index);
            auto doc = encode_vectors(block);
            invlists->add(index, doc);
        }
    }

    // we need to encode the vectors into their code representations.
    // coarse_idx is the inverse list we should assign to.
    // NOTE: EmbeddingBlocks are column major and the quantizer expects row major.
    std::unique_ptr<EncodedDocument> IndexIVF::encode_vectors(EmbeddingBlock& block) const {
        LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());

        auto num_tokens = block.dims(lintdb::TOKEN_DIMENSION);

        // the shape of block after tranpsoing should be dim x num_tokens. While this is backwards,
        // we can read the memory as num_token x dim thanks to the disk layout.
        auto row_major_data = af::transpose(block);

        auto row_major_data_ptr = row_major_data.device<float>();
        // use the quantizer to assign coarse indexes to each vector in the block.
        std::vector<idx_t> coarse_idx(num_tokens);
        quantizer.assign(num_tokens, row_major_data_ptr, coarse_idx.data());
        // compute residual
        std::vector<float> raw_residuals(num_tokens * dim);
        std::vector<residual_t> residual_codes(num_tokens * binarizer.sa_code_size());
        for(size_t i = 0; i < num_tokens; i++) {
            quantizer.compute_residual(
                row_major_data_ptr + i * dim, 
                raw_residuals.data() + i * dim, 
                coarse_idx[i]
            );
            binarizer.sa_encode(
                1,
                raw_residuals.data() + i * dim, 
                residual_codes.data() + i * binarizer.sa_code_size()
            );
        }
        af::freeHost(row_major_data_ptr);

        std::vector<code_t> token_coarse_idx;
        std::transform(coarse_idx.begin(), coarse_idx.end(), std::back_inserter(token_coarse_idx), [](idx_t idx) {
            return static_cast<code_t>(idx);
        });
        return std::make_unique<EncodedDocument>(EncodedDocument(token_coarse_idx, residual_codes, num_tokens, 0, ""));
    }

    // NOTE: EmbeddingBlocks are column major and the quantizer expects row major.
    EmbeddingBlock IndexIVF::decode_vectors(EncodedDocument& doc) const {
        std::vector<float> decoded_embeddings(dim * doc.num_tokens);
        for (size_t i = 0; i < doc.num_tokens; i++) {
            auto centroid_id = doc.codes[i];

            // add the centroid to the decoded embedding.
            quantizer.reconstruct(centroid_id, decoded_embeddings.data() + i*dim);
            
            // decode the residual code from the binary code.
            std::vector<float> decoded_residuals(dim);
            binarizer.sa_decode(1, doc.residuals.data() + i * binarizer.sa_code_size(), decoded_residuals.data());

            // add the residual to the decoded embedding.
            for (size_t j = 0; j < dim; j++) {
                decoded_embeddings[i * dim + j] += decoded_residuals[j];
            }

        }

        auto row_major_data = af::array(dim, doc.num_tokens, decoded_embeddings.data());
        auto decoded_block = af::transpose(row_major_data);
        return decoded_block;
    }
}