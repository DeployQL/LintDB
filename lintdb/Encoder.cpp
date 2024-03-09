#include "lintdb/Encoder.h"
#include "lintdb/exception.h"
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <glog/logging.h>
#include <faiss/Clustering.h>
#include "lintdb/assert.h"
#include "lintdb/util.h"
#include <cassert>
#include <cblas.h>

namespace lintdb {
    DefaultEncoder::DefaultEncoder(std::string path, size_t nlist, size_t nbits, size_t niter, size_t dim)
        : Encoder(), path(path), nlist(nlist), nbits(nbits), niter(niter), dim(dim) {
        
        // colBERT uses L2 during clustering.
        // for normalized vectors, this should be the same as IP.
        this->quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
    }

    std::unique_ptr<EncodedDocument> DefaultEncoder::encode_vectors(
            const RawPassage& doc) const {
        LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());
        auto num_tokens = doc.embedding_block.num_tokens;

        const float* data_ptr = doc.embedding_block.embeddings.data();

        // get the centroids closest to each token.
        std::vector<idx_t> coarse_idx(num_tokens, 0);
        std::vector<float> distances(num_tokens, 0);
        // quantizer->assign(num_tokens, data_ptr, coarse_idx.data());
        quantizer->search(
                num_tokens,
                data_ptr,
                1,
                distances.data(),
                coarse_idx.data());

        assert(coarse_idx.size() == num_tokens);

        // compute residual
        std::vector<float> raw_residuals(num_tokens * dim);
        for (size_t i = 0; i < num_tokens; i++) {
            quantizer->compute_residual(
                    data_ptr + i * dim,
                    raw_residuals.data() + i * dim,
                    coarse_idx[i]);
        }
        std::vector<code_t> token_coarse_idx;
        std::transform(
                coarse_idx.begin(),
                coarse_idx.end(),
                std::back_inserter(token_coarse_idx),
                [](idx_t idx) { return static_cast<code_t>(idx); });

        // LOG(INFO) << "ntotal " << quantizer->ntotal;
        // for(int i=0; i<token_coarse_idx.size();i++){
        //     LOG(INFO) << "Token " << i << " coarse_idx: " << coarse_idx[i] << " distance: "  << distances[i]; 
        //     // LINTDB_THROW_IF_NOT(coarse_idx[i] != -1);
        //     if (coarse_idx[i] == -1) {
        //         LOG(ERROR) << "coarse_idx is -1";
        //     }
        // }

        std::vector<residual_t> residual_codes(raw_residuals.begin(), raw_residuals.end());

        return std::make_unique<EncodedDocument>(EncodedDocument(
                token_coarse_idx, residual_codes, num_tokens, doc.id, doc.doc_id, doc.text));
    }

    std::vector<float> DefaultEncoder::decode_vectors(
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

            // add the residual to the decoded embedding.
            for (size_t j = 0; j < dim; j++) {
                decoded_embeddings[i * dim + j] += residuals[j];
            }
        }
        return decoded_embeddings;
    }

    void DefaultEncoder::search(
            float* data,
            int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            size_t k_top_centroids,
            float centroid_threshold
    ) {
        // we get back the k top centroid matches per token.
        quantizer->search(
                n,
                data,
                k_top_centroids,
                distances.data(),
                coarse_idx.data());
    }

    void DefaultEncoder::save(std::string path) {
        auto quantizer_path = path + "/"+ QUANTIZER_FILENAME;
        faiss::write_index(quantizer.get(), quantizer_path.c_str());
    }

    std::unique_ptr<Encoder> DefaultEncoder::load(std::string path, EncoderConfig& config) {
        std::unique_ptr<faiss::Index> quantizer;

         if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            quantizer = std::unique_ptr<faiss::Index>(faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str()));
        } else {
            throw LintDBException("Index not found at path: " + path);
        }

        auto encoder = std::make_unique<DefaultEncoder>(
                DefaultEncoder(path, config.nlist, config.nbits, config.niter, config.dim));
        encoder->quantizer = std::move(quantizer);
        encoder->is_trained = true;
        return std::move(encoder);
    }

    void DefaultEncoder::train(float* embeddings, size_t n, size_t dim) {
        try {
            faiss::ClusteringParameters cp;
            cp.niter = this->niter;
            cp.nredo = 1;
            cp.seed = 123;
            faiss::Clustering clus(dim, nlist, cp);
            clus.verbose = true;    

            // clustering uses L2 distance.
            faiss::IndexFlatL2 assigner(dim);
            clus.train(n, embeddings, assigner);

            normalize_vector(clus.centroids.data(), nlist, dim);

            quantizer->add(nlist, clus.centroids.data());

            this->is_trained = true;
        } catch (const faiss::FaissException& e) {
            LOG(ERROR) << "Faiss exception: " << e.what();
            throw LintDBException(e.what());
        }
    }

    void DefaultEncoder::set_centroids(float* data, int n, int dim) {
        LINTDB_THROW_IF_NOT(n == nlist);
        LINTDB_THROW_IF_NOT(dim == this->dim);

        quantizer->reset();
        quantizer->add(n, data);

        this->is_trained = true;
    }
}