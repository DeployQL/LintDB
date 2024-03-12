#include "lintdb/Encoder.h"
#include "lintdb/exception.h"
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <glog/logging.h>
#include <faiss/Clustering.h>
#include <faiss/IndexLSH.h>
#include "lintdb/assert.h"
#include "lintdb/util.h"
#include <cassert>
#include <cblas.h>

namespace lintdb {
    DefaultEncoder::DefaultEncoder(std::string path, size_t nlist, size_t nbits, size_t niter, size_t dim, bool use_compression)
        : Encoder(), path(path), nlist(nlist), nbits(nbits), niter(niter), dim(dim), use_compression(use_compression) {
        
        // colBERT uses L2 during clustering.
        // for normalized vectors, this should be the same as IP.
        this->quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
        if(use_compression) {
            this->binarizer = std::make_unique<faiss::IndexLSH>(dim, nbits);
        }
    }

    std::unique_ptr<EncodedDocument> DefaultEncoder::encode_vectors(
            const RawPassage& doc) const {
        LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());
        auto num_tokens = doc.embedding_block.num_tokens;

        const float* data_ptr = doc.embedding_block.embeddings.data();

        // get the centroids closest to each token.
        std::vector<idx_t> coarse_idx(num_tokens, 0);
        quantizer->assign(num_tokens, data_ptr, coarse_idx.data());

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
        if (use_compression) {
            std::vector<residual_t> residual_codes(num_tokens * nbits);
            binarizer->sa_encode(num_tokens, raw_residuals.data(), residual_codes.data());
            
            return std::make_unique<EncodedDocument>(EncodedDocument(
                token_coarse_idx, residual_codes, num_tokens, doc.id));
        } else {
            // uint8_t* begin = reinterpret_cast<uint8_t*>(raw_residuals.data());
            // auto size = raw_residuals.size() * (sizeof(float)/sizeof(uint8_t));
            // std::vector<residual_t> residual_codes(begin, begin + size);
            std::vector<residual_t> residual_codes(raw_residuals.begin(), raw_residuals.end());

            return std::make_unique<EncodedDocument>(EncodedDocument(
                token_coarse_idx, residual_codes, num_tokens, doc.id));
        }

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

            if (use_compression) {
                std::vector<float> decoded_residuals(dim);
                binarizer->sa_decode(1, residuals.data(), decoded_residuals.data());
                for (size_t j = 0; j < dim; j++) {
                    decoded_embeddings[i * dim + j] += decoded_residuals[j];
                }
            } else {
                // add the residual to the decoded embedding.
                // float* casted = reinterpret_cast<float*>(const_cast<residual_t*>(residuals.data() + i * dim));
                // for (size_t j = 0; j < dim; j++) {
                //     decoded_embeddings[i * dim + j] += casted[j];
                // }
                for (size_t j = 0; j < dim; j++) {
                    decoded_embeddings[i * dim + j] += residuals[i * dim + j];
                }
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

        if (use_compression) {
            auto binarizer_path = path + "/"+ BINARIZER_FILENAME;
            faiss::write_index(binarizer.get(), binarizer_path.c_str());
        }
    }

    std::unique_ptr<Encoder> DefaultEncoder::load(std::string path, EncoderConfig& config) {
        std::unique_ptr<faiss::Index> quantizer;

        if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            quantizer = std::unique_ptr<faiss::Index>(faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str()));
        } else {
            throw LintDBException("Quantizer not found at path: " + path);
        }

        auto encoder = std::make_unique<DefaultEncoder>(
                DefaultEncoder(path, config.nlist, config.nbits, config.niter, config.dim));
        encoder->quantizer = std::move(quantizer);

        if(config.use_compression) {
            std::unique_ptr<faiss::Index> binarizer;
            if (config.use_compression) {
                if (FILE *file = fopen((path + "/" + BINARIZER_FILENAME).c_str(), "r")) {
                    fclose(file);
                    binarizer = std::unique_ptr<faiss::Index>(faiss::read_index((path + "/" + BINARIZER_FILENAME).c_str()));
                } else {
                    throw LintDBException("Binarizer not found at path: " + path);
                }
            }

            encoder->binarizer = std::move(binarizer);
        }
        encoder->use_compression = config.use_compression;
        encoder->nlist = config.nlist;
        encoder->nbits = config.nbits;
        encoder->niter = config.niter;
        encoder->dim = config.dim;
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

            if (use_compression) {
                LOG(INFO) << "Training binarizer with " << n << " embeddings.";
                //train binarizer on residuals.
                std::vector<idx_t> assign(n);
                quantizer->assign(n, embeddings, assign.data());

                std::vector<float> residuals(n * dim);
                quantizer->compute_residual_n(n, embeddings, residuals.data(), assign.data());

                binarizer->train(n, residuals.data());
            }

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