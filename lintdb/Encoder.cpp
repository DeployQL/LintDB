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
    DefaultEncoder::DefaultEncoder(size_t nlist, size_t nbits, size_t niter, size_t dim, bool use_compression)
        : Encoder(), nlist(nlist), nbits(nbits), niter(niter), dim(dim), use_compression(use_compression) {
        
        // colBERT uses L2 during clustering.
        // for normalized vectors, this should be the same as IP.
        this->quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
        if(use_compression) {
            this->binarizer = std::make_unique<Binarizer>(nbits, dim);
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

        if (use_compression) {
            std::vector<residual_t> residual_codes(num_tokens * (dim / 8 * nbits));
            binarizer->sa_encode(num_tokens, raw_residuals.data(), residual_codes.data());
            
            return std::make_unique<EncodedDocument>(EncodedDocument(
                coarse_idx, residual_codes, num_tokens, doc.id));
        } else {
            std::vector<residual_t> residual_codes(raw_residuals.begin(), raw_residuals.end());

            return std::make_unique<EncodedDocument>(EncodedDocument(
                coarse_idx, residual_codes, num_tokens, doc.id));
        }

    }

    std::vector<float> DefaultEncoder::decode_vectors(
            const gsl::span<const code_t> codes,
            const gsl::span<const residual_t> residuals,
            const size_t num_tokens,
            const size_t dim) const {
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
                std::memcpy(decoded_embeddings.data(), residuals.data(), dim * num_tokens * sizeof(float));
            }
        }
        return decoded_embeddings;
    }

    void DefaultEncoder::search(
            const float* data, // size: (num_query_tok, dim)
            const int num_query_tok,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids,
            const float centroid_threshold
    ) {
        // we get back the k top centroid matches per token.
        // faiss' quantizer->search() is slightly slower than doing this ourselves.
        // therefore, we will write our own to do our own matmul.
        // quantizer->search(
        //         num_query_tok,
        //         data,
        //         k_top_centroids,
        //         distances.data(),
        //         coarse_idx.data());
        std::vector<float> query_scores(num_query_tok * nlist, 0);

        cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                num_query_tok,
                nlist,
                dim,
                1.0,
                data,
                dim,
                quantizer->get_xb(),
                dim,
                0.0,
                query_scores.data(), // size: (num_query_tok x nlist)
                nlist);

        auto comparator = [](std::pair<float, idx_t> p1, std::pair<float, idx_t> p2) {
            return p1.first > p2.first;
        };

        std::vector<std::pair<float, idx_t>> centroid_scores;
        centroid_scores.reserve(num_query_tok*k_top_centroids);

        std::vector<std::pair<float, idx_t>> token_centroid_scores;
        token_centroid_scores.reserve(k_top_centroids);

        for(int i=0; i < num_query_tok; i++) {
            for (int j=0; j < nlist; j++) {
                idx_t key = j;
                float score = query_scores[i * nlist + j];
                if (token_centroid_scores.size() < k_top_centroids) {
                    token_centroid_scores.push_back(std::pair<float, idx_t>(score, key));

                    if (token_centroid_scores.size() == k_top_centroids) {
                        std::make_heap(token_centroid_scores.begin(), token_centroid_scores.end(), comparator);
                    }
                } else if (score > token_centroid_scores.front().first) {
                    std::pop_heap(token_centroid_scores.begin(), token_centroid_scores.end(), comparator);
                    token_centroid_scores.front() = std::pair<float, idx_t>(score, key);
                    std::push_heap(token_centroid_scores.begin(), token_centroid_scores.end(), comparator);
                }
            }
            
            for(idx_t k=0; k < k_top_centroids; k++) {
                centroid_scores.emplace_back(token_centroid_scores[k]);
            }

            token_centroid_scores.clear();
        }

        // std::sort(centroid_scores.begin(), centroid_scores.end(), comparator);

        for (int i = 0; i < num_query_tok; i++) {
            for (int j = 0; j < k_top_centroids; j++) {
                auto idx = i * k_top_centroids + j;
                coarse_idx[idx] = centroid_scores[idx].second;
                distances[idx] = centroid_scores[idx].first;
            }
        }
    }

    void DefaultEncoder::save(std::string path) {
        auto quantizer_path = path + "/"+ QUANTIZER_FILENAME;
        faiss::write_index(quantizer.get(), quantizer_path.c_str());

        if (use_compression) {
            binarizer->save(path);
        }
    }

    std::unique_ptr<Encoder> DefaultEncoder::load(std::string path, EncoderConfig& config) {
        std::unique_ptr<faiss::IndexFlat> quantizer;

        if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            auto qptr = std::unique_ptr<faiss::Index>(faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str()));
            quantizer = std::unique_ptr<faiss::IndexFlat>(static_cast<faiss::IndexFlat*>(qptr.release()));
        } else {
            throw LintDBException("Quantizer not found at path: " + path);
        }

        auto encoder = std::make_unique<DefaultEncoder>(
                DefaultEncoder(config.nlist, config.nbits, config.niter, config.dim));
        encoder->quantizer = std::move(quantizer);

        if(config.use_compression) {
            encoder->binarizer = Binarizer::load(path);
        }
        encoder->use_compression = config.use_compression;
        encoder->nlist = config.nlist;
        encoder->nbits = config.nbits;
        encoder->niter = config.niter;
        encoder->dim = config.dim;
        encoder->is_trained = true;
        return std::move(encoder);
    }

    void DefaultEncoder::train(const float* embeddings, const size_t n, const size_t dim) {
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

                binarizer->train(n, residuals.data(), dim);
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

    void DefaultEncoder::set_weights(const std::vector<float>& weights, const std::vector<float>& cutoffs, const float avg_residual) {
        binarizer->set_weights(weights, cutoffs, avg_residual);
    }
}