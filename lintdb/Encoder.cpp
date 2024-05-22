#include "lintdb/Encoder.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/impl/FaissException.h>
#include <faiss/index_io.h>
#include <glog/logging.h>
#include <cassert>
#include "lintdb/SearchOptions.h"
#include "lintdb/assert.h"
#include "lintdb/exception.h"
#include "lintdb/quantizers/io.h"
#include "lintdb/util.h"

namespace lintdb {

extern "C" {
    // this is to keep the clang syntax checker happy
    #ifndef FINTEGER
    #define FINTEGER int
    #endif

    /* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

    int sgemm_(
            const char* transa,
            const char* transb,
            FINTEGER* m,
            FINTEGER* n,
            FINTEGER* k,
            const float* alpha,
            const float* a,
            FINTEGER* lda,
            const float* b,
            FINTEGER* ldb,
            float* beta,
            float* c,
            FINTEGER* ldc);
}

DefaultEncoder::DefaultEncoder(
        size_t nlist,
        size_t nbits,
        size_t niter,
        size_t dim,
        size_t num_subquantizers,
        IndexEncoding type)
        : Encoder(),
          nbits(nbits),
          dim(dim),
          num_subquantizers(num_subquantizers),
          quantizer_type(type) {
    this->nlist = nlist;
    this->niter = niter;
    // colBERT uses L2 during clustering.
    // for normalized vectors, this should be the same as IP.
    this->coarse_quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
    auto quantizer_config = QuantizerConfig{
            .nbits = nbits,
            .dim = dim,
            .num_subquantizers = num_subquantizers,
    };

    this->quantizer = create_quantizer(type, quantizer_config);
}

DefaultEncoder::DefaultEncoder(
        size_t nbits,
        size_t dim,
        size_t num_subquantizers,
        IndexEncoding type)
        : Encoder(),
          nbits(nbits),
          dim(dim),
          num_subquantizers(num_subquantizers),
          quantizer_type(type) {
    // colBERT uses L2 during clustering.
    // for normalized vectors, this should be the same as IP.
    this->coarse_quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
    auto quantizer_config = QuantizerConfig{
            .nbits = nbits,
            .dim = dim,
            .num_subquantizers = num_subquantizers,
    };
    this->quantizer = create_quantizer(type, quantizer_config);
}

std::unique_ptr<EncodedDocument> DefaultEncoder::encode_vectors(
        const EmbeddingPassage& doc) {
    LINTDB_THROW_IF_NOT(nlist <= std::numeric_limits<code_t>::max());
    auto num_tokens = doc.embedding_block.num_tokens;

    const float* data_ptr = doc.embedding_block.embeddings.data();

    // get the centroids closest to each token.
    std::vector<idx_t> coarse_idx(num_tokens, 0);
    coarse_quantizer->assign(num_tokens, data_ptr, coarse_idx.data());

    assert(coarse_idx.size() == num_tokens);

    // compute residual
    std::vector<float> raw_residuals(num_tokens * dim);
    for (size_t i = 0; i < num_tokens; i++) {
        coarse_quantizer->compute_residual(
                data_ptr + i * dim,
                raw_residuals.data() + i * dim,
                coarse_idx[i]);
    }

    if (quantizer != nullptr) {
        std::vector<residual_t> residual_codes(num_tokens * (dim / 8 * nbits));
        quantizer->sa_encode(
                num_tokens, raw_residuals.data(), residual_codes.data());

        return std::make_unique<EncodedDocument>(EncodedDocument(
                coarse_idx, residual_codes, num_tokens, doc.id, doc.metadata));
    } else {
        const residual_t* byte_ptr =
                reinterpret_cast<const residual_t*>(raw_residuals.data());
        std::vector<residual_t> residual_codes(
                byte_ptr, byte_ptr + sizeof(float) * raw_residuals.size());

        return std::make_unique<EncodedDocument>(EncodedDocument(
                coarse_idx, residual_codes, num_tokens, doc.id, doc.metadata));
    }
}

std::vector<float> DefaultEncoder::decode_vectors(
        const gsl::span<const code_t> codes,
        const gsl::span<const residual_t> residuals,
        const size_t num_tokens) const {
    std::vector<float> decoded_embeddings(dim * num_tokens);

    for (size_t i = 0; i < num_tokens; i++) {
        auto centroid_id = codes[i];

        // add the centroid to the decoded embedding.
        coarse_quantizer->reconstruct(
                centroid_id, decoded_embeddings.data() + i * dim);

        if (quantizer != nullptr) {
            std::vector<float> decoded_residuals(dim);
            // TODO (mbarta): we can do a better job of hiding this offset
            // information. if we move the use_compression check above the for
            // loop, we'd have better success at batching this and moving the
            // offset into the binarizer.
            quantizer->sa_decode(
                    1,
                    residuals.data() + i * dim / 8 / nbits,
                    decoded_residuals.data());
            for (size_t j = 0; j < dim; j++) {
                decoded_embeddings[i * dim + j] += decoded_residuals[j];
            }
        } else {
            const float* residual_ptr = reinterpret_cast<const float*>(
                    residuals.data() + i * dim * sizeof(float));

            for (size_t j = 0; j < dim; j++) {
                decoded_embeddings[i * dim + j] += residual_ptr[j];
            }
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
        const float centroid_threshold) {
    std::vector<float> query_scores(num_query_tok * nlist, 0);

    float alpha = 1.0;
    float beta = 0.0;

    FINTEGER m = FINTEGER(num_query_tok);
    FINTEGER n = FINTEGER(nlist);
    FINTEGER k = FINTEGER(dim);

    FINTEGER lda = FINTEGER(dim);
    FINTEGER ldb = FINTEGER(dim);
    FINTEGER ldc = FINTEGER(nlist);
    // we need to treat this as operating in column major format.
    // we want data x centroids^T = C, but have row major data.
    // because of that, we want to calculate centroids x data^T = C^T
    sgemm_(
        "T",
        "N",
        &n,
        &m,
        &k,
        &alpha,
        coarse_quantizer->get_xb(), // size: (nlist x dim). transposed = (dim x nlist)
        &lda,
        data, // size: (num_query_tok x dim). transposed = (dim x num_query_tok)
        &ldb,
        &beta,
        query_scores.data(), // size: (nlist x num_query_tok)
        &ldc);

    auto comparator = [](std::pair<float, idx_t> p1,
                         std::pair<float, idx_t> p2) {
        return p1.first < p2.first;
    };

    std::vector<std::pair<float, idx_t>> centroid_scores(num_query_tok * k_top_centroids);

#pragma omp parallel
{
    std::vector<std::pair<float, idx_t>> token_centroid_scores;
    token_centroid_scores.reserve(k_top_centroids);

#pragma omp for nowait schedule(dynamic, 1) 
    for (int i = 0; i < num_query_tok; i++) {
        for (int j = 0; j < nlist; j++) {
            idx_t key = j;
            float score = query_scores[i * nlist + j];
            if (token_centroid_scores.size() < k_top_centroids) {
                token_centroid_scores.push_back(
                        std::pair<float, idx_t>(score, key));

                if (token_centroid_scores.size() == k_top_centroids) {
                    std::make_heap(
                            token_centroid_scores.begin(),
                            token_centroid_scores.end(),
                            comparator);
                }
            } else if (score > token_centroid_scores.front().first) {
                std::pop_heap(
                        token_centroid_scores.begin(),
                        token_centroid_scores.end(),
                        comparator);
                token_centroid_scores.front() =
                        std::pair<float, idx_t>(score, key);
                std::push_heap(
                        token_centroid_scores.begin(),
                        token_centroid_scores.end(),
                        comparator);
            }
        }

        for (idx_t k = 0; k < k_top_centroids; k++) {
            auto top = token_centroid_scores[k];
            float score = top.first;
            idx_t idx = top.second;
            centroid_scores[i * k_top_centroids + k] = std::pair<float, idx_t>(score, idx);
        }
        token_centroid_scores.clear();
    } // end for loop
} // end parallel

    for (int i = 0; i < num_query_tok; i++) {
        for (int j = 0; j < k_top_centroids; j++) {
            auto pair = centroid_scores[i * k_top_centroids + j];
            distances[i * k_top_centroids + j] = pair.first;
            coarse_idx[i * k_top_centroids + j] = pair.second;
        }
    }
}

float* DefaultEncoder::get_centroids() const {
    return coarse_quantizer->get_xb();
}

void DefaultEncoder::search_quantizer(
        const float* data, // size: (num_query_tok, dim)
        const int num_query_tok,
        std::vector<idx_t>& coarse_idx,
        std::vector<float>& distances,
        const size_t k_top_centroids,
        const float centroid_threshold) {
    // we get back the k top centroid matches per token.
    // faiss' quantizer->search() is slightly slower than doing this ourselves.
    // therefore, we will write our own to do our own matmul.
    coarse_quantizer->search(
            num_query_tok,
            data,
            k_top_centroids,
            distances.data(), // size: (num_query_tok, k_top_centroids)
            coarse_idx.data());
}

void DefaultEncoder::save(std::string path) {
    auto quantizer_path = path + "/" + ENCODER_FILENAME;
    faiss::write_index(coarse_quantizer.get(), quantizer_path.c_str());

    save_quantizer(path, quantizer.get());
}

std::unique_ptr<Encoder> DefaultEncoder::load(
        std::string path,
        EncoderConfig& config) {
    std::unique_ptr<faiss::IndexFlat> coarse_quantizer;

    if (FILE* file = fopen((path + "/" + ENCODER_FILENAME).c_str(), "r")) {
        fclose(file);
        auto qptr = std::unique_ptr<faiss::Index>(
                faiss::read_index((path + "/" + ENCODER_FILENAME).c_str()));
        coarse_quantizer = std::unique_ptr<faiss::IndexFlat>(
                static_cast<faiss::IndexFlat*>(qptr.release()));
    } else {
        throw LintDBException("coarse_quantizer not found at path: " + path);
    }

    auto encoder = std::make_unique<DefaultEncoder>(DefaultEncoder(
            config.nbits,
            config.dim,
            config.num_subquantizers,
            config.type));
    encoder->coarse_quantizer = std::move(coarse_quantizer);

    auto quantizer_config = QuantizerConfig{
            .nbits = config.nbits,
            .dim = config.dim,
            .num_subquantizers = config.num_subquantizers,
    };
    encoder->quantizer = load_quantizer(path, config.type, quantizer_config);
    encoder->nlist = config.nlist;
    encoder->nbits = config.nbits;
    encoder->niter = config.niter;
    encoder->dim = config.dim;
    encoder->is_trained = true;
    encoder->quantizer_type = config.type;

    return std::move(encoder);
}

void DefaultEncoder::train(
        const float* embeddings,
        const size_t n,
        const size_t dim,
        const size_t n_list,
        const size_t n_iter) {
    try {
        faiss::ClusteringParameters cp;
        if (this->niter != 0 && n_iter == 0) {
            cp.niter = this->niter;
        } else {
            cp.niter = n_iter;
        }
        if(this->nlist == 0 && n_list != 0) {
            this->nlist = n_list;
        }

        LINTDB_THROW_IF_NOT(this->nlist != 0);

        cp.nredo = 1;
        cp.seed = 123;
        faiss::Clustering clus(dim, this->nlist, cp);
        clus.verbose = true;

        // clustering uses L2 distance.
        faiss::IndexFlatL2 assigner(dim);
        clus.train(n, embeddings, assigner);

        normalize_vector(clus.centroids.data(), nlist, dim);

        coarse_quantizer->add(nlist, clus.centroids.data());

        if (quantizer != nullptr) {
            // residual quantizers are trained on residuals. we aren't
            // supporting training directly on embeddings.
            LOG(INFO) << "Training quantizer with " << n << " embeddings.";
            // train binarizer on residuals.
            std::vector<idx_t> assign(n);
            coarse_quantizer->assign(n, embeddings, assign.data());

            std::vector<float> residuals(n * dim);
            coarse_quantizer->compute_residual_n(
                    n, embeddings, residuals.data(), assign.data());

            quantizer->train(n, residuals.data(), dim);
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

    coarse_quantizer->reset();
    coarse_quantizer->add(n, data);

    this->is_trained = true;
}

void DefaultEncoder::set_weights(
        const std::vector<float>& weights,
        const std::vector<float>& cutoffs,
        const float avg_residual) {
    if (this->quantizer_type == IndexEncoding::BINARIZER) {
        auto binarizer = dynamic_cast<Binarizer*>(quantizer.get());
        binarizer->set_weights(weights, cutoffs, avg_residual);
    }
}
} // namespace lintdb