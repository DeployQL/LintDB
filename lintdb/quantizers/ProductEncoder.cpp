#include "lintdb/quantizers/ProductEncoder.h"
#include <faiss/index_io.h>
#include "lintdb/exception.h"
#include "lintdb/assert.h"
#include <memory>


namespace lintdb {
    ProductEncoder::ProductEncoder(
        size_t dim,
        size_t nbits, 
        size_t num_subquantizers = 16
    ) : Quantizer(), nbits(nbits), dim(dim) {
        this->pq = std::make_unique<faiss::IndexPQ>(
            dim /*input dimensions*/, 
            num_subquantizers /* number of sub quantizers */, 
            nbits /* number of bits per subquantizer index */,
            faiss::METRIC_INNER_PRODUCT);
    }

    void ProductEncoder::sa_encode(size_t n, const float* x, residual_t* codes) {
        pq->sa_encode(n, x, codes);
    }
    void ProductEncoder::sa_decode(size_t n, const residual_t* codes, float* x) {
        pq->sa_decode(n, codes, x);
    }

    void ProductEncoder::save(std::string path) {
        auto quantizer_path = path + "/"+ QUANTIZER_FILENAME;
        faiss::write_index(pq.get(), quantizer_path.c_str());
    }

    std::unique_ptr<ProductEncoder> ProductEncoder::load(std::string path, QuantizerConfig& config) {
        std::unique_ptr<faiss::IndexPQ> quantizer;

        if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            auto qptr = std::unique_ptr<faiss::Index>(faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str()));
            quantizer = std::unique_ptr<faiss::IndexPQ>(static_cast<faiss::IndexPQ*>(qptr.release()));
        } else {
            throw LintDBException("Quantizer not found at path: " + path);
        }

        auto encoder = std::make_unique<ProductEncoder>(config.dim, config.nbits, config.num_subquantizers);
        encoder->pq = std::move(quantizer);

        encoder->nbits = config.nbits;
        encoder->dim = config.dim;
        encoder->is_trained = true;
        return encoder;
    }

    QuantizerType ProductEncoder::get_type() {
        return PRODUCT_ENCODER;
    }

    void ProductEncoder::train(const size_t n, const float* embeddings, const size_t dim) {
        pq->train(n, embeddings);
    }

}