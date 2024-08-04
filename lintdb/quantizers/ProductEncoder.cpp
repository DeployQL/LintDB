#include "lintdb/quantizers/ProductEncoder.h"
#include <faiss/index_io.h>
#include <faiss/IndexPQ.h>
#include <faiss/utils/distances.h>
#include <glog/logging.h>
#include <algorithm>
#include <list>
#include <memory>
#include "lintdb/assert.h"
#include "lintdb/exception.h"

namespace lintdb {
ProductEncoder::ProductEncoder(
        size_t dim,
        size_t nbits,
        size_t num_subquantizers = 16)
        : Quantizer(),
          nbits(nbits),
          dim(dim),
          num_subquantizers(num_subquantizers) {
    this->pq = std::make_unique<faiss::IndexPQ>(
            dim /*input dimensions*/,
            num_subquantizers /* number of sub quantizers */,
            nbits /* number of bits per subquantizer index */,
            faiss::METRIC_INNER_PRODUCT);
    dsub = pq->pq.dsub;
    ksub = pq->pq.ksub;
}

void ProductEncoder::sa_encode(size_t n, const float* x, residual_t* codes) {
    pq->sa_encode(n, x, codes);
}
void ProductEncoder::sa_decode(size_t n, const residual_t* codes, float* x) {
    pq->sa_decode(n, codes, x);
}

size_t ProductEncoder::code_size() {
    return pq->sa_code_size();
}

void ProductEncoder::save(std::string path) {
    auto quantizer_path = path;
    faiss::write_index(pq.get(), quantizer_path.c_str());
}

std::unique_ptr<ProductEncoder> ProductEncoder::load(
        std::string path,
        QuantizerConfig& config) {
    std::unique_ptr<faiss::IndexPQ> quantizer;

    if (FILE* file = fopen((path).c_str(), "r")) {
        fclose(file);
        auto qptr = std::unique_ptr<faiss::Index>(
                faiss::read_index((path).c_str()));
        quantizer = std::unique_ptr<faiss::IndexPQ>(
                static_cast<faiss::IndexPQ*>(qptr.release()));
    } else {
        throw LintDBException("Quantizer not found at path: " + path);
    }

    auto encoder = std::make_unique<ProductEncoder>(
            config.dim, config.nbits, config.num_subquantizers);
    encoder->is_trained = quantizer->is_trained;
    encoder->pq = std::move(quantizer);

    encoder->nbits = config.nbits;
    encoder->dim = config.dim;
    encoder->num_subquantizers = config.num_subquantizers;
    encoder->ksub = encoder->pq->pq.ksub;
    encoder->dsub = encoder->pq->pq.dsub;

    return encoder;
}

QuantizerType ProductEncoder::get_type() {
    return PRODUCT_ENCODER;
}

void ProductEncoder::train(
        const size_t n,
        const float* embeddings,
        const size_t dim) {
    pq->train(n, embeddings);
}

std::unique_ptr<PQDistanceTables> ProductEncoder::get_distance_tables(
        const float* query_data,
        size_t num_tokens) const {
    return std::make_unique<PQDistanceTables>(
            query_data, num_tokens, dim, this->pq, true);
}

ProductEncoder::ProductEncoder(const ProductEncoder& other) {
    this->pq = std::make_unique<faiss::IndexPQ>(
            other.dim /*input dimensions*/,
            other.num_subquantizers /* number of sub quantizers */,
            other.nbits /* number of bits per subquantizer index */,
            faiss::METRIC_INNER_PRODUCT);
    this->pq->pq = other.pq->pq;
    this->dsub = other.pq->pq.dsub;
    this->ksub = other.pq->pq.ksub;
    this->nbits = other.nbits;
    this->dim = other.dim;
    this->num_subquantizers = other.num_subquantizers;
}

void swap(ProductEncoder& lhs, ProductEncoder& rhs) {
    std::swap(lhs.pq, rhs.pq);
    std::swap(lhs.nbits, rhs.nbits);
    std::swap(lhs.dim, rhs.dim);
    std::swap(lhs.num_subquantizers, rhs.num_subquantizers);
    std::swap(lhs.dsub, rhs.dsub);
    std::swap(lhs.ksub, rhs.ksub);
}

} // namespace lintdb