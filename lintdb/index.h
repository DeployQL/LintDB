#ifndef LINTDB_INDEX_IVF_H
#define LINTDB_INDEX_IVF_H

#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/invlists/DirectMap.h>
#include <rocksdb/db.h>
#include <gsl/span>
#include <unordered_set>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/RawPassage.h"
#include "lintdb/api.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/InvertedList.h"

namespace lintdb {
static const std::string QUANTIZER_FILENAME = "_quantizer.bin";
static const std::string BINARIZER_FILENAME = "_binarizer.bin";

struct SearchResult {
    idx_t id;
    float distance;
};

/**
 * IndexIVF controls the inverted file structure.
 *
 * It's training will be a k-means clustering. We borrow naming conventions from
 * faiss.
 */
struct IndexIVF {
    // number of clusters.
    size_t nlist; // number of centroids to use in L1 quantizing.
    size_t nbits; // number of bits used in binarizing the residuals.

    IndexIVF(
            std::string path, // path to the database.
            size_t nlist,     // number of centroids to use in L1 quantizing.
            size_t dim,       // number of dimensions per embedding.
            size_t binarize_nbits // nbits used in the LSH encoding for
                                  // residuals.
    );

    IndexIVF(
            std::string path,
            faiss::Index* quantizer,
            faiss::Index* binarizer);

    // train will learn a k-means clustering for document assignment.
    // train operators directly on individual embeddings.
    void train(size_t n, std::vector<float>& embeddings);
    // this endpoint is meant to work with python bindings and numpy.
    void train(float* embeddings, size_t n, size_t dim);
    // try another with int dimensions for numpy.
    void train(float* embeddings, int n, int dim);

    /**
     * search will find the nearest neighbors for a vector block.
     *
     * @param block the block of embeddings to search.
     * @param n_probe the number of nearest neighbors to find.
     * @param k the top k results to return.
     */
    std::vector<SearchResult> search(
            EmbeddingBlock& block,
            size_t n_probe,
            size_t k) const;

    /**
     * add will add a block of embeddings to the index.
     *
     * @param docs a vector of RawPassages. This includes embeddings and ids.
     * @param ids the ids of the embeddings.
     */
    void add(const std::vector<RawPassage>& docs);
    void add_single(const RawPassage& doc);
    /**
     * remove deletes documents from the index by id.
     *
     * void remove(const std::vector<int64_t>& ids) works if SWIG complains
     * about idx_t.
     */
    void remove(const std::vector<idx_t>& ids);

    /**
     * Update is a convenience function for remove and add.
     */
    void update(const std::vector<RawPassage>& docs);

    /**
     * Index should be able to resume from a previous state.
     * We want to persist the centroids, the quantizer, and the inverted lists.
     *
     * Inverted lists are persisted to the database.
     */
    void save(std::string dir);
    static IndexIVF load(std::string path);

    ~IndexIVF() {
        delete invlists;
        for (auto cf : column_families) {
            db->DestroyColumnFamilyHandle(cf);
        }
        delete db;
        delete quantizer;
        delete binarizer;
    }

   private:
    rocksdb::DB* db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;
    // quantizer encodes and decodes the blocks of embeddings. L2 of
    // quantization.
    faiss::Index* quantizer;
    // binarizer encodes the residuals. End stage of quantization.
    faiss::Index* binarizer;

    size_t dim; // number of dimensions per embedding.

    bool is_trained = false;

    // the inverted list data structure.
    InvertedList* invlists;
    std::unordered_set<idx_t> get_pids(idx_t ivf_id) const;

    /**
     * Encode vectors translates the embeddings given to us in RawPassage to
     * the internal representation that we expect to see in the inverted lists.
     */
    std::unique_ptr<EncodedDocument> encode_vectors(
            const RawPassage& doc) const;

    /**
     * Decode vectors translates out of our internal representation.
     *
     * Note: The interface to this has been changing -- it depends on the
     * structures we use to retrieve the data, which is still in flux.
     */
    std::vector<float> decode_vectors(
            gsl::span<code_t> codes,
            gsl::span<residual_t> residuals,
            size_t num_tokens,
            size_t dim) const;
};

} // namespace lintdb

#endif