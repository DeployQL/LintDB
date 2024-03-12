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
#include "lintdb/Encoder.h"

namespace lintdb {
// static const std::string QUANTIZER_FILENAME = "_quantizer.bin";
// static const std::string BINARIZER_FILENAME = "_binarizer.bin";
static const std::string METADATA_FILENAME = "_lintdb_metadata.json";

struct SearchResult {
    idx_t id;
    float distance;
};

struct Configuration {
    size_t nlist = 256;
    size_t nbits = 2;
    size_t niter = 10;
    bool use_ivf = true;
    bool use_compression = true;
};

struct SearchOptions {
    idx_t expected_id = -1;

    SearchOptions(): expected_id(-1) {};
};

/**
 * IndexIVF controls the inverted file structure.
 *
 * It's training will be a k-means clustering. We borrow naming conventions from
 * faiss.
 */
struct IndexIVF {
    size_t nlist; // number of centroids to use in L1 quantizing.
    size_t nbits; // number of bits used in binarizing the residuals.
    size_t niter; // number of iterations to use in k-means clustering.
    bool use_ivf; // whether to use the inverted file structure.
    bool use_compression; // whether to use the LSH encoding for residuals.

    // load an existing index.
    IndexIVF(std::string path);

    IndexIVF(std::string path, size_t dim, Configuration& config);

    IndexIVF(
            std::string path, // path to the database.
            size_t nlist,     // number of centroids to use in L1 quantizing.
            size_t dim,       // number of dimensions per embedding.
            size_t binarize_nbits=2, // nbits used in the LSH encoding for esiduals.
            size_t niter = 10,
            bool use_compression = true,
            bool use_ivf = true
    );

    // train will learn a k-means clustering for document assignment.
    // train operators directly on individual embeddings.
    void train(size_t n, std::vector<float>& embeddings);
    // this endpoint is meant to work with python bindings and numpy.
    void train(float* embeddings, size_t n, size_t dim);
    // try another with int dimensions for numpy.
    void train(float* embeddings, int n, int dim);

    /**
     * set_centroids overwrites the centroids in the encoder.
     * 
     * This is useful if you want to parallelize index writing and merge indices later.
    */
    void set_centroids(float* data, int n, int dim);

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
        size_t k,
        SearchOptions opts=SearchOptions()) const;

    // search method to accept a numpy array from python.
    std::vector<SearchResult> search(
        float* data,
        int n,
        int dim,
        size_t n_probe,
        size_t k,
        SearchOptions opts=SearchOptions()) const;

    // accesses the inverted list for a given ivf_id and returns the pids.
    std::vector<idx_t> lookup_pids(idx_t ivf_id) const;

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
    void save();

    ~IndexIVF() {
        for (auto cf : column_families) {
            db->DestroyColumnFamilyHandle(cf);
        }
    }

   private:
    std::string path;
    std::unique_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;

    std::unique_ptr<Encoder> encoder;

    size_t dim; // number of dimensions per embedding.

    // the inverted list data structure.
    std::unique_ptr<ForwardIndex> index_;
    std::unordered_set<idx_t> get_pids(idx_t ivf_id) const;

    // /**
    //  * Encode vectors translates the embeddings given to us in RawPassage to
    //  * the internal representation that we expect to see in the inverted lists.
    //  */
    // std::unique_ptr<EncodedDocument> encode_vectors(
    //         const RawPassage& doc) const;

    // /**
    //  * Decode vectors translates out of our internal representation.
    //  *
    //  * Note: The interface to this has been changing -- it depends on the
    //  * structures we use to retrieve the data, which is still in flux.
    //  */
    // std::vector<float> decode_vectors(
    //         gsl::span<const code_t> codes,
    //         gsl::span<const residual_t> residuals,
    //         size_t num_tokens,
    //         size_t dim) const;

    void write_metadata();
    void read_metadata(std::string path);
};

} // namespace lintdb

#endif