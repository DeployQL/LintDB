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

static const std::string METADATA_FILENAME = "_lintdb_metadata.json";

/**
 * SearchResult is a simple struct to hold the results of a search.
 * 
*/
struct SearchResult {
    idx_t id; /* the document id being returned. */
    float score; /* the score for the given document as compared to the query. */
};

struct Configuration {
    size_t nlist = 256; /* the number of centroids to train. */
    size_t nbits = 2; /* the number of bits to use in residual compression. */
    size_t niter = 10; /* the number of iterations to use during training. */
    size_t dim; /* the dimensions expected for incoming vectors. */
    bool use_compression = false; /* whether to compress residuals. */

    inline bool operator==(const Configuration& other) const {
        return nlist == other.nlist && 
            nbits == other.nbits && 
            niter == other.niter && 
            dim == other.dim && 
            use_compression == other.use_compression;
    }
};

struct SearchOptions {
    idx_t expected_id = -1; /* expects a document id in the return result. prints additional information during execution. useful for debugging.*/
    float centroid_score_threshold = 0.45; /* the threshold for centroid scores. */
    size_t k_top_centroids = 2; /* the number of top centroids to consider. */
    size_t num_second_pass = 1024; /* the number of second pass candidates to consider. */
    
    SearchOptions(): expected_id(-1) {};
};

/**
 * IndexIVF controls the inverted file structure.
 *
 * It's training will be a k-means clustering. We borrow naming conventions from
 * faiss.
 */
struct IndexIVF {
    size_t nlist; /// number of centroids to use in L1 quantizing.
    size_t nbits; /// number of bits used in binarizing the residuals.
    size_t niter; /// number of iterations to use in k-means clustering.
    bool use_ivf; /// whether to use the inverted file structure.
    bool use_compression; /// whether to use the LSH encoding for residuals.
    bool read_only; /// whether to open the index in read-only mode.

    /// load an existing index.
    IndexIVF(std::string path, bool read_only=false);

    IndexIVF(std::string path, size_t dim, Configuration& config);

    IndexIVF(
            std::string path, /// path to the database.
            size_t nlist,     /// number of centroids to use in L1 quantizing.
            size_t dim,       /// number of dimensions per embedding.
            size_t binarize_nbits=2, /// nbits used in the LSH encoding for esiduals.
            size_t niter = 10,
            bool use_compression = false,
            bool read_only = false
    );

    /**
     * Train will learn quantization and compression parameters from the given data.
     * 
     * @param n the number of embeddings to train on.
     * @param embeddings the embeddings to train on.
    */
    void train(size_t n, std::vector<float>& embeddings);
    void train(float* embeddings, size_t n, size_t dim);
    void train(float* embeddings, int n, int dim);

    /**
     * set_centroids overwrites the centroids in the encoder.
     * 
     * This is useful if you want to parallelize index writing and merge indices later.
    */
    void set_centroids(float* data, int n, int dim);

    /**
     * set_weights overwrites the compression weights in the encoder, if using compression.
    */
    void set_weights(const std::vector<float> weights, const std::vector<float> cutoffs, const float avg_residual);

    /**
     * search will find the nearest neighbors for a vector block.
     *
     * @param block the block of embeddings to search.
     * @param n_probe the number of nearest neighbors to find.
     * @param k the top k results to return.
     */
    std::vector<SearchResult> search(
        const uint64_t tenant,
        EmbeddingBlock& block,
        size_t n_probe,
        size_t k,
        SearchOptions opts=SearchOptions()) const;
    std::vector<SearchResult> search(
        const uint64_t tenant,
        float* data,
        int n,
        int dim,
        size_t n_probe,
        size_t k,
        SearchOptions opts=SearchOptions()) const;

    /**
     * lookup_pids accesses the inverted list for a given ivf_id and returns the passage ids.
     * 
    */
    std::vector<idx_t> lookup_pids(const uint64_t tenant, idx_t ivf_id) const;

    /**
     * add will add a block of embeddings to the index.
     *
     * @param docs a vector of RawPassages. This includes embeddings and ids.
     * @param ids the ids of the embeddings.
     */
    void add(const uint64_t tenant, const std::vector<RawPassage>& docs);
    void add_single(const uint64_t tenant, const RawPassage& doc);
    /**
     * remove deletes documents from the index by id.
     *
     * void remove(const std::vector<int64_t>& ids) works if SWIG complains
     * about idx_t.
     */
    void remove(const uint64_t tenant, const std::vector<idx_t>& ids);

    /**
     * Update is a convenience function for remove and add.
     */
    void update(const uint64_t tenant, const std::vector<RawPassage>& docs);

    /**
     * Merge will combine the index with another index.
     * 
     * We verify that the configuration of each index is correct, but this doesn't
     * prevent you from merging indices with different centroids. There will be
     * subtle ways for this to break, but this can enable easier multiprocess
     * building of indices.
    */
    void merge(const std::string path);

    /**
     * Index should be able to resume from a previous state.
     * Any quantization and compression will be saved within the Index's path.
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
    std::shared_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;

    std::unique_ptr<Encoder> encoder;

    size_t dim; /// number of dimensions per embedding.

    // helper to initialize the inverted list.
    void initialize_inverted_list();

    /// the inverted list data structure.
    std::unique_ptr<ForwardIndex> index_;
    std::unordered_set<idx_t> get_pids(idx_t ivf_id) const;
    std::vector<std::pair<float, idx_t>> get_top_centroids( 
        std::vector<idx_t>& coarse_idx,
        std::vector<float>& distances, 
        size_t n,
        const size_t total_centroids_to_calculate,
        float centroid_score_threshold,
        size_t k_top_centroids,
        size_t n_probe) const;

    /**
     * Write_metadata (and read) are helper methods to persist metadata attributes.
     * 
    */
    void write_metadata();
    Configuration read_metadata(std::string path);
};

} /// namespace lintdb

#endif