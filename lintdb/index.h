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
    idx_t id; 
    float score;
};

/**
 * Configuration of the Index. 
 * 
*/
struct Configuration {
    size_t nlist = 256; /// the number of centroids to train. 
    size_t nbits = 2; /// the number of bits to use in residual compression. 
    size_t niter = 10; /// the number of iterations to use during training. 
    size_t dim; /// the dimensions expected for incoming vectors. 
    bool use_compression = true; /// whether to compress residuals. 

    inline bool operator==(const Configuration& other) const {
        return nlist == other.nlist && 
            nbits == other.nbits && 
            niter == other.niter && 
            dim == other.dim && 
            use_compression == other.use_compression;
    }
};

/**
 * SearchOptions enables custom searching behavior. 
 * 
 * These options expose ways to tradeoff recall and latency at different levels of retrieval.
 * Searching more centroids:
 * - decrease centroid_score_threshold and increase k_top_centroids.
 * - increase n_probe in search()
 * 
 * Decreasing latency:
 * - increase centroid_score_threshold and decrease k_top_centroids.
 * - decrease n_probe in search()
*/
struct SearchOptions {
    idx_t expected_id = -1; /// expects a document id in the return result. prints additional information during execution. useful for debugging.
    float centroid_score_threshold = 0.45; /// the threshold for centroid scores. 
    size_t k_top_centroids = 2; /// the number of top centroids to consider. 
    size_t num_second_pass = 1024; /// the number of second pass candidates to consider. 

    SearchOptions(): expected_id(-1) {};
};

/**
 * IndexIVF is a multi vector index with an inverted file structure. 
 * 
 * This relies on pretrained centroids to accurately retrieve the closest documents.
 * 
 * 
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

    IndexIVF(std::string path, Configuration& config);

    IndexIVF(
            std::string path, /// path to the database.
            size_t nlist,     /// number of centroids to use in L1 quantizing.
            size_t dim,       /// number of dimensions per embedding.
            size_t binarize_nbits=2, /// nbits used in the LSH encoding for esiduals.
            size_t niter = 10,
            bool use_compression = true,
            bool read_only = false
    );

    // TODO(mbarta): this breaks SWIG. SWIG needs to ignore this, but won't.
    /**
     * Copy creates a new index at the given path from a trained index. The copy
     * will always be writeable.
     * 
     * Will throw an exception if the index isn't trained when this method is called.
     * 
     * @param path the path to initialize the index.
    */
    IndexIVF(const IndexIVF& other, const std::string path);

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
     * @param tenant the tenant the document belongs to.
     * @param block the block of embeddings to search.
     * @param n_probe the number of centroids to search.
     * @param k the top k results to return.
     * @param opts any search options to use during searching.
     */
    std::vector<SearchResult> search(
        const uint64_t tenant,
        const float* data,
        const int n,
        const int dim,
        const size_t n_probe,
        const size_t k,
        const SearchOptions& opts=SearchOptions()) const;

    std::vector<SearchResult> search(
        const uint64_t tenant,
        const EmbeddingBlock& block,
        const size_t n_probe,
        const size_t k,
        const SearchOptions& opts=SearchOptions()) const;

    /**
     * Add will add a block of embeddings to the index.
     *
     * @param tenant the tenant to assign the document to.
     * @param docs a vector of RawPassages. This includes embeddings and ids.
     */
    void add(const uint64_t tenant, const std::vector<RawPassage>& docs);

    /**
     * Add a single document.
    */
    void add_single(const uint64_t tenant, const RawPassage& doc);

    /**
     * Remove deletes documents from the index by id.
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
     *
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
    std::unordered_set<idx_t> get_pids(const idx_t ivf_id) const;
    std::vector<std::pair<float, idx_t>> get_top_centroids( 
        const std::vector<idx_t>& coarse_idx,
        const std::vector<float>& distances, 
        const size_t n,
        const size_t total_centroids_to_calculate,
        const float centroid_score_threshold,
        const size_t k_top_centroids,
        const size_t n_probe) const;

    /**
     * Flush data to disk.
     * 
     * Note: currently not used. We may want to expose this in the future.
    */
    void flush();

    /**
     * lookup_pids accesses the inverted list for a given ivf_id and returns the passage ids.
     * 
    */
    std::vector<idx_t> lookup_pids(const uint64_t tenant, const idx_t ivf_id) const;

    /**
     * Write_metadata (and read) are helper methods to persist metadata attributes.
     * 
    */
    void write_metadata();
    Configuration read_metadata(std::string path);


};

} /// namespace lintdb

#endif