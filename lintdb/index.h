#ifndef LINTDB_INDEX_IVF_H
#define LINTDB_INDEX_IVF_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <unordered_set>
#include "lintdb/api.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/IndexWriter.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/query/Query.h"
#include "lintdb/retrievers/Retriever.h"
#include "lintdb/schema/DocProcessor.h"
#include "lintdb/schema/Document.h"
#include "lintdb/schema/FieldMapper.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/SearchResult.h"
#include "lintdb/version.h"

// forward declare these classes and avoid including the rocksdb headers.
namespace rocksdb {
class DB;
class ColumnFamilyHandle;

} // namespace rocksdb

namespace lintdb {

static const std::string METADATA_FILENAME = "_lintdb_metadata.json";

/**
 * Configuration of the Index.
 *
 */
struct Configuration {
    Version lintdb_version =
            LINTDB_VERSION; /// the current version of the index. Used
                            /// internally for feature compatibility.

    inline bool operator==(const Configuration& other) const {
        return lintdb_version == other.lintdb_version;
    }

    Configuration() = default;
};

/**
 * IndexIVF is a multi vector index with an inverted file structure.
 *
 * This relies on pretrained centroids to accurately retrieve the closest
 * documents.
 *
 *
 */
struct IndexIVF {
    Configuration config;
    bool read_only; /// whether to open the index in read-only mode.

    friend struct Collection; // our Collection wants access to the index.

    /// load an existing index.
    IndexIVF(const std::string& path, bool read_only = false);

    IndexIVF(
            const std::string& path,
            const Schema& schema,
            const Configuration& config);

    /**
     * Copy creates a new index at the given path from a trained index. The copy
     * will always be writeable.
     *
     * Will throw an exception if the index isn't trained when this method is
     * called.
     *
     * @param path the path to initialize the index.
     */
    IndexIVF(const IndexIVF& other, const std::string& path);

    /**
     * Train will learn quantization and compression parameters from the given
     * data.
     *
     */
    void train(const std::vector<Document>& docs);

    void set_quantizer(
            const std::string& field,
            std::shared_ptr<Quantizer> quantizer);
    void set_coarse_quantizer(
            const std::string& field,
            std::shared_ptr<ICoarseQuantizer> quantizer);

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
            const Query& query,
            const size_t k,
            const SearchOptions& opts = SearchOptions()) const;

    /**
     * Add will add a block of embeddings to the index.
     *
     * @param tenant the tenant to assign the document to.
     * @param docs a vector of EmbeddingPassages. This includes embeddings and
     * ids.
     */
    void add(const uint64_t tenant, const std::vector<Document>& docs);

    /**
     * Add a single document.
     */
    void add_single(const uint64_t tenant, const Document& doc);

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
    void update(const uint64_t tenant, const std::vector<Document>& docs);

    /**
     * Merge will combine the index with another index.
     *
     * We verify that the configuration of each index is correct, but this
     * doesn't prevent you from merging indices with different centroids. There
     * will be subtle ways for this to break, but this can enable easier
     * multiprocess building of indices.
     *
     */
    void merge(const std::string& path);

    /**
     * Index should be able to resume from a previous state.
     * Any quantization and compression will be saved within the Index's path.
     *
     * Inverted lists are persisted to the database.
     */
    void save();

    void close();

    ~IndexIVF() {
        for (auto& cf : column_families) {
            if (cf) {
                auto status = db->DestroyColumnFamilyHandle(cf);
                assert(status.ok());
            }
        }
    }

   private:
    std::string path;
    std::shared_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;
    Schema schema;
    std::shared_ptr<FieldMapper> field_mapper;

    std::unordered_map<std::string, std::shared_ptr<ICoarseQuantizer>>
            coarse_quantizer_map;
    std::unordered_map<std::string, std::shared_ptr<Quantizer>> quantizer_map;
    std::unique_ptr<Retriever> retriever;

    std::shared_ptr<DocumentProcessor> document_processor;
    // Note: invertedList and ForwardIndex are becoming read-only classes for
    // retrieval. writing is done through the index writer. Merging/Removing
    // will likely move to the writer as well.
    std::shared_ptr<InvertedList> inverted_list_;
    std::shared_ptr<ForwardIndex> index_;

    // helper to initialize the inverted list.
    void initialize_inverted_list(const Version& version);
    // helper to initialize the encoder, quantizer, and retrievers. These are
    // all inter-related.
    void initialize_retrieval();
    // instead of initializing, load from disk.
    void load_retrieval(const std::string& path, const Configuration& config);

    /**
     * Flush data to disk.
     *
     * Note: currently not used. We may want to expose this in the future.
     */
    void flush();

    /**
     * Write_metadata (and read) are helper methods to persist metadata
     * attributes.
     *
     */
    void write_metadata();
    Configuration read_metadata(const std::string& path);
};

} // namespace lintdb

#endif