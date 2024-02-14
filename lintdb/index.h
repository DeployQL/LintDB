#ifndef LINTDB_INDEX_IVF_H
#define LINTDB_INDEX_IVF_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "lintdb/api.h"
#include "lintdb/invlists/InvertedList.h"
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/IndexLSH.h>
#include "lintdb/EmbeddingBlock.h"
#include <faiss/Clustering.h>
#include <faiss/IndexPQ.h>
#include <rocksdb/db.h>

namespace lintdb {

struct SearchResults {
    std::vector<float> distances;
    std::vector<idx_t> labels;
};
/**
 * IndexIVF controls the inverted file structure. 
 * 
 * It's training will be a k-means clustering. We borrow naming conventions from faiss.
*/
struct IndexIVF {
    const std::string CENTROIDS_FILENAME = "_centroids.bin";
    const std::string QUANTIZER_FILENAME = "_quantizer.bin";
    const std::string BINARIZER_FILENAME = "_binarizer.bin";

    // the path to the database.
    std::string path;

    rocksdb::DB* db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;
    // quantizer encodes and decodes the blocks of embeddings. L2 of quantization.
    faiss::IndexPQ quantizer;
    // binarizer encodes the residuals. End stage of quantization.
    faiss::IndexLSH binarizer;
    // keep a copy of the centroids so that we can do a coarse quantization on queries.
    af::array centroids;
    size_t dim; // number of dimensions per embedding.

    // number of clusters.
    size_t nlist;
    bool is_trained = false;

    //the inverted list data structure.
    std::unique_ptr<InvertedList> invlists;

    IndexIVF(
        std::string path, // path to the database.
        size_t nlist, // number of centroids to use in L1 quantizing.
        size_t dim, // number of dimensions per embedding.
        size_t code_nbits, // nbits used in the PQ encoding for codes.
        size_t binarize_nbits // nbits used in the LSH encoding for residuals.
    );

    // train will learn a k-means clustering for document assignment.
    // train operators directly on individual embeddings.
    void train(size_t n,std::vector<float>& embeddings);

    /**
     * search will find the nearest neighbors for a vector block.
     * 
     * @param block the block of embeddings to search.
     * @param n the number of nearest neighbors to find.
     * @param distances the output array of distances.
     * @param labels the output array of labels.
    */
    SearchResults search(EmbeddingBlock& block, size_t n_probe) const;
    SearchResults search_single_list(size_t idx, EmbeddingBlock& block, af::array& nlist_maxes) const;
    /**
     * add will add a block of embeddings to the index.
     * 
     * @param block the block of embeddings to add.
     * @param ids the ids of the embeddings.
    */
    void add(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> ids);

    std::unique_ptr<EncodedDocument> encode_vectors(EmbeddingBlock& block) const;
    EmbeddingBlock decode_vectors(EncodedDocument& doc) const;

    /**
     * Index should be able to resume from a previous state.
     * We want to persist the centroids, the quantizer, and the inverted lists.
     * 
     * Inverted lists are persisted to the database.
    */
    void save();

    ~IndexIVF() {
        for(auto cf : column_families) {
            db->DestroyColumnFamilyHandle(cf);
        }
        delete db;
    }
};

} // namespace lintdb

#endif