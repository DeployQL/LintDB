#ifndef LINTDB_INDEX_IVF_H
#define LINTDB_INDEX_IVF_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lintdb/quantizer.h"
#include "lintdb/ResidualQuantizer.h"
#include "lintdb/api.h"
#include "lintdb/invlists/InvertedLists.h"
#include <faiss/invlists/DirectMap.h>
#include "lintdb/EmbeddingBlock.h"
#include <faiss/Clustering.h>
#include "lintdb/schema/schema_generated.h"

namespace lintdb {

/**
 * IndexIVF controls the inverted file structure. 
 * 
 * It's training will be a k-means clustering. We borrow naming conventions from faiss.
*/
struct IndexIVF {
    // quantizer encodes and decodes the blocks of embeddings.
    Quantizer* quantizer;

    // number of clusters.
    size_t nlist;
    bool is_trained = false;

    // total number of documents in the index.
    size_t ntotal = 0;

    //the inverted list data structure.
    InvertedLists invlists;
    bool own_invlists = true;

     /** optional map that maps back ids to invlist entries. This
     *  enables reconstruct() */
    faiss::DirectMap direct_map;

    /// do the codes in the invlists encode the vectors relative to the
    /// centroids?
    bool by_residual = true;

    IndexIVF(Quantizer* quantizer, size_t nlist);

    // train will learn a k-means clustering for document assignment.
    // train operators directly on individual embeddings.
    void train(size_t n,std::vector<float> embeddings);

    /**
     * search will find the nearest neighbors for a vector block.
     * 
     * @param block the block of embeddings to search.
     * @param n the number of nearest neighbors to find.
     * @param distances the output array of distances.
     * @param labels the output array of labels.
    */
    void search(EmbeddingBlock& block, size_t n_probe, std::vector<float> distances, std::vector<idx_t> labels) const;

    /**
     * add will add a block of embeddings to the index.
     * 
     * @param block the block of embeddings to add.
     * @param ids the ids of the embeddings.
    */
    void add(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> ids);
    void add_core(std::vector<EmbeddingBlock>, std::vector<idx_t> xids, std::vector<idx_t> coarse_idx);

    std::vector<std::unique_ptr<EncodedDocument>> encode_vectors(std::vector<EmbeddingBlock> blocks) const;
};

    size_t total_embeddings_in_blocks(std::vector<EmbeddingBlock> blocks);

} // namespace lintdb

#endif