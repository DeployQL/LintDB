
#include "lintdb/index.h"
#include "lintdb/assert.h"
#include <omp.h>
#include "lintdb/schema/util.h"

namespace lintdb {
    IndexIVF::IndexIVF(
        Quantizer* quantizer,
        size_t nlist
        ):  nlist(nlist),
            quantizer(quantizer),
            invlists(nlist){
            
            is_trained = quantizer->is_trained() && (quantizer->n_total() == nlist);
        }

    void IndexIVF::train(size_t n, std::vector<float> embeddings) {
        // perform a q1 clustering training.
        faiss::ClusteringParameters cp;
        faiss::Clustering clus(quantizer->dimensions(), nlist, cp);
        quantizer->get_inner_quantizer()->reset();

        // train using a faiss quantizer. faiss' quantizers extend indices, so the quantizer can be used directly to train.
        clus.train(n, embeddings.data(), *quantizer->get_inner_quantizer());
        quantizer->get_inner_quantizer()->is_trained = true;
    }

    void IndexIVF::search(EmbeddingBlock& block, size_t n_probe, std::vector<float> distances, std::vector<idx_t> labels) const {
        // we need to decode the block of embeddings.
        // we need to find the nearest neighbors.
        // we need to return the distances and labels.
        std::vector<EmbeddingBlock> blocks = { block };
        std::vector<idx_t> coarse_idx(n_probe);
        // auto indices = quantizer->assign(blocks);
    }

    void IndexIVF::add(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> ids) {
        // we encode all blocks into their code representations.
        // coarse_idx should point from each individual token embedding to its centroid.
        // therefore, we have block->len * n size of coarse_idx.
        size_t total = total_embeddings_in_blocks(blocks);
        // std::unique_ptr<idx_t[]> coarse_idx(new idx_t[total]);
        std::vector<idx_t> coarse_idx(total);
        quantizer->assign(blocks, coarse_idx);
        add_core(blocks, ids, coarse_idx);
    }

    // we need to encode the vectors into their code representations.
    // coarse_idx is the inverse list we should assign to.
    std::vector<std::unique_ptr<EncodedDocument>> IndexIVF::encode_vectors(std::vector<EmbeddingBlock> blocks) const {
        std::vector<std::unique_ptr<EncodedDocument>> codes;
        for(auto block : blocks) {
            size_t code_size = quantizer->code_size();
            std::vector<uint8_t> block_codes(block.len * code_size);
            quantizer->encode(block.len, block.data, block_codes);

            // create the inverted doc for the list.
            // auto doc_builder = create_inverted_document(block, block_codes.get(), code_size);
            auto embedded_doc = EncodedDocument(block_codes.data(), code_size, block.len, 0, "");
            codes.push_back(std::make_unique<EncodedDocument>(embedded_doc));
        }
        return codes;
    }

    void IndexIVF::add_core(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> xids, std::vector<idx_t> coarse_idx) {
        LINTDB_THROW_IF_NOT(is_trained);

        direct_map.check_can_add(xids.data());

        size_t nadd = 0, nminus1 = 0;
        size_t total = total_embeddings_in_blocks(blocks);
        for (size_t i = 0; i < total; ++i) {
            if (coarse_idx[i] < 0) {
                    nminus1++;
                }
        }

        auto docs = encode_vectors(blocks);

        faiss::DirectMapAdd dm_adder(direct_map, blocks.size(), xids.data());

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < blocks.size(); i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids.empty() ? xids[i] : ntotal + i;
                invlists.add_entry(
                        list_no, id, std::move(docs[i]));

                dm_adder.add(i, list_no, 0);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    // if (verbose) {
    //     printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
    //            nadd,
    //            n,
    //            nminus1);
    // }

    ntotal += blocks.size();
    }

    size_t total_embeddings_in_blocks(std::vector<EmbeddingBlock> blocks) {
        size_t total = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            total += blocks[i].len;
        }
        return total;
    }
}