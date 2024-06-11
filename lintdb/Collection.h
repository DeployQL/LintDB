#pragma once

#include "lintdb/index.h"
#include "lintdb/index_builder/EmbeddingModel.h"
#include "lintdb/index_builder/Tokenizer.h"
#include "lintdb/Passages.h"
#include "lintdb/api.h"
#include <string>
#include <vector>
#include <map>
#include "lintdb/SearchOptions.h"

namespace lintdb {
    struct CollectionOptions {
        std::string model_file;
        std::string tokenizer_file;
        size_t max_length = 512;
    };

    struct TokenScore {
        std::string token;
        float score;
    };

    /**
     * Collection is a collection of documents. Instead of dealing directly with vectors, this
     * class allows you to add and search for documents by text.
    */
    struct Collection {
        Collection(IndexIVF* index, const CollectionOptions& opts);

        /**
         * Add a text document to the index.
         * 
         * @param tenant The tenant id.
         * @param id The document id.
         * @param text The text to add.
         * @param metadata a dictionary of metadata to store with the document. only accepts strings.
        */
        void add(const uint64_t tenant, const uint64_t id, const std::string& text, const std::map<std::string, std::string>& metadata) const;

        /**
         * Add a batch of texts to the index.
         *
         * @param tenant The tenant id.
         * @param passages A list of EmbeddingPassage objects to add.
        */
        void add_batch(
            const uint64_t tenant,
            const std::vector<TextPassage> passages
        ) const;
        /**
         * Search the index for similar documents.
         * 
         * @param tenant The tenant id.
         * @param text The text to search for.
         * @param k The number of results to return.
         * @param opts Any search options to use.
        */
        std::vector<SearchResult> search(
            const uint64_t tenant, 
            const std::string& text, 
            const size_t k, 
            const SearchOptions& opts=SearchOptions()) const;

        std::vector<TokenScore> interpret(
            const std::string& text,
            const std::vector<float> scores
        );

        void train(const std::vector<std::string> texts, size_t nlist=0, size_t niter=0);

        private:
            IndexIVF* index;
            std::unique_ptr<EmbeddingModel> model;
            std::unique_ptr<Tokenizer> tokenizer;
    };
}