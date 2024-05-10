#pragma once

#include "lintdb/index.h"
#include "lintdb/index_builder/EmbeddingModel.h"
#include "lintdb/index_builder/Tokenizer.h"
#include "lintdb/api.h"
#include <string>
#include <vector>
#include <map>

namespace lintdb {
    struct CollectionOptions {
        std::string model_file;
        std::string tokenizer_file;
        size_t max_length = 512;
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
        */
        void add(const uint64_t tenant, const uint64_t id, const std::string& text, const std::map<std::string, std::string>& metadata) const;

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

        void train(const std::vector<std::string> texts);

        private:
            IndexIVF* index;
            std::unique_ptr<EmbeddingModel> model;
            std::unique_ptr<Tokenizer> tokenizer;
    };
}