#include "lintdb/Collection.h"
#include "lintdb/RawPassage.h"
#include <glog/logging.h>
#include <iostream>
#include "lintdb/utils/progress_bar.h"
#include <chrono>

namespace lintdb {
    Collection::Collection(IndexIVF* index, const CollectionOptions& opts) {
        this->index = index;
        this->model = std::make_unique<EmbeddingModel>(opts.model_file);
        this->tokenizer = std::make_unique<Tokenizer>(opts.tokenizer_file, opts.max_length);
    }

    void Collection::add(const uint64_t tenant, const uint64_t id, const std::string& text, const std::map<std::string, std::string>& metadata) const {
        auto ids = tokenizer->encode(text);

        ModelInput input;
        input.input_ids = ids;

        std::vector<int32_t> attn;
        for(auto id: ids) {
            if(id == 0) {
                attn.push_back(0);
            } else {
                attn.push_back(1);
            }
        }
        input.attention_mask = attn;

        auto output = model->encode(input);
        
        auto passage = RawPassage(output.data(), ids.size(), model->get_dims(), id, metadata);
        index->add(tenant, {passage});
    }

    std::vector<SearchResult> Collection::search(
            const uint64_t tenant, 
            const std::string& text, 
            const size_t k, 
            const SearchOptions& opts) const {
        auto ids = tokenizer->encode(text);

        ModelInput input;
        input.input_ids = ids;

        std::vector<int32_t> attn;
        for(auto id: ids) {
            if(id == 0) {
                attn.push_back(0);
            } else {
                attn.push_back(1);
            }
        }
        input.attention_mask = attn;
        auto output = model->encode(input);

        return index->search(tenant, output.data(), ids.size(), model->get_dims(), opts.n_probe, k, opts);
    }

    std::vector<TokenScore> Collection::interpret(
        const std::string& text,
        const std::vector<float> scores
    ) {
        auto ids = tokenizer->encode(text);
        std::vector<std::string> tokens;
        for(auto id: ids) {
            tokens.push_back(tokenizer->decode({id}));
        }

        std::vector<TokenScore> results;
        for(size_t i = 0; i < ids.size(); i++) {
            results.push_back({tokens[i], scores[i]});
        }

        return results;
    }

    void Collection::train(const std::vector<std::string> texts) {
        std::vector<float> embeddings;
        size_t num_embeddings = 0;

        progressbar bar(texts.size());
        bar.set_todo_char(" ");
        bar.set_done_char("█");

        for(auto text: texts) {
            bar.update();

            auto ids = tokenizer->encode(text);

            ModelInput input;
            input.input_ids = ids;

            std::vector<int32_t> attn;
            for(auto id: ids) {
                if(id == 0) {
                    attn.push_back(0);
                } else {
                    attn.push_back(1);
                }
            }
            input.attention_mask = attn;
            auto output = model->encode(input);

            embeddings.insert(embeddings.end(), output.begin(), output.end());
            num_embeddings += ids.size();
        }

        index->train(num_embeddings, embeddings);
    }
}