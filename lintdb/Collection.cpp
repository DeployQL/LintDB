#include "lintdb/Collection.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/assert.h"
#include <glog/logging.h>
#include <iostream>
#include "lintdb/utils/progress_bar.h"
#include <chrono>
#include "lintdb/util.h"

namespace lintdb {
    Collection::Collection(IndexIVF* index, const CollectionOptions& opts) {
        this->index = index;
        this->model = std::make_unique<EmbeddingModel>(opts.model_file);

        LINTDB_THROW_IF_NOT_MSG(index->config.dim == model->get_dims(), "model dimensions don't match index dimensions");

        bool is_xtr = index->config.quantizer_type == IndexEncoding::XTR;
        this->tokenizer = std::make_unique<Tokenizer>(opts.tokenizer_file, opts.max_length, is_xtr);
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
//        normalize_vector(output.data(), ids.size(), model->get_dims());
    
        auto passage = EmbeddingPassage(output.data(), ids.size(), model->get_dims(), id, metadata);
        index->add(tenant, {passage});
    }

    void Collection::add_batch(
        const uint64_t tenant,
        const std::vector<TextPassage>& passages
        ) const {

        for(const auto& passage: passages) {
            this->add(tenant, passage.id, passage.data, passage.metadata);
        }
    }

    std::vector<SearchResult> Collection::search(
            const uint64_t tenant, 
            const std::string& text, 
            const size_t k, 
            const SearchOptions& opts) const {
        auto ids = tokenizer->encode(text, true);

        ModelInput input;
        input.input_ids = ids;

        std::vector<int32_t> attn;
        int size = 0;
        for(auto id: ids) {
            if(id == 0) {
                attn.push_back(0);
            } else {
                attn.push_back(1);
                size++;
            }
        }
        input.attention_mask = attn;
        auto output = model->encode(input);

        std::vector<float> query_data = output;
//        normalize_vector(query_data.data(), ids.size(), model->get_dims());

        return index->search(tenant, query_data.data(), size, model->get_dims(), opts.n_probe, k, opts);
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
            if (tokenizer->is_special(ids[i])) {
                continue;
            }
            results.push_back({tokens[i], scores[i]});
        }

        return results;
    }

    void Collection::train(const std::vector<std::string> texts, size_t nlist, size_t niter) {
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

        index->train(num_embeddings, embeddings, nlist, niter);
    }
}