#include "lintdb/index_builder/Collection.h"
#include "lintdb/RawPassage.h"

namespace lintdb {
    Collection::Collection(IndexIVF* index, const CollectionOptions& opts) {
        this->index = index;
        this->model = std::make_unique<EmbeddingModel>(opts.model_file);
        this->tokenizer = std::make_unique<Tokenizer>(opts.tokenizer_file);
    }

    void Collection::add(const uint64_t tenant, const uint64_t id, const std::string& text) const {
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
        auto passage = RawPassage(output.data(), ids.size(), model->get_dims(), id);
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
}