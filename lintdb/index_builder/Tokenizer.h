#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

typedef std::vector<int32_t> InputIds;
typedef std::vector<int32_t> AttentionMask;

namespace tokenizers {
class Tokenizer;
}

namespace lintdb {
/**
 * Tokenizer takes a string and returns a list of token ids.
 *
 * This equates to running a huggingface tokenizer.
 */
struct Tokenizer {
    const int cls_token = 101;
    const int sep_token = 102;
    const int mask_token = 103;
    const int pad_token = 0;
    const int query_token = 1;
    const int doc_token = 2;
    const int xtr_eos_token = 1;

    std::set<int> skip_tokens;

    Tokenizer(
            const std::string& path,
            const size_t max_length,
            const bool is_xtr = false);
    InputIds encode(const std::string& text, bool is_query = false) const;
    std::string decode(const InputIds& ids) const;

    bool is_special(const int id) const;

    ~Tokenizer();

   private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    const size_t max_length;
    const bool is_xtr;

    InputIds modify_ids_for_encoding(const InputIds& ids, bool is_query) const;
    InputIds modify_ids_for_xtr(const InputIds& ids, bool is_query) const;
    InputIds modify_ids_for_colbert(const InputIds& ids, bool is_query) const;
};

std::string LoadBytesFromFile(const std::string& path);
} // namespace lintdb