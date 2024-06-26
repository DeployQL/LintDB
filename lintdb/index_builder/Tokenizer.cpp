#include "lintdb/index_builder/Tokenizer.h"
#include <glog/logging.h>
#include <tokenizers_cpp.h>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>

namespace lintdb {
Tokenizer::Tokenizer(
        const std::string& path,
        const size_t max_length,
        const bool is_xtr)
        : max_length(max_length), is_xtr(is_xtr) {
    auto blob = LoadBytesFromFile(path);
    if (is_xtr) {
        tokenizer = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
    } else {
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
    }

    for (const char& c : "!\"#$%&'()*+,-./:;<=>? @[\\]^_`{|}~") {
        skip_tokens.insert(tokenizer->TokenToId(std::string(1, c)));
    }
}

/**
 * Modify the input ids for encoding by adding special tokens.
 */
InputIds Tokenizer::modify_ids_for_encoding(const InputIds& ids, bool is_query)
        const {
    if (is_xtr) {
        return modify_ids_for_xtr(ids, is_query);
    }
    return modify_ids_for_colbert(ids, is_query);
}

InputIds Tokenizer::modify_ids_for_xtr(const InputIds& ids, bool is_query)
        const {
    InputIds new_ids;

    for (auto id : ids) {
        if (new_ids.size() >= max_length - 1) {
            break;
        }
        if (is_query) {
            if (skip_tokens.find(id) == skip_tokens.end()) {
                new_ids.push_back(id);
            }
        } else {
            new_ids.push_back(id);
        }
    }
    new_ids.push_back(xtr_eos_token);

    return new_ids;
}

InputIds Tokenizer::modify_ids_for_colbert(const InputIds& ids, bool is_query)
        const {
    InputIds new_ids;
    new_ids.push_back(cls_token);
    new_ids.push_back(is_query ? query_token : doc_token);

    for (auto id : ids) {
        if (new_ids.size() >= max_length - 1) {
            break;
        }
        if (is_query) {
            if (skip_tokens.find(id) == skip_tokens.end()) {
                new_ids.push_back(id);
            }
        } else {
            new_ids.push_back(id);
        }
    }

    new_ids.push_back(sep_token);
    return new_ids;
}

InputIds Tokenizer::encode(const std::string& text, bool is_query) const {
    InputIds ids = tokenizer->Encode(text);

    return modify_ids_for_encoding(ids, is_query);
}

bool Tokenizer::is_special(const int id) const {
    if (id == cls_token || id == sep_token || id == query_token ||
        id == doc_token || id == mask_token || id == pad_token) {
        return true;
    }
    return false;
}

std::string Tokenizer::decode(const InputIds& ids) const {
    return tokenizer->Decode(ids);
}

std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

Tokenizer::~Tokenizer() = default;
} // namespace lintdb