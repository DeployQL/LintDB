
#ifndef LINTDB_INVERTEDLISTS_H
#define LINTDB_INVERTEDLISTS_H

#include "lintdb/schema/schema_generated.h"
#include "lintdb/api.h"
#include "lintdb/schema/util.h"

namespace lintdb {

struct EncodedDocument {
    EncodedDocument(
        const uint8_t* codes, 
        size_t code_size, 
        size_t len, 
        idx_t id, 
        std::string doc_id);

    const uint8_t* codes;
    size_t code_size;
    size_t len;
    idx_t id;
    std::string doc_id;
};

struct InvertedLists {
    InvertedLists(size_t list_no);

    size_t list_size(size_t list_no) const;
    const std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>>& get_codes(size_t list_no) const;
    const std::vector<idx_t>& get_ids(size_t list_no) const;

    void add_entries(
            size_t list_no,
            size_t n_entry,
            std::vector<idx_t> &ids,
            std::vector<EncodedDocument> &docs);

    void add_entry(
            size_t list_no,
            idx_t id,
            std::unique_ptr<EncodedDocument> code);

    void update_entries(
            size_t list_no,
            size_t n_entry,
            std::vector<idx_t> &ids,
            std::vector<EncodedDocument> &docs);

    void resize(size_t list_no, size_t new_size);


    private:
    size_t list_no;
    std::vector<std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>>> codes;
    std::vector<std::vector<idx_t>> ids;
};

}

#endif

