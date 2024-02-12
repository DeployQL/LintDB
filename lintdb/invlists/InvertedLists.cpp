#include "lintdb/invlists/InvertedLists.h"

namespace lintdb {
    EncodedDocument::EncodedDocument(
        const uint8_t* codes, 
        size_t code_size, 
        size_t len, 
        idx_t id, 
        std::string doc_id) {
        this->codes = codes;
        this->code_size = code_size;
        this->len = len;
    }

    InvertedLists::InvertedLists(size_t list_no) {
        this->list_no = list_no;

        codes.resize(list_no);
        ids.resize(list_no);
    }

    size_t InvertedLists::list_size(size_t list_no) const {
        return codes[list_no].size();
    };

    const std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>>& InvertedLists::get_codes(size_t list_no) const {
        return codes.at(list_no);
    }

    const std::vector<idx_t>& InvertedLists::get_ids(size_t list_no) const {
        return ids.at(list_no);
    }

    void InvertedLists::add_entries(
            size_t list_no,
            size_t n_entry,
            std::vector<idx_t> &ids,
            std::vector<EncodedDocument> &docs) {
        for(auto doc : docs) {
            auto doc_ptr = create_inverted_document(doc.len, doc.id, doc.doc_id, doc.codes, doc.code_size);
            this->codes[list_no].push_back(std::move(doc_ptr));
            this->ids[list_no].push_back(doc.id);
        }
    }

    void InvertedLists::add_entry(
            size_t list_no,
            idx_t id,
            std::unique_ptr<EncodedDocument> doc) {
        auto doc_ptr = create_inverted_document(doc->len, doc->id, doc->doc_id, doc->codes, doc->code_size);
        this->codes[list_no].push_back(std::move(doc_ptr));
        this->ids[list_no].push_back(doc->id);
    }

    void InvertedLists::update_entries(
            size_t list_no,
            size_t n_entry,
            std::vector<idx_t> &ids,
            std::vector<EncodedDocument> &docs) {
        
        for (size_t i = 0; i < n_entry; i++) {
            auto doc = docs[i];
            auto doc_ptr = create_inverted_document(doc.len, doc.id, doc.doc_id, doc.codes, doc.code_size);
            codes[list_no].insert(codes[list_no].begin() +ids[i], std::move(doc_ptr));
        }
    }

    void InvertedLists::resize(size_t list_no, size_t new_size) {
        codes[list_no].resize(new_size);
        ids[list_no].resize(new_size);
    }
}