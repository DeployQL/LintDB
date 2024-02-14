#include "lintdb/invlists/RocksdbList.h"
#include "lintdb/schema/util.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/constants.h"
#include "lintdb/assert.h"
#include <iostream>
#include <glog/logging.h>

namespace lintdb {
    RocksDBInvertedList::RocksDBInvertedList(
        rocksdb::DB& db, 
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families
    ): db_(db), column_families(column_families) {}

    std::unique_ptr<Iterator>  RocksDBInvertedList::get_iterator(size_t list_no) const {
        return std::make_unique<RocksDBIterator>(RocksDBIterator(db_, column_families[kIndexColumnIndex], list_no));
    }

    void RocksDBInvertedList::add(
            size_t list_no,
            std::unique_ptr<EncodedDocument>& doc){
        rocksdb::WriteOptions wo;
        
        auto doc_ptr = create_inverted_index_document(
            doc->codes.data(), 
            doc->codes.size()
        );

        auto forward_doc_ptr = create_forward_index_document(
            doc->num_tokens,  
            doc->doc_id, 
            doc->residuals.data(),
            doc->residuals.size()
        );

        Key key = Key{kDefaultTenant, list_no, doc->id};

        auto* ptr = doc_ptr->GetBufferPointer();
        auto size = doc_ptr->GetSize();
        const rocksdb::Slice slice(reinterpret_cast<const char*>(ptr), size);
        std::string k_string = key.serialize();
        rocksdb::Status status = db_.Put(
                wo,
                column_families[kIndexColumnIndex],
                rocksdb::Slice(k_string),
                slice
        );
        assert(status.ok());

        auto* forward_ptr = forward_doc_ptr->GetBufferPointer();
        auto forward_size = forward_doc_ptr->GetSize();
        const rocksdb::Slice forward_slice(reinterpret_cast<const char*>(forward_ptr), forward_size);
    
        rocksdb::Status forward_status = db_.Put(
                wo,
                column_families[kForwardColumnIndex],
                rocksdb::Slice(k_string),
                forward_slice
        );
        assert(forward_status.ok());

        VLOG(1) << "Added document with id: " << doc->id << " to list: " << list_no << " to rocksdb. key: " << k_string;
    };

    std::vector<std::unique_ptr<EncodedDocument>> RocksDBInvertedList::get(std::vector<idx_t> ids) const {
        std::vector<rocksdb::Slice> keys;
        for (auto id : ids) {
            auto key = ForwardIndexKey{kDefaultTenant, id};
            auto serialized_key = key.serialize();
            keys.push_back(rocksdb::Slice(serialized_key.data(), serialized_key.size()));
        }

        std::vector<std::unique_ptr<EncodedDocument>> docs;
        rocksdb::ReadOptions ro;
        std::vector<rocksdb::PinnableSlice> values;
        std::vector<rocksdb::Status> statuses;

        db_.MultiGet(
            ro, 
            column_families[kForwardColumnIndex],
            ids.size(), 
            keys.data(), 
            values.data(),
            statuses.data()
        );

        const code_t* empty_codes = nullptr;

        for (size_t i = 0; i < ids.size(); i++) {
            if (statuses[i].ok()) {
                auto doc = GetForwardIndexDocument(values[i].data());
                auto ptr = std::make_unique<EncodedDocument>(
                        EncodedDocument(
                            empty_codes,
                            0,
                            doc->residuals()->data(),
                            doc->residuals()->size(),
                            doc->num_tokens(),
                            ids[i],
                            doc->doc_id()->str()
                        )
                    );
                docs.push_back(std::move(ptr));
                // release the memory used by rocksdb for this value.
                values[i].Reset();
            } else {
                LOG(WARNING) << "Could not find document with id: " << ids[i] << " in the forward index.";
                docs.push_back(nullptr);
            }
        }

        return docs;
    }

    void RocksDBInvertedList::delete_entry(
            size_t list_no,
            idx_t id) {
            
        Key key = Key{kDefaultTenant, list_no, id};
        rocksdb::WriteOptions wo;
        std::string k_string = key.serialize();
        rocksdb::Status status = db_.Delete(
            wo,
            rocksdb::Slice(k_string)
        );

        assert(status.ok());
    };

}