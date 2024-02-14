#ifndef LINTDB_INVLISTS_INVERTED_LIST_H
#define LINTDB_INVLISTS_INVERTED_LIST_H

#include <stddef.h>
#include "lintdb/api.h"
#include <memory>
#include "lintdb/invlists/EncodedDocument.h"
#include <vector>
#include "lintdb/constants.h"
#include "lintdb/invlists/Iterator.h"

namespace lintdb {
    /**
     * InvertedList manages the storage of centroid -> codes mappping.
     * 
     * It's also handling forward index behavior, and we should eventually split this out into a separate class.
    */
    struct InvertedList {
        virtual void add(
            size_t list_no,
            std::unique_ptr<EncodedDocument>& docs) = 0;

        virtual void delete_entry(
                size_t list_no,
                idx_t id) = 0;

        virtual std::unique_ptr<Iterator> get_iterator(size_t list_no) const = 0;

        /**
         * get retrieves from the forward index only. The inverted index should only be iterated on.
        */
        virtual std::vector<std::unique_ptr<EncodedDocument>> get(std::vector<idx_t> ids) const = 0;

        virtual ~InvertedList() = default;

    };
}

#endif