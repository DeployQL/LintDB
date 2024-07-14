#ifndef LINTDB_INVLISTS_ITERATOR_H
#define LINTDB_INVLISTS_ITERATOR_H

#include "lintdb/invlists/KeyBuilder.h"
#include <string>

namespace lintdb {
struct Iterator {
    virtual bool is_valid() = 0;
    virtual void next() = 0;

    virtual InvertedIndexKey get_key() const = 0;
    virtual std::string get_value() const = 0;

    virtual ~Iterator() = default;
};
} // namespace lintdb

#endif