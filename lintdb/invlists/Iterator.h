#ifndef LINTDB_INVLISTS_ITERATOR_H
#define LINTDB_INVLISTS_ITERATOR_H

#include "lintdb/invlists/EncodedDocument.h"

namespace lintdb {
struct Iterator {
    virtual bool has_next() const = 0;
    virtual void next() = 0;
    virtual EncodedDocument get() const = 0;

    virtual ~Iterator() = default;
};
}

#endif