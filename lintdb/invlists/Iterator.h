#ifndef LINTDB_INVLISTS_ITERATOR_H
#define LINTDB_INVLISTS_ITERATOR_H

#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/keys.h"

namespace lintdb {
struct Iterator {
    virtual bool has_next() = 0;
    virtual void next() = 0;

    virtual Key get_key() const = 0;
    virtual TokenKey get_token_key() const = 0;
    virtual PartialDocumentCodes get_value() const = 0;

    virtual ~Iterator() = default;
};
} // namespace lintdb

#endif