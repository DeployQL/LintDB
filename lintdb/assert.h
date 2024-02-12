
#ifndef LINTDB_ASSERT_H
#define LINTDB_ASSERT_H

#include "lintdb/exception.h"
#include <faiss/impl/platform_macros.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// #define __PRETTY_FUNCTION__ __FUNCSIG__

#define LINTDB_THROW_FMT(FMT, ...)                              \
    do {                                                       \
        std::string __s;                                       \
        int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);   \
        __s.resize(__size + 1);                                \
        snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);       \
        throw lintdb::LintDBException(                           \
                __s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

///
/// Exceptions thrown upon a conditional failure
///

#define LINTDB_THROW_IF_NOT(X)                          \
    do {                                               \
        if (!(X)) {                                    \
            LINTDB_THROW_FMT("Error: '%s' failed", #X); \
        }                                              \
    } while (false)

#define LINTDB_THROW_IF_NOT_MSG(X, MSG)                       \
    do {                                                     \
        if (!(X)) {                                          \
            LINTDB_THROW_FMT("Error: '%s' failed: " MSG, #X); \
        }                                                    \
    } while (false)

#define LINTDB_THROW_IF_NOT_FMT(X, FMT, ...)                               \
    do {                                                                  \
        if (!(X)) {                                                       \
            LINTDB_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                 \
    } while (false)

#endif