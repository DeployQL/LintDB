#ifndef LINTDB_CONSTANTS_H
#define LINTDB_CONSTANTS_H

#include <vector>
#include <string>


namespace lintdb {
    using std::string;
    static const string kIndexColumnFamily = "index";
    static const string kForwardColumnFamily = "forward";

    typedef idx_t column_index_t;
    static const column_index_t kIndexColumnIndex = 1;
    static const column_index_t kForwardColumnIndex = 2;

    static const uint64_t kDefaultTenant = 1;
}

#endif