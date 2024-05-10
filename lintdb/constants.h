#ifndef LINTDB_CONSTANTS_H
#define LINTDB_CONSTANTS_H

#include <string>
#include <vector>
#include "lintdb/api.h"

namespace lintdb {
using std::string;
static const string kIndexColumnFamily = "index";
static const string kCodesColumnFamily = "codes";
static const string kResidualsColumnFamily = "residuals";
static const string kForwardColumnFamily = "forward";
static const string kMappingColumnFamily = "mapping";
static const string kDocColumnFamily = "doc";

typedef idx_t column_index_t;
static const column_index_t kIndexColumnIndex = 1;
static const column_index_t kForwardColumnIndex = 2;
static const column_index_t kCodesColumnIndex = 3;
static const column_index_t kResidualsColumnIndex = 4;
static const column_index_t kMappingColumnIndex = 5;
static const column_index_t kDocColumnIndex = 6;

[[deprecated]]
static const uint64_t kDefaultTenant = 0;
} // namespace lintdb

#endif