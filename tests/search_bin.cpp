#define private public
#include "lintdb/index.h"
#include <faiss/utils/random.h>
#include <chrono>
#include <iostream>
#include <cblas.h>
#include <lintdb/invlists/RocksdbList.h>
#include <glog/logging.h>
#include "lintdb/invlists/util.h"
// #include "lintdb/cf.h"
#include "lintdb/invlists/RocksdbList.h"

int main() {
    auto index = lintdb::IndexIVF("/Users/matt/deployql/LintDB/experiments/test_bench2");
    std::vector<idx_t> ids = {1, 2, 3, 4, 5};
    index.index_->get_codes(0, ids);
}

// int main() {
//     // auto config =lintdb::Configuration();
//     // auto index = lintdb::IndexIVF("/Users/matt/deployql/LintDB/experiments/test_bench2", 128, config);

//     // LOG(INFO) << "created db";
//     size_t dim = 128;
//     size_t num_docs = 100;
//     size_t num_tokens = 100;
//     size_t nlist = 65000;

//     lintdb::SearchOptions opts;
//     opts.centroid_score_threshold = 0;
//     opts.k_top_centroids = 64;
//     opts.num_second_pass = 1024;

//     size_t num_bits = 2;

//     rocksdb::ReadOptions ro;
//     // works as expected
//     // auto it = index.db->NewIterator(ro, index.column_families[lintdb::kCodesColumnIndex]);
//     // it->SeekToFirst();

//     // while(it->Valid()) {
//     //     rocksdb::Slice key = it->key();
//     //     LOG(INFO) << "iterator key: " << key.ToString(true);
//     //     auto ourkey = lintdb::ForwardIndexKey::from_slice(key);
//     //     LOG(INFO) << "doc id: " << ourkey.id;
//     //     LOG(INFO) << "doc tenant: " << ourkey.tenant;
//     //     LOG(INFO) << "doc size: " << it->value().size();

//     //     it->Next();
//     // }

//     rocksdb::Options options;
//     options.create_if_missing = true;
//     options.create_missing_column_families = true;

//     std::vector<rocksdb::ColumnFamilyHandle*> column_families;

//     auto cfs = lintdb::create_column_families();
//     rocksdb::OptimisticTransactionDB* db;
//     rocksdb::Status s = rocksdb::OptimisticTransactionDB::Open(
//             options, "/Users/matt/deployql/LintDB/experiments/test_bench3", cfs, &column_families, &db);
//     assert(s.ok());
//     auto owned_ptr = std::shared_ptr<rocksdb::OptimisticTransactionDB>(db);
//     auto index = std::make_unique<lintdb::WritableRocksDBInvertedList>(lintdb::WritableRocksDBInvertedList(owned_ptr, column_families));

//     LOG(INFO) << "number of column families created: " << column_families.size();

//     // std::vector<idx_t> ids = {1, 2, 3, 4, 5};
//     // auto codes = index.index_->get_codes(0, ids);

//     std::vector<std::string> key_strings;
//     std::vector<rocksdb::Slice> keys;
//     std::vector<rocksdb::Slice> new_keys;
 
//     // rocksdb slices don't take ownership of the underlying data, so we need to
//     // keep the strings around.
//     for (idx_t i = 0; i < 5; i++) {
//         // auto id = ids[i];
//         // auto key = lintdb::ForwardIndexKey{0, id};
//         // std::string kk = key.serialize();
//         std::string kk = lintdb::ForwardIndexKey{0, i}.serialize();
//         key_strings.emplace_back(kk);
//         keys.push_back(rocksdb::Slice(kk));
//         std::string kt = lintdb::ForwardIndexKey{0, i}.serialize();
//         new_keys.push_back(rocksdb::Slice(kt));
//         LOG(INFO) << "key string: " << key_strings[i] << " len: " << key_strings[i].size();
//         LOG(INFO) << "rocksdb key: " << keys[i].ToString() << " len: " << keys[i].size();
//         LOG(INFO) << "rocksdb slice data: " << keys[i].data();
//     }

//     // keys and new keys seem to work ok. this might suggest that serialization itself is nondeterministic.
//     auto keyone = lintdb::ForwardIndexKey{0, 1}.serialize();
//     LOG(INFO) << "keyone: " << keyone;
//     auto keytwo = lintdb::ForwardIndexKey{0, 1}.serialize();
//     LOG(INFO) << "keytwo: " << keytwo;

//     assert(keyone == keytwo);

//     rocksdb::WriteOptions wo;
//     LOG(INFO) << "writing";
//     db->Put(wo, column_families[lintdb::kCodesColumnIndex], keys[0], "e");
//     db->Put(wo, column_families[lintdb::kCodesColumnIndex], keys[1], "s");
//     db->Put(wo, column_families[lintdb::kCodesColumnIndex], keys[2], "t");
//     db->Put(wo, column_families[lintdb::kCodesColumnIndex], keys[3], "s");
//     LOG(INFO) << "wrote data";

//     // for(int i = 0; i < 5; i++) {
//     //     auto key_a = keys[i].data();
//     //     auto key_b = key_strings[i].data();
//     //     assert(strcmp(key_a, key_b) == 0);
//     // }

//     // assert(key_strings.size() == ids.size());
//     // VLOG(100) << "Getting num docs: " << key_strings.size()
//     //           << " from the forward index.";

//     // rocksdb::ReadOptions ro;
//     std::vector<rocksdb::PinnableSlice> values(keys.size());
//     std::vector<rocksdb::Status> statuses(keys.size());

//     // NOTE (MB): for some reason, multiget doesn't work. The issue is that
//     // the rocksdb::Slices don't seem to retain the string information **before** we
//     // make this call. I can't figure out why this is happening here, but not with individual
//     // get() calls.
//     // Multiget also works fine in an isolated binary, but copying that code here results in failure as well.
//     // I'm guessing there is an issue with this invlist accessing the memory of the strings for some reason.
//     db->MultiGet(
//             ro,
//             column_families[lintdb::kCodesColumnIndex],
//             new_keys.size(),
//             new_keys.data(),
//             values.data(),
//             statuses.data());

//     for (size_t i = 0; i < keys.size(); i++) {
//         if (statuses[i].ok()) {
//             // auto doc = GetInvertedIndexDocument(values[i].data());
//             // auto ptr = std::make_unique<DocumentCodes>(DocumentCodes(
//             //         ids[i], doc->codes()->data(), doc->codes()->size(), doc->codes()->size()
//             // ));
//             // docs.push_back(std::move(ptr));
//             // // release the memory used by rocksdb for this value.
//             // values[i].Reset();
//         } else {
//             LOG(ERROR) << "Could not find codes for doc id: " << key_strings[i];
//             LOG(ERROR) << "rocksdb: " << statuses[i].ToString();
//             // docs.push_back(nullptr);
//         }
//     }

//     //works as expected
//     // for (int i=1; i <= 5; i++ ) {
//     //     auto key = lintdb::ForwardIndexKey{0, i};
//     //     std::string value;
//     //     auto status = index.db->Get(ro, index.column_families[lintdb::kCodesColumnIndex], key.serialize(), &value);
//     //     if(!status.ok()) {
//     //         LOG(ERROR) << "not ok";
//     //     }

//     //     LOG(INFO) << "value: " << value.size();
//     // }

//     // does not work as epected
//     // std::vector<rocksdb::Slice> kkeys;
//     // for (int i=1; i <= 5; i++ ) {
//     //     auto key = lintdb::ForwardIndexKey{0, i};
//     //     kkeys.push_back(key.serialize());
//     // }

//     // std::vector<rocksdb::PinnableSlice> values_test(ids.size());
//     // std::vector<rocksdb::Status> statuses_test(ids.size());

//     // index.db->MultiGet(
//     //         ro,
//     //         index.column_families[lintdb::kCodesColumnIndex],
//     //         kkeys.size(),
//     //         kkeys.data(),
//     //         values_test.data(),
//     //         statuses_test.data());

//     // for (int i =0; i <statuses_test.size(); i++){
//     //     LOG(INFO) << "status i: " << i;
//     //     if(!statuses_test[i].ok()){
//     //         LOG(ERROR) << "not ok.";
//     //     }
//     // }

//     // std::vector<float> query(num_tokens * dim);
//     // faiss::rand_smooth_vectors(num_tokens, dim, query.data(), 1234);

//     // auto block = lintdb::EmbeddingBlock(query.data(), num_tokens, dim);
//     // // auto t_start = std::chrono::high_resolution_clock::now();
//     // auto results = index.search(0, block, 64, 100, opts);
//     // auto t_end = std::chrono::high_resolution_clock::now();
//     // double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

//     // std::cout << "Elapsed time: " << elapsed_time_ms << "ms" << std::endl;
//     // std::cout << "Results: " << results.size() << std::endl;


//     /// test cblas.
//     //   std::vector<float> query(dim * num_tokens);
//     // faiss::rand_smooth_vectors(num_tokens, dim, query.data(), 1234);

//     // std::vector<float> clusters(nlist * dim);
//     // faiss::rand_smooth_vectors(nlist, dim, clusters.data(), 1234);
    
//     // std::vector<float> query_scores(num_tokens * nlist, 0);
//     // auto t_start = std::chrono::high_resolution_clock::now();

//     // cblas_sgemm(
//     //         CblasRowMajor,
//     //         CblasNoTrans,
//     //         CblasTrans,
//     //         num_tokens,
//     //         nlist,
//     //         dim,
//     //         1.0,
//     //         query.data(),
//     //         dim,
//     //         clusters.data(),
//     //         dim,
//     //         0.0,
//     //         query_scores.data(), // size: (num_query_tok x nlist)
//     //         nlist);
// }