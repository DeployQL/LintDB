#include "lintdb/index.h"
#include <glog/logging.h>
#include <json/reader.h>
#include <json/writer.h>
#include <omp.h>
#include <rocksdb/db.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <gsl/span>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/assert.h"
#include "lintdb/cf.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/quantizers/io.h"
#include "lintdb/util.h"
#include "lintdb/version.h"
#include "lintdb/schema/FieldMapper.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/query/QueryExecutor.h"
#include "lintdb/scoring/Scorer.h"
#include <filesystem>

namespace lintdb {

// env var to set the number of threads for processing.
const char* PROCESSING_THREADS = "LINTDB_NUM_THREADS";

IndexIVF::IndexIVF(const std::string& path, bool read_only)
        : read_only(read_only), path(path) {
    // check that path exists as a directory
    if (!std::filesystem::is_directory(path)) {
        throw LintDBException("Path does not exist: " + path);
    }

    // set all of our individual attributes.
    this->config = this->read_metadata(path);

    Json::Value schema_root = loadJson(path + "/" + "_schema.json");
    this->schema = Schema::fromJson(schema_root);

    Json::Value field_mapper_root = loadJson(path + "/" + "_field_mapper.json");
    this->field_mapper = FieldMapper::fromJson(field_mapper_root);

    initialize_inverted_list(config.lintdb_version);
    load_retrieval(path, config);
}

IndexIVF::IndexIVF(const std::string& path, const Schema& schema, const Configuration& config) {
    this->config = config;
    this->path = path;
    this->schema = schema;
    this->read_only = false;

    this->field_mapper = std::make_shared<FieldMapper>();
    this->field_mapper->addSchema(schema);

    initialize_inverted_list(config.lintdb_version);
    initialize_retrieval();
}

IndexIVF::IndexIVF(const IndexIVF& other, const std::string& path) {
    // we'll leverage the loading methods and construct the index components
    // from files on disk.
    this->config = Configuration(other.config);
    this->schema = other.schema;
    this->read_only = false; // copying an existing index will always create a
                             // writeable blank index.
    FieldMapper* theirs = other.field_mapper.get();
    this->field_mapper = std::make_shared<FieldMapper>(*theirs);

    this->path = path;
    this->initialize_inverted_list(this->config.lintdb_version);
    load_retrieval(other.path, other.config);
    this->save();
}

void IndexIVF::initialize_retrieval() {
    // set omp threads
    if (std::getenv(PROCESSING_THREADS) != nullptr) {
        omp_set_num_threads(std::stoi(std::getenv(PROCESSING_THREADS)));
    }
    // load coarse quantizers
    for (const auto& field : this->schema.fields) {
        if (field.data_type != DataType::TENSOR && field.data_type != DataType::QUANTIZED_TENSOR) {
            continue;
        }
        std::shared_ptr<CoarseQuantizer> cq = std::make_shared<CoarseQuantizer>(field.parameters.dimensions);
        this->coarse_quantizer_map[field.name] = std::move(cq);
    }

    // load quantizers
    for (const auto& field : this->schema.fields) {
        if ((field.data_type != DataType::TENSOR &&
             field.data_type != DataType::QUANTIZED_TENSOR)) {
            continue;
        }
        QuantizerConfig qc{field.parameters.nbits, field.parameters.dimensions, field.parameters.num_subquantizers};
        std::shared_ptr<Quantizer> quantizer = create_quantizer(field.parameters.quantization, qc);
        this->quantizer_map[field.name] = std::move(quantizer);
    }
}

void IndexIVF::load_retrieval(const std::string& existing_path, const Configuration& config) {
    // load coarse quantizers
    for (const auto& field : this->schema.fields) {
        if (field.data_type != DataType::TENSOR && field.data_type != DataType::QUANTIZED_TENSOR) {
            continue;
        }
        std::string cqp = existing_path + "/" + field.name + "_coarse_quantizer";
        std::shared_ptr<CoarseQuantizer> cq = CoarseQuantizer::deserialize(cqp, config.lintdb_version);
        this->coarse_quantizer_map[field.name] = std::move(cq);
    }

    // load quantizers
    for (const auto& field : this->schema.fields) {
        if ((field.data_type != DataType::TENSOR &&
            field.data_type != DataType::QUANTIZED_TENSOR)) {
            continue;
        }
        LOG(INFO) << "loading quantizer for field: " << field.name;
        std::string qp = existing_path + "/" + field.name + "_quantizer";
        QuantizerConfig qc{field.parameters.nbits, field.parameters.dimensions, field.parameters.num_subquantizers};
        std::shared_ptr<Quantizer> quantizer = load_quantizer(qp, field.parameters.quantization, qc);
        this->quantizer_map[field.name] = std::move(quantizer);
        LOG(INFO) << "done loading quantizer for field: " << field.name;
    }
}

void IndexIVF::initialize_inverted_list(const Version& version) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();

    rocksdb::DB* ptr;
    rocksdb::Status s;
    if (!read_only) {
        s = rocksdb::DB::Open(
                options, path, cfs, &(this->column_families), &ptr);
    } else {
        s = rocksdb::DB::OpenForReadOnly(
                options, path, cfs, &(this->column_families), &ptr);
    }
    if (!s.ok()) {
        LOG(ERROR) << s.ToString();
        throw LintDBException("Could not open database at path: " + path);
    }
    assert(s.ok());

    this->db = std::shared_ptr<rocksdb::DB>(ptr);
    auto index_writer = std::make_unique<IndexWriter>(db, column_families, version);
    this->document_processor = std::make_shared<DocumentProcessor>(
        this->schema,
        this->quantizer_map,
        this->coarse_quantizer_map,
        this->field_mapper,
        std::move(index_writer)
    );

    this->index_ = std::make_shared<RocksdbForwardIndex>(
            this->db, this->column_families, version);
    this->inverted_list_ = std::make_shared<RocksdbInvertedList>(
            this->db, this->column_families, version);
}

void IndexIVF::train(const std::vector<Document>& docs) {
    LOG(INFO) << "num docs: " << docs.size();

    for(const auto& field: schema.fields) {
        // only train fields that are tensors and require indexing.
        if ((field.data_type == DataType::TENSOR ||
            field.data_type == DataType::QUANTIZED_TENSOR ) &&
                (std::find(field.field_types.begin(), field.field_types.end(), FieldType::Indexed) != field.field_types.end() ||
                        std::find(field.field_types.begin(), field.field_types.end(), FieldType::Colbert) != field.field_types.end()) ) {
            LOG(INFO) << "training field: " << field.name;

            LINTDB_THROW_IF_NOT(field.parameters.num_centroids != 0);

            // we've already initialized untrained quantizers in the constructor.
            std::shared_ptr<ICoarseQuantizer> cq = this->coarse_quantizer_map[field.name];

            // pull out the embeddings from the documents.
            std::vector<float> embeddings;
            size_t num_embeddings = 0;
            for (auto doc : docs) {
                FieldValue* fv;
                for(auto& doc_fields: doc.fields) {
                    if (doc_fields.name == field.name) {
                        fv = &doc_fields;
                        break;
                    }
                }
                if (fv == nullptr) {
                    continue;
                }

                auto field_value = *fv;
                LINTDB_THROW_IF_NOT(field_value.data_type == DataType::TENSOR || field_value.data_type == DataType::QUANTIZED_TENSOR);
                auto tensor = std::get<Tensor>(field_value.value);

                LINTDB_THROW_IF_NOT(tensor.size() % field.parameters.dimensions == 0);
                LINTDB_THROW_IF_NOT(tensor.size() > 0);

                num_embeddings += tensor.size() / field.parameters.dimensions;
                embeddings.insert(embeddings.end(), tensor.begin(), tensor.end());
            }

            cq->train(num_embeddings, embeddings.data(), field.parameters.num_centroids, field.parameters.num_iterations);

            if (field.parameters.quantization != QuantizerType::NONE) {
                // randomly sample embeddings to train the quantizer on.
                // we'll use the coarse quantizer to assign the embeddings to centroids.
                if(num_embeddings > 1e5) {
                    std::vector<size_t> sampled_ids = subsample(num_embeddings, std::sqrt(num_embeddings));
                    num_embeddings = sampled_ids.size();

                    std::vector<float> sampled_embeddings(num_embeddings * field.parameters.dimensions, 0);
                    size_t count = 0;
                    for(const auto& idx: sampled_ids) {
                        for(size_t i = 0; i < field.parameters.dimensions; i++) {
                            sampled_embeddings[count * field.parameters.dimensions + i] = embeddings[idx * field.parameters.dimensions + i];
                        }
                        count++;
                    }
                    embeddings = sampled_embeddings;
                }


                QuantizerConfig qc = {field.parameters.nbits, field.parameters.dimensions, field.parameters.num_subquantizers};
                std::unique_ptr<Quantizer> quantizer = create_quantizer(field.parameters.quantization, qc);

                LOG(INFO) << "Training quantizer for field: " << field.name;
                std::vector<idx_t> assign(num_embeddings, 0);

                cq->assign(num_embeddings, embeddings.data(), assign.data());

                std::vector<float> residuals(num_embeddings * field.parameters.dimensions, 0);
                cq->compute_residual_n(
                        num_embeddings, embeddings.data(), residuals.data(), assign.data());

                quantizer->train(num_embeddings, residuals.data(), field.parameters.dimensions);

                this->quantizer_map[field.name] = std::move(quantizer);
            }
        }
    }

    this->save();

    LOG(INFO) << "done training";
}

void IndexIVF::save() {
    // save schema using json cpp

    auto saveJson = [](std::string& path, Json::Value& root) {
        std::ofstream out(path);
        Json::StyledWriter writer;
        if (out.is_open()) {
            out << writer.write(root);
            out.close();
        } else {
            LOG(ERROR) << "Unable to open file for writing: "
                       << path;
        }
    };

    std::string schema_path = path + "/" + "_schema.json";
    Json::Value schema_root = schema.toJson();
    saveJson(schema_path, schema_root);

    std::string field_mapper_path = path + "/" + "_field_mapper.json";
    Json::Value field_mapper_root = field_mapper->toJson();
    saveJson(field_mapper_path, field_mapper_root);

    // save coarse quantizers
    for(const auto& [name, quantizer]: this->coarse_quantizer_map) {
        std::string cqp = this->path + "/" + name + "_coarse_quantizer";
        quantizer->serialize(cqp);
    }

    // save quantizers
    for(const auto& [name, quantizer]: this->quantizer_map) {
        std::string qp = this->path + "/" + name + "_quantizer";
        save_quantizer(qp, quantizer.get());
    }

    this->write_metadata();
}

void IndexIVF::flush() {
    rocksdb::FlushOptions fo;
    this->db->Flush(fo, column_families);
}


/**
 * Implementation note:
 *
 * when we look at what IVF lists to search, we have several parameters that
 * will influence this.
 * 1. k_top_centroids: responsible for how many centroids per token we include
 * in our search before sorting.
 * 2. n_probe: the number of lists we search on after sorting.
 */
std::vector<SearchResult> IndexIVF::search(
        const uint64_t tenant,
        const Query& query,
        const size_t k,
        const SearchOptions& opts) const {

    uint8_t colbert_field_id = this->field_mapper->getFieldID(opts.colbert_field);
    size_t colbert_code_size = this->quantizer_map.at(opts.colbert_field)->code_size();

    auto fm = field_mapper;
    QueryContext context(tenant, opts.colbert_field, this->inverted_list_, fm, coarse_quantizer_map, quantizer_map);
    PlaidScorer scorer(context);
    ColBERTScorer ranker(context);
    QueryExecutor executor(scorer, ranker);

    std::vector<ScoredDocument> results = executor.execute(context, query, k, opts);

    std::vector<SearchResult> search_results;
    for (const auto& result : results) {
        SearchResult sr;
        sr.id = result.doc_id;
        sr.score = result.score;
        search_results.push_back(sr);
    }

    return search_results;
}

void IndexIVF::set_quantizer(const std::string& field, const std::shared_ptr<Quantizer> quantizer) {
    this->quantizer_map.insert({field, std::move(quantizer)});
}

void IndexIVF::set_coarse_quantizer(const std::string& field, const std::shared_ptr<CoarseQuantizer> quantizer) {
    this->coarse_quantizer_map.insert({field, std::move(quantizer)});
}

void IndexIVF::add(
        const uint64_t tenant,
        const std::vector<Document>& docs) {

    for (auto doc : docs) {
        add_single(tenant, doc);
    }
}

void IndexIVF::add_single(const uint64_t tenant, const Document& doc) {
    this->document_processor->processDocument(tenant, doc);
}

void IndexIVF::remove(const uint64_t tenant, const std::vector<idx_t>& ids) {
    for( const auto& field: schema.fields) {
        uint8_t field_id = field_mapper->getFieldID(field.name);
        inverted_list_->remove(tenant, ids, field_id, field.data_type, field.field_types);
        index_->remove(tenant, ids);
    }
}

void IndexIVF::update(
        const uint64_t tenant,
        const std::vector<Document>& docs) {
    std::vector<idx_t> ids;
    for (auto doc : docs) {
        ids.push_back(doc.id);
    }
    remove(tenant, ids);
    add(tenant, docs);
}

void IndexIVF::merge(const std::string& path) {
    Configuration incoming_config = read_metadata(path);

    LINTDB_THROW_IF_NOT(this->config == incoming_config);

    rocksdb::Options options;
    options.create_if_missing = false;
    options.create_missing_column_families = true;

    auto cfs = create_column_families();
    std::vector<rocksdb::ColumnFamilyHandle*> other_cfs;

    rocksdb::DB* ptr;
    rocksdb::Status s =
            rocksdb::DB::OpenForReadOnly(options, path, cfs, &other_cfs, &ptr);
    assert(s.ok());

    inverted_list_->merge(ptr, other_cfs);
    index_->merge(ptr, other_cfs);

    for (auto cf : other_cfs) {
        db->DestroyColumnFamilyHandle(cf);
    }
}

void IndexIVF::close() {
    for (auto& cf : column_families) {
        db->DestroyColumnFamilyHandle(cf);
    }
    auto status = db->Close();
    assert(status.ok());

    column_families.clear();
}

void IndexIVF::write_metadata() {
    std::string out_path = path + "/" + METADATA_FILENAME;
    std::ofstream out(out_path);

    Json::Value metadata;

    metadata["lintdb_version"] = Json::String(LINTDB_VERSION_STRING);

    Json::StyledWriter writer;
    out << writer.write(metadata);
    out.close();
}

Configuration IndexIVF::read_metadata(const std::string& p) {
    this->path = p;
    std::string in_path = p + "/" + METADATA_FILENAME;
    std::ifstream in(in_path);
    if (!in) {
        throw LintDBException("Could not read metadata from path: " + in_path);
    }

    Json::Reader reader;
    Json::Value metadata;
    reader.parse(in, metadata);

    Configuration config;

    std::string version = metadata.get("lintdb_version", "0.0.0").asString();
    config.lintdb_version = Version(version);

    return config;
}

} // namespace lintdb