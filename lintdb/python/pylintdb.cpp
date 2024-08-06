#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "lintdb/api.h"
#include "lintdb/index.h"
#include "lintdb/quantizers/Binarizer.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/query/Query.h"
#include "lintdb/query/QueryNode.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Document.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/SearchResult.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace lintdb {

// Helper function to convert std::chrono::time_point to a Python datetime
// object
nb::object toPythonDatetime(const DateTime& datetime) {
    auto duration = datetime.time_since_epoch();
    auto millis =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                    .count();
    return nb::cast(millis);
}

// Helper function to convert Python datetime object to std::chrono::time_point
DateTime fromPythonDatetime(const nb::object& obj) {
    auto millis = nb::cast<int64_t>(obj);
    return DateTime(Duration(millis));
}

// Wrapper functions
FieldValue IntFieldValue(std::string name, int value) {
    return FieldValue(name, value);
}

FieldValue FloatFieldValue(std::string name, float value) {
    return FieldValue(name, value);
}

FieldValue TextFieldValue(std::string name, const std::string& value) {
    return FieldValue(name, value);
}

FieldValue TensorFieldValue(
        std::string name,
        nb::ndarray<float, nb::ndim<2>, nb::device::cpu> value) {
    std::vector<float> vec =
            std::vector<float>(value.data(), value.data() + value.size());
    return FieldValue(name, vec, value.shape(0));
}

FieldValue QuantizedTensorFieldValue(
        std::string name,
        nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cpu> value) {
    std::vector<uint8_t> vec =
            std::vector<uint8_t>(value.data(), value.data() + value.size());
    return FieldValue(name, vec, value.shape(0));
}

FieldValue DateFieldValue(std::string name, const nb::object& value) {
    DateTime datetime = fromPythonDatetime(value);
    return FieldValue(name, datetime);
}

// Function to create FieldParameters from a dictionary
FieldParameters create_field_parameters_from_dict(const nb::dict& params) {
    FieldParameters fp;
    if (params.contains("dimensions"))
        fp.dimensions = nb::cast<size_t>(params["dimensions"]);
    if (params.contains("analyzer"))
        fp.analyzer = nb::cast<std::string>(params["analyzer"]);
    if (params.contains("quantization"))
        fp.quantization = nb::cast<QuantizerType>(params["quantization"]);
    if (params.contains("num_centroids"))
        fp.num_centroids = nb::cast<size_t>(params["num_centroids"]);
    if (params.contains("num_iterations"))
        fp.num_iterations = nb::cast<size_t>(params["num_iterations"]);
    if (params.contains("num_subquantizers"))
        fp.num_subquantizers = nb::cast<size_t>(params["num_subquantizers"]);
    if (params.contains("nbits"))
        fp.nbits = nb::cast<size_t>(params["nbits"]);
    return fp;
}

SearchOptions create_search_options_from_dict(const nb::dict& dict) {
    SearchOptions opts;
    if (dict.contains("expected_id"))
        opts.expected_id = nb::cast<idx_t>(dict["expected_id"]);
    if (dict.contains("centroid_score_threshold"))
        opts.centroid_score_threshold =
                nb::cast<float>(dict["centroid_score_threshold"]);
    if (dict.contains("k_top_centroids"))
        opts.k_top_centroids = nb::cast<size_t>(dict["k_top_centroids"]);
    if (dict.contains("num_second_pass"))
        opts.num_second_pass = nb::cast<size_t>(dict["num_second_pass"]);
    if (dict.contains("n_probe"))
        opts.n_probe = nb::cast<size_t>(dict["n_probe"]);
    if (dict.contains("nearest_tokens_to_fetch"))
        opts.nearest_tokens_to_fetch =
                nb::cast<size_t>(dict["nearest_tokens_to_fetch"]);
    if (dict.contains("colbert_field"))
        opts.colbert_field = nb::cast<std::string>(dict["colbert_field"]);
    return opts;
}

// Wrapper functions for Field constructors
Field PyField(
        const std::string& name,
        DataType data_type,
        const std::vector<FieldType>& field_types,
        const nb::dict& params) {
    FieldParameters fp = create_field_parameters_from_dict(params);
    return Field(name, data_type, field_types, fp);
}

IndexedField PyIndexedField(
        const std::string& name,
        DataType data_type,
        const nb::dict& params) {
    FieldParameters fp = create_field_parameters_from_dict(params);
    return IndexedField(name, data_type, fp);
}

ContextField PyContextField(
        const std::string& name,
        DataType data_type,
        const nb::dict& params) {
    FieldParameters fp = create_field_parameters_from_dict(params);
    return ContextField(name, data_type, fp);
}

StoredField PyStoredField(
        const std::string& name,
        DataType data_type,
        const nb::dict& params) {
    FieldParameters fp = create_field_parameters_from_dict(params);
    return StoredField(name, data_type, fp);
}

ColbertField PyColbertField(
        const std::string& name,
        DataType data_type,
        const nb::dict& params) {
    FieldParameters fp = create_field_parameters_from_dict(params);
    return ColbertField(name, data_type, fp);
}

// Query wrappers
std::unique_ptr<QueryNode> CreateTermQueryNode(FieldValue& value) {
    return std::make_unique<TermQueryNode>(value);
}

std::unique_ptr<QueryNode> CreateVectorQueryNode(FieldValue& value) {
    return std::make_unique<VectorQueryNode>(value);
}

std::unique_ptr<QueryNode> CreateAndQueryNode(
        std::vector<std::unique_ptr<QueryNode>> its) {
    return std::make_unique<AndQueryNode>(std::move(its));
}

std::shared_ptr<CoarseQuantizer> CreateCoarseQuantizer(
        const nb::ndarray<float, nb::device::cpu> centroids) {
    std::vector<float> vec = std::vector<float>(
            centroids.data(), centroids.data() + centroids.size());
    return std::make_shared<CoarseQuantizer>(
            centroids.shape(1), vec, centroids.shape(0));
}

std::shared_ptr<FaissCoarseQuantizer> CreateFaissCoarseQuantizer(
        const nb::ndarray<float, nb::device::cpu> centroids) {
    std::vector<float> vec = std::vector<float>(
            centroids.data(), centroids.data() + centroids.size());
    return std::make_shared<FaissCoarseQuantizer>(
            centroids.shape(1), vec, centroids.shape(0));
}

std::shared_ptr<Binarizer> CreateBinarizer(
        const std::vector<float>& bucket_cutoffs,
        const std::vector<float>& bucket_weights,
        float avg_residual,
        size_t nbits,
        size_t dim) {
    return std::make_shared<Binarizer>(
            bucket_cutoffs, bucket_weights, avg_residual, nbits, dim);
}

NB_MODULE(core, m) {
    // Enum DataType
    nb::enum_<DataType>(m, "DataType")
            .value("TENSOR", DataType::TENSOR, "Tensor data type")
            .value("QUANTIZED_TENSOR",
                   DataType::QUANTIZED_TENSOR,
                   "Quantized tensor data type")
            .value("INTEGER", DataType::INTEGER, "Integer data type")
            .value("FLOAT", DataType::FLOAT, "Float data type")
            .value("TEXT", DataType::TEXT, "Text data type")
            .value("DATETIME", DataType::DATETIME, "Datetime data type")
            .value("COLBERT", DataType::COLBERT, "Colbert data type");

    nb::enum_<FieldType>(m, "FieldType")
            .value("Indexed", FieldType::Indexed, "Indexed field type")
            .value("Context", FieldType::Context, "Context field type")
            .value("Stored", FieldType::Stored, "Stored field type")
            .value("Colbert", FieldType::Colbert, "Colbert field type");

    nb::class_<FieldParameters>(
            m, "FieldParameters", "Field parameters for configuration")
            .

            def(nb::init<>(),

                "Default constructor")
            .def_rw("dimensions",
                    &FieldParameters::dimensions,
                    "Number of dimensions")
            .def_rw("analyzer", &FieldParameters::analyzer, "Analyzer type")
            .def_rw("quantization",
                    &FieldParameters::quantization,
                    "Quantization type")
            .def_rw("num_centroids",
                    &FieldParameters::num_centroids,
                    "Number of centroids")
            .def_rw("num_iterations",
                    &FieldParameters::num_iterations,
                    "Number of iterations")
            .def_rw("num_subquantizers",
                    &FieldParameters::num_subquantizers,
                    "Number of subquantizers")
            .def_rw("nbits", &FieldParameters::nbits, "Number of bits");

    nb::class_<Field>(m, "__Field", "Field configuration")
            .

            def(nb::init<>(),

                "Default constructor")
            .

            def(nb::init<
                        const std::string&,
                        const DataType,
                        std::vector<FieldType>,
                        const FieldParameters&>(),

                "Constructor with parameters")
            .def_rw("name", &Field::name, "Field name")
            .def_rw("data_type", &Field::data_type, "Field data type")
            .def_rw("field_types", &Field::field_types, "Field types")
            .def_rw("parameters", &Field::parameters, "Field parameters")
            .def("toJson", &Field::toJson, "Convert field to JSON")
            .def_static("fromJson", &Field::fromJson, "Create field from JSON")
            .def("add_field_type", &Field::add_field_type, "Add a field type");

    // We don't expose these classes directly, but they get returned by wrapper
    // functions. Struct IndexedField
    nb::class_<IndexedField, Field>(
            m, "__IndexedField", "Indexed field configuration")
            .

            def(nb::init<
                        const std::string&,
                        const DataType,
                        const FieldParameters&>(),

                "Constructor with parameters");

    // Struct ContextField
    nb::class_<ContextField, Field>(
            m, "__ContextField", "Context field configuration")
            .

            def(nb::init<
                        const std::string&,
                        const DataType,
                        const FieldParameters&>(),

                "Constructor with parameters");

    // Struct StoredField
    nb::class_<StoredField, Field>(
            m, "__StoredField", "Stored field configuration")
            .

            def(nb::init<
                        const std::string&,
                        const DataType,
                        const FieldParameters&>(),

                "Constructor with parameters");

    // Struct ColbertField
    nb::class_<ColbertField, Field>(
            m, "__ColbertField", "Colbert field configuration")
            .

            def(nb::init<
                        const std::string&,
                        const DataType,
                        const FieldParameters&>(),

                "Constructor with parameters");

    m.def("Field", &PyField, "Create a Field object");
    m.def("IndexedField", &PyIndexedField, "Create an IndexedField object");
    m.def("ContextField", &PyContextField, "Create a ContextField object");
    m.def("StoredField", &PyStoredField, "Create a StoredField object");
    m.def("ColbertField", &PyColbertField, "Create a ColbertField object");

    // Struct Schema
    nb::class_<Schema>(m, "Schema", "Schema configuration")
            .

            def(nb::init<>(),

                "Default constructor")
            .

            def(nb::init<const std::vector<Field>&>(),

                "Constructor with fields")
            .def_rw("fields", &Schema::fields, "Fields in the schema")
            .def("toJson", &Schema::toJson, "Convert schema to JSON")
            .def_static(
                    "fromJson", &Schema::fromJson, "Create schema from JSON")
            .def("add_field", &Schema::add_field, "Add a field to the schema");

    /**
     * DataTypes
     */
    // Type aliases
    using Tensor = std::vector<float>;
    using QuantizedTensor = std::vector<uint8_t>;

    // Duration and DateTime
    nb::class_<Duration>(m, "Duration", "Duration type");
    nb::class_<DateTime>(m, "DateTime", "DateTime type")
            .

            def(nb::init<>(),

                "Default constructor")
            .def("to_python", &toPythonDatetime, "Convert to Python datetime")
            .def_static(
                    "from_python",
                    &fromPythonDatetime,
                    "Convert from Python datetime");

    // Struct ColBERTContextData
    nb::class_<ColBERTContextData>(
            m, "ColBERTContextData", "ColBERT context data")
            .

            def(nb::init<>(),

                "Default constructor")
            .def_rw("doc_codes",
                    &ColBERTContextData::doc_codes,
                    "Document codes")
            .def_rw("doc_residuals",
                    &ColBERTContextData::doc_residuals,
                    "Document residuals");

    // SupportedTypes
    nb::class_<SupportedTypes>(m, "SupportedTypes", "Supported data types");

    // Struct FieldValue
    nb::class_<FieldValue>(
            m, "FieldValue", "FieldValue for storing different types of data.")
            .

            def(nb::init<>(),

                "Default constructor.")
            .

            def(nb::init<std::string, int>(),
                nb::arg("name"),
                nb::arg("value"),

                "Constructor with integer value.\n\n"
                ":param name: Field name.\n"
                ":param value: Integer value.")
            .

            def(nb::init<std::string, float>(),
                nb::arg("name"),
                nb::arg("value"),

                "Constructor with float value.\n\n"
                ":param name: Field name.\n"
                ":param value: Float value.")
            .

            def(nb::init<std::string, std::string>(),
                nb::arg("name"),
                nb::arg("value"),

                "Constructor with string value.\n\n"
                ":param name: Field name.\n"
                ":param value: String value.")
            .

            def(nb::init<std::string, DateTime>(),
                nb::arg("name"),
                nb::arg("value"),

                "Constructor with DateTime value.\n\n"
                ":param name: Field name.\n"
                ":param value: DateTime value.")
            .

            def(nb::init<std::string, Tensor>(),
                nb::arg("name"),
                nb::arg("value"),

                "Constructor with Tensor value.\n\n"
                ":param name: Field name.\n"
                ":param value: Tensor value.")
            .

            def(nb::init<std::string, Tensor, size_t>(),
                nb::arg("name"),
                nb::arg("value"),
                nb::arg("num_tensors"),

                "Constructor with Tensor value and number of tensors.\n\n"
                ":param name: Field name.\n"
                ":param value: Tensor value.\n"
                ":param num_tensors: Number of tensors.")
            .

            def(nb::init<std::string, QuantizedTensor, size_t>(),
                nb::arg("name"),
                nb::arg("value"),
                nb::arg("num_tensors"),

                "Constructor with QuantizedTensor value and number of tensors.\n\n"
                ":param name: Field name.\n"
                ":param value: QuantizedTensor value.\n"
                ":param num_tensors: Number of tensors.")
            .

            def(nb::init<std::string, ColBERTContextData, size_t>(),
                nb::arg("name"),
                nb::arg("value"),
                nb::arg("num_tensors"),

                "Constructor with ColBERTContextData value and number of tensors.\n\n"
                ":param name: Field name.\n"
                ":param value: ColBERTContextData value.\n"
                ":param num_tensors: Number of tensors.")
            .def_rw("name", &FieldValue::name, "Field name.")
            .def_rw("data_type", &FieldValue::data_type, "Field data type.")
            .def_rw("num_tensors",
                    &FieldValue::num_tensors,
                    "Number of tensors.")
            .def_rw("value", &FieldValue::value, "Field value.");

    // Wrapper functions
    m.def("IntFieldValue", &IntFieldValue, "Create FieldValue from integer");
    m.def("FloatFieldValue", &FloatFieldValue, "Create FieldValue from float");
    m.def("TextFieldValue", &TextFieldValue, "Create FieldValue from string");
    m.def("TensorFieldValue",
          &TensorFieldValue,
          "Create FieldValue from Tensor");
    m.def("QuantizedTensorFieldValue",
          &QuantizedTensorFieldValue,
          "Create FieldValue from QuantizedTensor");
    m.def("DateFieldValue", &DateFieldValue, "Create FieldValue from DateTime");

    nb::class_<Document>(
            m,
            "Document",
            "Document for storing multiple fields and a unique ID.")
            .

            def(nb::init<idx_t, const std::vector<FieldValue>&>(),
                nb::arg("id"),
                nb::arg("fields"),

                "Constructor with ID and fields.\n\n"
                ":param id: Unique ID of the document.\n"
                ":param fields: List of FieldValue objects.")
            .def_rw("id", &Document::id, "Unique ID of the document.")
            .def_rw("fields",
                    &Document::fields,
                    "List of FieldValue objects in the document.");
    /**
     * Search Options and Indexes
     */
    nb::class_<SearchOptions>(
            m,
            "SearchOptions",
            "SearchOptions enables custom searching behavior.\n\n"
            "These options expose ways to tradeoff recall and latency at different levels of retrieval.\n"
            "To search more centroids:\n"
            "- Decrease `centroid_score_threshold` and increase `k_top_centroids`.\n"
            "- Increase `n_probe` in search().\n"
            "To decrease latency:\n"
            "- Increase `centroid_score_threshold` and decrease `k_top_centroids`.\n"
            "- Decrease `n_probe` in search().")
            .

            def(nb::init<>(),

                "Default constructor")
            .def_rw("expected_id",
                    &SearchOptions::expected_id,
                    "Expects a document ID in the return result. Prints additional information during execution. Useful for debugging.")
            .def_rw("centroid_score_threshold",
                    &SearchOptions::centroid_score_threshold,
                    "The threshold for centroid scores. Lower values mean more centroids are considered.")
            .def_rw("k_top_centroids",
                    &SearchOptions::k_top_centroids,
                    "The number of top centroids to consider per token. Higher values mean more centroids are considered.")
            .def_rw("num_second_pass",
                    &SearchOptions::num_second_pass,
                    "The number of second pass candidates to consider. Higher values mean more candidates are considered.")
            .def_rw("n_probe",
                    &SearchOptions::n_probe,
                    "The number of centroids to search overall. Higher values mean more centroids are searched.")
            .def_rw("nearest_tokens_to_fetch",
                    &SearchOptions::nearest_tokens_to_fetch,
                    "The number of nearest tokens to fetch in XTR. Higher values mean more tokens are fetched.")
            .def_rw("colbert_field",
                    &SearchOptions::colbert_field,
                    "The field to use for ColBERT (Contextualized Late Interaction over BERT).");

    // Struct SearchResult
    nb::class_<SearchResult>(m, "SearchResult", "Search result")
            .

            def(nb::init<>(),

                "Default constructor")
            .def_rw("id", &SearchResult::id, "Document ID")
            .def_rw("score", &SearchResult::score, "Final score")
            .def_rw("metadata",
                    &SearchResult::metadata,
                    "Metadata for the document")
            .def_rw("token_scores",
                    &SearchResult::token_scores,
                    "Document token scores")
            .def("__lt__",
                 &SearchResult::operator<,
                 "Less than comparison operator")
            .def("__gt__",
                 &SearchResult::operator>,
                 "Greater than comparison operator");

    nb::class_<Configuration>(m, "Configuration", "Configuration for the index")
            .

            def(nb::init<>(),

                "Default constructor")
            .def_rw("lintdb_version",
                    &Configuration::lintdb_version,
                    "LintDB version")
            .def("__eq__",
                 &Configuration::operator==,
                 "Equality comparison operator");

    nb::class_<IndexIVF>(
            m,
            "IndexIVF",
            "IndexIVF is a multi-vector index with an inverted file structure.")
            .

            def(nb::init<const std::string&, bool>(),
                nb::arg("path"),
                nb::arg("read_only")

                = false,
                "Load an existing index.\n\n"
                ":param path: The path to the index.\n"
                ":param read_only: Whether to open the index in read-only mode.")
            .

            def(nb::init<
                        const std::string&,
                        const Schema&,
                        const Configuration&>(),
                nb::arg("path"),
                nb::arg("schema"),
                nb::arg("config"),

                "Create a new index with the given path, schema, and configuration.\n\n"
                ":param path: The path to initialize the index.\n"
                ":param schema: The schema for the index.\n"
                ":param config: The configuration for the index.")
            .

            def(nb::init<const IndexIVF&, const std::string&>(),
                nb::arg("other"),
                nb::arg("path"),

                "Create a copy of a trained index at the given path. The copy will always be writable.\n\n"
                "Throws an exception if the index isn't trained when this method is called.\n\n"
                ":param other: The other IndexIVF to copy.\n"
                ":param path: The path to initialize the new index.")
            .def("train",
                 &IndexIVF::train,
                 nb::arg("docs"),
                 "Train the index with the given documents to learn quantization and compression parameters.\n\n"
                 ":param docs: The documents to use for training.")
            .def("set_quantizer",
                 &IndexIVF::set_quantizer,
                 nb::arg("field"),
                 nb::arg("quantizer"),
                 "Set the quantizer for a field.\n\n"
                 ":param field: The field to set the quantizer for.\n"
                 ":param quantizer: The quantizer to set.")
            .def("set_coarse_quantizer",
                 &IndexIVF::set_coarse_quantizer,
                 nb::arg("field"),
                 nb::arg("quantizer"),
                 "Set the coarse quantizer for a field.\n\n"
                 ":param field: The field to set the coarse quantizer for.\n"
                 ":param quantizer: The coarse quantizer to set.")
            .def(
                    "search",
                    [](IndexIVF& self,
                       const uint64_t tenant,
                       const Query& query,
                       size_t k,
                       const nb::dict& opts_dict) {
                        SearchOptions opts =
                                create_search_options_from_dict(opts_dict);
                        return self.search(tenant, query, k, opts);
                    },
                    nb::arg("tenant"),
                    nb::arg("query"),
                    nb::arg("k"),
                    nb::arg("opts") = nb::dict(),

                    "Find the nearest neighbors for a vector block.\n\n"
                    ":param tenant: The tenant the document belongs to.\n"
                    ":param query: The query to search with.\n"
                    ":param k: The number of top results to return.\n"
                    ":param opts: Search options to use during searching.")
            .def("add",
                 &IndexIVF::add,
                 nb::arg("tenant"),
                 nb::arg("docs"),
                 "Add a block of embeddings to the index.\n\n"
                 ":param tenant: The tenant to assign the documents to.\n"
                 ":param docs: A vector of documents to add.")
            .def("add_single",
                 &IndexIVF::add_single,
                 nb::arg("tenant"),
                 nb::arg("doc"),
                 "Add a single document to the index.\n\n"
                 ":param tenant: The tenant to assign the document to.\n"
                 ":param doc: The document to add.")
            .def("remove",
                 &IndexIVF::remove,
                 nb::arg("tenant"),
                 nb::arg("ids"),
                 "Remove documents from the index by their IDs.\n\n"
                 ":param tenant: The tenant the documents belong to.\n"
                 ":param ids: The IDs of the documents to remove.")
            .def("update",
                 &IndexIVF::update,
                 nb::arg("tenant"),
                 nb::arg("docs"),
                 "Update documents in the index. This is a convenience function for remove and add.\n\n"
                 ":param tenant: The tenant the documents belong to.\n"
                 ":param docs: The documents to update.")
            .def("merge",
                 &IndexIVF::merge,
                 nb::arg("path"),
                 "Merge the index with another index.\n\n"
                 "This enables easier multiprocess building of indices but can have subtle issues if indices have different centroids.\n\n"
                 ":param path: The path to the other index.")
            .def("save",
                 &IndexIVF::save,
                 "Save the current state of the index. Quantization and compression will be saved within the Index's path.")
            .def("close",
                 &IndexIVF::close,
                 "Close the index, releasing any resources.")
            .def(
                    "__del__",
                    [](IndexIVF* self) { delete self; },
                    "Destructor to clean up resources.")
            .def_rw("config", &IndexIVF::config, "Configuration of the index.")
            .def_rw("read_only",
                    &IndexIVF::read_only,
                    "Flag indicating whether the index is read-only.");

    // Enum QuantizerType
    nb::enum_<QuantizerType>(
            m, "QuantizerType", "Enumeration of quantizer types.")
            .value("UNKNOWN", QuantizerType::UNKNOWN, "Unknown quantizer type.")
            .value("NONE", QuantizerType::NONE, "No quantizer.")
            .value("BINARIZER",
                   QuantizerType::BINARIZER,
                   "Binarizer quantizer.")
            .value("PRODUCT_ENCODER",
                   QuantizerType::PRODUCT_ENCODER,
                   "Product encoder quantizer.");

    // Bindings for Quantizer
    nb::class_<Quantizer>(
            m,
            "Quantizer",
            "Abstract Quantizer class providing an interface for quantization operations.")
            .def("train",
                 &Quantizer::train,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("dim"),
                 "Train the quantizer with the given data.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param dim: Dimension size.")
            .def("save",
                 &Quantizer::save,
                 nb::arg("path"),
                 "Save the quantizer to the specified path.\n\n"
                 ":param path: Path to save the quantizer.")
            .def("sa_encode",
                 &Quantizer::sa_encode,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("codes"),
                 "Encode the given data using the quantizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param codes: Pointer to encoded data.")
            .def("sa_decode",
                 &Quantizer::sa_decode,
                 nb::arg("n"),
                 nb::arg("codes"),
                 nb::arg("x"),
                 "Decode the given data using the quantizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param codes: Pointer to encoded data.\n"
                 ":param x: Pointer to decoded data.")
            .def("code_size",
                 &Quantizer::code_size,
                 "Get the size of the code.\n\n"
                 ":return: Size of the code.")
            .def("get_nbits",
                 &Quantizer::get_nbits,
                 "Get the number of bits.\n\n"
                 ":return: Number of bits.")
            .def("get_type",
                 &Quantizer::get_type,
                 "Get the type of quantizer.\n\n"
                 ":return: Quantizer type.");

    // Binarizer
    nb::class_<Binarizer, Quantizer>(
            m, "_Binarizer", "Binarizer class derived from Quantizer.")
            .

            def(nb::init<size_t, size_t>(),
                nb::arg("nbits"),
                nb::arg("dim"),

                "Constructor with nbits and dimension.\n\n"
                ":param nbits: Number of bits.\n"
                ":param dim: Dimension size.")
            .

            def(nb::init<
                        const std::vector<float>&,
                        const std::vector<float>&,
                        float,
                        size_t,
                        size_t>(),
                nb::arg("bucket_cutoffs"),
                nb::arg("bucket_weights"),
                nb::arg("avg_residual"),
                nb::arg("nbits"),
                nb::arg("dim"),

                "Constructor with bucket cutoffs, bucket weights, average residual, nbits, and dimension.\n\n"
                ":param bucket_cutoffs: Bucket cutoffs.\n"
                ":param bucket_weights: Bucket weights.\n"
                ":param avg_residual: Average residual.\n"
                ":param nbits: Number of bits.\n"
                ":param dim: Dimension size.")
            .def("binarize",
                 &Binarizer::binarize,
                 nb::arg("residuals"),
                 "Binarize the given residuals.\n\n"
                 ":param residuals: Vector of residuals to binarize.\n"
                 ":return: Binarized version of the residuals.")
            // we overwrite the train method to accept a numpy array
            .def(
                    "train",
                    [](Binarizer& self, nb::ndarray<float>& x) {
                        self.train(
                                x.shape(0),
                                x.

                                data(),
                                x

                                        .shape(1));
                    },
                    nb::arg("x"),
                    "Train the Binarizer with the given data.")
            .def("save",
                 &Binarizer::save,
                 nb::arg("path"),
                 "Save the Binarizer to the specified path.\n\n"
                 ":param path: Path to save the Binarizer.")
            .def("sa_encode",
                 &Binarizer::sa_encode,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("codes"),
                 "Encode the given data using the Binarizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param codes: Pointer to encoded data.")
            .def("sa_decode",
                 &Binarizer::sa_decode,
                 nb::arg("n"),
                 nb::arg("codes"),
                 nb::arg("x"),
                 "Decode the given data using the Binarizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param codes: Pointer to encoded data.\n"
                 ":param x: Pointer to decoded data.")
            .def("code_size",
                 &Binarizer::code_size,
                 "Get the size of the code.\n\n"
                 ":return: Size of the code.")
            .def("get_nbits",
                 &Binarizer::get_nbits,
                 "Get the number of bits.\n\n"
                 ":return: Number of bits.")
            .def_static(
                    "load",
                    &Binarizer::load,
                    nb::arg("path"),
                    "Load a Binarizer from the specified path.\n\n"
                    ":param path: Path to load the Binarizer from.\n"
                    ":return: Loaded Binarizer.")
            .def("get_type",
                 &Binarizer::get_type,
                 "Get the type of quantizer.\n\n"
                 ":return: Quantizer type.");

    m.def("Binarizer", &CreateBinarizer, "Create a Binarizer object");

    // Bindings for ICoarseQuantizer
    nb::class_<ICoarseQuantizer>(
            m,
            "ICoarseQuantizer",
            "Abstract ICoarseQuantizer class providing an interface for coarse quantization operations.")
            .def("train",
                 &ICoarseQuantizer::train,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("k"),
                 nb::arg("num_iter"),
                 "Train the coarse quantizer with the given data.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param k: Number of centroids.\n"
                 ":param num_iter: Number of iterations.")
            .def("save",
                 &ICoarseQuantizer::save,
                 nb::arg("path"),
                 "Save the coarse quantizer to the specified path.\n\n"
                 ":param path: Path to save the quantizer.")
            .def("assign",
                 &ICoarseQuantizer::assign,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("codes"),
                 "Assign the nearest centroids to the given data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param codes: Pointer to assigned codes.")
            .def("sa_decode",
                 &ICoarseQuantizer::sa_decode,
                 nb::arg("n"),
                 nb::arg("codes"),
                 nb::arg("x"),
                 "Decode the given codes to data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param codes: Pointer to codes.\n"
                 ":param x: Pointer to decoded data.")
            .def("compute_residual",
                 &ICoarseQuantizer::compute_residual,
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_id"),
                 "Compute the residual vector for a given data point and centroid.\n\n"
                 ":param vec: Pointer to data point.\n"
                 ":param residual: Pointer to residual vector.\n"
                 ":param centroid_id: Centroid ID.")
            .def("compute_residual_n",
                 &ICoarseQuantizer::compute_residual_n,
                 nb::arg("n"),
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_ids"),
                 "Compute the residual vectors for multiple data points and centroids.\n\n"
                 ":param n: Number of data points.\n"
                 ":param vec: Pointer to data points.\n"
                 ":param residual: Pointer to residual vectors.\n"
                 ":param centroid_ids: Pointer to centroid IDs.")
            .def("reconstruct",
                 &ICoarseQuantizer::reconstruct,
                 nb::arg("centroid_id"),
                 nb::arg("embedding"),
                 "Reconstruct the embedding for a given centroid.\n\n"
                 ":param centroid_id: Centroid ID.\n"
                 ":param embedding: Pointer to reconstructed embedding.")
            .def("search",
                 &ICoarseQuantizer::search,
                 nb::arg("num_query_tok"),
                 nb::arg("data"),
                 nb::arg("k_top_centroids"),
                 nb::arg("distances"),
                 nb::arg("coarse_idx"),
                 "Search for the nearest centroids to the given data points.\n\n"
                 ":param num_query_tok: Number of query tokens.\n"
                 ":param data: Pointer to data points.\n"
                 ":param k_top_centroids: Number of top centroids to search.\n"
                 ":param distances: Pointer to distances to nearest centroids.\n"
                 ":param coarse_idx: Pointer to coarse indices of nearest centroids.")
            .def("reset",
                 &ICoarseQuantizer::reset,
                 "Reset the coarse quantizer.")
            .def("add",
                 &ICoarseQuantizer::add,
                 nb::arg("n"),
                 nb::arg("data"),
                 "Add new data points to the coarse quantizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param data: Pointer to data points.")
            .def("code_size",
                 &ICoarseQuantizer::code_size,
                 "Get the size of the code.\n\n"
                 ":return: Size of the code.")
            .def("num_centroids",
                 &ICoarseQuantizer::num_centroids,
                 "Get the number of centroids.\n\n"
                 ":return: Number of centroids.")
            .def("get_xb",
                 &ICoarseQuantizer::get_xb,
                 "Get the centroids.\n\n"
                 ":return: Pointer to centroids.")
            .def("serialize",
                 &ICoarseQuantizer::serialize,
                 nb::arg("filename"),
                 "Serialize the coarse quantizer to a file.\n\n"
                 ":param filename: File name to serialize to.")
            .def("is_trained",
                 &ICoarseQuantizer::is_trained,
                 "Check if the coarse quantizer is trained.\n\n"
                 ":return: True if trained, False otherwise.");

    //// Bindings for CoarseQuantizer
    nb::class_<CoarseQuantizer, ICoarseQuantizer>(
            m,
            "_CoarseQuantizer",
            "CoarseQuantizer class derived from ICoarseQuantizer.")
            .

            def(nb::init<size_t>(),
                nb::arg("d"),

                "Constructor with dimensionality.\n\n"
                ":param d: Dimensionality of data points.")
            .

            def(nb::init<size_t, const std::vector<float>&, size_t>(),
                nb::arg("d"),
                nb::arg("centroids"),
                nb::arg("k"),

                "Constructor with dimensionality, centroids, and number of centroids.\n\n"
                ":param d: Dimensionality of data points.\n"
                ":param centroids: Initial centroids.\n"
                ":param k: Number of centroids.")
            //.def("train", &CoarseQuantizer::train, nb::arg("n"), nb::arg("x"),
            //nb::arg("k"), nb::arg("num_iter") = 10, "Train the quantizer with
            //the given data.\n\n"
            //":param n: Number of data points.\n"
            //":param x: Pointer to data.\n"
            //":param k: Number of centroids.\n"
            //":param num_iter: Number of iterations (default: 10).")
            .def(
                    "train",
                    [](CoarseQuantizer& self,
                       nb::ndarray<float, nb::ndim<2>, nb::device::cpu>& x,
                       size_t k,
                       size_t num_iter) {
                        self.train(
                                x.shape(0),
                                x.

                                data(),
                                k,
                                num_iter

                        );
                    },
                    nb::arg("x"),
                    nb::arg("k"),
                    nb::arg("num_iter") = 10,
                    "Train the CoarseQuantizer with the given data.")
            .def("save",
                 &CoarseQuantizer::save,
                 nb::arg("path"),
                 "Save the quantizer to the specified path.\n\n"
                 ":param path: Path to save the quantizer.")
            .def("assign",
                 &CoarseQuantizer::assign,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("codes"),
                 "Assign the nearest centroids to the given data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param codes: Pointer to assigned codes.")
            .def("sa_decode",
                 &CoarseQuantizer::sa_decode,
                 nb::arg("n"),
                 nb::arg("codes"),
                 nb::arg("x"),
                 "Decode the given codes to data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param codes: Pointer to codes.\n"
                 ":param x: Pointer to decoded data.")
            .def("compute_residual",
                 &CoarseQuantizer::compute_residual,
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_id"),
                 "Compute the residual vector for a given data point and centroid.\n\n"
                 ":param vec: Pointer to data point.\n"
                 ":param residual: Pointer to residual vector.\n"
                 ":param centroid_id: Centroid ID.")
            .def("compute_residual_n",
                 &CoarseQuantizer::compute_residual_n,
                 nb::arg("n"),
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_ids"),
                 "Compute the residual vectors for multiple data points and centroids.\n\n"
                 ":param n: Number of data points.\n"
                 ":param vec: Pointer to data points.\n"
                 ":param residual: Pointer to residual vectors.\n"
                 ":param centroid_ids: Pointer to centroid IDs.")
            .def("reconstruct",
                 &CoarseQuantizer::reconstruct,
                 nb::arg("centroid_id"),
                 nb::arg("embedding"),
                 "Reconstruct the embedding for a given centroid.\n\n"
                 ":param centroid_id: Centroid ID.\n"
                 ":param embedding: Pointer to reconstructed embedding.")
            .def("search",
                 &CoarseQuantizer::search,
                 nb::arg("num_query_tok"),
                 nb::arg("data"),
                 nb::arg("k_top_centroids"),
                 nb::arg("distances"),
                 nb::arg("coarse_idx"),
                 "Search for the nearest centroids to the given data points.\n\n"
                 ":param num_query_tok: Number of query tokens.\n"
                 ":param data: Pointer to data points.\n"
                 ":param k_top_centroids: Number of top centroids to search.\n"
                 ":param distances: Pointer to distances to nearest centroids.\n"
                 ":param coarse_idx: Pointer to coarse indices of nearest centroids.")
            .def("reset", &CoarseQuantizer::reset, "Reset the quantizer.")
            .def("add",
                 &CoarseQuantizer::add,
                 nb::arg("n"),
                 nb::arg("data"),
                 "Add new data points to the quantizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param data: Pointer to data points.")
            .def("code_size",
                 &CoarseQuantizer::code_size,
                 "Get the size of the code.\n\n"
                 ":return: Size of the code.")
            .def("num_centroids",
                 &CoarseQuantizer::num_centroids,
                 "Get the number of centroids.\n\n"
                 ":return: Number of centroids.")
            .def("get_xb",
                 &CoarseQuantizer::get_xb,
                 "Get the centroids.\n\n"
                 ":return: Pointer to centroids.")
            .def("serialize",
                 &CoarseQuantizer::serialize,
                 nb::arg("filename"),
                 "Serialize the quantizer to a file.\n\n"
                 ":param filename: File name to serialize to.")
            .def_static(
                    "deserialize",
                    &CoarseQuantizer::deserialize,
                    nb::arg("filename"),
                    nb::arg("version"),
                    "Deserialize a quantizer from a file.\n\n"
                    ":param filename: File name to deserialize from.\n"
                    ":param version: Version of the quantizer.\n"
                    ":return: Deserialized quantizer.")
            .def("is_trained",
                 &CoarseQuantizer::is_trained,
                 "Check if the quantizer is trained.\n\n"
                 ":return: True if trained, False otherwise.");

    //// Bindings for FaissCoarseQuantizer
    nb::class_<FaissCoarseQuantizer, ICoarseQuantizer>(
            m,
            "_FaissCoarseQuantizer",
            "FaissCoarseQuantizer class derived from ICoarseQuantizer.")
            .

            def(nb::init<size_t>(),
                nb::arg("d"),

                "Constructor with dimensionality.\n\n"
                ":param d: Dimensionality of data points.")
            .

            def(nb::init<size_t, const std::vector<float>&, size_t>(),
                nb::arg("d"),
                nb::arg("centroids"),
                nb::arg("k"),

                "Constructor with dimensionality, centroids, and number of centroids.\n\n"
                ":param d: Dimensionality of data points.\n"
                ":param centroids: Initial centroids.\n"
                ":param k: Number of centroids.")
            //.def("train", &CoarseQuantizer::train, nb::arg("n"), nb::arg("x"),
            //nb::arg("k"), nb::arg("num_iter") = 10, "Train the quantizer with
            //the given data.\n\n"
            //":param n: Number of data points.\n"
            //":param x: Pointer to data.\n"
            //":param k: Number of centroids.\n"
            //":param num_iter: Number of iterations (default: 10).")
            .def(
                    "train",
                    [](FaissCoarseQuantizer& self,
                       nb::ndarray<float, nb::ndim<2>, nb::device::cpu>& x,
                       size_t k,
                       size_t num_iter) {
                        self.train(
                                x.shape(0),
                                x.

                                data(),
                                k,
                                num_iter

                        );
                    },
                    nb::arg("x"),
                    nb::arg("k"),
                    nb::arg("num_iter") = 10,
                    "Train the FaissCoarseQuantizer with the given data.")
            .def("save",
                 &FaissCoarseQuantizer::save,
                 nb::arg("path"),
                 "Save the quantizer to the specified path.\n\n"
                 ":param path: Path to save the quantizer.")
            .def("assign",
                 &FaissCoarseQuantizer::assign,
                 nb::arg("n"),
                 nb::arg("x"),
                 nb::arg("codes"),
                 "Assign the nearest centroids to the given data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param x: Pointer to data.\n"
                 ":param codes: Pointer to assigned codes.")
            .def("sa_decode",
                 &FaissCoarseQuantizer::sa_decode,
                 nb::arg("n"),
                 nb::arg("codes"),
                 nb::arg("x"),
                 "Decode the given codes to data points.\n\n"
                 ":param n: Number of data points.\n"
                 ":param codes: Pointer to codes.\n"
                 ":param x: Pointer to decoded data.")
            .def("compute_residual",
                 &FaissCoarseQuantizer::compute_residual,
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_id"),
                 "Compute the residual vector for a given data point and centroid.\n\n"
                 ":param vec: Pointer to data point.\n"
                 ":param residual: Pointer to residual vector.\n"
                 ":param centroid_id: Centroid ID.")
            .def("compute_residual_n",
                 &FaissCoarseQuantizer::compute_residual_n,
                 nb::arg("n"),
                 nb::arg("vec"),
                 nb::arg("residual"),
                 nb::arg("centroid_ids"),
                 "Compute the residual vectors for multiple data points and centroids.\n\n"
                 ":param n: Number of data points.\n"
                 ":param vec: Pointer to data points.\n"
                 ":param residual: Pointer to residual vectors.\n"
                 ":param centroid_ids: Pointer to centroid IDs.")
            .def("reconstruct",
                 &FaissCoarseQuantizer::reconstruct,
                 nb::arg("centroid_id"),
                 nb::arg("embedding"),
                 "Reconstruct the embedding for a given centroid.\n\n"
                 ":param centroid_id: Centroid ID.\n"
                 ":param embedding: Pointer to reconstructed embedding.")
            .def("search",
                 &FaissCoarseQuantizer::search,
                 nb::arg("num_query_tok"),
                 nb::arg("data"),
                 nb::arg("k_top_centroids"),
                 nb::arg("distances"),
                 nb::arg("coarse_idx"),
                 "Search for the nearest centroids to the given data points.\n\n"
                 ":param num_query_tok: Number of query tokens.\n"
                 ":param data: Pointer to data points.\n"
                 ":param k_top_centroids: Number of top centroids to search.\n"
                 ":param distances: Pointer to distances to nearest centroids.\n"
                 ":param coarse_idx: Pointer to coarse indices of nearest centroids.")
            .def("reset", &FaissCoarseQuantizer::reset, "Reset the quantizer.")
            .def("add",
                 &FaissCoarseQuantizer::add,
                 nb::arg("n"),
                 nb::arg("data"),
                 "Add new data points to the quantizer.\n\n"
                 ":param n: Number of data points.\n"
                 ":param data: Pointer to data points.")
            .def("code_size",
                 &FaissCoarseQuantizer::code_size,
                 "Get the size of the code.\n\n"
                 ":return: Size of the code.")
            .def("num_centroids",
                 &FaissCoarseQuantizer::num_centroids,
                 "Get the number of centroids.\n\n"
                 ":return: Number of centroids.")
            .def("get_xb",
                 &FaissCoarseQuantizer::get_xb,
                 "Get the centroids.\n\n"
                 ":return: Pointer to centroids.")
            .def("serialize",
                 &FaissCoarseQuantizer::serialize,
                 nb::arg("filename"),
                 "Serialize the quantizer to a file.\n\n"
                 ":param filename: File name to serialize to.")
            .def_static(
                    "deserialize",
                    &FaissCoarseQuantizer::deserialize,
                    nb::arg("filename"),
                    nb::arg("version"),
                    "Deserialize a quantizer from a file.\n\n"
                    ":param filename: File name to deserialize from.\n"
                    ":param version: Version of the quantizer.\n"
                    ":return: Deserialized quantizer.")
            .def("is_trained",
                 &FaissCoarseQuantizer::is_trained,
                 "Check if the quantizer is trained.\n\n"
                 ":return: True if trained, False otherwise.");

    // m.def("CoarseQuantizer", &CreateCoarseQuantizer, nb::arg("centroids"));
    m.def("FaissCoarseQuantizer",
          &CreateFaissCoarseQuantizer,
          nb::arg("centroids"));

    // Version
    nb::class_<Version>(
            m,
            "Version",
            "Version class representing LintDB's version with major, minor, revision, and build numbers.")
            .

            def(nb::init<>(),

                "Default constructor.")
            .

            def(nb::init<std::string>(),
                nb::arg("versionStr"),

                "Constructor with version string.\n\n"
                ":param versionStr: Version string in the format 'major.minor.revision'.")
            .def("__eq__",
                 &Version::operator==,
                 nb::arg("otherVersion"),
                 "Equality comparison operator.\n\n"
                 ":param otherVersion: The other version to compare with.\n"
                 ":return: True if versions are equal, False otherwise.")
            .def("__lt__",
                 &Version::operator<,
                 nb::arg("otherVersion"),
                 "Less than comparison operator.\n\n"
                 ":param otherVersion: The other version to compare with.\n"
                 ":return: True if this version is less than the other version, False otherwise.")
            .def_rw("metadata_enabled",
                    &Version::metadata_enabled,
                    "Flag indicating if metadata is enabled.")
            .def_rw("major", &Version::major, "Major version number.")
            .def_rw("minor", &Version::minor, "Minor version number.")
            .def_rw("revision", &Version::revision, "Revision number.")
            .def_rw("build", &Version::build, "Build number.");

    /**
     * Query nodes
     */
    nb::enum_<QueryNodeType>(m, "QueryNodeType", "Types of query nodes.")
            .value("TERM", QueryNodeType::TERM, "Term query node.")
            .value("VECTOR", QueryNodeType::VECTOR, "Vector query node.")
            .value("AND", QueryNodeType::AND, "AND query node.");

    nb::class_<QueryNode>(m, "__QueryNode", "Base class for query nodes.");

    // nb::class_<TermQueryNode, QueryNode>(m,
    //"__TermQueryNode", "Term query node.")
    //.def(nb::init<FieldValue &>(), nb::arg("value"),
    //"Constructor with FieldValue.\n\n"
    //":param value: FieldValue instance.");
    //
    // nb::class_<MultiQueryNode, QueryNode>(m, "__MultiQueryNode", "Multi query
    // node.") .def(nb::init<QueryNodeType>(), nb::arg("op"), "Constructor with
    //query node type.\n\n"
    //":param op: Query node type.");
    //
    // nb::class_<VectorQueryNode, QueryNode>(m, "__VectorQueryNode", "Vector
    // query node.") .def(nb::init<FieldValue&>(), nb::arg("value"),
    //"Constructor with FieldValue.\n\n"
    //":param value: FieldValue instance.");
    //
    // nb::class_<AndQueryNode, MultiQueryNode>(m, "__AndQueryNode", "AND query
    // node.") .def(nb::init<std::vector<std::unique_ptr<QueryNode>>>(),
    //nb::arg("its"), "Constructor with a list of child query nodes.\n\n"
    //":param its: List of child query nodes.");

    m.def("AndQueryNode", &CreateAndQueryNode, nb::arg("its"));
    m.def("TermQueryNode", &CreateTermQueryNode, nb::arg("value"));
    m.def("VectorQueryNode", &CreateVectorQueryNode, nb::arg("value"));

    nb::class_<Query>(m, "Query", "Query object containing a root query node.")
            .def(nb::init<std::unique_ptr<QueryNode>>(),
                 nb::arg("root"),
                 "Constructor with a unique pointer to the root query node.\n\n"
                 ":param root: Unique pointer to the root query node.");
} // namespace lintdb

} // namespace lintdb