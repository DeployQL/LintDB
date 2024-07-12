#include "DocProcessor.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <bitsery/adapter/buffer.h>
#include "lintdb/invlists/keys.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/invlists/PostingData.h"
#include "lintdb/schema/DocEncoder.h"
#include <glog/logging.h>

namespace lintdb {

DocumentProcessor::DocumentProcessor(
        const Schema& schema,
        const std::unordered_map<std::string, std::shared_ptr<Quantizer>>& quantizer_map,
        const std::unordered_map<std::string, std::shared_ptr<CoarseQuantizer>>& coarse_quantizer_map,
        const std::shared_ptr<FieldMapper> field_mapper,
        std::unique_ptr<IndexWriter> index_writer
    ) : schema(schema), field_mapper(field_mapper), quantizer_map(quantizer_map), coarse_quantizer_map(coarse_quantizer_map), index_writer(std::move(index_writer)) {
    for (const auto& field : schema.fields) {
        field_map[field.name] = field;
    }
}

void DocumentProcessor::processDocument(const uint64_t tenant, const Document& document) {
    std::vector<ProcessedData> inverted_data;
    std::vector<ProcessedData> context_data;
    std::vector<ProcessedData> stored_data;
    std::vector<ProcessedData> colbert_data;

    for (const auto& [name, value] : document.fields) {
        if (field_map.find(name) == field_map.end()) {
            throw std::invalid_argument(
                    "Field " + name + " not defined in schema.");
        }
        const Field& field = field_map[name];
        validateField(field, value);

        FieldValue quantizedValue = quantizeField(field, value);

        ProcessedData processed_data;
        processed_data.centroid_ids = assignIVFCentroids(field, value);

        processed_data.value = quantizedValue;
        processed_data.doc_id = document.id;
        processed_data.tenant = tenant;

        uint8_t field_id = field_mapper->getFieldID(field.name);
        processed_data.field = field_id;

        for(const FieldType type : field.field_types) {
            switch(type) {
                case FieldType::Colbert:
                    colbert_data.push_back(processed_data);
                    break;
                case FieldType::Indexed:
                    inverted_data.push_back(processed_data);
                    break;
                case FieldType::Context:
                    context_data.push_back(processed_data);
                    break;
                case FieldType::Stored:
                    stored_data.push_back(processed_data);
                    break;
            }
        }
    }

    BatchPostingData posting_data;
    //process colbert data.
    for(ProcessedData& data: colbert_data) {
        // store all of the token codes in the context index.
        auto cd = DocEncoder::encode_context_data(data);
        posting_data.context.push_back(cd);

        // quantizers must exist for colbert data. either an identity quantizer or otherwise.
        size_t code_size = quantizer_map.at(field_mapper->getFieldName(data.field))->code_size();
        // for colbert fields, don't store data into the inverted index itself. we'll strip that out.
        data.value = FieldValue("");

        std::vector<PostingData> encoded_data = DocEncoder::encode_inverted_data(data, code_size);
        // for colbert data types, we strip the value out of the inverted index data.

        posting_data.inverted.reserve(posting_data.inverted.size() + encoded_data.size());
        posting_data.inverted.insert(posting_data.inverted.end(), encoded_data.begin(), encoded_data.end());

        // now, we store all of the values together in the context field.

    }

    // inverted data produces multiple posting data elements -- one for each centroid assigned to the vectors.
    for(ProcessedData& data : inverted_data) {
        // TODO(mbarta): code size is only used for tensors. we should be able to split this out at some point.
        size_t code_size = 0;
        if(data.value.data_type == DataType::TENSOR || data.value.data_type == DataType::QUANTIZED_TENSOR) {
            code_size = quantizer_map.at(field_mapper->getFieldName(data.field))->code_size();
        }

        std::vector<PostingData> encoded_data = DocEncoder::encode_inverted_data(data, code_size);

        posting_data.inverted.reserve(posting_data.inverted.size() + encoded_data.size());
        posting_data.inverted.insert(posting_data.inverted.end(), encoded_data.begin(), encoded_data.end());

        std::vector<PostingData> mapping_data = DocEncoder::encode_inverted_mapping_data(data);
        posting_data.inverted_mapping.insert(posting_data.inverted_mapping.end(), mapping_data.begin(), mapping_data.end());
    }

    posting_data.context.reserve(posting_data.context.size() + context_data.size());
    for(const ProcessedData& data : context_data) {
        auto cd = DocEncoder::encode_context_data(data);
        posting_data.context.push_back(cd);
    }

    PostingData forward_data = DocEncoder::encode_forward_data(stored_data);
    posting_data.forward = forward_data;

    index_writer->write(posting_data);

}

std::vector<idx_t> DocumentProcessor::assignIVFCentroids(const Field& field, const FieldValue& value) {
    if (field.data_type == DataType::TENSOR || field.data_type == DataType::QUANTIZED_TENSOR) {
        std::shared_ptr<CoarseQuantizer> encoder = coarse_quantizer_map.at(field.name);
        assert(encoder->is_trained);

        Tensor tensor = std::get<Tensor>(value.value);
        std::vector<idx_t> centroids(value.num_tensors);
        encoder->assign(value.num_tensors, tensor.data(), centroids.data());

        return centroids;
    }
    return {};
}

void DocumentProcessor::validateField(
        const Field& field,
        const FieldValue& value) {
    if (field.data_type != value.data_type) {
        throw std::invalid_argument("Field " + field.name + " type mismatch.");
    }
    // Add further validation based on FieldParameters if necessary
}

FieldValue DocumentProcessor::quantizeField(const Field& field, const FieldValue& value) {
    if (field.data_type == DataType::TENSOR || field.data_type == DataType::QUANTIZED_TENSOR) {
        // Check if quantizer exists for the field
        LINTDB_THROW_IF_NOT(quantizer_map.count(field.name) > 0);

        std::shared_ptr<Quantizer> quantizer = quantizer_map.at(field.name);

        Tensor tensor = std::get<Tensor>(value.value);
        std::vector<residual_t > codes(value.num_tensors * quantizer->code_size());
        quantizer->sa_encode(value.num_tensors, tensor.data(), codes.data());
        return {codes, value.num_tensors};
    } else {
        return value; // No quantization needed for other data types
    }
}

}
