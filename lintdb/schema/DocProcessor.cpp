#include "DocumentProcessor.h"

namespace lintdb {

DocumentProcessor::DocumentProcessor(
        const Schema& schema,
        std::unordered_map<std::string, std::shared_ptr<Quantizer>> quantizer_map,
        std::unordered_map<std::string, std::shared_ptr<CoarseQuantizer>> coarse_quantizer_map
    ) : schema(schema), coarse_quantizer(coarse_quantizer), quantizer(quantizer) {
    for (const auto& field : schema.fields) {
        field_map[field.name] = field;
    }
}

EncodedDocument DocumentProcessor::processDocument(const Document& document) {
    EncodedDocument encodedDoc;

    for (const auto& [name, value] : document.fields) {
        if (field_map.find(name) == field_map.end()) {
            throw std::invalid_argument(
                    "Field " + name + " not defined in schema.");
        }
        const Field& field = field_map[name];
        validateField(field, value);

        std::vector<idx_t> centroids;
        if (field.data_type == DataType::TENSOR || field.data_type == DataType::TENSOR_ARRAY) {
            centroids = assignIVFCentroids(field, value);
            encodedDoc.addCentroids(name, centroids);
        }

        FieldValue quantizedValue = quantizeField(field, value);
        encodedDoc.addField(name, encodeField(field, quantizedValue));
    }

    return encodedDoc;
}

std::vector<idx_t> DocumentProcessor::assignIVFCentroids(const Field& field, const FieldValue& value) {
    auto encoder = coarse_quantizer_map[field.name];
    if (field.data_type == DataType::TENSOR) {
        Tensor tensor = std::get<Tensor>(value.value);
        return encoder->assignCentroids(tensor, field.parameters.dimensions);
    } else if (field.data_type == DataType::TENSOR_ARRAY) {
        TensorArray tensorArray = std::get<TensorArray>(value.value);
        return encoder->assignCentroidsArray(tensorArray, field.parameters.dimensions);
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

std::string DocumentProcessor::encodeField(
        const Field& field,
        const FieldValue& value) {
    std::ostringstream oss;
    bitsery::Serializer<bitsery::OutputBufferedStreamAdapter> ser(oss);

    // Serialize the FieldValue
    ser.object(value);
    ser.adapter().flush();

    return oss.str();
}

void DocumentProcessor::serializeEncodedDocument(const EncodedDocument& encodedDoc, InvertedIndexDocument& invertedDoc, ForwardIndexDocument& forwardDoc, ContextIndexDocument& contextDoc) {
    for (const auto& [name, value] : encodedDoc.encoded_fields) {
        const Field& field = field_map[name];
        if (field.field_type == Context) {
            contextDoc.addField(name, value);
        }
        if (field.field_type == Indexed) {
            invertedDoc.addField(name, value);
        }
        if (field.field_type == Stored) {
            forwardDoc.addField(name, value);
        }
    }
}

FieldValue DocumentProcessor::quantizeField(const Field& field, const FieldValue& value) {
    if (field.parameters.quantization_type == QuantizationType::NONE) {
        return value; // No quantization needed
    }

    if (field.data_type == DataType::TENSOR) {
        Tensor tensor = std::get<Tensor>(value.value);
        QuantizedTensor quantized = quantizer.quantizeTensor(tensor, field.parameters.quantization_type);
        return FieldValue(quantized);
    } else if (field.data_type == DataType::TENSOR_ARRAY) {
        size_t numTensors = tensorArray.size() / dim;
        TensorArray tensorArray = std::get<TensorArray>(value.value);
        for(size_t i = 0; i < numTensors; ++i) {
            gsl::span<float> tensor(tensorArray.data() + i * dim, tensorArray.data() + (i + 1) * dim);
            quantizedArray.push_back(quantizeTensor(tensor, type));
        }
        return FieldValue(quantizedArray);
    }
    return value; // No quantization needed for other data types
}

}
