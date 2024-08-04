#include "DocEncoder.h"
#include <vector>
#include <variant>
#include <map>
#include <unordered_set>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include "lintdb/schema/DataTypes.h"
#include <glog/logging.h>
#include "lintdb/invlists/KeyBuilder.h"


namespace lintdb {
    std::vector<PostingData> DocEncoder::encode_inverted_data(const ProcessedData &data, size_t code_size) {
        std::vector<std::string> keys;

        using Buffer = std::vector<uint8_t>;
        using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

        std::vector<std::string> values;
        switch(data.value.data_type) {
            case DataType::TENSOR: {
                // a raw tensor should never be here. it means that it skipped quantization.
                // all tensors should look like QuantizedTensor going into the index.
                LOG(WARNING) << "Datatype is TENSOR. This should not happen.";

                assert(data.value.num_tensors == data.centroid_ids.size());

                std::map<idx_t, std::vector<idx_t>> centroid_to_tokens;
                for(idx_t i = 0; i < data.centroid_ids.size(); i++) {
                    centroid_to_tokens[data.centroid_ids[i]].push_back(i);
                }

                Tensor tensor_arr = std::get<Tensor>(data.value.value);

                for(const auto& [centroid_id, token_ids] : centroid_to_tokens) {
                    std::string key = create_index_id(data.tenant, data.field, DataType::TENSOR, centroid_id,
                                                      data.doc_id);

                    keys.push_back(key);

                    Tensor centroid_data;
                    for (size_t i = 0; i < token_ids.size(); i++) {
                        idx_t token_id = token_ids[i];

                        centroid_data.insert(
                                centroid_data.end(),
                                tensor_arr.data() + token_id * code_size,
                                tensor_arr.data() + (token_id + 1) * code_size);

                    }
                    Buffer buf;
                    SupportedTypes tt = tensor_arr;
                    auto written = bitsery::quickSerialization(OutputAdapter{buf}, tt);
                    auto st = std::string(buf.begin(), buf.begin() + written);
                    values.push_back(st);

                }
                break;
            }

            case DataType::QUANTIZED_TENSOR: {
                assert(data.value.num_tensors == data.centroid_ids.size());

                std::map<idx_t, std::vector<idx_t>> centroid_to_tokens;
                for(idx_t i = 0; i < data.centroid_ids.size(); i++) {
                    centroid_to_tokens[data.centroid_ids[i]].push_back(i);
                }

                QuantizedTensor tensor_arr = std::get<QuantizedTensor>(data.value.value);

                for(const auto& [centroid_id, token_ids] : centroid_to_tokens) {
                    std::string key = create_index_id(data.tenant, data.field, DataType::QUANTIZED_TENSOR, centroid_id, data.doc_id);

                    keys.push_back(key);

                    QuantizedTensor centroid_data;
                    for (size_t i = 0; i < token_ids.size(); i++) {
                        idx_t token_id = token_ids[i];

                        centroid_data.insert(
                                centroid_data.end(),
                                tensor_arr.data() + token_id * code_size,
                                tensor_arr.data() + (token_id + 1) * code_size);

                    }
                    Buffer buf;
                    SupportedTypes tt = tensor_arr;
                    auto written = bitsery::quickSerialization(OutputAdapter{buf}, tt);
                    auto st = std::string(buf.begin(), buf.begin() + written);
                    values.push_back(st);

                }
                break;
            }
            case DataType::COLBERT: {
                assert(data.value.num_tensors == data.centroid_ids.size());

                std::map<idx_t, std::vector<idx_t>> centroid_to_tokens;
                for(idx_t i = 0; i < data.centroid_ids.size(); i++) {
                    centroid_to_tokens[data.centroid_ids[i]].push_back(i);
                }

                for(const auto& [centroid_id, token_ids] : centroid_to_tokens) {
                    std::string key = create_index_id(data.tenant, data.field, DataType::QUANTIZED_TENSOR, centroid_id, data.doc_id);

                    keys.push_back(key);
                    // ColBERT implicitly skips values. This is because the value isn't stored in the inverted index, but we need
                    // to build a posting list for it.
                    values.emplace_back("");
                }
                break;
            }
            case DataType::DATETIME:{

                DateTime dv = std::get<DateTime>(data.value.value);

                std::string key = create_index_id(data.tenant, data.field, DataType::DATETIME, dv, data.doc_id);

                keys.push_back(key);

                Buffer buf;
                size_t written = bitsery::quickSerialization(OutputAdapter{buf}, data.value.value);

                auto st = std::string(buf.begin(), buf.begin() + written);
                values.push_back(st);

                break;
            }
            case DataType::FLOAT: {
                std::string key = create_index_id(data.tenant, data.field, DataType::FLOAT, data.value.value, data.doc_id);

                keys.push_back(key);

                Buffer buf;
                size_t written = bitsery::quickSerialization(OutputAdapter{buf}, data.value.value);

                auto st = std::string(buf.begin(), buf.begin() + written);
                values.push_back(st);

                break;
            }
            case DataType::INTEGER: {
                std::string key = create_index_id(data.tenant, data.field, DataType::INTEGER, data.value.value, data.doc_id);

                keys.push_back(key);

                Buffer buf;
                size_t written = bitsery::quickSerialization(OutputAdapter{buf}, data.value.value);

                auto st = std::string(buf.begin(), buf.begin() + written);
                values.push_back(st);

                break;
            }
            case DataType::TEXT: {
                std::string key = create_index_id(data.tenant, data.field, DataType::TEXT, data.value.value, data.doc_id);

                keys.push_back(key);

                Buffer buf;
                size_t written = bitsery::quickSerialization(OutputAdapter{buf}, data.value.value);

                auto st = std::string(buf.begin(), buf.begin() + written);
                values.push_back(st);
                break;
            }
        };

        assert(keys.size() == values.size());

        std::vector<PostingData> results;
        for(size_t i = 0; i < keys.size(); i++) {
            results.push_back({keys[i], values[i]});
        }
        return results;
    }

    std::vector<PostingData> DocEncoder::encode_inverted_mapping_data(const ProcessedData& data) {
        std::vector<PostingData> results;

        std::string key = create_forward_index_id(data.tenant, data.doc_id);

        using Buffer = std::vector<uint8_t>;
        using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

        // create unique list of inverted index ids.
        std::unordered_set<idx_t> inverted_mapping_ids;
        for(const auto& id : data.centroid_ids) {
            inverted_mapping_ids.insert(id);
        }

        Buffer buf;
        auto written = bitsery::quickSerialization(OutputAdapter{buf}, std::vector<idx_t>(inverted_mapping_ids.begin(), inverted_mapping_ids.end()));
        auto st = std::string(buf.begin(), buf.begin() + written);

        results.push_back({key, st});

        return results;
    }

    // Encode the data for the forward index. There's only one key-value pair per document.
    PostingData DocEncoder::encode_forward_data(const std::vector<ProcessedData>& data) {
        if (data.size() == 0) {
            return {};
        }
        std::string key = create_forward_index_id(data[0].tenant, data[0].doc_id);

        using Buffer = std::vector<uint8_t>;
        using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

        std::map<uint8_t, SupportedTypes> forward_data;
        for(const auto& processed : data) {
            forward_data[processed.field] = processed.value.value;
        }

        Buffer buf;
        auto written = bitsery::quickSerialization(OutputAdapter{buf}, forward_data);
        auto st = std::string(buf.begin(), buf.begin() + written);

        return PostingData{key, st};

    }

    // Encode the data for the context index. There's one key-value pair per context feature.
    PostingData DocEncoder::encode_context_data(const ProcessedData& data) {
        std::string key = create_context_id(data.tenant, data.field, data.doc_id);

        using Buffer = std::vector<uint8_t>;
        using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

        Buffer buf;
        auto written = bitsery::quickSerialization(OutputAdapter{buf}, data.value.value);
        auto st = std::string(buf.begin(), buf.begin() + written);

        return PostingData{key, st};

    }

    SupportedTypes DocEncoder::decode_supported_types(std::string& data) {
        using Buffer = std::string;
        using InputAdapter = bitsery::InputBufferAdapter<Buffer>;

        SupportedTypes res;
        auto state = bitsery::quickDeserialization<InputAdapter>({data.begin(), data.size()}, res);
        assert(state.first == bitsery::ReaderError::NoError && state.second);

        return res;
    }

    std::map<uint8_t, SupportedTypes> DocEncoder::decode_forward_data(std::string& data) {
        using Buffer = std::string;
        using InputAdapter = bitsery::InputBufferAdapter<Buffer>;

        std::map<uint8_t, SupportedTypes> res;
        auto state = bitsery::quickDeserialization<InputAdapter>({data.begin(), data.size()}, res);
        assert(state.first == bitsery::ReaderError::NoError && state.second);

        return res;
    }

    std::vector<idx_t> DocEncoder::decode_inverted_mapping_data(std::string& data) {
        using Buffer = std::string;
        using InputAdapter = bitsery::InputBufferAdapter<Buffer>;

        std::vector<idx_t> res;
        auto state = bitsery::quickDeserialization<InputAdapter>({data.begin(), data.size()}, res);
        assert(state.first == bitsery::ReaderError::NoError && state.second);

        return res;
    }

} // lintdb