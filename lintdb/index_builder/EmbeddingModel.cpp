#include "lintdb/index_builder/EmbeddingModel.h"
#include "lintdb/exception.h"
#include <glog/logging.h>
#include <onnxruntime_cxx_api.h>
#include <cassert>

namespace lintdb {
    std::string print_shape(const std::vector<std::int64_t>& v) {
        std::stringstream ss("");
        for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
        ss << v[v.size() - 1];
        return ss.str();
    }

    Ort::Value vec_to_tensor(std::vector<int64_t>& data, const std::vector<std::int64_t>& shape) {
        auto mem_info_att =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<int64_t> att_shape = {1, static_cast<std::int64_t>(data.size())};

        auto att_tensor = Ort::Value::CreateTensor<int64_t>(mem_info_att, data.data(), data.size(), shape.data(), shape.size());
        return att_tensor;
    }

    EmbeddingModel::EmbeddingModel(const std::string& path) {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "lintdb");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(*env.get(), path.data(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> input_names;
        std::vector<std::int64_t> input_shapes;
        /**
         * For colbert model:
             Input Node Name/Shape (0):
                    input_ids : -1x-1
                    attention_mask : -1x-1
             Output Node Name/Shape (0):
                    contextual : -1x-1x128
        */
        VLOG(20) << "Input Node Name/Shape (" << input_names.size() << "):";
        for (std::size_t i = 0; i < session->GetInputCount(); i++) {
            input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
            input_shapes = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            VLOG(20) << "\t" << input_names.at(i) << " : " << print_shape(input_shapes);
        }
        // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
        for (auto& s : input_shapes) {
            if (s < 0) {
            s = 1;
            }
        }

        // print name/shape of outputs
        std::vector<std::string> output_names;
        VLOG(20) << "Output Node Name/Shape (" << output_names.size() << "):";
        for (std::size_t i = 0; i < session->GetOutputCount(); i++) {
            output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
            auto output_shapes = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            VLOG(20) << "\t" << output_names.at(i) << " : " << print_shape(output_shapes);
        }

        // Assume model has 2 input node and 1 output node.
        assert(input_names.size() == 2 && output_names.size() == 1);

        this->input_names = input_names;
        this->output_names = output_names;

    }

    std::vector<float> EmbeddingModel::encode(ModelInput& inputs) const {
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(this->input_names.size());

        std::vector<int64_t> casted_ids;
        std::transform(std::begin(inputs.input_ids), std::end(inputs.input_ids), std::back_inserter(casted_ids), [](int32_t i) { return int64_t(i); });
        auto id_shape = std::vector<std::int64_t>{1, static_cast<std::int64_t>(inputs.input_ids.size())};
        input_tensors.push_back(vec_to_tensor(casted_ids, id_shape));

        std::vector<int64_t> casted_attn;
        std::transform(std::begin(inputs.attention_mask), std::end(inputs.attention_mask), std::back_inserter(casted_attn), [](int32_t i) { return int64_t(i); });
        auto attn_shape = std::vector<std::int64_t>{1, static_cast<std::int64_t>(inputs.attention_mask.size())};
        input_tensors.push_back(vec_to_tensor(casted_attn, attn_shape));

        std::vector<const char*> input_names_char(input_names.size());
        std::transform(std::begin(this->input_names), std::end(this->input_names), std::begin(input_names_char),
            [&](const std::string& str) { return str.c_str(); });

        std::vector<const char*> output_names_char(output_names.size());
        std::transform(std::begin(this->output_names), std::end(this->output_names), std::begin(output_names_char),
            [&](const std::string& str) { return str.c_str(); });

        try{
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr}, 
                input_names_char.data(), 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names_char.data(), 
                output_names_char.size()
            );


            assert(output_tensors.size() == 1);

            std::vector<int64_t> tensor_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

            auto data = output_tensors.front().GetTensorMutableData<float>();

            assert(tensor_shape.size() == 3); // (1, 1, 128)
            auto size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];

            std::vector<float> vec(data, data + size);

            return vec;

        } catch (const Ort::Exception& e) {
            LOG(ERROR) << "Error while running model: " << e.what();
            throw LintDBException("Error while running model");
        }

        return {};
    }

    size_t EmbeddingModel::get_dims() const {
        // assuming we have one output node
        return session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back();
    }

    EmbeddingModel::~EmbeddingModel() = default;
}