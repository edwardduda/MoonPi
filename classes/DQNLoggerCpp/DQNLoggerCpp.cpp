// dqn_logger.cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>  // For std::min

namespace py = pybind11;

class DQNLogger {
public:
    std::vector<torch::Tensor> temporal_attention_buffer;
    std::vector<torch::Tensor> feature_attention_buffer;
    std::vector<torch::Tensor> technical_attention_buffer;
    int attention_buffer_count = 0;
    const int max_attention_samples = 1000;

    // Simplified initialization: assumes the buffers are empty.
    void initialize_attention_buffers(const std::vector<torch::Tensor>& temporal_weights,
                                      const std::vector<torch::Tensor>& feature_weights,
                                      const std::vector<torch::Tensor>* technical_weights = nullptr) {
        if (temporal_attention_buffer.empty()) {
            for (const auto& weight : temporal_weights) {
                temporal_attention_buffer.push_back(torch::zeros_like(weight));
            }
        }
        if (feature_attention_buffer.empty()) {
            for (const auto& weight : feature_weights) {
                feature_attention_buffer.push_back(torch::zeros_like(weight));
            }
        }
        if (technical_weights && technical_attention_buffer.empty()) {
            for (const auto& weight : *technical_weights) {
                technical_attention_buffer.push_back(torch::zeros_like(weight));
            }
        }
    }

    void update_attention_buffers(const std::vector<torch::Tensor>& temporal_weights,
                                  const std::vector<torch::Tensor>& feature_weights,
                                  const std::vector<torch::Tensor>* technical_weights = nullptr) {
        torch::NoGradGuard no_grad;  // Disables gradient tracking
        initialize_attention_buffers(temporal_weights, feature_weights, technical_weights);
        attention_buffer_count = std::min(attention_buffer_count + 1, max_attention_samples);
        double alpha = 1.0 / attention_buffer_count;

        // Update temporal buffers
        for (size_t i = 0; i < temporal_weights.size(); ++i) {
            // Mimic selecting the first element like [0] in Python.
            torch::Tensor current = temporal_weights[i].select(0, 0);
            temporal_attention_buffer[i] = (1 - alpha) * temporal_attention_buffer[i] + alpha * current;
        }
        // Update feature buffers
        for (size_t i = 0; i < feature_weights.size(); ++i) {
            torch::Tensor current = feature_weights[i].select(0, 0);
            feature_attention_buffer[i] = (1 - alpha) * feature_attention_buffer[i] + alpha * current;
        }
        // Update technical buffers if provided
        if (technical_weights) {
            const auto& tech_weights = *technical_weights;
            for (size_t i = 0; i < tech_weights.size(); ++i) {
                torch::Tensor current = tech_weights[i].select(0, 0);
                technical_attention_buffer[i] = (1 - alpha) * technical_attention_buffer[i] + alpha * current;
            }
        }
    }

    std::vector<torch::Tensor> get_temporal_attention_buffer() const {
        return temporal_attention_buffer;
    }
    
    std::vector<torch::Tensor> get_feature_attention_buffer() const {
        return feature_attention_buffer;
    }
    
    std::vector<torch::Tensor> get_technical_attention_buffer() const {
        return technical_attention_buffer;
    }
};

PYBIND11_MODULE(DQNLoggerCpp, m) {
    py::class_<DQNLogger> dqn_logger(m, "DQNLogger");
    dqn_logger.def(py::init<>())
              .def("update_attention_buffers", &DQNLogger::update_attention_buffers)
              .def("get_temporal_attention_buffer", &DQNLogger::get_temporal_attention_buffer)
              .def("get_feature_attention_buffer", &DQNLogger::get_feature_attention_buffer)
              .def("get_technical_attention_buffer", &DQNLogger::get_technical_attention_buffer);
}
