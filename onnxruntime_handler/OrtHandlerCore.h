#pragma once

#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "OrtHandler.h"


class OrtHandlerCore {
public:
    OrtHandlerCore();
    ~OrtHandlerCore() = default;

    void LoadModel(
            const std::string& modelPath, InferenceOption inferenceOption);

    std::vector<const char*> GetInputNames();
    std::vector<const char*> GetOutputNames();

    std::vector<Tensor<float>> Run(Tensor<float>& tensor);

    static Tensor<float> ToTensor(
            float *data,
            int rows, int cols,
            const std::vector<float>& mean = { 0.f, 0.f, 0.f },
            const std::vector<float>& std = { 1.f, 1.f, 1.f },
            bool swapRB = false,
            bool expandDim = true);

    static void blobFromImageData(
            float* data, int rows, int cols,
            const std::vector<float>& mean = { 0.f, 0.f, 0.f },
            const std::vector<float>& std = { 1.f, 1.f, 1.f },
            bool swapRB = false);

private:
    std::unique_ptr<Ort::Env> _ort_env;
    std::unique_ptr<Ort::Session> _ort_session;
    std::unique_ptr<Ort::SessionOptions> _ort_session_options;
    Ort::MemoryInfo _ort_mem_info{nullptr};

    OrtCUDAProviderOptions _ort_cuda_provider_options;

private:
    std::vector<const char*> _input_name;
    std::vector<const char*> _output_name;

    std::vector<const char*> _getInputName();
    std::vector<const char*> _getOutputName();

    template<typename T>
    Ort::Value _tensorToOrtValue(Tensor<T>& tensor);

    template<typename T>
    Tensor<T> _ortValueToTensor(Ort::Value& ortValue);

    template<typename T>
    std::vector<Tensor<T>> _ortValuesToTensors(std::vector<Ort::Value>& ortValues);

    void _setInferenceOption(InferenceOption inferenceOption);
    void _setGraphOptimizationLevel(GraphOptimization graphOptimization);
    void _setRunMode(RunMode runMode);
    void _setIntraOpNumThread(int n);
    void _setInterOpNumThread(int n);
    void _setCudaProvider(int gpuIndex);
};
