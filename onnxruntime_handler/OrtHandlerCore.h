#pragma once

#include <iostream>
#include "onnxruntime_cxx_api.h"


class OrtHandlerCore {
public:
    OrtHandlerCore();
    ~OrtHandlerCore();

    void LoadModel(std::string model_path);


    std::vector<const char*> GetInputNames();
    std::vector<const char*> GetOutputNames();

private:
    std::unique_ptr<Ort::Env> _ort_env;
    std::unique_ptr<Ort::Session> _ort_session;
    std::unique_ptr<Ort::SessionOptions> _ort_session_options;
    std::unique_ptr<Ort::MemoryInfo> _ort_mem_info;

private:
    std::vector<const char*> _getInputName();
    std::vector<const char*> _getOutputName();
};
