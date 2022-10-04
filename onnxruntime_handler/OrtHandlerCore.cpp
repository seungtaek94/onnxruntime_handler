#include "OrtHandlerCore.h"
#include "onnxruntime_session_options_config_keys.h"

#include <sstream>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


OrtHandlerCore::OrtHandlerCore()
{
    _ort_session_options = std::make_unique<Ort::SessionOptions>();
    _ort_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _ort_session_options->SetInterOpNumThreads(1);

    _ort_env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Inference");

    _ort_mem_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU));
}

OrtHandlerCore::~OrtHandlerCore()
{

}

void OrtHandlerCore::LoadModel(std::string model_path)
{
#ifdef _WIN32
    std::wstring w_model_path;
    w_model_path.assign(model_path.begin(), model_path.end());
    _ort_session = std::make_unique<Ort::Session>(*_ort_env, w_model_path.c_str(), *_ort_session_options);
#else
    this->_ort_session = std::make_unique<Ort::Session>(*this->_ort_env, model_path.c_str(), *this->_ort_session_options);
#endif
}


std::vector<const char*> OrtHandlerCore::GetInputNames()
{
    return this->_getInputName();
}


std::vector<const char*> OrtHandlerCore::GetOutputNames()
{
    return  this->_getOutputName();
}


std::vector<const char*> OrtHandlerCore::_getInputName()
{
    Ort::AllocatorWithDefaultOptions allocator;

    size_t input_num = this->_ort_session->GetInputCount();

    std::vector<const char*> input_names(input_num);

    for (int i = 0; i < input_num; i++)
    {
        input_names[i] = this->_ort_session->GetInputName(i, allocator);
    }

    return input_names;
}


std::vector<const char*> OrtHandlerCore::_getOutputName()
{
    Ort::AllocatorWithDefaultOptions allocator;

    size_t output_num = this->_ort_session->GetOutputCount();

    std::vector<const char*> output_names(output_num);

    for (int i = 0; i < output_num; i++)
    {
        output_names[i] = this->_ort_session->GetOutputName(i, allocator);
    }

    return output_names;
}