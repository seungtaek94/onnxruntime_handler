#include <algorithm>
#include <memory>
#include <sstream>

#include "OrtHandlerCore.h"


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

    _ort_mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
}


void OrtHandlerCore::LoadModel(const std::string& model_path)
{
#ifdef _WIN32
    std::wstring w_model_path;
    w_model_path.assign(model_path.begin(), model_path.end());
    _ort_session = std::make_unique<Ort::Session>(*_ort_env, w_model_path.c_str(), *_ort_session_options);
#else
    this->_ort_session = std::make_unique<Ort::Session>(*this->_ort_env, model_path.c_str(), *this->_ort_session_options);
#endif

    this->_input_name = _getInputName();
    this->_output_name = _getOutputName();
}


std::vector<const char*> OrtHandlerCore::GetInputNames()
{
    return this->_input_name;
}


std::vector<const char*> OrtHandlerCore::GetOutputNames()
{
    return  this->_output_name;
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


std::vector<Tensor<float>> OrtHandlerCore::Run(Tensor<float>& tensor)
{
    auto ortValue = this->_tensorToOrtValue(tensor);

    std::vector<Ort::Value> results = _ort_session->Run(
                Ort::RunOptions{nullptr},
                this->_input_name.data(),
                &ortValue,
                this->_input_name.size(),
                this->_output_name.data(),
                this->_output_name.size());

    return this->_ortValuesToTensors<float>(results);
}


Tensor<float> OrtHandlerCore::ToTensor(
        float *data,
        int rows, int cols,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const bool swapRB,
        const bool expandDim)
{
    OrtHandlerCore::blobFromImageData(data, rows, cols, mean, std, swapRB);

    std::vector<int64_t> dims;
    if(expandDim)
        dims = {1, 3, rows, cols};
    else
        dims = {3, rows, cols};

    size_t size = 1;
    for(auto & dim : dims)
    {
        size *= dim;
    }

    return Tensor<float> {data, size, dims};
}


void OrtHandlerCore::blobFromImageData(
        float *data, const int rows, const int cols,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const bool swapRB)
{
    const size_t nPixels = rows * cols;

    auto *flatten = new float[nPixels * 3];

    int nPos = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col += 1)
        {
            int nIdx = 0;
            nIdx = row * cols * 3 + col * 3;

            flatten[nPos] = (data[nIdx] - mean[0]) / std[0];                      // Blue
            flatten[nPos + nPixels] = (data[nIdx + 1] - mean[1]) / std[1];        // Grean
            flatten[nPos + (nPixels * 2)] = (data[nIdx + 2] - mean[2]) / std[2];  // Red

            if (swapRB) // Swap B <-> R Channel
            {
                std::swap(flatten[nPos], flatten[nPos + (nPixels * 2)]);
            }
            nPos++;
        }
    }
    std::copy_n(flatten, (nPixels * 3), data);

    delete[] flatten;
}


template<typename T>
Ort::Value OrtHandlerCore::_tensorToOrtValue(Tensor<T> &tensor)
{
    return Ort::Value::CreateTensor<T>(
            this->_ort_mem_info,
            tensor.data,
            tensor.size,
            tensor.dims.data(),
            tensor.dims.size()
    );
}


template<typename T>
Tensor<T> OrtHandlerCore::_ortValueToTensor(Ort::Value &ortValue)
{
    size_t size = ortValue.GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<int64_t> dims = ortValue.GetTensorTypeAndShapeInfo().GetShape();

    return Tensor<T> {
            ortValue.GetTensorMutableData<T>(),
            size,
            dims
    };
}


template<typename T>
std::vector<Tensor<T>> OrtHandlerCore::_ortValuesToTensors(std::vector<Ort::Value>& ortValues)
{
    std::vector<Tensor<T>> tensors;

    for(auto& ortValue : ortValues)
    {
        tensors.emplace_back(this->template _ortValueToTensor<T>(ortValue));
    }

    return tensors;
}