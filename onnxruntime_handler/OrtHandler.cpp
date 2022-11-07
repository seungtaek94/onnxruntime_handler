
#include <algorithm>
#include <memory>

#include "onnxruntime_cxx_api.h"
#include "OrtHandler.h"
#include "OrtHandlerCore.h"


namespace Ort {
    Handler::Handler()
    {
        core = (OrtHandlerCore *) new OrtHandlerCore();
    }


    Handler::~Handler()
    {
        delete (OrtHandlerCore *) core;
    }


    std::unique_ptr<Handler> Handler::LoadModel(
            const std::string& modelPath, InferenceOption inferenceOption)
    {
        std::unique_ptr<Handler> handler(new Handler);
        ((OrtHandlerCore *) handler->core)->LoadModel(modelPath, inferenceOption);

        return handler;
    }


    std::vector<const char *> Handler::GetInputNames() {
        return ((OrtHandlerCore *) this->core)->GetInputNames();
    }


    std::vector<const char *> Handler::GetOutputNames() {
        return ((OrtHandlerCore *) this->core)->GetOutputNames();
    }


    std::vector<Tensor<float>> Handler::Run(Tensor<float>& tensor)
    {
        return ((OrtHandlerCore *) this->core)->Run(tensor);
    }


    Tensor<float> Handler::ToTensor(
            float *data,
            int rows, int cols,
            const std::vector<float>& mean,
            const std::vector<float>& std,
            const bool swapRB,
            const bool expandDim)
    {
        return OrtHandlerCore::ToTensor(data, rows, cols, mean, std, swapRB, expandDim);
    }
}
