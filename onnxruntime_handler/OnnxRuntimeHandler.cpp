
#include "onnxruntime_cxx_api.h"
#include "OnnxRuntimeHandler.h"
#include "OnnxRuntimeHandlerCore.h"


OnnxRuntimeHandler::OnnxRuntimeHandler(std::string model_path)
{
    core = (OnnxRuntimeHandlerCore*) new OnnxRuntimeHandlerCore(model_path);
}

OnnxRuntimeHandler::~OnnxRuntimeHandler()
{
    delete (OnnxRuntimeHandlerCore*)core;
}

std::vector<const char*> OnnxRuntimeHandler::GetInputNames()
{
    return ((OnnxRuntimeHandlerCore*)core)->GetInputNames();
}

std::vector<const char*> OnnxRuntimeHandler::GetOutputNames()
{
    return ((OnnxRuntimeHandlerCore*)core)->GetOutputNames();
}