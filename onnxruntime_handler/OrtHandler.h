#pragma once

#if defined(_MSC_VER)
//  Microsoft
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#if EXPORT_ORT_HANDLE
    #define LIB_ORT_HANDLE EXPORT
#else
    #define LIB_ORT_HANDLE IMPORT
#endif


#include <iostream>
#include <vector>

//#include "Utils.h"

template <typename T>
struct Tensor
{
    T* data;
    size_t size;
    std::vector<int64_t> dims;
};


namespace Ort
{
    class LIB_ORT_HANDLE Handler
    {
    public:
        static std::unique_ptr<Handler> LoadModel(std::string model_path);

        std::vector<const char*> GetInputNames();
        std::vector<const char*> GetOutputNames();

        // (H,W,3) cv::Mat data to (3,H,W) or (1,3,H,W) tensor;
        static Tensor<float> ToTensor(
                float *data,
                int rows, int cols,
                const std::vector<float>& mean = { 0.f, 0.f, 0.f },
                const std::vector<float>& std = { 1.f, 1.f, 1.f },
                bool swapRB = false,
                bool expandDim = true);

    public:
        ~Handler();

    private:
        Handler();
        void* core;
    };
}

