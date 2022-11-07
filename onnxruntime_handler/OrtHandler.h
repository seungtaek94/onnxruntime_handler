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


template <typename T>
struct Tensor
{
    T* data;
    size_t size;
    std::vector<int64_t> dims;

    Tensor(T* data, size_t size, std::vector<int64_t>& dims)
    :data{data}, size{size}, dims{dims} { }
};


enum class GraphOptimization
{
    DISABLE = 0,
    BASIC = 1,
    EXTENDED = 2,
    ALL = 99
};


enum class RunMode
{
    SEQUENTIAL = 0,
    PARALLEL = 1
};


typedef struct InferenceOption
{
    GraphOptimization graphOptimization;
    RunMode runMode;

    // For parallelize the execution of the graph such as model.
    // RunMode Should be set RunMode::PARALLEL
    int interOpNumThread;

    // For parallelize the execution within nodes such as add operation.
    int intraOpNumThread;

    InferenceOption()
    :
    graphOptimization(GraphOptimization::DISABLE),
    runMode(RunMode::SEQUENTIAL),
    interOpNumThread(0),
    intraOpNumThread(0) { }

    InferenceOption(
            GraphOptimization graphOptimization,
            RunMode runMode,
            int interOpNumThread,
            int intraOpNumThread)
    :
    graphOptimization(graphOptimization),
    runMode(runMode),
    interOpNumThread(interOpNumThread),
    intraOpNumThread(intraOpNumThread) { }

    ~InferenceOption() = default;
}InferenceOption;


namespace Ort
{
    class LIB_ORT_HANDLE Handler
    {
    public:
        static std::unique_ptr<Handler> LoadModel(
                const std::string& modelPath, InferenceOption inferenceOption);

        std::vector<const char*> GetInputNames();
        std::vector<const char*> GetOutputNames();

        std::vector<Tensor<float>> Run(Tensor<float>& tensor);

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

