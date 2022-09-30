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

namespace Ort
{
    class LIB_ORT_HANDLE Handler
    {
    public:
        Handler(std::string model_path);
        ~Handler();

        std::vector<const char*> GetInputNames();
        std::vector<const char*> GetOutputNames();

        static void blobFromImageData(
                float* data, int rows, int cols, int ch = 3,
                std::vector<float> mean = { 0.f, 0.f, 0.f },
                std::vector<float> std = { 1.f, 1.f, 1.f },
                bool swapRB = false);

    protected:
        void* core;
    };



   //Utils Handler::tests;






}

