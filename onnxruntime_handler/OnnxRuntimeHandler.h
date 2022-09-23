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
    //  do nothing and hope for the best?
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
#include "onnxruntime_cxx_api.h"

class LIB_ORT_HANDLE OnnxRuntimeHandler
{
public:
    OnnxRuntimeHandler();
    ~OnnxRuntimeHandler();

    int add(int a, int b);
};
