#include "OnnxRuntimeHandler.h"

#include <iostream>

void hello() {
    std::cout << "Hello, World!" << std::endl;
}


OnnxRuntimeHandler::OnnxRuntimeHandler()
{

}

OnnxRuntimeHandler::~OnnxRuntimeHandler()
{

}

int OnnxRuntimeHandler::add(int a, int b)
{
    return a + b;
}
