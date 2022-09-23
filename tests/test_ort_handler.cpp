//
// Created by admin on 2022-09-22.
//

#include "gtest/gtest.h"

#include "OnnxRuntimeHandler.h"

std::unique_ptr<OnnxRuntimeHandler> orthandler;

// Test OnnxRuntimeHandler::OnnxRuntimeHandler(std::string)
TEST(OnnxRuntimeHandler, OnnxRuntimeHandler)
{
    orthandler = std::make_unique<OnnxRuntimeHandler>("../../assets/models/shufflenetv2_x0.5.onnx");
}

// Test OnnxRuntimeHandler::GetInputNames()
TEST(OnnxRuntimeHandler, GetInputNames)
{
    EXPECT_FALSE(std::strcmp("input", orthandler->GetInputNames()[0]));
}

// Test OnnxRuntimeHandler::GetOutputNames()
TEST(OnnxRuntimeHandler, GetOutputNames)
{
    EXPECT_FALSE(std::strcmp("output", orthandler->GetOutputNames()[0]));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}