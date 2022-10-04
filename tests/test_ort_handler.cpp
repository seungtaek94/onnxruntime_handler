//
// Created by admin on 2022-09-22.
//

#include <cstring>
#include "gtest/gtest.h"

#include "OrtHandler.h"

std::unique_ptr<Ort::Handler> orthandler;

// Test Ort::Handler::LoadModel()
// OrtHandler
TEST(OrtHandler, OrtHandler)
{
    ASSERT_NO_THROW(orthandler = Ort::Handler::LoadModel("../../assets/models/shufflenetv2_x0.5.onnx"));
}

// Test Ort::Handler::GetInputNames()
TEST(OrtHandler, GetInputNames)
{
    EXPECT_FALSE(std::strcmp("input", orthandler->GetInputNames()[0]));
}

// Test Ort::Handler::GetOutputNames()
TEST(OrtHandler, GetOutputNames)
{
    EXPECT_FALSE(std::strcmp("output", orthandler->GetOutputNames()[0]));
}

// Test Ort::Handler::blobFromImageData()
TEST(OrtHandler, blobFromImageData)
{
    float input[6][3] = {
        {1.1f, 1.2f, 1.3f},{2.1f, 2.2f, 2.3f},
        {3.1f, 3.2f, 3.3f},{4.1f, 4.2f, 4.3f},
        {5.1f, 5.2f, 5.3f},{6.1f, 6.2f, 6.3f},
    };

    float expect[6][3] = {
        {1.1f, 2.1f, 3.1f},{4.1f, 5.1f, 6.1f},
        {1.2f, 2.2f, 3.2f},{4.2f, 5.2f, 6.2f},
        {1.3f, 2.3f, 3.3f},{4.3f, 5.3f, 6.3f},
    };

    Ort::Handler::blobFromImageData((float*)input, 3, 2, 3);

    int nElement = sizeof(input) / sizeof(float);

    EXPECT_EQ(sizeof(input), sizeof(expect));

    float (*pInput)[3] = input;
    float (*pExpect)[3] = expect;

    for(int i=0; i < nElement; i++){
        EXPECT_FLOAT_EQ(*(*pInput + i), *(*pExpect + i));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}