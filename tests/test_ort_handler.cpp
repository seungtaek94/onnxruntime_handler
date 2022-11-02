//
// Created by admin on 2022-09-22.
//

#include <cstring>
#include "gtest/gtest.h"
#include "OrtHandler.h"

class TestOrtHandler : public ::testing::Test
{
protected:
    void SetUp() override
    {
        orthandler = Ort::Handler::LoadModel("../../assets/models/conv1x1.onnx");
    }

    void TearDown() override
    {
    }

    std::unique_ptr<Ort::Handler> orthandler;
};

// Test Ort::Handler::GetInputNames()
TEST_F(TestOrtHandler, GetInputNames)
{
    EXPECT_FALSE(std::strcmp("input", orthandler->GetInputNames()[0]));
}

// Test Ort::Handler::GetOutputNames()
TEST_F(TestOrtHandler, GetOutputNames)
{
    EXPECT_FALSE(std::strcmp("output", orthandler->GetOutputNames()[0]));
}


TEST_F(TestOrtHandler, ToTensorWithoutSwapRB)
{
    // HWC
    float input[6][3] = {
            {1.1f, 1.2f, 1.3f},{2.1f, 2.2f, 2.3f},
            {3.1f, 3.2f, 3.3f},{4.1f, 4.2f, 4.3f},
            {5.1f, 5.2f, 5.3f},{6.1f, 6.2f, 6.3f},
    };

    // CHW with swapRB==false
    float expect_without_swapRB[6][3] = {
            {1.1f, 2.1f, 3.1f},{4.1f, 5.1f, 6.1f},
            {1.2f, 2.2f, 3.2f},{4.2f, 5.2f, 6.2f},
            {1.3f, 2.3f, 3.3f},{4.3f, 5.3f, 6.3f},
    };

    Tensor<float> tensor  = Ort::Handler::ToTensor((float*)input, 2, 3);

    int nElement = sizeof(expect_without_swapRB) / sizeof(float);

    EXPECT_EQ(tensor.size, nElement);
    float (*pExpect)[3] = expect_without_swapRB;

    for(int i=0; i < nElement; i++){
        EXPECT_FLOAT_EQ(tensor.data[i], *(*pExpect + i));
    }
}


TEST_F(TestOrtHandler, ToTensorWithSwapRB)
{
    // HWC
    float input[6][3] = {
            {1.1f, 1.2f, 1.3f},{2.1f, 2.2f, 2.3f},
            {3.1f, 3.2f, 3.3f},{4.1f, 4.2f, 4.3f},
            {5.1f, 5.2f, 5.3f},{6.1f, 6.2f, 6.3f},
    };

    // CHW with swapRB==true
    float expect_with_swapRB[6][3] = {
            {1.3f, 2.3f, 3.3f},{4.3f, 5.3f, 6.3f},
            {1.2f, 2.2f, 3.2f},{4.2f, 5.2f, 6.2f},
            {1.1f, 2.1f, 3.1f},{4.1f, 5.1f, 6.1f},
    };

    Tensor<float> tensor  = Ort::Handler::ToTensor((float*)input, 2, 3, {0.f, 0.f, 0.f} , {1.f, 1.f, 1.f}, true);

    int nElement = sizeof(expect_with_swapRB) / sizeof(float);

    EXPECT_EQ(tensor.size, nElement);
    float (*pExpect)[3] = expect_with_swapRB;

    for(int i=0; i < nElement; i++){
        EXPECT_FLOAT_EQ(tensor.data[i], *(*pExpect + i));
    }
}


TEST_F(TestOrtHandler, RUN)
{
    // HWC
    float input[6][3] = {
            {1.1f, 1.2f, 1.3f},{2.1f, 2.2f, 2.3f},
            {3.1f, 3.2f, 3.3f},{4.1f, 4.2f, 4.3f},
            {5.1f, 5.2f, 5.3f},{6.1f, 6.2f, 6.3f},
    };

    std::vector<int64_t> expect_output_tensor_dims {1, 1, 2, 3};
    size_t expect_output_tensor_size = 6;

    Tensor<float> input_tensor = Ort::Handler::ToTensor((float*)input, 2, 3);

    std::vector<Tensor<float>> output_tensor = orthandler->Run(input_tensor);

    EXPECT_EQ(expect_output_tensor_size, output_tensor[0].size);
    EXPECT_EQ(expect_output_tensor_dims, output_tensor[0].dims);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}