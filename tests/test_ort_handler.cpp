//
// Created by admin on 2022-09-22.
//

#include "gtest/gtest.h"

#include "OnnxRuntimeHandler.h"

TEST(teat_a, teat1)
{
    OnnxRuntimeHandler orthandle;
    EXPECT_EQ(orthandle.add(1, 1), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}