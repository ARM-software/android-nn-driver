//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../UnidirectionalSequenceLstm.hpp"

using namespace armnn_driver;

DOCTEST_TEST_SUITE("UnidirectionalSequenceLstmTests_1.2_CpuRef")
{

    DOCTEST_TEST_CASE("UnidirectionalSequenceLstmLayerFloat32Test_1.2_CpuRef")
    {
        UnidirectionalSequenceLstmLayerFloat32TestImpl<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("UnidirectionalSequenceLstmLayerFloat32TimeMajorTest_1.2_CpuRef")
    {
        UnidirectionalSequenceLstmLayerFloat32TimeMajorTestImpl<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionTest_1.2_CpuRef")
    {
        UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionTestImpl<hal_1_2::HalPolicy>
            (armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTest_1.2_CpuRef")
    {
        UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTestImpl<hal_1_2::HalPolicy>
            (armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTest_1.2_CpuRef")
    {
        UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTestImpl<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

}