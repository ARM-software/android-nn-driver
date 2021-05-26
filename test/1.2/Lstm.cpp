//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Lstm.hpp"

using namespace armnn_driver;

TEST_SUITE("LstmTests_1.2_CpuRef")
{
    TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.2_armnn::Compute::CpuRef")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }
    TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.2_CpuRef")
    {
        LstmCifgPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }
    TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.2_CpuRef")
    {
        LstmNoCifgPeepholeProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }
    TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.2_CpuRef")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }
    TEST_CASE("QuantizedLstmTest_1.2_CpuRef")
    {
        QuantizedLstm<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }
}

#if defined(ARMCOMPUTECL_ENABLED)
TEST_SUITE("LstmTests_1.2_GpuAcc")
{
    TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.2_GpuAcc")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }
    TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.2_GpuAcc")
    {
        LstmCifgPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }
    TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.2_GpuAcc")
    {
        LstmNoCifgPeepholeProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }
    TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.2_GpuAcc")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }
        TEST_CASE("QuantizedLstmTest_1.2_GpuAcc")
    {
        QuantizedLstm<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }
}
#endif
