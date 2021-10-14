//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Lstm.hpp"

using namespace armnn_driver;

DOCTEST_TEST_SUITE("LstmTests_1.0_CpuRef")
{

    DOCTEST_TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.0_armnn::Compute::CpuRef")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_0::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.0_CpuRef")
    {
        LstmCifgPeepholeNoProjection<hal_1_0::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.0_CpuRef")
    {
        LstmNoCifgPeepholeProjection<hal_1_0::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.0_CpuRef")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_0::HalPolicy>(armnn::Compute::CpuRef);
    }

}

#if defined(ARMCOMPUTECL_ENABLED)
DOCTEST_TEST_SUITE("LstmTests_1.0_GpuAcc")
{

    DOCTEST_TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.0_GpuAcc")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_0::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.0_GpuAcc")
    {
        LstmCifgPeepholeNoProjection<hal_1_0::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.0_GpuAcc")
    {
        LstmNoCifgPeepholeProjection<hal_1_0::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.0_GpuAcc")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_0::HalPolicy>(armnn::Compute::GpuAcc);
    }

}
#endif
