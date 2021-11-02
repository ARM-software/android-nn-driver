//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Lstm.hpp"

using namespace armnn_driver;

#if defined(ARMNNREF_ENABLED)
DOCTEST_TEST_SUITE("LstmTests_1.2_CpuRef")
{

    DOCTEST_TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.2_armnn::Compute::CpuRef")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.2_CpuRef")
    {
        LstmCifgPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.2_CpuRef")
    {
        LstmNoCifgPeepholeProjection<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.2_CpuRef")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("QuantizedLstmTest_1.2_CpuRef")
    {
        QuantizedLstm<hal_1_2::HalPolicy>(armnn::Compute::CpuRef);
    }

}
#endif

#if defined(ARMCOMPUTECL_ENABLED)
DOCTEST_TEST_SUITE("LstmTests_1.2_GpuAcc")
{

    DOCTEST_TEST_CASE("LstmNoCifgNoPeepholeNoProjectionTest_1.2_GpuAcc")
    {
        LstmNoCifgNoPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionTest_1.2_GpuAcc")
    {
        LstmCifgPeepholeNoProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmNoCifgPeepholeProjectionTest_1.2_GpuAcc")
    {
        LstmNoCifgPeepholeProjection<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("LstmCifgPeepholeNoProjectionBatch2Test_1.2_GpuAcc")
    {
        LstmCifgPeepholeNoProjectionBatch2<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("QuantizedLstmTest_1.2_GpuAcc")
    {
        QuantizedLstm<hal_1_2::HalPolicy>(armnn::Compute::GpuAcc);
    }

}
#endif
