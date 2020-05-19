//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Lstm.hpp"

#include <boost/test/data/test_case.hpp>

BOOST_AUTO_TEST_SUITE(LstmTests)

using namespace armnn_driver;

BOOST_DATA_TEST_CASE(LstmNoCifgNoPeepholeNoProjectionTest, COMPUTE_DEVICES)
{
    LstmNoCifgNoPeepholeNoProjection<hal_1_2::HalPolicy>(sample);
}

BOOST_DATA_TEST_CASE(LstmCifgPeepholeNoProjectionTest, COMPUTE_DEVICES)
{
    LstmCifgPeepholeNoProjection<hal_1_2::HalPolicy>(sample);
}

BOOST_DATA_TEST_CASE(LstmNoCifgPeepholeProjectionTest, COMPUTE_DEVICES)
{
    LstmNoCifgPeepholeProjection<hal_1_2::HalPolicy>(sample);
}

BOOST_DATA_TEST_CASE(LstmCifgPeepholeNoProjectionBatch2Test, COMPUTE_DEVICES)
{
    LstmCifgPeepholeNoProjectionBatch2<hal_1_2::HalPolicy>(sample);
}

BOOST_DATA_TEST_CASE(LstmNoCifgPeepholeProjectionNoClippingLayerNormTest, COMPUTE_DEVICES)
{
    LstmNoCifgPeepholeProjectionNoClippingLayerNorm<hal_1_2::HalPolicy>(sample);
}

BOOST_DATA_TEST_CASE(LstmCifgPeepholeProjectionNoClippingLayerNormTest, COMPUTE_DEVICES)
{
    LstmCifgPeepholeProjectionNoClippingLayerNorm<hal_1_2::HalPolicy>(sample);
}

#if defined(ARMCOMPUTECL_ENABLED)
BOOST_DATA_TEST_CASE(QuantizedLstmTest, COMPUTE_DEVICES)
{
    QuantizedLstm<hal_1_2::HalPolicy>(sample);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
