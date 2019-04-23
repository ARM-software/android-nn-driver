//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../../1.0/FullyConnected.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(FullyConnectedReshapeTests)

BOOST_AUTO_TEST_CASE(TestFlattenFullyConnectedInput)
{
    using armnn::TensorShape;
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({97,1,1,2048}), TensorShape({512, 2048})) ==
               TensorShape({97, 2048}));
}

BOOST_AUTO_TEST_SUITE_END()
