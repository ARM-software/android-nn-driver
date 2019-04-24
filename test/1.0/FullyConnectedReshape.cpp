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

    // Pass through 2d input
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({2,2048}), TensorShape({512, 2048})) ==
               TensorShape({2, 2048}));

    // Trivial flattening of batched channels
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({97,1,1,2048}), TensorShape({512, 2048})) ==
               TensorShape({97, 2048}));

    // Flatten single batch of rows
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({1,97,1,2048}), TensorShape({512, 2048})) ==
               TensorShape({97, 2048}));

    // Flatten single batch of columns
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({1,1,97,2048}), TensorShape({512, 2048})) ==
               TensorShape({97, 2048}));

    // Move batches into input dimension
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({50,1,1,10}), TensorShape({512, 20})) ==
               TensorShape({25, 20}));

    // Flatten single batch of 3D data (e.g. convolution output)
    BOOST_TEST(FlattenFullyConnectedInput(TensorShape({1,16,16,10}), TensorShape({512, 2560})) ==
               TensorShape({1, 2560}));
}

BOOST_AUTO_TEST_SUITE_END()
