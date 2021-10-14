//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"

DOCTEST_TEST_SUITE("FullyConnectedReshapeTests")
{
DOCTEST_TEST_CASE("TestFlattenFullyConnectedInput")
{
    using armnn::TensorShape;

    // Pass through 2d input
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({2,2048}),
                                             TensorShape({512, 2048})) == TensorShape({2, 2048}));

    // Trivial flattening of batched channels
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({97,1,1,2048}),
                                             TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Flatten single batch of rows
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({1,97,1,2048}),
                                             TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Flatten single batch of columns
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({1,1,97,2048}),
                                             TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Move batches into input dimension
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({50,1,1,10}),
                                             TensorShape({512, 20})) == TensorShape({25, 20}));

    // Flatten single batch of 3D data (e.g. convolution output)
    DOCTEST_CHECK(FlattenFullyConnectedInput(TensorShape({1,16,16,10}),
                                             TensorShape({512, 2560})) == TensorShape({1, 2560}));
}

}
