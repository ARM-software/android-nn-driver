//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../../1.0/FullyConnected.hpp"

#include <doctest/doctest.h>

TEST_SUITE("FullyConnectedReshapeTests")
{
TEST_CASE("TestFlattenFullyConnectedInput")
{
    using armnn::TensorShape;

    // Pass through 2d input
    CHECK(FlattenFullyConnectedInput(TensorShape({2,2048}), TensorShape({512, 2048})) == TensorShape({2, 2048}));

    // Trivial flattening of batched channels
    CHECK(FlattenFullyConnectedInput(TensorShape({97,1,1,2048}), TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Flatten single batch of rows
    CHECK(FlattenFullyConnectedInput(TensorShape({1,97,1,2048}), TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Flatten single batch of columns
    CHECK(FlattenFullyConnectedInput(TensorShape({1,1,97,2048}), TensorShape({512, 2048})) == TensorShape({97, 2048}));

    // Move batches into input dimension
    CHECK(FlattenFullyConnectedInput(TensorShape({50,1,1,10}), TensorShape({512, 20})) == TensorShape({25, 20}));

    // Flatten single batch of 3D data (e.g. convolution output)
    CHECK(FlattenFullyConnectedInput(TensorShape({1,16,16,10}), TensorShape({512, 2560})) == TensorShape({1, 2560}));
}

}
