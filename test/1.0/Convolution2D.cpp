//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../Convolution2D.hpp"
#include "../../1.0/HalPolicy.hpp"

#include <doctest/doctest.h>
#include <log/log.h>

#include <OperationsUtils.h>


using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

namespace driverTestHelpers
{

void SetModelFp16Flag(V1_0::Model&, bool)
{
    // Nothing to do, the V1_0::Model does not support fp16 precision relaxation.
    // This function is used for compatibility only.
}

} // namespace driverTestHelpers

TEST_SUITE("Convolution2DTests_1.0")
{


TEST_CASE("ConvValidPadding_Hal_1_0")
{
    PaddingTestImpl<hal_1_0::HalPolicy>(android::nn::kPaddingValid);
}

TEST_CASE("ConvSamePadding_Hal_1_0")
{
    PaddingTestImpl<hal_1_0::HalPolicy>(android::nn::kPaddingSame);
}

}
