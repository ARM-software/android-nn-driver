//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../Convolution2D.hpp"
#include "../../1.1/HalPolicy.hpp"

#include <doctest/doctest.h>
#include <log/log.h>

#include <OperationsUtils.h>



using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

namespace driverTestHelpers
{

void SetModelFp16Flag(V1_1::Model& model, bool fp16Enabled)
{
    // Set the fp16 flag in the given model
    model.relaxComputationFloat32toFloat16 = fp16Enabled;
}

} // namespace driverTestHelpers


TEST_SUITE("Convolution2DTests_1.1")
{

TEST_CASE("ConvValidPadding_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingValid);
}

TEST_CASE("ConvSamePadding_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingSame);
}

TEST_CASE("ConvValidPaddingFp16Flag_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingValid, true);
}

TEST_CASE("ConvSamePaddingFp16Flag_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingSame, true);
}

}
