//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Convolution2D.hpp"

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


DOCTEST_TEST_SUITE("Convolution2DTests_1.1")
{

DOCTEST_TEST_CASE("ConvValidPadding_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingValid);
}

DOCTEST_TEST_CASE("ConvSamePadding_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingSame);
}

DOCTEST_TEST_CASE("ConvValidPaddingFp16Flag_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingValid, true);
}

DOCTEST_TEST_CASE("ConvSamePaddingFp16Flag_Hal_1_1")
{
    PaddingTestImpl<hal_1_1::HalPolicy>(android::nn::kPaddingSame, true);
}

}
