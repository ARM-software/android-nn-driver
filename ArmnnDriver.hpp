//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include <log/log.h>

#if defined(ARMNN_ANDROID_NN_V1_1)

#include "1.1/ArmnnDriver.hpp"

namespace armnn_driver
{

class ArmnnDriver : public hal_1_1::ArmnnDriver
{
public:
    ArmnnDriver(DriverOptions options)
        : hal_1_1::ArmnnDriver(std::move(options))
    {
        ALOGV("ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}
};

} // namespace armnn_driver

#else // Fallback to ::android::hardware::neuralnetworks::V1_0.

#include "1.0/ArmnnDriver.hpp"

namespace armnn_driver
{

class ArmnnDriver : public hal_1_0::ArmnnDriver
{
public:
    ArmnnDriver(DriverOptions options)
        : hal_1_0::ArmnnDriver(std::move(options))
    {
        ALOGV("ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}
};

} // namespace armnn_driver

#endif
