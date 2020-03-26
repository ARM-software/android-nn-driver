//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include <log/log.h>

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_3

#include "1.1/ArmnnDriver.hpp"
#include "1.2/ArmnnDriver.hpp"
#include "1.3/ArmnnDriver.hpp"

namespace armnn_driver
{

class ArmnnDriver : public hal_1_3::ArmnnDriver
{
public:
    ArmnnDriver(DriverOptions options)
        : hal_1_3::ArmnnDriver(std::move(options))
    {
        ALOGV("ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}
};

} // namespace armnn_driver
#elif ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2

#include "1.1/ArmnnDriver.hpp"
#include "1.2/ArmnnDriver.hpp"

namespace armnn_driver
{

class ArmnnDriver : public hal_1_2::ArmnnDriver
{
public:
    ArmnnDriver(DriverOptions options)
        : hal_1_2::ArmnnDriver(std::move(options))
    {
        ALOGV("ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}
};

} // namespace armnn_driver
#elif ARMNN_ANDROID_NN_V1_1 // Using ::android::hardware::neuralnetworks::V1_1

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

#else // Fallback to ::android::hardware::neuralnetworks::V1_0

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
