//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriverTests"
#include <log/log.h>

#ifndef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif

#include "DriverTestHelpers.hpp"

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

DOCTEST_TEST_SUITE("DriverTests")
{

DOCTEST_TEST_CASE("Init")
{
    // Making the driver object on the stack causes a weird libc error, so make it on the heap instead
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::DeviceStatus status = driver->getStatus();
    // Note double-parentheses to avoid compile error from doctest trying to printf the DeviceStatus
    DOCTEST_CHECK((status == V1_0::DeviceStatus::AVAILABLE));
}

DOCTEST_TEST_CASE("TestCapabilities")
{
    // Making the driver object on the stack causes a weird libc error, so make it on the heap instead
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus error;
    V1_0::Capabilities cap;

    auto cb = [&](V1_0::ErrorStatus status, const V1_0::Capabilities& capabilities)
    {
        error = status;
        cap = capabilities;
    };

    driver->getCapabilities(cb);

    DOCTEST_CHECK((int)error == (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(cap.float32Performance.execTime > 0.f);
    DOCTEST_CHECK(cap.float32Performance.powerUsage > 0.f);
    DOCTEST_CHECK(cap.quantized8Performance.execTime > 0.f);
    DOCTEST_CHECK(cap.quantized8Performance.powerUsage > 0.f);
}

}
