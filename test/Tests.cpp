//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#define LOG_TAG "ArmnnDriverTests"
#define BOOST_TEST_MODULE armnn_driver_tests
#include <boost/test/unit_test.hpp>
#include <log/log.h>

#include "DriverTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(DriverTests)

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

BOOST_AUTO_TEST_CASE(Init)
{
    // Making the driver object on the stack causes a weird libc error, so make it on the heap instead
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    DeviceStatus status = driver->getStatus();
    // Note double-parentheses to avoid compile error from Boost trying to printf the DeviceStatus
    BOOST_TEST((status == DeviceStatus::AVAILABLE));
}

BOOST_AUTO_TEST_CASE(TestCapabilities)
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

    BOOST_TEST((int)error == (int)V1_0::ErrorStatus::NONE);
    BOOST_TEST(cap.float32Performance.execTime > 0.f);
    BOOST_TEST(cap.float32Performance.powerUsage > 0.f);
    BOOST_TEST(cap.quantized8Performance.execTime > 0.f);
    BOOST_TEST(cap.quantized8Performance.powerUsage > 0.f);
}

BOOST_AUTO_TEST_SUITE_END()
