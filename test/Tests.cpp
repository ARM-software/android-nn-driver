//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#define LOG_TAG "ArmnnDriverTests"
#define BOOST_TEST_MODULE armnn_driver_tests
#include <boost/test/unit_test.hpp>
#include <log/log.h>

#include "DriverTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(DriverTests)

using ArmnnDriver = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using namespace driverTestHelpers;

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

    ErrorStatus error;
    V1_0::Capabilities cap;

    ArmnnDriver::getCapabilities_cb cb = [&](ErrorStatus status, const V1_0::Capabilities& capabilities)
    {
        error = status;
        cap = capabilities;
    };

    driver->getCapabilities(cb);

    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(cap.float32Performance.execTime > 0.f);
    BOOST_TEST(cap.float32Performance.powerUsage > 0.f);
    BOOST_TEST(cap.quantized8Performance.execTime > 0.f);
    BOOST_TEST(cap.quantized8Performance.powerUsage > 0.f);
}

BOOST_AUTO_TEST_SUITE_END()
