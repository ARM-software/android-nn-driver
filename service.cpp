//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDriver.hpp"

#include <hidl/LegacySupport.h>
#include <log/log.h>

#include <string>

using namespace armnn_driver;
using namespace std;

int main(int argc, char** argv)
{
    android::sp<ArmnnDriver> driver;
    DriverOptions driverOptions(argc, argv);

    if (driverOptions.ShouldExit())
    {
        return driverOptions.GetExitCode();
    }
    try
    {
        driver = new ArmnnDriver(DriverOptions(argc, argv));
    }
    catch (const std::exception& e)
    {
        ALOGE("Could not create driver: %s", e.what());
        std::cout << "Unable to start:" << std::endl
                  << "Could not create driver: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    android::hardware::configureRpcThreadpool(1, true);
    android::status_t status = android::UNKNOWN_ERROR;
    try
    {
        status = driver->registerAsService(driverOptions.GetServiceName());
    }
    catch (const std::exception& e)
    {
        ALOGE("Could not register service: %s", e.what());
        std::cout << "Unable to start:" << std::endl
                  << "Could not register service: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (status != android::OK)
    {
        ALOGE("Could not register service");
        std::cout << "Unable to start:" << std::endl
                  << "Could not register service" << std::endl;
        return EXIT_FAILURE;
    }
    android::hardware::joinRpcThreadpool();
    ALOGW("Service exited!");
    return EXIT_SUCCESS;
}
