//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDriver.hpp"

#include <hidl/LegacySupport.h>
#include <log/log.h>

#include <string>
#include <vector>

using namespace armnn_driver;
using namespace std;

int main(int argc, char** argv)
{
    android::sp<ArmnnDriver> driver;
    try
    {
        driver = new ArmnnDriver(DriverOptions(argc, argv));
    }
    catch (const std::exception& e)
    {
        ALOGE("Could not create driver: %s", e.what());
        return EXIT_FAILURE;
    }

    android::hardware::configureRpcThreadpool(1, true);
    android::status_t status = android::UNKNOWN_ERROR;
    try
    {
        status = driver->registerAsService("armnn");
    }
    catch (const std::exception& e)
    {
        ALOGE("Could not register service: %s", e.what());
        return EXIT_FAILURE;
    }
    if (status != android::OK)
    {
        ALOGE("Could not register service");
        return EXIT_FAILURE;
    }

    android::hardware::joinRpcThreadpool();
    ALOGE("Service exited!");
    return EXIT_FAILURE;
}
