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
    android::sp<ArmnnDriver> driver = new ArmnnDriver(DriverOptions(argc, argv));

    android::hardware::configureRpcThreadpool(1, true);
    if (driver->registerAsService("armnn") != android::OK)
    {
        ALOGE("Could not register service");
        return 1;
    }
    android::hardware::joinRpcThreadpool();
    ALOGE("Service exited!");
    return 1;
}
