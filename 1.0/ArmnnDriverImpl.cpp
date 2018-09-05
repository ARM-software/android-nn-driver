//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmnnDriverImpl.hpp"
#include "../SystemPropertiesUtils.hpp"

#include <log/log.h>

using namespace std;
using namespace android;
using namespace android::nn;
using namespace android::hardware;

namespace
{

const char *g_Float32PerformanceExecTimeName = "ArmNN.float32Performance.execTime";
const char *g_Float32PerformancePowerUsageName = "ArmNN.float32Performance.powerUsage";
const char *g_Quantized8PerformanceExecTimeName = "ArmNN.quantized8Performance.execTime";
const char *g_Quantized8PerformancePowerUsageName = "ArmNN.quantized8Performance.powerUsage";

} // anonymous namespace

namespace armnn_driver
{
namespace V1_0
{

Return<void> ArmnnDriverImpl::getCapabilities(
        const armnn::IRuntimePtr& runtime,
        neuralnetworks::V1_0::IDevice::getCapabilities_cb cb)
{
    ALOGV("V1_0::ArmnnDriverImpl::getCapabilities()");

    neuralnetworks::V1_0::Capabilities capabilities;
    if (runtime)
    {
        capabilities.float32Performance.execTime =
            ParseSystemProperty(g_Float32PerformanceExecTimeName, .1f);

        capabilities.float32Performance.powerUsage =
            ParseSystemProperty(g_Float32PerformancePowerUsageName, .1f);

        capabilities.quantized8Performance.execTime =
            ParseSystemProperty(g_Quantized8PerformanceExecTimeName, .1f);

        capabilities.quantized8Performance.powerUsage =
            ParseSystemProperty(g_Quantized8PerformancePowerUsageName, .1f);

        cb(ErrorStatus::NONE, capabilities);
    }
    else
    {
        capabilities.float32Performance.execTime = 0;
        capabilities.float32Performance.powerUsage = 0;
        capabilities.quantized8Performance.execTime = 0;
        capabilities.quantized8Performance.powerUsage = 0;

        cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }

    return Void();
}

} // namespace armnn_driver::V1_0
} // namespace armnn_driver
