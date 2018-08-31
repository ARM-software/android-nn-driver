//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <HalInterfaces.h>

#include "ArmnnDriverImpl.hpp"
#include "ArmnnDevice.hpp"

#include <log/log.h>

namespace armnn_driver
{
namespace V1_0
{

class ArmnnDriver : public ArmnnDevice, public ::android::hardware::neuralnetworks::V1_0::IDevice
{
public:
    ArmnnDriver(DriverOptions options)
        : ArmnnDevice(std::move(options))
    {
        ALOGV("V1_0::ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}

public:
    Return<void> getCapabilities(
            ::android::hardware::neuralnetworks::V1_0::IDevice::getCapabilities_cb cb)
    {
        ALOGV("V1_0::ArmnnDriver::getCapabilities()");

        return ArmnnDriverImpl::getCapabilities(m_Runtime, cb);
    }

    Return<void> getSupportedOperations(
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            ::android::hardware::neuralnetworks::V1_0::IDevice::getSupportedOperations_cb cb)
    {
        ALOGV("V1_0::ArmnnDriver::getSupportedOperations()");

        return ArmnnDriverImpl::getSupportedOperations(m_Runtime, m_Options, model, cb);
    }

    Return<ErrorStatus> prepareModel(
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            const android::sp<IPreparedModelCallback>& cb)
    {
        ALOGV("V1_0::ArmnnDriver::prepareModel()");

        return ArmnnDriverImpl::prepareModel(m_Runtime, m_ClTunedParameters, m_Options, model, cb);
    }

    Return<DeviceStatus> getStatus()
    {
        ALOGV("V1_0::ArmnnDriver::getStatus()");

        return ArmnnDriverImpl::getStatus();
    }
};

} // armnn_driver::namespace V1_0
} // namespace armnn_driver
