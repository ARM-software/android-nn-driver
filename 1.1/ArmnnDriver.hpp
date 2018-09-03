//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <HalInterfaces.h>

#include "ArmnnDevice.hpp"
#include "../ArmnnDriverImpl.hpp"

#include <log/log.h>

namespace armnn_driver
{
namespace V1_1
{

class ArmnnDriver : public ArmnnDevice, public ::android::hardware::neuralnetworks::V1_1::IDevice
{
public:
    ArmnnDriver(DriverOptions options)
        : ArmnnDevice(std::move(options))
    {
        ALOGV("V1_1::ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}

public:
    Return<void> getCapabilities(
            ::android::hardware::neuralnetworks::V1_0::IDevice::getCapabilities_cb cb) override
    {
        ALOGV("V1_1::ArmnnDriver::getCapabilities()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_0>::getCapabilities(m_Runtime,
                                                                              cb);
    }

    Return<void> getSupportedOperations(
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            ::android::hardware::neuralnetworks::V1_0::IDevice::getSupportedOperations_cb cb) override
    {
        ALOGV("V1_1::ArmnnDriver::getSupportedOperations()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_0>::getSupportedOperations(m_Runtime,
                                                                                     m_Options,
                                                                                     model,
                                                                                     cb);
    }

    Return<ErrorStatus> prepareModel(
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            const android::sp<IPreparedModelCallback>& cb) override
    {
        ALOGV("V1_1::ArmnnDriver::prepareModel()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_0>::prepareModel(m_Runtime,
                                                                           m_ClTunedParameters,
                                                                           m_Options,
                                                                           model,
                                                                           cb);
    }

    Return<void> getCapabilities_1_1(
            ::android::hardware::neuralnetworks::V1_1::IDevice::getCapabilities_1_1_cb cb) override
    {
        ALOGV("V1_1::ArmnnDriver::getCapabilities_1_1()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_1>::getCapabilities(m_Runtime,
                                                                              cb);
    }

    Return<void> getSupportedOperations_1_1(
            const ::android::hardware::neuralnetworks::V1_1::Model& model,
            ::android::hardware::neuralnetworks::V1_1::IDevice::getSupportedOperations_1_1_cb cb) override
    {
        ALOGV("V1_1::ArmnnDriver::getSupportedOperations_1_1()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_1>::getSupportedOperations(m_Runtime,
                                                                                     m_Options,
                                                                                     model,
                                                                                     cb);
    }

    Return<ErrorStatus> prepareModel_1_1(
            const ::android::hardware::neuralnetworks::V1_1::Model& model,
            ::android::hardware::neuralnetworks::V1_1::ExecutionPreference preference,
            const android::sp<IPreparedModelCallback>& cb) override
    {
        ALOGV("V1_1::ArmnnDriver::prepareModel_1_1()");

        if (!(preference == ExecutionPreference::LOW_POWER ||
              preference == ExecutionPreference::FAST_SINGLE_ANSWER ||
              preference == ExecutionPreference::SUSTAINED_SPEED))
        {
            ALOGV("V1_1::ArmnnDriver::prepareModel_1_1: Invalid execution preference");
            cb->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
            return ErrorStatus::INVALID_ARGUMENT;
        }

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_1>::prepareModel(m_Runtime,
                                                                           m_ClTunedParameters,
                                                                           m_Options,
                                                                           model,
                                                                           cb);
    }

    Return<DeviceStatus> getStatus() override
    {
        ALOGV("V1_1::ArmnnDriver::getStatus()");

        return armnn_driver::ArmnnDriverImpl<HalVersion_1_1>::getStatus();
    }
};

} // armnn_driver::namespace V1_1
} // namespace armnn_driver
