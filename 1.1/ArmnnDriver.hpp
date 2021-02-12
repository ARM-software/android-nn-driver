//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include "../ArmnnDevice.hpp"
#include "ArmnnDriverImpl.hpp"
#include "HalPolicy.hpp"

#include "../ArmnnDriverImpl.hpp"
#include "../1.0/ArmnnDriverImpl.hpp"
#include "../1.0/HalPolicy.hpp"

#include <log/log.h>

namespace armnn_driver
{
namespace hal_1_1
{

class ArmnnDriver : public ArmnnDevice, public V1_1::IDevice
{
public:
    ArmnnDriver(DriverOptions options)
        : ArmnnDevice(std::move(options))
    {
        ALOGV("hal_1_1::ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}

public:

    Return<void> getCapabilities(V1_0::IDevice::getCapabilities_cb cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::getCapabilities()");

        return hal_1_0::ArmnnDriverImpl::getCapabilities(m_Runtime, cb);
    }

    Return<void> getSupportedOperations(const V1_0::Model& model,
                                        V1_0::IDevice::getSupportedOperations_cb cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::getSupportedOperations()");

        return armnn_driver::ArmnnDriverImpl<hal_1_0::HalPolicy>::getSupportedOperations(m_Runtime,
                                                                                         m_Options,
                                                                                         model,
                                                                                         cb);
    }

    Return<V1_0::ErrorStatus> prepareModel(const V1_0::Model& model,
                                           const android::sp<V1_0::IPreparedModelCallback>& cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::prepareModel()");

        return armnn_driver::ArmnnDriverImpl<hal_1_0::HalPolicy>::prepareModel(m_Runtime,
                                                                               m_ClTunedParameters,
                                                                               m_Options,
                                                                               model,
                                                                               cb);
    }

    Return<void> getCapabilities_1_1(V1_1::IDevice::getCapabilities_1_1_cb cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::getCapabilities_1_1()");

        return hal_1_1::ArmnnDriverImpl::getCapabilities_1_1(m_Runtime, cb);
    }

    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                            V1_1::IDevice::getSupportedOperations_1_1_cb cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::getSupportedOperations_1_1()");

        return armnn_driver::ArmnnDriverImpl<hal_1_1::HalPolicy>::getSupportedOperations(m_Runtime,
                                                                                         m_Options,
                                                                                         model,
                                                                                         cb);
    }

    Return<V1_0::ErrorStatus> prepareModel_1_1(const V1_1::Model& model,
                                               V1_1::ExecutionPreference preference,
                                               const android::sp<V1_0::IPreparedModelCallback>& cb) override
    {
        ALOGV("hal_1_1::ArmnnDriver::prepareModel_1_1()");

        if (!(preference == V1_1::ExecutionPreference::LOW_POWER ||
              preference == V1_1::ExecutionPreference::FAST_SINGLE_ANSWER ||
              preference == V1_1::ExecutionPreference::SUSTAINED_SPEED))
        {
            ALOGV("hal_1_1::ArmnnDriver::prepareModel_1_1: Invalid execution preference");
            cb->notify(V1_0::ErrorStatus::INVALID_ARGUMENT, nullptr);
            return V1_0::ErrorStatus::INVALID_ARGUMENT;
        }

        return armnn_driver::ArmnnDriverImpl<hal_1_1::HalPolicy>::prepareModel(m_Runtime,
                                                                               m_ClTunedParameters,
                                                                               m_Options,
                                                                               model,
                                                                               cb,
                                                                               model.relaxComputationFloat32toFloat16
                                                                               && m_Options.GetFp16Enabled());
    }

    Return<V1_0::DeviceStatus> getStatus() override
    {
        ALOGV("hal_1_1::ArmnnDriver::getStatus()");

        return armnn_driver::ArmnnDriverImpl<hal_1_1::HalPolicy>::getStatus();
    }
};

} // namespace hal_1_1
} // namespace armnn_driver
