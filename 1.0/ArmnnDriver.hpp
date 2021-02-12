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

#include <log/log.h>

namespace armnn_driver
{
namespace hal_1_0
{

class ArmnnDriver : public ArmnnDevice, public V1_0::IDevice
{
public:
    ArmnnDriver(DriverOptions options)
        : ArmnnDevice(std::move(options))
    {
        ALOGV("hal_1_0::ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}

public:
    Return<void> getCapabilities(V1_0::IDevice::getCapabilities_cb cb) override
    {
        ALOGV("hal_1_0::ArmnnDriver::getCapabilities()");

        return hal_1_0::ArmnnDriverImpl::getCapabilities(m_Runtime, cb);
    }

    Return<void> getSupportedOperations(const V1_0::Model& model,
                                        V1_0::IDevice::getSupportedOperations_cb cb) override
    {
        ALOGV("hal_1_0::ArmnnDriver::getSupportedOperations()");

        return armnn_driver::ArmnnDriverImpl<HalPolicy>::getSupportedOperations(m_Runtime, m_Options, model, cb);
    }

    Return<V1_0::ErrorStatus> prepareModel(const V1_0::Model& model,
                                           const android::sp<V1_0::IPreparedModelCallback>& cb) override
    {
        ALOGV("hal_1_0::ArmnnDriver::prepareModel()");

        return armnn_driver::ArmnnDriverImpl<HalPolicy>::prepareModel(m_Runtime,
                                                                      m_ClTunedParameters,
                                                                      m_Options,
                                                                      model,
                                                                      cb);
    }

    Return<V1_0::DeviceStatus> getStatus() override
    {
        ALOGV("hal_1_0::ArmnnDriver::getStatus()");

        return armnn_driver::ArmnnDriverImpl<HalPolicy>::getStatus();
    }
};

} // namespace hal_1_0
} // namespace armnn_driver
