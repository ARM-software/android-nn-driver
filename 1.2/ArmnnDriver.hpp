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
#include "../1.2/ArmnnDriverImpl.hpp"
#include "../1.2/HalPolicy.hpp"
#include "../1.1/ArmnnDriverImpl.hpp"
#include "../1.1/HalPolicy.hpp"
#include "../1.0/ArmnnDriverImpl.hpp"
#include "../1.0/HalPolicy.hpp"

#include <armnn/BackendHelper.hpp>

#include <log/log.h>

namespace armnn_driver
{
namespace hal_1_2
{

class ArmnnDriver : public ArmnnDevice, public V1_2::IDevice
{
public:

    ArmnnDriver(DriverOptions options)
        : ArmnnDevice(std::move(options))
    {
        ALOGV("hal_1_2::ArmnnDriver::ArmnnDriver()");
    }
    ~ArmnnDriver() {}

    using HidlToken = android::hardware::hidl_array<uint8_t, ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN>;

public:
    Return<void> getCapabilities(V1_0::IDevice::getCapabilities_cb cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::getCapabilities()");

        return hal_1_0::ArmnnDriverImpl::getCapabilities(m_Runtime, cb);
    }

    Return<void> getSupportedOperations(const V1_0::Model& model,
                                        V1_0::IDevice::getSupportedOperations_cb cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::getSupportedOperations()");

        return armnn_driver::ArmnnDriverImpl<hal_1_0::HalPolicy>::getSupportedOperations(m_Runtime,
                                                                                         m_Options,
                                                                                         model,
                                                                                         cb);
    }

    Return<V1_0::ErrorStatus> prepareModel(const V1_0::Model& model,
                                           const android::sp<V1_0::IPreparedModelCallback>& cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::prepareModel()");

        return armnn_driver::ArmnnDriverImpl<hal_1_0::HalPolicy>::prepareModel(m_Runtime,
                                                                               m_ClTunedParameters,
                                                                               m_Options,
                                                                               model,
                                                                               cb);
    }

    Return<void> getCapabilities_1_1(V1_1::IDevice::getCapabilities_1_1_cb cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::getCapabilities_1_1()");

        return hal_1_1::ArmnnDriverImpl::getCapabilities_1_1(m_Runtime, cb);
    }

    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                            V1_1::IDevice::getSupportedOperations_1_1_cb cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::getSupportedOperations_1_1()");
        return armnn_driver::ArmnnDriverImpl<hal_1_1::HalPolicy>::getSupportedOperations(m_Runtime,
                                                                                         m_Options,
                                                                                         model,
                                                                                         cb);
    }

    Return<V1_0::ErrorStatus> prepareModel_1_1(const V1_1::Model& model,
                                               V1_1::ExecutionPreference preference,
                                               const android::sp<V1_0::IPreparedModelCallback>& cb) override
    {
        ALOGV("hal_1_2::ArmnnDriver::prepareModel_1_1()");

        if (!(preference == V1_1::ExecutionPreference::LOW_POWER ||
              preference == V1_1::ExecutionPreference::FAST_SINGLE_ANSWER ||
              preference == V1_1::ExecutionPreference::SUSTAINED_SPEED))
        {
            ALOGV("hal_1_2::ArmnnDriver::prepareModel_1_1: Invalid execution preference");
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
        ALOGV("hal_1_2::ArmnnDriver::getStatus()");

        return armnn_driver::ArmnnDriverImpl<hal_1_2::HalPolicy>::getStatus();
    }

    Return<void> getVersionString(getVersionString_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getVersionString()");

        cb(V1_0::ErrorStatus::NONE, "ArmNN");
        return Void();
    }

    Return<void> getType(getType_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getType()");
        const auto device_type = hal_1_2::HalPolicy::GetDeviceTypeFromOptions(this->m_Options);
        cb(V1_0::ErrorStatus::NONE, device_type);
        return Void();
    }

    Return<V1_0::ErrorStatus> prepareModelFromCache(
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
        const HidlToken& token,
        const android::sp<V1_2::IPreparedModelCallback>& cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::prepareModelFromCache()");
        return ArmnnDriverImpl::prepareModelFromCache(m_Runtime,
                                                      m_Options,
                                                      modelCacheHandle,
                                                      dataCacheHandle,
                                                      token,
                                                      cb);
    }

    Return<V1_0::ErrorStatus> prepareModel_1_2(
        const V1_2::Model& model, V1_1::ExecutionPreference preference,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
        const HidlToken& token,
        const android::sp<V1_2::IPreparedModelCallback>& cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::prepareModel_1_2()");

        if (!(preference == V1_1::ExecutionPreference::LOW_POWER ||
              preference == V1_1::ExecutionPreference::FAST_SINGLE_ANSWER ||
              preference == V1_1::ExecutionPreference::SUSTAINED_SPEED))
        {
            ALOGV("hal_1_2::ArmnnDriver::prepareModel_1_2: Invalid execution preference");
            cb->notify(V1_0::ErrorStatus::INVALID_ARGUMENT, nullptr);
            return V1_0::ErrorStatus::INVALID_ARGUMENT;
        }

        return ArmnnDriverImpl::prepareArmnnModel_1_2(m_Runtime,
                                                      m_ClTunedParameters,
                                                      m_Options,
                                                      model,
                                                      modelCacheHandle,
                                                      dataCacheHandle,
                                                      token,
                                                      cb,
                                                      model.relaxComputationFloat32toFloat16
                                                      && m_Options.GetFp16Enabled());
    }

    Return<void> getSupportedExtensions(getSupportedExtensions_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getSupportedExtensions()");
        cb(V1_0::ErrorStatus::NONE, {/* No extensions. */});
        return Void();
    }

    Return<void> getCapabilities_1_2(getCapabilities_1_2_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getCapabilities()");

        return hal_1_2::ArmnnDriverImpl::getCapabilities_1_2(m_Runtime, cb);
    }

    Return<void> getSupportedOperations_1_2(const V1_2::Model& model,
                                            getSupportedOperations_1_2_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getSupportedOperations()");

        return armnn_driver::ArmnnDriverImpl<hal_1_2::HalPolicy>::getSupportedOperations(m_Runtime,
                                                                                         m_Options,
                                                                                         model,
                                                                                         cb);
    }

    Return<void> getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb cb)
    {
        ALOGV("hal_1_2::ArmnnDriver::getSupportedExtensions()");
        unsigned int numberOfCachedModelFiles = 0;
        for (auto& backend : m_Options.GetBackends())
        {
            numberOfCachedModelFiles += GetNumberOfCacheFiles(backend);
        }
        cb(V1_0::ErrorStatus::NONE, numberOfCachedModelFiles,   1ul);
        return Void();
    }
};

} // namespace hal_1_2
} // namespace armnn_driver
