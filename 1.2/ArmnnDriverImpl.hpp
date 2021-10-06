//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include "../CacheDataHandler.hpp"
#include "../DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

#include <NeuralNetworks.h>

#ifdef ARMNN_ANDROID_R
using namespace android::nn::hal;
#endif

#ifdef ARMNN_ANDROID_S
using namespace android::hardware;
#endif

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;

namespace armnn_driver
{
namespace hal_1_2
{

class ArmnnDriverImpl
{
public:
    using HidlToken = android::hardware::hidl_array<uint8_t, ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN>;

    static Return<V1_0::ErrorStatus> prepareArmnnModel_1_2(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const V1_2::Model& model,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
        const HidlToken& token,
        const android::sp<V1_2::IPreparedModelCallback>& cb,
        bool float32ToFloat16 = false);

    static Return<V1_0::ErrorStatus> prepareModelFromCache(
        const armnn::IRuntimePtr& runtime,
        const DriverOptions& options,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
        const HidlToken& token,
        const android::sp<V1_2::IPreparedModelCallback>& cb,
        bool float32ToFloat16 = false);

    static Return<void> getCapabilities_1_2(const armnn::IRuntimePtr& runtime,
                                            V1_2::IDevice::getCapabilities_1_2_cb cb);
};

} // namespace hal_1_2
} // namespace armnn_driver