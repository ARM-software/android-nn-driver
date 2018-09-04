//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <HalInterfaces.h>

#include "DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

namespace armnn_driver
{

struct HalVersion_1_0
{
    using Model = ::android::hardware::neuralnetworks::V1_0::Model;
    using getSupportedOperations_cb = ::android::hardware::neuralnetworks::V1_0::IDevice::getSupportedOperations_cb;
};

#if defined(ARMNN_ANDROID_NN_V1_1) // Using ::android::hardware::neuralnetworks::V1_1.
struct HalVersion_1_1
{
    using Model = ::android::hardware::neuralnetworks::V1_1::Model;
    using getSupportedOperations_cb = ::android::hardware::neuralnetworks::V1_1::IDevice::getSupportedOperations_1_1_cb;
};
#endif

template <typename HalVersion>
class ArmnnDriverImpl
{
public:
    using HalModel = typename HalVersion::Model;
    using HalGetSupportedOperations_cb = typename HalVersion::getSupportedOperations_cb;

    static Return<void> getSupportedOperations(
            const armnn::IRuntimePtr& runtime,
            const DriverOptions& options,
            const HalModel& model,
            HalGetSupportedOperations_cb);
    static Return<ErrorStatus> prepareModel(
            const armnn::IRuntimePtr& runtime,
            const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
            const DriverOptions& options,
            const HalModel& model,
            const android::sp<IPreparedModelCallback>& cb,
            bool float32ToFloat16 = false);
    static Return<DeviceStatus> getStatus();
};

} // namespace armnn_driver
