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
namespace V1_0
{

class ArmnnDriverImpl
{
public:
    static Return<void> getCapabilities(
            const armnn::IRuntimePtr& runtime,
            ::android::hardware::neuralnetworks::V1_0::IDevice::getCapabilities_cb cb);
    static Return<void> getSupportedOperations(
            const armnn::IRuntimePtr& runtime,
            const DriverOptions& options,
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            ::android::hardware::neuralnetworks::V1_0::IDevice::getSupportedOperations_cb cb);
    static Return<ErrorStatus> prepareModel(
            const armnn::IRuntimePtr& runtime,
            const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
            const DriverOptions& options,
            const ::android::hardware::neuralnetworks::V1_0::Model& model,
            const android::sp<IPreparedModelCallback>& cb,
            bool  float32ToFloat16 = false);
    static Return<DeviceStatus> getStatus();
};

} // namespace armnn_driver::V1_0
} // namespace armnn_driver
