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
namespace V1_1
{

class ArmnnDriverImpl
{
public:
    static Return<void> getCapabilities_1_1(
            const armnn::IRuntimePtr& runtime,
            ::android::hardware::neuralnetworks::V1_1::IDevice::getCapabilities_1_1_cb cb);
    static Return<void> getSupportedOperations_1_1(
            const armnn::IRuntimePtr& runtime,
            const DriverOptions& options,
            const ::android::hardware::neuralnetworks::V1_1::Model& model,
            ::android::hardware::neuralnetworks::V1_1::IDevice::getSupportedOperations_1_1_cb cb);
    static Return<ErrorStatus> prepareModel_1_1(
            const armnn::IRuntimePtr& runtime,
            const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
            const DriverOptions& options,
            const ::android::hardware::neuralnetworks::V1_1::Model& model,
            const android::sp<IPreparedModelCallback>& cb);
};

} // namespace armnn_driver::V1_1
} // namespace armnn_driver
