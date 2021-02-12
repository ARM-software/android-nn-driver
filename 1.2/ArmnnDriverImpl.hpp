//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include "../DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

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
    static Return<V1_0::ErrorStatus> prepareArmnnModel_1_2(const armnn::IRuntimePtr& runtime,
                                                           const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
                                                           const DriverOptions& options,
                                                           const V1_2::Model& model,
                                                           const android::sp<V1_2::IPreparedModelCallback>& cb,
                                                           bool float32ToFloat16 = false);

    static Return<void> getCapabilities_1_2(const armnn::IRuntimePtr& runtime,
                                            V1_2::IDevice::getCapabilities_1_2_cb cb);
};

} // namespace hal_1_2
} // namespace armnn_driver