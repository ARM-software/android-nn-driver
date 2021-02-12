//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverOptions.hpp"

#include <HalInterfaces.h>

#ifdef ARMNN_ANDROID_R
using namespace android::nn::hal;
#endif

#ifdef ARMNN_ANDROID_S
using namespace android::hardware;
#endif

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;
namespace V1_1 = ::android::hardware::neuralnetworks::V1_1;

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;
#endif

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_3
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;
namespace V1_3 = ::android::hardware::neuralnetworks::V1_3;
#endif

namespace armnn_driver
{

template <typename Callback, typename Context>
struct CallbackContext
{
    Callback callback;
    Context ctx;
};

template<typename HalPolicy>
class ArmnnDriverImpl
{
public:
    using HalModel                     = typename HalPolicy::Model;
    using HalGetSupportedOperations_cb = typename HalPolicy::getSupportedOperations_cb;
    using HalErrorStatus               = typename HalPolicy::ErrorStatus;

    static Return<void> getSupportedOperations(
            const armnn::IRuntimePtr& runtime,
            const DriverOptions& options,
            const HalModel& model,
            HalGetSupportedOperations_cb);

    static Return<V1_0::ErrorStatus> prepareModel(
            const armnn::IRuntimePtr& runtime,
            const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
            const DriverOptions& options,
            const HalModel& model,
            const android::sp<V1_0::IPreparedModelCallback>& cb,
            bool float32ToFloat16 = false);

    static Return<V1_0::DeviceStatus> getStatus();

};

} // namespace armnn_driver
