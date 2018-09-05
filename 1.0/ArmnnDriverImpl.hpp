//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include "../DriverOptions.hpp"

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
};

} // namespace armnn_driver::V1_0
} // namespace armnn_driver
