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
namespace hal_1_1
{

class ArmnnDriverImpl
{
public:
    static Return<void> getCapabilities_1_1(const armnn::IRuntimePtr& runtime,
                                            V1_1::IDevice::getCapabilities_1_1_cb cb);
};

} // namespace hal_1_1
} // namespace armnn_driver
