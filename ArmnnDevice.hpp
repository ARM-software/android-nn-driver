//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

namespace armnn_driver
{

class ArmnnDevice
{
protected:
    ArmnnDevice(DriverOptions options);
    virtual ~ArmnnDevice() {}

protected:
    armnn::IRuntimePtr m_Runtime;
    armnn::IGpuAccTunedParametersPtr m_ClTunedParameters;
    DriverOptions m_Options;
};

} // namespace armnn_driver
