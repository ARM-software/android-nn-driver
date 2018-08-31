//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/ArmNN.hpp>

#include <set>
#include <string>

namespace armnn_driver
{

class DriverOptions
{
public:
    DriverOptions(armnn::Compute computeDevice);
    DriverOptions(int argc, char** argv);
    DriverOptions(DriverOptions&& other) = default;

    armnn::Compute GetComputeDevice() const { return m_ComputeDevice; }
    bool IsVerboseLoggingEnabled() const { return m_VerboseLogging; }
    const std::string& GetRequestInputsAndOutputsDumpDir() const { return m_RequestInputsAndOutputsDumpDir; }
    const std::set<unsigned int>& GetForcedUnsupportedOperations() const { return m_ForcedUnsupportedOperations; }
    const std::string& GetClTunedParametersFile() const { return m_ClTunedParametersFile; }
    armnn::IGpuAccTunedParameters::Mode GetClTunedParametersMode() const { return m_ClTunedParametersMode; }
    bool IsGpuProfilingEnabled() const { return m_EnableGpuProfiling; }
    bool GetFp16Enabled() const { return m_fp16Enabled; }

private:
    armnn::Compute m_ComputeDevice;
    bool m_VerboseLogging;
    std::string m_RequestInputsAndOutputsDumpDir;
    std::set<unsigned int> m_ForcedUnsupportedOperations;
    std::string m_ClTunedParametersFile;
    armnn::IGpuAccTunedParameters::Mode m_ClTunedParametersMode;
    bool m_EnableGpuProfiling;
    bool m_fp16Enabled;
};

} // namespace armnn_driver
