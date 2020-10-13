//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

#include <set>
#include <string>
#include <vector>

namespace armnn_driver
{

class DriverOptions
{
public:
    DriverOptions(armnn::Compute computeDevice, bool fp16Enabled = false);
    DriverOptions(const std::vector<armnn::BackendId>& backends, bool fp16Enabled);
    DriverOptions(int argc, char** argv);
    DriverOptions(DriverOptions&& other) = default;

    const std::vector<armnn::BackendId>& GetBackends() const { return m_Backends; }
    bool IsVerboseLoggingEnabled() const { return m_VerboseLogging; }
    const std::string& GetRequestInputsAndOutputsDumpDir() const { return m_RequestInputsAndOutputsDumpDir; }
    const std::string& GetServiceName() const { return m_ServiceName; }
    const std::set<unsigned int>& GetForcedUnsupportedOperations() const { return m_ForcedUnsupportedOperations; }
    const std::string& GetClTunedParametersFile() const { return m_ClTunedParametersFile; }
    armnn::IGpuAccTunedParameters::Mode GetClTunedParametersMode() const { return m_ClTunedParametersMode; }
    armnn::IGpuAccTunedParameters::TuningLevel GetClTuningLevel() const { return m_ClTuningLevel; }
    bool IsGpuProfilingEnabled() const { return m_EnableGpuProfiling; }
    bool IsFastMathEnabled() const { return m_FastMathEnabled; }
    bool GetFp16Enabled() const { return m_fp16Enabled; }
    void SetBackends(const std::vector<armnn::BackendId>& backends) { m_Backends = backends; }
    bool ShouldExit() const { return m_ShouldExit; }
    int GetExitCode() const { return m_ExitCode; }

private:
    std::vector<armnn::BackendId> m_Backends;
    bool m_VerboseLogging;
    std::string m_RequestInputsAndOutputsDumpDir;
    std::string m_ServiceName;
    std::set<unsigned int> m_ForcedUnsupportedOperations;
    std::string m_ClTunedParametersFile;
    armnn::IGpuAccTunedParameters::Mode m_ClTunedParametersMode;
    armnn::IGpuAccTunedParameters::TuningLevel m_ClTuningLevel;
    bool m_EnableGpuProfiling;
    bool m_fp16Enabled;
    bool m_FastMathEnabled;
    bool m_ShouldExit;
    int m_ExitCode;
};

} // namespace armnn_driver
