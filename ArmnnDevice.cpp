//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDevice.hpp"

#include <OperationsUtils.h>

#include <log/log.h>

#include <memory>

using namespace android;

namespace armnn_driver
{

ArmnnDevice::ArmnnDevice(DriverOptions options)
    : m_Runtime(nullptr, nullptr)
    , m_ClTunedParameters(nullptr)
    , m_Options(std::move(options))
{
    ALOGV("ArmnnDevice::ArmnnDevice()");

    armnn::ConfigureLogging(false, m_Options.IsVerboseLoggingEnabled(), armnn::LogSeverity::Trace);
    if (m_Options.IsVerboseLoggingEnabled())
    {
        SetMinimumLogSeverity(base::VERBOSE);
    }
    else
    {
        SetMinimumLogSeverity(base::INFO);
    }

#if defined(ARMCOMPUTECL_ENABLED)
    try
    {
        armnn::IRuntime::CreationOptions options;
        if (!m_Options.GetClTunedParametersFile().empty())
        {
            m_ClTunedParameters = armnn::IGpuAccTunedParameters::Create(m_Options.GetClTunedParametersMode());
            try
            {
                m_ClTunedParameters->Load(m_Options.GetClTunedParametersFile().c_str());
            }
            catch (const armnn::Exception& error)
            {
                // This is only a warning because the file won't exist the first time you are generating it.
                ALOGW("ArmnnDevice: Failed to load CL tuned parameters file '%s': %s",
                      m_Options.GetClTunedParametersFile().c_str(), error.what());
            }
            options.m_GpuAccTunedParameters = m_ClTunedParameters;
        }

        options.m_EnableGpuProfiling = m_Options.IsGpuProfilingEnabled();

        m_Runtime = armnn::IRuntime::Create(options);
    }
    catch (const armnn::ClRuntimeUnavailableException& error)
    {
        ALOGE("ArmnnDevice: Failed to setup CL runtime: %s. Device will be unavailable.", error.what());
    }
#endif
}

} // namespace armnn_driver
