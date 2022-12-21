//
// Copyright © 2017, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDevice.hpp"

#include <OperationsUtils.h>

#include <log/log.h>

#include <memory>
#include <string>

using namespace android;

namespace
{

std::string GetBackendString(const armnn_driver::DriverOptions& options)
{
    std::stringstream backends;
    for (auto&& b : options.GetBackends())
    {
        backends << b << " ";
    }
    return backends.str();
}

} // anonymous namespace

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

    armnn::IRuntime::CreationOptions runtimeOptions;

#if defined(ARMCOMPUTECL_ENABLED)
    try
    {
        if (!m_Options.GetClTunedParametersFile().empty())
        {
            m_ClTunedParameters = armnn::IGpuAccTunedParameters::Create(m_Options.GetClTunedParametersMode(),
                                                                        m_Options.GetClTuningLevel());
            try
            {
                m_ClTunedParameters->Load(m_Options.GetClTunedParametersFile().c_str());
            }
            catch (std::exception& error)
            {
                // This is only a warning because the file won't exist the first time you are generating it.
                ALOGW("ArmnnDevice: Failed to load CL tuned parameters file '%s': %s",
                      m_Options.GetClTunedParametersFile().c_str(), error.what());
            }
            runtimeOptions.m_GpuAccTunedParameters = m_ClTunedParameters;
        }
    }
    catch (const armnn::ClRuntimeUnavailableException& error)
    {
        ALOGE("ArmnnDevice: Failed to setup CL runtime: %s. Device will be unavailable.", error.what());
    }
    catch (std::exception& error)
    {
        ALOGE("ArmnnDevice: Unknown exception: %s. Device will be unavailable.", error.what());
    }
#endif
    runtimeOptions.m_EnableGpuProfiling = m_Options.IsGpuProfilingEnabled();
    m_Runtime = armnn::IRuntime::Create(runtimeOptions);

    std::vector<armnn::BackendId> backends;

    if (m_Runtime)
    {
        const armnn::BackendIdSet supportedDevices = m_Runtime->GetDeviceSpec().GetSupportedBackends();
        for (auto &backend : m_Options.GetBackends())
        {
            if (std::find(supportedDevices.cbegin(), supportedDevices.cend(), backend) == supportedDevices.cend())
            {
                ALOGW("ArmnnDevice: Requested unknown backend %s", backend.Get().c_str());
            }
            else
            {
                if (m_Options.isAsyncModelExecutionEnabled() &&
                    armnn::HasCapability(armnn::BackendOptions::BackendOption{"AsyncExecution", false}, backend))
                {
                    ALOGV("ArmnnDevice: ArmNN does not support AsyncExecution with the following backend: %s",
                          backend.Get().c_str());
                }
                else
                {
                    backends.push_back(backend);
                }
            }
        }
    }

    if (backends.empty())
    {
        // No known backend specified
        throw armnn::InvalidArgumentException("ArmnnDevice: No known backend specified.");
    }

    m_Options.SetBackends(backends);
    ALOGV("ArmnnDevice: Created device with the following backends: %s",
          GetBackendString(m_Options).c_str());
}

} // namespace armnn_driver
