//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "DriverOptions.hpp"
#include "Utils.hpp"

#include <log/log.h>
#include "SystemPropertiesUtils.hpp"

#include <OperationsUtils.h>

#include <cxxopts/cxxopts.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <sstream>

using namespace android;
using namespace std;

namespace armnn_driver
{

DriverOptions::DriverOptions(armnn::Compute computeDevice, bool fp16Enabled)
    : m_Backends({computeDevice})
    , m_VerboseLogging(false)
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(fp16Enabled)
    , m_FastMathEnabled(false)
{
}

DriverOptions::DriverOptions(const std::vector<armnn::BackendId>& backends, bool fp16Enabled)
    : m_Backends(backends)
    , m_VerboseLogging(false)
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(fp16Enabled)
    , m_FastMathEnabled(false)
{
}

DriverOptions::DriverOptions(int argc, char** argv)
    : m_VerboseLogging(false)
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(false)
    , m_FastMathEnabled(false)
{
    std::string unsupportedOperationsAsString;
    std::string clTunedParametersModeAsString;
    std::string clTuningLevelAsString;
    std::vector<std::string> backends;

    cxxopts::Options optionsDesc("Options");
    try
    {
        optionsDesc.add_options()
        ("c,compute",
         "Comma separated list of backends to run layers on. Examples of possible values are: CpuRef, CpuAcc, GpuAcc",
         cxxopts::value<std::vector<std::string>>(backends))

        ("v,verbose-logging", "Turns verbose logging on",
         cxxopts::value<bool>(m_VerboseLogging)->default_value("false"))

        ("d,request-inputs-and-outputs-dump-dir",
         "If non-empty, the directory where request inputs and outputs should be dumped",
         cxxopts::value<std::string>(m_RequestInputsAndOutputsDumpDir)->default_value(""))

        ("n,service-name",
         "If non-empty, the driver service name to be registered",
         cxxopts::value<std::string>(m_ServiceName)->default_value("armnn"))

        ("u,unsupported-operations",
         "If non-empty, a comma-separated list of operation indices which the driver will forcibly "
         "consider unsupported",
         cxxopts::value<std::string>(unsupportedOperationsAsString)->default_value(""))

        ("t,cl-tuned-parameters-file",
         "If non-empty, the given file will be used to load/save CL tuned parameters. "
         "See also --cl-tuned-parameters-mode",
         cxxopts::value<std::string>(m_ClTunedParametersFile)->default_value(""))

        ("m,cl-tuned-parameters-mode",
         "If 'UseTunedParameters' (the default), will read CL tuned parameters from the file specified by "
         "--cl-tuned-parameters-file. "
         "If 'UpdateTunedParameters', will also find the optimum parameters when preparing new networks and update "
         "the file accordingly.",
         cxxopts::value<std::string>(clTunedParametersModeAsString)->default_value("UseTunedParameters"))

        ("o,cl-tuning-level",
         "exhaustive: all lws values are tested "
         "normal: reduced number of lws values but enough to still have the performance really close to the "
         "exhaustive approach "
         "rapid: only 3 lws values should be tested for each kernel ",
         cxxopts::value<std::string>(clTuningLevelAsString)->default_value("rapid"))

        ("a,fast-math", "Turns FastMath on",
         cxxopts::value<bool>(m_FastMathEnabled)->default_value("false"))

        ("p,gpu-profiling", "Turns GPU profiling on",
         cxxopts::value<bool>(m_EnableGpuProfiling)->default_value("false"))

        ("f,fp16-enabled", "Enables support for relaxed computation from Float32 to Float16",
         cxxopts::value<bool>(m_fp16Enabled)->default_value("false"));
    }
    catch (const std::exception& e)
    {
        ALOGW("An error occurred attempting to construct options: %s", e.what());
    }


    try
    {
        cxxopts::ParseResult result = optionsDesc.parse(argc, argv);
        // If no backends have been specified then the default value is GpuAcc.
        if (backends.empty())
        {
            backends.push_back("GpuAcc");
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        ALOGW("An error occurred attempting to parse program options: %s", e.what());
    }

    // Convert the string backend names into backendId's.
    m_Backends.reserve(backends.size());
    for (auto&& backend : backends)
    {
            m_Backends.emplace_back(backend);
    }

    if (!unsupportedOperationsAsString.empty())
    {
        std::istringstream argStream(unsupportedOperationsAsString);

        std::string s;
        while (!argStream.eof())
        {
            std::getline(argStream, s, ',');
            try
            {
                unsigned int operationIdx = std::stoi(s);
                m_ForcedUnsupportedOperations.insert(operationIdx);
            }
            catch (const std::invalid_argument&)
            {
                ALOGW("Ignoring invalid integer argument in -u/--unsupported-operations value: %s", s.c_str());
            }
        }
    }

    if (!m_ClTunedParametersFile.empty())
    {
        // The mode is only relevant if the file path has been provided
        if (clTunedParametersModeAsString == "UseTunedParameters")
        {
            m_ClTunedParametersMode = armnn::IGpuAccTunedParameters::Mode::UseTunedParameters;
        }
        else if (clTunedParametersModeAsString == "UpdateTunedParameters")
        {
            m_ClTunedParametersMode = armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters;
        }
        else
        {
            ALOGW("Requested unknown cl-tuned-parameters-mode '%s'. Defaulting to UseTunedParameters",
                clTunedParametersModeAsString.c_str());
        }

        if (clTuningLevelAsString == "exhaustive")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Exhaustive;
        }
        else if (clTuningLevelAsString == "normal")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Normal;
        }
        else if (clTuningLevelAsString == "rapid")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Rapid;
        }
        else
        {
            ALOGW("Requested unknown cl-tuner-mode '%s'. Defaulting to rapid",
                  clTuningLevelAsString.c_str());
        }
    }
}

} // namespace armnn_driver
