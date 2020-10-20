//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "DriverOptions.hpp"
#include "Utils.hpp"

#include <armnn/Version.hpp>
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
    , m_ShouldExit(false)
{
    std::string unsupportedOperationsAsString;
    std::string clTunedParametersModeAsString;
    std::string clTuningLevelAsString;
    std::vector<std::string> backends;
    bool showHelp;
    bool showVersion;

    cxxopts::Options optionsDesc(argv[0], "ArmNN Android NN driver for the Android Neural Networks API. The Android NN "
                                          "driver will convert Android NNAPI requests and delegate them to available "
                                          "ArmNN backends.");
    try
    {
        optionsDesc.add_options()

        ("a,enable-fast-math", "Enables fast_math options in backends that support it. Using the fast_math flag can "
                               "lead to performance improvements but may result in reduced or different precision.",
         cxxopts::value<bool>(m_FastMathEnabled)->default_value("false"))

        ("c,compute",
         "Comma separated list of backends to run layers on. Examples of possible values are: CpuRef, CpuAcc, GpuAcc",
         cxxopts::value<std::vector<std::string>>(backends))

        ("d,request-inputs-and-outputs-dump-dir",
         "If non-empty, the directory where request inputs and outputs should be dumped",
         cxxopts::value<std::string>(m_RequestInputsAndOutputsDumpDir)->default_value(""))

        ("f,fp16-enabled", "Enables support for relaxed computation from Float32 to Float16",
         cxxopts::value<bool>(m_fp16Enabled)->default_value("false"))

        ("h,help", "Show this help",
         cxxopts::value<bool>(showHelp)->default_value("false"))

        ("m,cl-tuned-parameters-mode",
         "If 'UseTunedParameters' (the default), will read CL tuned parameters from the file specified by "
         "--cl-tuned-parameters-file. "
         "If 'UpdateTunedParameters', will also find the optimum parameters when preparing new networks and update "
         "the file accordingly.",
         cxxopts::value<std::string>(clTunedParametersModeAsString)->default_value("UseTunedParameters"))

        ("n,service-name",
         "If non-empty, the driver service name to be registered",
         cxxopts::value<std::string>(m_ServiceName)->default_value("armnn"))

        ("o,cl-tuning-level",
         "exhaustive: all lws values are tested "
         "normal: reduced number of lws values but enough to still have the performance really close to the "
         "exhaustive approach "
         "rapid: only 3 lws values should be tested for each kernel ",
         cxxopts::value<std::string>(clTuningLevelAsString)->default_value("rapid"))

        ("p,gpu-profiling", "Turns GPU profiling on",
         cxxopts::value<bool>(m_EnableGpuProfiling)->default_value("false"))

        ("t,cl-tuned-parameters-file",
         "If non-empty, the given file will be used to load/save CL tuned parameters. "
         "See also --cl-tuned-parameters-mode",
         cxxopts::value<std::string>(m_ClTunedParametersFile)->default_value(""))

        ("u,unsupported-operations",
         "If non-empty, a comma-separated list of operation indices which the driver will forcibly "
         "consider unsupported",
         cxxopts::value<std::string>(unsupportedOperationsAsString)->default_value(""))

        ("v,verbose-logging", "Turns verbose logging on",
         cxxopts::value<bool>(m_VerboseLogging)->default_value("false"))

        ("V,version", "Show version information",
         cxxopts::value<bool>(showVersion)->default_value("false"));
    }
    catch (const std::exception& e)
    {
        ALOGE("An error occurred attempting to construct options: %s", e.what());
        std::cout << "An error occurred attempting to construct options: %s" << std::endl;
        m_ExitCode = EXIT_FAILURE;
        return;
    }

    try
    {
        cxxopts::ParseResult result = optionsDesc.parse(argc, argv);
    }
    catch (const cxxopts::OptionException& e)
    {
        ALOGW("An exception occurred attempting to parse program options: %s", e.what());
        std::cout << optionsDesc.help() << std::endl
                  << "An exception occurred while parsing program options: " << std::endl
                  << e.what() << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_FAILURE;
        return;
    }
    if (showHelp)
    {
        ALOGW("Showing help and exiting");
        std::cout << optionsDesc.help() << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_SUCCESS;
        return;
    }
    if (showVersion)
    {
        ALOGW("Showing version and exiting");
        std::cout << "ArmNN Android NN driver for the Android Neural Networks API.\n"
                     "ArmNN v" << ARMNN_VERSION << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_SUCCESS;
        return;
    }

    // Convert the string backend names into backendId's.
    m_Backends.reserve(backends.size());
    for (auto&& backend : backends)
    {
        m_Backends.emplace_back(backend);
    }

    // If no backends have been specified then the default value is GpuAcc.
    if (backends.empty())
    {
        ALOGE("No backends have been specified:");
        std::cout << optionsDesc.help() << std::endl
                  << "Unable to start:" << std::endl
                  << "No backends have been specified" << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_FAILURE;
        return;
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
