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

#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>

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
{
}

DriverOptions::DriverOptions(const std::vector<armnn::BackendId>& backends, bool fp16Enabled)
    : m_Backends(backends)
    , m_VerboseLogging(false)
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(fp16Enabled)
{
}

DriverOptions::DriverOptions(int argc, char** argv)
    : m_VerboseLogging(false)
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(false)
{
    namespace po = boost::program_options;

    std::string unsupportedOperationsAsString;
    std::string clTunedParametersModeAsString;
    std::string clTuningLevelAsString;

    po::options_description optionsDesc("Options");
    optionsDesc.add_options()
        ("compute,c",
         po::value<std::vector<std::string>>()->
            multitoken()->default_value(std::vector<std::string>{"GpuAcc"}, "{GpuAcc}"),
         "Which backend to run layers on. Examples of possible values are: CpuRef, CpuAcc, GpuAcc")

        ("verbose-logging,v",
         po::bool_switch(&m_VerboseLogging),
         "Turns verbose logging on")

        ("request-inputs-and-outputs-dump-dir,d",
         po::value<std::string>(&m_RequestInputsAndOutputsDumpDir)->default_value(""),
         "If non-empty, the directory where request inputs and outputs should be dumped")

        ("unsupported-operations,u",
         po::value<std::string>(&unsupportedOperationsAsString)->default_value(""),
         "If non-empty, a comma-separated list of operation indices which the driver will forcibly "
         "consider unsupported")

        ("cl-tuned-parameters-file,t",
         po::value<std::string>(&m_ClTunedParametersFile)->default_value(""),
         "If non-empty, the given file will be used to load/save CL tuned parameters. "
         "See also --cl-tuned-parameters-mode")

        ("cl-tuned-parameters-mode,m",
         po::value<std::string>(&clTunedParametersModeAsString)->default_value("UseTunedParameters"),
         "If 'UseTunedParameters' (the default), will read CL tuned parameters from the file specified by "
         "--cl-tuned-parameters-file. "
         "If 'UpdateTunedParameters', will also find the optimum parameters when preparing new networks and update "
         "the file accordingly.")

        ("cl-tuning-level,o",
         po::value<std::string>(&clTuningLevelAsString)->default_value("rapid"),
         "exhaustive: all lws values are tested "
         "normal: reduced number of lws values but enough to still have the performance really close to the "
         "exhaustive approach "
         "rapid: only 3 lws values should be tested for each kernel ")

        ("gpu-profiling,p",
         po::bool_switch(&m_EnableGpuProfiling),
         "Turns GPU profiling on")

        ("fp16-enabled,f",
         po::bool_switch(&m_fp16Enabled),
         "Enables support for relaxed computation from Float32 to Float16");

    po::variables_map variablesMap;
    try
    {
        po::store(po::parse_command_line(argc, argv, optionsDesc), variablesMap);
        po::notify(variablesMap);
    }
    catch (const po::error& e)
    {
        ALOGW("An error occurred attempting to parse program options: %s", e.what());
    }

    const std::vector<std::string> backends = variablesMap["compute"].as<std::vector<std::string>>();
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
