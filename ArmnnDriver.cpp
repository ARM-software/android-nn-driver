//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDriver.hpp"
#include "ArmnnPreparedModel.hpp"
#include "ModelToINetworkConverter.hpp"
#include "Utils.hpp"

#include <log/log.h>
#include "SystemPropertiesUtils.hpp"

#include "OperationsUtils.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>

#include <cassert>
#include <functional>
#include <string>
#include <sstream>

using namespace android;
using namespace std;

namespace
{

const char *g_Float32PerformanceExecTimeName = "ArmNN.float32Performance.execTime";
const char *g_Float32PerformancePowerUsageName = "ArmNN.float32Performance.powerUsage";
const char *g_Quantized8PerformanceExecTimeName = "ArmNN.quantized8Performance.execTime";
const char *g_Quantized8PerformancePowerUsageName = "ArmNN.quantized8Performance.powerUsage";

}; //namespace

namespace armnn_driver
{

DriverOptions::DriverOptions(armnn::Compute computeDevice)
: m_ComputeDevice(computeDevice)
, m_VerboseLogging(false)
, m_UseAndroidNnCpuExecutor(false)
{
}

DriverOptions::DriverOptions(int argc, char** argv)
: m_ComputeDevice(armnn::Compute::GpuAcc)
, m_VerboseLogging(false)
, m_UseAndroidNnCpuExecutor(false)
, m_ClTunedParametersMode(armnn::IClTunedParameters::Mode::UseTunedParameters)
{
    namespace po = boost::program_options;

    std::string computeDeviceAsString;
    std::string unsupportedOperationsAsString;
    std::string clTunedParametersModeAsString;

    po::options_description optionsDesc("Options");
    optionsDesc.add_options()
        ("compute,c",
         po::value<std::string>(&computeDeviceAsString)->default_value("GpuAcc"),
         "Which device to run layers on by default. Possible values are: CpuRef, CpuAcc, GpuAcc")

        ("verbose-logging,v",
         po::bool_switch(&m_VerboseLogging),
         "Turns verbose logging on")

        ("use-androidnn-cpu-executor,e",
         po::bool_switch(&m_UseAndroidNnCpuExecutor),
         "Forces the driver to satisfy requests via the Android-provided CpuExecutor")

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
         "the file accordingly.");


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

    if (computeDeviceAsString == "CpuRef")
    {
        m_ComputeDevice = armnn::Compute::CpuRef;
    }
    else if (computeDeviceAsString == "GpuAcc")
    {
        m_ComputeDevice = armnn::Compute::GpuAcc;
    }
    else if (computeDeviceAsString == "CpuAcc")
    {
        m_ComputeDevice = armnn::Compute::CpuAcc;
    }
    else
    {
        ALOGW("Requested unknown compute device %s. Defaulting to compute id %s",
            computeDeviceAsString.c_str(), GetComputeDeviceAsCString(m_ComputeDevice));
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
            m_ClTunedParametersMode = armnn::IClTunedParameters::Mode::UseTunedParameters;
        }
        else if (clTunedParametersModeAsString == "UpdateTunedParameters")
        {
            m_ClTunedParametersMode = armnn::IClTunedParameters::Mode::UpdateTunedParameters;
        }
        else
        {
            ALOGW("Requested unknown cl-tuned-parameters-mode '%s'. Defaulting to UseTunedParameters",
                clTunedParametersModeAsString.c_str());
        }
    }
}

ArmnnDriver::ArmnnDriver(DriverOptions options)
 : m_Runtime(nullptr, nullptr)
 , m_ClTunedParameters(nullptr, nullptr)
 , m_Options(std::move(options))
{
    ALOGV("ArmnnDriver::ArmnnDriver()");

    armnn::ConfigureLogging(false, m_Options.IsVerboseLoggingEnabled(), armnn::LogSeverity::Trace);
    if (m_Options.IsVerboseLoggingEnabled())
    {
        SetMinimumLogSeverity(base::VERBOSE);
    }
    else
    {
        SetMinimumLogSeverity(base::INFO);
    }

    try
    {
        armnn::IRuntime::CreationOptions options(m_Options.GetComputeDevice());
        options.m_UseCpuRefAsFallback = false;
        if (!m_Options.GetClTunedParametersFile().empty())
        {
            m_ClTunedParameters = armnn::IClTunedParameters::Create(m_Options.GetClTunedParametersMode());
            try
            {
                m_ClTunedParameters->Load(m_Options.GetClTunedParametersFile().c_str());
            }
            catch (const armnn::Exception& error)
            {
                // This is only a warning because the file won't exist the first time you are generating it.
                ALOGW("ArmnnDriver: Failed to load CL tuned parameters file '%s': %s",
                    m_Options.GetClTunedParametersFile().c_str(), error.what());
            }
            options.m_ClTunedParameters = m_ClTunedParameters.get();
        }
        m_Runtime = armnn::IRuntime::Create(options);
    }
    catch (const armnn::ClRuntimeUnavailableException& error)
    {
        ALOGE("ArmnnDriver: Failed to setup CL runtime: %s. Device will be unavailable.", error.what());
    }
}

Return<void> ArmnnDriver::getCapabilities(getCapabilities_cb cb)
{
    ALOGV("ArmnnDriver::getCapabilities()");

    Capabilities capabilities;
    if (m_Runtime)
    {
        capabilities.float32Performance.execTime =
            ParseSystemProperty(g_Float32PerformanceExecTimeName, 1.0f);

        capabilities.float32Performance.powerUsage =
            ParseSystemProperty(g_Float32PerformancePowerUsageName, 1.0f);

        capabilities.quantized8Performance.execTime =
            ParseSystemProperty(g_Quantized8PerformanceExecTimeName, 1.0f);

        capabilities.quantized8Performance.powerUsage =
            ParseSystemProperty(g_Quantized8PerformancePowerUsageName, 1.0f);

        cb(ErrorStatus::NONE, capabilities);
    }
    else
    {
        capabilities.float32Performance.execTime = 0;
        capabilities.float32Performance.powerUsage = 0;
        capabilities.quantized8Performance.execTime = 0;
        capabilities.quantized8Performance.powerUsage = 0;

        cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }

    return Void();
}

Return<void> ArmnnDriver::getSupportedOperations(const Model& model, getSupportedOperations_cb cb)
{
    ALOGV("ArmnnDriver::getSupportedOperations()");

    std::vector<bool> result;

    if (!m_Runtime)
    {
        cb(ErrorStatus::DEVICE_UNAVAILABLE, result);
        return Void();
    }

    // Run general model validation, if this doesn't pass we shouldn't analyse the model anyway
    if (!android::nn::validateModel(model))
    {
        cb(ErrorStatus::INVALID_ARGUMENT, result);
        return Void();
    }

    // Attempt to convert the model to an ArmNN input network (INetwork).
    ModelToINetworkConverter modelConverter(m_Runtime->GetDeviceSpec().DefaultComputeDevice, model,
        m_Options.GetForcedUnsupportedOperations());

    if (modelConverter.GetConversionResult() != ConversionResult::Success
        && modelConverter.GetConversionResult() != ConversionResult::UnsupportedFeature)
    {
        cb(ErrorStatus::GENERAL_FAILURE, result);
        return Void();
    }

    // Check each operation if it was converted successfully and copy the flags
    // into the result (vector<bool>) that we need to return to Android
    result.reserve(model.operations.size());
    for (uint32_t operationIdx = 0; operationIdx < model.operations.size(); operationIdx++)
    {
        bool operationSupported = modelConverter.IsOperationSupported(operationIdx);
        result.push_back(operationSupported);
    }

    cb(ErrorStatus::NONE, result);
    return Void();
}

namespace
{

void NotifyCallbackAndCheck(const sp<IPreparedModelCallback>& callback, ErrorStatus errorStatus,
                            const ::android::sp<IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriver::prepareModel: hidl callback failed to return properly: %s ",
            returned.description().c_str());
    }
}

Return<ErrorStatus> FailPrepareModel(ErrorStatus error,
    const std::string& message,
    const sp<IPreparedModelCallback>& callback)
{
    ALOGW("ArmnnDriver::prepareModel: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

}

Return<ErrorStatus> ArmnnDriver::prepareModel(const Model& model,
    const sp<IPreparedModelCallback>& cb)
{
    ALOGV("ArmnnDriver::prepareModel()");

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriver::prepareModel: Invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!m_Runtime)
    {
        return FailPrepareModel(ErrorStatus::DEVICE_UNAVAILABLE, "ArmnnDriver::prepareModel: Device unavailable", cb);
    }

    if (!android::nn::validateModel(model))
    {
        return FailPrepareModel(ErrorStatus::INVALID_ARGUMENT,
            "ArmnnDriver::prepareModel: Invalid model passed as input", cb);
    }

    if (m_Options.UseAndroidNnCpuExecutor())
    {
        sp<AndroidNnCpuExecutorPreparedModel> preparedModel = new AndroidNnCpuExecutorPreparedModel(model,
            m_Options.GetRequestInputsAndOutputsDumpDir());
        if (preparedModel->Initialize())
        {
            NotifyCallbackAndCheck(cb, ErrorStatus::NONE, preparedModel);
            return ErrorStatus::NONE;
        }
        else
        {
            NotifyCallbackAndCheck(cb, ErrorStatus::INVALID_ARGUMENT, preparedModel);
            return ErrorStatus::INVALID_ARGUMENT;
        }
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    std::set<unsigned int> unsupportedOperations;
    ModelToINetworkConverter modelConverter(m_Runtime->GetDeviceSpec().DefaultComputeDevice, model,
        unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        return FailPrepareModel(ErrorStatus::GENERAL_FAILURE, "ModelToINetworkConverter failed", cb);
    }

    // optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(), m_Runtime->GetDeviceSpec());
    }
    catch (armnn::Exception& e)
    {
        std::stringstream message;
        message << "armnn::Exception ("<<e.what()<<") caught from optimize.";
        return FailPrepareModel(ErrorStatus::GENERAL_FAILURE, message.str(), cb);
    }

    // load it into the runtime
    armnn::NetworkId netId = 0;
    try
    {
        if (m_Runtime->LoadNetwork(netId, std::move(optNet)) != armnn::Status::Success)
        {
            return FailPrepareModel(ErrorStatus::GENERAL_FAILURE,
                "ArmnnDriver::prepareModel: Network could not be loaded", cb);
        }
    }
    catch (armnn::Exception& e)
    {
        std::stringstream message;
        message << "armnn::Exception (" << e.what()<< ") caught from LoadNetwork.";
        return FailPrepareModel(ErrorStatus::GENERAL_FAILURE, message.str(), cb);
    }

    std::unique_ptr<ArmnnPreparedModel> preparedModel(new ArmnnPreparedModel(
        netId,
        m_Runtime.get(),
        model,
        m_Options.GetRequestInputsAndOutputsDumpDir()
    ));

    // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
    // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
    preparedModel->ExecuteWithDummyInputs();

    if (m_ClTunedParameters &&
        m_Options.GetClTunedParametersMode() == armnn::IClTunedParameters::Mode::UpdateTunedParameters)
    {
        // Now that we've done one inference the CL kernel parameters will have been tuned, so save the updated file.
        try
        {
            m_ClTunedParameters->Save(m_Options.GetClTunedParametersFile().c_str());
        }
        catch (const armnn::Exception& error)
        {
            ALOGE("ArmnnDriver: Failed to save CL tuned parameters file '%s': %s",
                m_Options.GetClTunedParametersFile().c_str(), error.what());
        }
    }

    NotifyCallbackAndCheck(cb, ErrorStatus::NONE, preparedModel.release());

    return ErrorStatus::NONE;
}

Return<DeviceStatus> ArmnnDriver::getStatus()
{
    ALOGV("ArmnnDriver::getStatus()");
    return DeviceStatus::AVAILABLE;
}

}
