//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmnnDriverImpl.hpp"
#include "../ArmnnPreparedModel_1_3.hpp"
#include "../ModelToINetworkConverter.hpp"
#include "../SystemPropertiesUtils.hpp"

#include <log/log.h>

namespace
{
const char *g_RelaxedFloat32toFloat16PerformanceExecTime    = "ArmNN.relaxedFloat32toFloat16Performance.execTime";
const char *g_RelaxedFloat32toFloat16PerformancePowerUsage  = "ArmNN.relaxedFloat32toFloat16Performance.powerUsage";

const char *g_ifPerformanceExecTime                         = "ArmNN.ifPerformance.execTime";
const char *g_ifPerformancePowerUsage                       = "ArmNN.ifPerformance.powerUsage";

const char *g_whilePerformanceExecTime                      = "ArmNN.whilePerformance.execTime";
const char *g_whilePerformancePowerUsage                    = "ArmNN.whilePerformance.powerUsage";

const char *g_OperandTypeTensorFloat32PerformanceExecTime   = "Armnn.operandTypeTensorFloat32Performance.execTime";
const char *g_OperandTypeTensorFloat32PerformancePowerUsage = "Armnn.operandTypeTensorFloat32Performance.powerUsage";

const char *g_OperandTypeFloat32PerformanceExecTime         = "Armnn.operandTypeFloat32Performance.execTime";
const char *g_OperandTypeFloat32PerformancePowerUsage       = "Armnn.operandTypeFloat32Performance.powerUsage";

const char *g_OperandTypeTensorFloat16PerformanceExecTime   = "Armnn.operandTypeTensorFloat16Performance.execTime";
const char *g_OperandTypeTensorFloat16PerformancePowerUsage = "Armnn.operandTypeTensorFloat16Performance.powerUsage";

const char *g_OperandTypeFloat16PerformanceExecTime         = "Armnn.operandTypeFloat16Performance.execTime";
const char *g_OperandTypeFloat16PerformancePowerUsage       = "Armnn.operandTypeFloat16Performance.powerUsage";

const char *g_OperandTypeTensorQuant8AsymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant8AsymmPerformance.execTime";
const char *g_OperandTypeTensorQuant8AsymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant8AsymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant8AsymmSignedPerformanceExecTime =
    "Armnn.operandTypeTensorQuant8AsymmSignedPerformance.execTime";
const char *g_OperandTypeTensorQuant8AsymmSignedPerformancePowerUsage =
    "Armnn.operandTypeTensorQuant8AsymmSignedPerformance.powerUsage";

const char *g_OperandTypeTensorQuant16SymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant16SymmPerformance.execTime";
const char *g_OperandTypeTensorQuant16SymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant16SymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant8SymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant8SymmPerformance.execTime";
const char *g_OperandTypeTensorQuant8SymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant8SymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant8SymmPerChannelPerformanceExecTime =
    "Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.execTime";
const char *g_OperandTypeTensorQuant8SymmPerChannelPerformancePowerUsage =
    "Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.powerUsage";


const char *g_OperandTypeTensorInt32PerformanceExecTime     = "Armnn.operandTypeTensorInt32Performance.execTime";
const char *g_OperandTypeTensorInt32PerformancePowerUsage   = "Armnn.operandTypeTensorInt32Performance.powerUsage";

const char *g_OperandTypeInt32PerformanceExecTime           = "Armnn.operandTypeInt32Performance.execTime";
const char *g_OperandTypeInt32PerformancePowerUsage         = "Armnn.operandTypeInt32Performance.powerUsage";


void NotifyCallbackAndCheck(const sp<V1_3::IPreparedModelCallback>& callback,
                            V1_3::ErrorStatus errorStatus,
                            const sp<V1_3::IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify_1_3(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriverImpl::prepareModel: hidl callback failed to return properly: %s ",
              returned.description().c_str());
    }
}

Return<V1_3::ErrorStatus> FailPrepareModel(V1_3::ErrorStatus error,
                                           const std::string& message,
                                           const sp<V1_3::IPreparedModelCallback>& callback)
{
    ALOGW("ArmnnDriverImpl::prepareModel: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

} // anonymous namespace

namespace armnn_driver
{
namespace hal_1_3
{

Return<V1_3::ErrorStatus> ArmnnDriverImpl::prepareArmnnModel_1_3(
       const armnn::IRuntimePtr& runtime,
       const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
       const DriverOptions& options,
       const V1_3::Model& model,
       const sp<V1_3::IPreparedModelCallback>& cb,
       bool float32ToFloat16,
       V1_3::Priority priority)
{
    ALOGV("ArmnnDriverImpl::prepareArmnnModel_1_3()");

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriverImpl::prepareModel: Invalid callback passed to prepareModel");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!runtime)
    {
        return FailPrepareModel(V1_3::ErrorStatus::DEVICE_UNAVAILABLE, "Device unavailable", cb);
    }

    if (!android::nn::validateModel(model))
    {
        return FailPrepareModel(V1_3::ErrorStatus::INVALID_ARGUMENT, "Invalid model passed as input", cb);
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    std::set<unsigned int> unsupportedOperations;
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetBackends(),
                                                       model,
                                                       unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, "ModelToINetworkConverter failed", cb);
        return V1_3::ErrorStatus::NONE;
    }

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptions OptOptions;
    OptOptions.m_ReduceFp32ToFp16 = float32ToFloat16;

    armnn::BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() }
    });
    armnn::BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() }
    });
    OptOptions.m_ModelOptions.push_back(gpuAcc);
    OptOptions.m_ModelOptions.push_back(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what() << ") caught from optimize.";
        FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_3::ErrorStatus::NONE;
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        std::stringstream message;
        message << "Invalid optimized network";
        for (const std::string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_3::ErrorStatus::NONE;
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet,
                                                               options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    try
    {
        if (runtime->LoadNetwork(netId, move(optNet)) != armnn::Status::Success)
        {
            return FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, "Network could not be loaded", cb);
        }
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what()<< ") caught from LoadNetwork.";
        FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_3::ErrorStatus::NONE;
    }

    // Now that we have a networkId for the graph rename the dump file to use it
    // so that we can associate the graph file and the input/output tensor dump files
    RenameGraphDotFile(dotGraphFileName,
                       options.GetRequestInputsAndOutputsDumpDir(),
                       netId);

    std::unique_ptr<ArmnnPreparedModel_1_3<hal_1_3::HalPolicy>> preparedModel(
            new ArmnnPreparedModel_1_3<hal_1_3::HalPolicy>(
                    netId,
                    runtime.get(),
                    model,
                    options.GetRequestInputsAndOutputsDumpDir(),
                    options.IsGpuProfilingEnabled(),
                    priority));

    // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
    // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
    if (!preparedModel->ExecuteWithDummyInputs())
    {
        return FailPrepareModel(V1_3::ErrorStatus::GENERAL_FAILURE, "Network could not be executed", cb);
    }

    if (clTunedParameters &&
        options.GetClTunedParametersMode() == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters)
    {
        // Now that we've done one inference the CL kernel parameters will have been tuned, so save the updated file.
        try
        {
            clTunedParameters->Save(options.GetClTunedParametersFile().c_str());
        }
        catch (std::exception& error)
        {
            ALOGE("ArmnnDriverImpl::prepareModel: Failed to save CL tuned parameters file '%s': %s",
                  options.GetClTunedParametersFile().c_str(), error.what());
        }
    }

    NotifyCallbackAndCheck(cb, V1_3::ErrorStatus::NONE, preparedModel.release());

    return V1_3::ErrorStatus::NONE;
}

Return<void> ArmnnDriverImpl::getCapabilities_1_3(const armnn::IRuntimePtr& runtime,
                                                  V1_3::IDevice::getCapabilities_1_3_cb cb)
{
    ALOGV("hal_1_3::ArmnnDriverImpl::getCapabilities()");

    V1_3::Capabilities capabilities;

    float defaultValue = .1f;

    if (runtime)
    {
        capabilities.relaxedFloat32toFloat16PerformanceScalar.execTime =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformanceExecTime, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceScalar.powerUsage =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformancePowerUsage, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceTensor.execTime =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformanceExecTime, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceTensor.powerUsage =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformancePowerUsage, defaultValue);

        capabilities.ifPerformance.execTime =
                ParseSystemProperty(g_ifPerformanceExecTime, defaultValue);

        capabilities.ifPerformance.powerUsage =
                ParseSystemProperty(g_ifPerformancePowerUsage, defaultValue);

        capabilities.whilePerformance.execTime =
                ParseSystemProperty(g_whilePerformanceExecTime, defaultValue);

        capabilities.whilePerformance.powerUsage =
                ParseSystemProperty(g_whilePerformancePowerUsage, defaultValue);

        // Set the base value for all operand types
        capabilities.operandPerformance = nonExtensionOperandPerformance<HalVersion::V1_3>({FLT_MAX, FLT_MAX});

        // Load supported operand types
        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_FLOAT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorFloat32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorFloat32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::FLOAT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeFloat32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeFloat32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_FLOAT16,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorFloat16PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorFloat16PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::FLOAT16,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeFloat16PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeFloat16PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_QUANT8_ASYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmPerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_QUANT8_SYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerformancePowerUsage, defaultValue)
                });
        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
               {
                   .execTime = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmSignedPerformanceExecTime,
                   defaultValue),
                   .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmSignedPerformancePowerUsage,
                   defaultValue)
               });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_QUANT16_SYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant16SymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant16SymmPerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
               {
                   .execTime =
                   ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerChannelPerformanceExecTime, defaultValue),
                   .powerUsage =
                   ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerChannelPerformancePowerUsage, defaultValue)
               });

        update(&capabilities.operandPerformance, V1_3::OperandType::TENSOR_INT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorInt32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorInt32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_3::OperandType::INT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeInt32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeInt32PerformancePowerUsage, defaultValue)
                });

        cb(V1_3::ErrorStatus::NONE, capabilities);
    }
    else
    {
        capabilities.relaxedFloat32toFloat16PerformanceScalar.execTime   = 0;
        capabilities.relaxedFloat32toFloat16PerformanceScalar.powerUsage = 0;
        capabilities.relaxedFloat32toFloat16PerformanceTensor.execTime   = 0;
        capabilities.relaxedFloat32toFloat16PerformanceTensor.powerUsage = 0;
        capabilities.ifPerformance.execTime      = 0;
        capabilities.ifPerformance.powerUsage    = 0;
        capabilities.whilePerformance.execTime   = 0;
        capabilities.whilePerformance.powerUsage = 0;

        // Set the base value for all operand types
        capabilities.operandPerformance = nonExtensionOperandPerformance<HalVersion::V1_3>({0.f, 0.0f});

        cb(V1_3::ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }

    return Void();
}

} // namespace hal_1_3
} // namespace armnn_driver