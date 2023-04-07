//
// Copyright © 2017, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDriverImpl.hpp"
#include "ArmnnPreparedModel.hpp"

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3) // Using ::android::hardware::neuralnetworks::V1_2
#include "ArmnnPreparedModel_1_2.hpp"
#endif

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_2
#include "ArmnnPreparedModel_1_3.hpp"
#endif

#include "Utils.hpp"

#include "ModelToINetworkConverter.hpp"
#include "SystemPropertiesUtils.hpp"

#include <ValidateHal.h>
#include <log/log.h>
#include <chrono>

using namespace std;
using namespace android;
using namespace android::nn;
using namespace android::hardware;

namespace
{

void NotifyCallbackAndCheck(const sp<V1_0::IPreparedModelCallback>& callback,
                            V1_0::ErrorStatus errorStatus,
                            const sp<V1_0::IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriverImpl::prepareModel: hidl callback failed to return properly: %s ",
              returned.description().c_str());
    }
}

Return<V1_0::ErrorStatus> FailPrepareModel(V1_0::ErrorStatus error,
                                           const string& message,
                                           const sp<V1_0::IPreparedModelCallback>& callback)
{
    ALOGW("ArmnnDriverImpl::prepareModel: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

} // namespace

namespace armnn_driver
{

template<typename HalPolicy>
Return<V1_0::ErrorStatus> ArmnnDriverImpl<HalPolicy>::prepareModel(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const HalModel& model,
        const sp<V1_0::IPreparedModelCallback>& cb,
        bool float32ToFloat16)
{
    ALOGV("ArmnnDriverImpl::prepareModel()");

    std::chrono::time_point<std::chrono::system_clock> prepareModelTimepoint = std::chrono::system_clock::now();

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriverImpl::prepareModel: Invalid callback passed to prepareModel");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!runtime)
    {
        return FailPrepareModel(V1_0::ErrorStatus::DEVICE_UNAVAILABLE, "Device unavailable", cb);
    }

    if (!android::nn::validateModel(model))
    {
        return FailPrepareModel(V1_0::ErrorStatus::INVALID_ARGUMENT, "Invalid model passed as input", cb);
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    set<unsigned int> unsupportedOperations;
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetBackends(),
                                                       model,
                                                       unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "ModelToINetworkConverter failed", cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Serialize the network graph to a .armnn file if an output directory
    // has been specified in the drivers' arguments.
    std::vector<uint8_t> dataCacheData;
    auto serializedNetworkFileName =
        SerializeNetwork(*modelConverter.GetINetwork(),
                         options.GetRequestInputsAndOutputsDumpDir(),
                         dataCacheData,
                         false);

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptionsOpaque OptOptions;
    OptOptions.SetReduceFp32ToFp16(float32ToFloat16);

    armnn::BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "SaveCachedNetwork", options.SaveCachedNetwork() },
        { "CachedNetworkFilePath", options.GetCachedNetworkFilePath() },
        { "MLGOTuningFilePath", options.GetClMLGOTunedParametersFile() }

    });

    armnn::BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "NumberOfThreads", options.GetNumberOfThreads() }
    });
    OptOptions.AddModelOption(gpuAcc);
    OptOptions.AddModelOption(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception &e)
    {
        stringstream message;
        message << "Exception (" << e.what() << ") caught from optimize.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        stringstream message;
        message << "Invalid optimized network";
        for (const string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet, options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    std::string msg;
    armnn::INetworkProperties networkProperties(options.isAsyncModelExecutionEnabled(),
                                                armnn::MemorySource::Undefined,
                                                armnn::MemorySource::Undefined);

    try
    {
        if (runtime->LoadNetwork(netId, move(optNet), msg, networkProperties) != armnn::Status::Success)
        {
            return FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Network could not be loaded", cb);
        }
    }
    catch (std::exception& e)
    {
        stringstream message;
        message << "Exception (" << e.what()<< ") caught from LoadNetwork.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Now that we have a networkId for the graph rename the exported files to use it
    // so that we can associate the graph file and the input/output tensor exported files
    RenameExportedFiles(serializedNetworkFileName,
                        dotGraphFileName,
                        options.GetRequestInputsAndOutputsDumpDir(),
                        netId);

    sp<ArmnnPreparedModel<HalPolicy>> preparedModel(
            new ArmnnPreparedModel<HalPolicy>(
                    netId,
                    runtime.get(),
                    model,
                    options.GetRequestInputsAndOutputsDumpDir(),
                    options.IsGpuProfilingEnabled(),
                    options.isAsyncModelExecutionEnabled(),
                    options.getNoOfArmnnThreads(),
                    options.isImportEnabled(),
                    options.isExportEnabled()));

    if (std::find(options.GetBackends().begin(),
                  options.GetBackends().end(),
                  armnn::Compute::GpuAcc) != options.GetBackends().end())
    {
        // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
        // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
        if (!preparedModel->ExecuteWithDummyInputs())
        {
            return FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Network could not be executed", cb);
        }

        if (clTunedParameters &&
            options.GetClTunedParametersMode() == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters)
        {
            // Now that we've done one inference the CL kernel parameters will have been tuned, so save the updated file
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
    }
    NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel);

    ALOGV("ArmnnDriverImpl::prepareModel cache timing = %lld µs", std::chrono::duration_cast<std::chrono::microseconds>
         (std::chrono::system_clock::now() - prepareModelTimepoint).count());

    return V1_0::ErrorStatus::NONE;
}

template<typename HalPolicy>
Return<void> ArmnnDriverImpl<HalPolicy>::getSupportedOperations(const armnn::IRuntimePtr& runtime,
                                                                const DriverOptions& options,
                                                                const HalModel& model,
                                                                HalGetSupportedOperations_cb cb)
{
    std::stringstream ss;
    ss << "ArmnnDriverImpl::getSupportedOperations()";
    std::string fileName;
    std::string timestamp;
    if (!options.GetRequestInputsAndOutputsDumpDir().empty())
    {
        ss << " : "
           << options.GetRequestInputsAndOutputsDumpDir()
           << "/"
           << GetFileTimestamp()
           << "_getSupportedOperations.txt";
    }
    ALOGV(ss.str().c_str());

    if (!options.GetRequestInputsAndOutputsDumpDir().empty())
    {
        //dump the marker file
        std::ofstream fileStream;
        fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);
        if (fileStream.good())
        {
            fileStream << timestamp << std::endl;
        }
        fileStream.close();
    }

    vector<bool> result;

    if (!runtime)
    {
        cb(HalErrorStatus::DEVICE_UNAVAILABLE, result);
        return Void();
    }

    // Run general model validation, if this doesn't pass we shouldn't analyse the model anyway.
    if (!android::nn::validateModel(model))
    {
        cb(HalErrorStatus::INVALID_ARGUMENT, result);
        return Void();
    }

    // Attempt to convert the model to an ArmNN input network (INetwork).
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetBackends(),
                                                       model,
                                                       options.GetForcedUnsupportedOperations());

    if (modelConverter.GetConversionResult() != ConversionResult::Success
            && modelConverter.GetConversionResult() != ConversionResult::UnsupportedFeature)
    {
        cb(HalErrorStatus::GENERAL_FAILURE, result);
        return Void();
    }

    // Check each operation if it was converted successfully and copy the flags
    // into the result (vector<bool>) that we need to return to Android.
    result.reserve(getMainModel(model).operations.size());
    for (uint32_t operationIdx = 0;
         operationIdx < getMainModel(model).operations.size();
         ++operationIdx)
    {
        bool operationSupported = modelConverter.IsOperationSupported(operationIdx);
        result.push_back(operationSupported);
    }

    cb(HalErrorStatus::NONE, result);
    return Void();
}

template<typename HalPolicy>
Return<V1_0::DeviceStatus> ArmnnDriverImpl<HalPolicy>::getStatus()
{
    ALOGV("ArmnnDriver::getStatus()");

    return V1_0::DeviceStatus::AVAILABLE;
}

///
/// Class template specializations
///

template class ArmnnDriverImpl<hal_1_0::HalPolicy>;

#ifdef ARMNN_ANDROID_NN_V1_1
template class ArmnnDriverImpl<hal_1_1::HalPolicy>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_2
template class ArmnnDriverImpl<hal_1_1::HalPolicy>;
template class ArmnnDriverImpl<hal_1_2::HalPolicy>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
template class ArmnnDriverImpl<hal_1_1::HalPolicy>;
template class ArmnnDriverImpl<hal_1_2::HalPolicy>;
template class ArmnnDriverImpl<hal_1_3::HalPolicy>;
#endif

} // namespace armnn_driver
