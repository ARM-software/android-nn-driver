//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnDriverImpl.hpp"
#include "ArmnnPreparedModel.hpp"
#include "ModelToINetworkConverter.hpp"
#include "SystemPropertiesUtils.hpp"

#if defined(ARMNN_ANDROID_P)
// The headers of the ML framework have changed between Android O and Android P.
// The validation functions have been moved into their own header, ValidateHal.h.
#include <ValidateHal.h>
#endif

#include <log/log.h>

using namespace std;
using namespace android;
using namespace android::nn;
using namespace android::hardware;

namespace
{

void NotifyCallbackAndCheck(const sp<IPreparedModelCallback>& callback,
                            ErrorStatus errorStatus,
                            const sp<IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriverImpl::prepareModel: hidl callback failed to return properly: %s ",
            returned.description().c_str());
    }
}

Return<ErrorStatus> FailPrepareModel(ErrorStatus error,
                                     const string& message,
                                     const sp<IPreparedModelCallback>& callback)
{
    ALOGW("ArmnnDriverImpl::prepareModel: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

} // namespace

namespace armnn_driver
{

template<typename HalPolicy>
Return<void> ArmnnDriverImpl<HalPolicy>::getSupportedOperations(const armnn::IRuntimePtr& runtime,
                                                                const DriverOptions& options,
                                                                const HalModel& model,
                                                                HalGetSupportedOperations_cb cb)
{
    ALOGV("ArmnnDriverImpl::getSupportedOperations()");

    vector<bool> result;

    if (!runtime)
    {
        cb(ErrorStatus::DEVICE_UNAVAILABLE, result);
        return Void();
    }

    // Run general model validation, if this doesn't pass we shouldn't analyse the model anyway.
    if (!android::nn::validateModel(model))
    {
        cb(ErrorStatus::INVALID_ARGUMENT, result);
        return Void();
    }

    // Attempt to convert the model to an ArmNN input network (INetwork).
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetComputeDevice(),
                                                        model,
                                                        options.GetForcedUnsupportedOperations());

    if (modelConverter.GetConversionResult() != ConversionResult::Success
            && modelConverter.GetConversionResult() != ConversionResult::UnsupportedFeature)
    {
        cb(ErrorStatus::GENERAL_FAILURE, result);
        return Void();
    }

    // Check each operation if it was converted successfully and copy the flags
    // into the result (vector<bool>) that we need to return to Android.
    result.reserve(model.operations.size());
    for (uint32_t operationIdx = 0; operationIdx < model.operations.size(); operationIdx++)
    {
        bool operationSupported = modelConverter.IsOperationSupported(operationIdx);
        result.push_back(operationSupported);
    }

    cb(ErrorStatus::NONE, result);
    return Void();
}

template<typename HalPolicy>
Return<ErrorStatus> ArmnnDriverImpl<HalPolicy>::prepareModel(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const HalModel& model,
        const sp<IPreparedModelCallback>& cb,
        bool float32ToFloat16)
{
    ALOGV("ArmnnDriverImpl::prepareModel()");

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriverImpl::prepareModel: Invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!runtime)
    {
        return FailPrepareModel(ErrorStatus::DEVICE_UNAVAILABLE,
                                "ArmnnDriverImpl::prepareModel: Device unavailable", cb);
    }

    if (!android::nn::validateModel(model))
    {
        return FailPrepareModel(ErrorStatus::INVALID_ARGUMENT,
                                "ArmnnDriverImpl::prepareModel: Invalid model passed as input", cb);
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    set<unsigned int> unsupportedOperations;
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetComputeDevice(),
                                                        model,
                                                        unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        FailPrepareModel(ErrorStatus::GENERAL_FAILURE,
                         "ArmnnDriverImpl::prepareModel: ModelToINetworkConverter failed", cb);
        return ErrorStatus::NONE;
    }

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptions OptOptions;
    OptOptions.m_ReduceFp32ToFp16 = float32ToFloat16;

    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(),
                                 {options.GetComputeDevice()},
                                 runtime->GetDeviceSpec(),
                                 OptOptions);
    }
    catch (armnn::Exception &e)
    {
        stringstream message;
        message << "ArmnnDriverImpl::prepareModel: armnn::Exception (" << e.what() << ") caught from optimize.";
        FailPrepareModel(ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return ErrorStatus::NONE;
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        FailPrepareModel(ErrorStatus::GENERAL_FAILURE,
                         "ArmnnDriverImpl::prepareModel: Invalid optimized network", cb);
        return ErrorStatus::NONE;
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    ExportNetworkGraphToDotFile<HalModel>(*optNet, options.GetRequestInputsAndOutputsDumpDir(), model);

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    try
    {
        if (runtime->LoadNetwork(netId, move(optNet)) != armnn::Status::Success)
        {
            return FailPrepareModel(ErrorStatus::GENERAL_FAILURE,
                                    "ArmnnDriverImpl::prepareModel: Network could not be loaded", cb);
        }
    }
    catch (armnn::Exception& e)
    {
        stringstream message;
        message << "ArmnnDriverImpl::prepareModel: armnn::Exception (" << e.what()<< ") caught from LoadNetwork.";
        FailPrepareModel(ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return ErrorStatus::NONE;
    }

    unique_ptr<ArmnnPreparedModel<HalPolicy>> preparedModel(
                new ArmnnPreparedModel<HalPolicy>(
                    netId,
                    runtime.get(),
                    model,
                    options.GetRequestInputsAndOutputsDumpDir(),
                    options.IsGpuProfilingEnabled()));

    // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
    // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
    preparedModel->ExecuteWithDummyInputs();

    if (clTunedParameters &&
        options.GetClTunedParametersMode() == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters)
    {
        // Now that we've done one inference the CL kernel parameters will have been tuned, so save the updated file.
        try
        {
            clTunedParameters->Save(options.GetClTunedParametersFile().c_str());
        }
        catch (const armnn::Exception& error)
        {
            ALOGE("ArmnnDriverImpl::prepareModel: Failed to save CL tuned parameters file '%s': %s",
                options.GetClTunedParametersFile().c_str(), error.what());
        }
    }

    NotifyCallbackAndCheck(cb, ErrorStatus::NONE, preparedModel.release());

    return ErrorStatus::NONE;
}

template<typename HalPolicy>
Return<DeviceStatus> ArmnnDriverImpl<HalPolicy>::getStatus()
{
    ALOGV("ArmnnDriver::getStatus()");

    return DeviceStatus::AVAILABLE;
}

///
/// Class template specializations
///

template class ArmnnDriverImpl<hal_1_0::HalPolicy>;

#ifdef ARMNN_ANDROID_NN_V1_1
template class ArmnnDriverImpl<hal_1_1::HalPolicy>;
#endif

} // namespace armnn_driver