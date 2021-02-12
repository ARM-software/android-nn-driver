//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Note: the ArmnnFencedExecutionCallback and code snippet in the executeFenced() function
//       in this file is based on Android code
//       under the Apache 2.0 license. See comments below for details.
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel_1_3.hpp"
#include "Utils.hpp"

#include <Utils.h>
#include <android/sync.h>
#include <log/log.h>
#include <OperationsUtils.h>
#include <ExecutionBurstServer.h>
#include <ValidateHal.h>

#include <cassert>
#include <cinttypes>

#ifdef ARMNN_ANDROID_S
#include <LegacyUtils.h>
#endif

using namespace android;
using namespace android::hardware;

namespace {

static const V1_2::Timing g_NoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};
using namespace armnn_driver;
using TimePoint = std::chrono::steady_clock::time_point;

TimePoint Now()
{
    return std::chrono::steady_clock::now();
}

unsigned long MicrosecondsDuration(TimePoint endPoint, TimePoint startPoint)
{
    return static_cast<unsigned long>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      endPoint - startPoint).count());
}

void NotifyCallbackAndCheck(const ::android::sp<V1_0::IExecutionCallback>& callback,
                            V1_3::ErrorStatus errorStatus,
                            std::vector<V1_2::OutputShape>,
                            const V1_2::Timing,
                            std::string callingFunction)
{
    Return<void> returned = callback->notify(convertToV1_0(errorStatus));
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriver::%s: hidl callback failed to return properly: %s",
              callingFunction.c_str(), returned.description().c_str());
    }
}

void NotifyCallbackAndCheck(const ::android::sp<V1_2::IExecutionCallback>& callback,
                            V1_3::ErrorStatus errorStatus,
                            std::vector<V1_2::OutputShape> outputShapes,
                            const V1_2::Timing timing,
                            std::string callingFunction)
{
    Return<void> returned = callback->notify_1_2(convertToV1_0(errorStatus), outputShapes, timing);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriver::%s: hidl callback failed to return properly: %s",
              callingFunction.c_str(), returned.description().c_str());
    }
}

void NotifyCallbackAndCheck(const ::android::sp<V1_3::IExecutionCallback>& callback,
                            V1_3::ErrorStatus errorStatus,
                            std::vector<V1_2::OutputShape> outputShapes,
                            const V1_2::Timing timing,
                            std::string callingFunction)
{
    Return<void> returned = callback->notify_1_3(errorStatus, outputShapes, timing);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriver::%s: hidl callback failed to return properly: %s",
              callingFunction.c_str(), returned.description().c_str());
    }
}

bool ValidateRequestArgument(const V1_0::RequestArgument& requestArg, const armnn::TensorInfo& tensorInfo)
{
    if (requestArg.dimensions.size() != 0)
    {
        if (requestArg.dimensions.size() != tensorInfo.GetNumDimensions())
        {
            ALOGE("Mismatched dimensions (request argument: %zu, expected: %u)",
                  requestArg.dimensions.size(), tensorInfo.GetNumDimensions());
            return false;
        }

        for (unsigned int d = 0; d < tensorInfo.GetNumDimensions(); ++d)
        {
            if (requestArg.dimensions[d] != 0 && requestArg.dimensions[d] != tensorInfo.GetShape()[d])
            {
                ALOGE("Mismatched size for dimension %d (request argument: %u, expected %u)",
                      d, requestArg.dimensions[d], tensorInfo.GetShape()[d]);
                return false;
            }
        }
    }

    return true;
}

armnn::Tensor GetTensorForRequestArgument(const V1_0::RequestArgument& requestArg,
                                          const armnn::TensorInfo& tensorInfo,
                                          const std::vector<::android::nn::RunTimePoolInfo>& requestPools)
{
    if (!ValidateRequestArgument(requestArg, tensorInfo))
    {
        return armnn::Tensor();
    }

    return armnn::Tensor(tensorInfo, GetMemoryFromPool(requestArg.location, requestPools));
}

inline std::string BuildTensorName(const char* tensorNamePrefix, std::size_t index)
{
    return tensorNamePrefix + std::to_string(index);
}

} // anonymous namespace

using namespace android::hardware;

namespace armnn_driver
{

template<typename HalVersion>
RequestThread_1_3<ArmnnPreparedModel_1_3, HalVersion, CallbackContext_1_3>
        ArmnnPreparedModel_1_3<HalVersion>::m_RequestThread;

template<typename HalVersion>
template<typename TensorBindingCollection>
void ArmnnPreparedModel_1_3<HalVersion>::DumpTensorsIfRequired(char const* tensorNamePrefix,
                                                               const TensorBindingCollection& tensorBindings)
{
    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        const std::string requestName = std::to_string(m_NetworkId) + "_" + std::to_string(m_RequestCount) + ".dump";
        for (std::size_t i = 0u; i < tensorBindings.size(); ++i)
        {
            DumpTensor(m_RequestInputsAndOutputsDumpDir,
                       requestName,
                       BuildTensorName(tensorNamePrefix, i),
                       tensorBindings[i].second);
        }
    }
}

template<typename HalVersion>
ArmnnPreparedModel_1_3<HalVersion>::ArmnnPreparedModel_1_3(armnn::NetworkId networkId,
                                                           armnn::IRuntime* runtime,
                                                           const V1_3::Model& model,
                                                           const std::string& requestInputsAndOutputsDumpDir,
                                                           const bool gpuProfilingEnabled,
                                                           V1_3::Priority priority)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_Model(model)
    , m_RequestCount(0)
    , m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir)
    , m_GpuProfilingEnabled(gpuProfilingEnabled)
    , m_ModelPriority(priority)
{
    // Enable profiling if required.
    m_Runtime->GetProfiler(m_NetworkId)->EnableProfiling(m_GpuProfilingEnabled);
}

template<typename HalVersion>
ArmnnPreparedModel_1_3<HalVersion>::~ArmnnPreparedModel_1_3()
{
    // Get a hold of the profiler used by this model.
    std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);

    // Unload the network associated with this model.
    m_Runtime->UnloadNetwork(m_NetworkId);

    // Dump the profiling info to a file if required.
    DumpJsonProfilingIfRequired(m_GpuProfilingEnabled, m_RequestInputsAndOutputsDumpDir, m_NetworkId, profiler.get());
}

template<typename HalVersion>
Return <V1_0::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::execute(const V1_0::Request& request,
        const ::android::sp<V1_0::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::execute invalid callback passed");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](V1_3::ErrorStatus errorStatus,
                         std::vector<V1_2::OutputShape> outputShapes,
                         const V1_2::Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };


    return convertToV1_0(Execute(convertToV1_3(request), V1_2::MeasureTiming::NO, cb));
}

template<typename HalVersion>
Return <V1_0::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::execute_1_2(
    const V1_0::Request& request,
    V1_2::MeasureTiming measureTiming,
    const sp<V1_2::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::execute_1_2 invalid callback passed");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](V1_3::ErrorStatus errorStatus,
                         std::vector<V1_2::OutputShape> outputShapes,
                         const V1_2::Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return convertToV1_0(Execute(convertToV1_3(request), measureTiming, cb));
}

template<typename HalVersion>
Return <V1_3::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::execute_1_3(
        const V1_3::Request& request,
        V1_2::MeasureTiming measureTiming,
        const V1_3::OptionalTimePoint&,
        const V1_3::OptionalTimeoutDuration&,
        const sp<V1_3::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::execute_1_3 invalid callback passed");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](V1_3::ErrorStatus errorStatus,
                         std::vector<V1_2::OutputShape> outputShapes,
                         const V1_2::Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return Execute(request, measureTiming, cb);
}

/// This class is inspired by the sample implementation in Android named SampleFencedExecutionCallback.
/// The original code is licensed under Apache-2.0 and can be found at the following link:
/// https://android.googlesource.com/platform/frameworks/ml/+/master/nn/driver/sample/SampleDriver.h
class ArmnnFencedExecutionCallback : public V1_3::IFencedExecutionCallback
{
public:
    ArmnnFencedExecutionCallback(V1_3::ErrorStatus errorStatus, V1_2::Timing timing, V1_2::Timing fenceTiming)
        : m_ErrorStatus(errorStatus), m_Timing(timing), m_FenceTiming(fenceTiming) {}
    ~ArmnnFencedExecutionCallback() {}

    Return<void> getExecutionInfo(getExecutionInfo_cb callback) override
    {
        callback(m_ErrorStatus, m_Timing, m_FenceTiming);
        return Void();
    }
private:
    V1_3::ErrorStatus m_ErrorStatus;
    V1_2::Timing m_Timing;
    V1_2::Timing m_FenceTiming;
};

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_3<HalVersion>::executeFenced(const V1_3::Request& request,
                                                               const hidl_vec<hidl_handle>& fenceWaitFor,
                                                               V1_2::MeasureTiming measureTiming,
                                                               const V1_3::OptionalTimePoint& deadline,
                                                               const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                                               const V1_3::OptionalTimeoutDuration&,
                                                               executeFenced_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_3::executeFenced(...)");
    if (cb == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::executeFenced invalid callback passed");
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
        return Void();
    }

    if (deadline.getDiscriminator() != V1_3::OptionalTimePoint::hidl_discriminator::none)
    {
        ALOGW("ArmnnPreparedModel_1_3::executeFenced parameter deadline is set but not supported.");
    }

    if (loopTimeoutDuration.getDiscriminator() != V1_3::OptionalTimeoutDuration::hidl_discriminator::none)
    {
        ALOGW("ArmnnPreparedModel_1_3::executeFenced parameter loopTimeoutDuration is set but not supported.");
    }

    if (!android::nn::validateRequest(request, m_Model, /*allowUnspecifiedOutput=*/false))
    {
        ALOGV("ArmnnPreparedModel_1_3::executeFenced outputs must be specified for fenced execution ");
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
        return Void();
    }

    ExecutionContext_1_3 ctx;
    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        ctx.measureTimings = measureTiming;
        ctx.driverStart = Now();
    }

    ALOGV("ArmnnPreparedModel_1_3::executeFenced(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        ALOGD("Dumping inputs and outputs for request %" PRIuPTR, reinterpret_cast<std::uintptr_t>(&cb));
    }

    // This code snippet is inspired by the sample implementation in Android named SampleDriver::executeFenced()
    // function. The original code is licensed under Apache-2.0 and can be found at the following link:
    // https://android.googlesource.com/platform/frameworks/ml/+/master/nn/driver/sample/SampleDriver.cpp
    const auto fenceSize = fenceWaitFor.size();
    for (unsigned int index = 0; index < fenceSize; ++index)
    {
        auto fenceNativeHandle = fenceWaitFor[index].getNativeHandle();
        if (!fenceNativeHandle)
        {
            cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
            return Void();
        }

        if (sync_wait(fenceNativeHandle->data[0], -1) < 0)
        {
            ALOGE("ArmnnPreparedModel_1_3::executeFenced sync fence failed.");
            cb(V1_3::ErrorStatus::GENERAL_FAILURE, hidl_handle(nullptr), nullptr);
            return Void();
        }
    }

    TimePoint fenceExecutionStart;
    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        fenceExecutionStart = Now();
    }

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputs = std::make_shared<armnn::InputTensors>();
    auto outputs = std::make_shared<armnn::OutputTensors>();

    auto [status, outShapes, timings, message] = PrepareMemoryForIO(*inputs, *outputs, *memPools, request);
    if (status != V1_3::ErrorStatus::NONE)
    {
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
        return Void();
    }

    ALOGV("ArmnnPreparedModel_1_3::executeFenced(...) before ExecuteGraph");

    // call it with nullCallback for now as we will report the error status from here..
    auto nullCallback = [](V1_3::ErrorStatus, std::vector<V1_2::OutputShape>, const V1_2::Timing&, std::string) {};
    CallbackContext_1_3 cbCtx;
    cbCtx.callback = nullCallback;
    cbCtx.ctx = ctx;

    auto errorStatus = ExecuteGraph(memPools, *inputs, *outputs, cbCtx);
    if (errorStatus != V1_3::ErrorStatus::NONE)
    {
        cb(errorStatus, hidl_handle(nullptr), nullptr);
        return Void();
    }
    ALOGV("ArmnnPreparedModel_1_3::executeFenced(...) after ExecuteGraph");

    V1_2::Timing timing = g_NoTiming;
    V1_2::Timing fenceTiming = g_NoTiming;
    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        fenceTiming.timeOnDevice = MicrosecondsDuration(ctx.deviceEnd, ctx.deviceStart);
        fenceTiming.timeInDriver = MicrosecondsDuration(ctx.driverEnd, fenceExecutionStart);
        ALOGV("ArmnnPreparedModel_1_3::fenceFinishExecutionTiming - Device = %lu Driver = %lu",
              fenceTiming.timeOnDevice, fenceTiming.timeInDriver);
    }

    sp<ArmnnFencedExecutionCallback> armnnFencedExecutionCallback =
        new ArmnnFencedExecutionCallback(V1_3::ErrorStatus::NONE, timing, fenceTiming);
    cb(V1_3::ErrorStatus::NONE, hidl_handle(nullptr), armnnFencedExecutionCallback);
    return Void();
}

template<typename HalVersion>
Return<V1_3::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::PrepareMemoryForInputs(
    armnn::InputTensors& inputs,
    const V1_3::Request& request,
    const std::vector<android::nn::RunTimePoolInfo>& memPools)
{
    inputs.reserve(request.inputs.size());
    for (unsigned int i = 0; i < request.inputs.size(); i++)
    {
        const auto& inputArg = request.inputs[i];

        const armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        const armnn::Tensor inputTensor = GetTensorForRequestArgument(inputArg, inputTensorInfo, memPools);

        if (inputTensor.GetMemoryArea() == nullptr)
        {
            ALOGE("Cannot execute request. Error converting request input %u to tensor", i);
            return V1_3::ErrorStatus::GENERAL_FAILURE;
        }

        inputs.emplace_back(i, inputTensor);
    }

    return V1_3::ErrorStatus::NONE;
}

template<typename HalVersion>
Return<V1_3::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::PrepareMemoryForOutputs(
    armnn::OutputTensors& outputs,
    std::vector<V1_2::OutputShape> &outputShapes,
    const V1_3::Request& request,
    const std::vector<android::nn::RunTimePoolInfo>& memPools)
{
    outputs.reserve(request.outputs.size());
    for (unsigned int i = 0; i < request.outputs.size(); i++)
    {
        const auto& outputArg = request.outputs[i];

        armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        const armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, memPools);
        if (outputTensor.GetMemoryArea() == nullptr)
        {
            ALOGE("Cannot execute request. Error converting request output %u to tensor", i);
            return V1_3::ErrorStatus::GENERAL_FAILURE;
        }

        const size_t outputSize = outputTensorInfo.GetNumBytes();

        unsigned int count = 0;
        std::for_each(outputArg.dimensions.begin(), outputArg.dimensions.end(), [&](auto dim)
        {
            if (dim != 0)
            {
                outputTensorInfo.GetShape()[count] = dim;
            }
            else
            {
                outputTensorInfo.GetShape()[count] = outputArg.dimensions.size();
            }

            count++;
        });

        outputs.emplace_back(i, outputTensor);
        outputShapes[i] = ComputeShape(outputTensorInfo);

        if (outputArg.location.length < outputSize)
        {
            ALOGW("ArmnnPreparedModel_1_3::Execute failed outputArg.location.length (%s) < outputSize (%s)",
                std::to_string(outputArg.location.length).c_str(), std::to_string(outputSize).c_str());
            outputShapes[i].isSufficient = false;
            return V1_3::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }

        size_t bufferSize = 0;
#if !defined(ARMNN_ANDROID_S)
        bufferSize = memPools.at(outputArg.location.poolIndex).getHidlMemory().size();
        if (bufferSize < outputSize)
#else
        bufferSize = memPools.at(outputArg.location.poolIndex).getMemory().size;
#endif
        if (bufferSize < outputSize)
        {
            ALOGW("ArmnnPreparedModel_1_3::Execute failed bufferSize (%s) < outputSize (%s)",
                  std::to_string(bufferSize).c_str(), std::to_string(outputSize).c_str());
            outputShapes[i].isSufficient = false;
            return V1_3::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }
    }

    return V1_3::ErrorStatus::NONE;
}

template<typename HalVersion>
std::tuple<V1_3::ErrorStatus, hidl_vec<V1_2::OutputShape>, V1_2::Timing, std::string>
    ArmnnPreparedModel_1_3<HalVersion>::PrepareMemoryForIO(armnn::InputTensors& inputs,
                                                           armnn::OutputTensors& outputs,
                                                           std::vector<android::nn::RunTimePoolInfo>& memPools,
                                                           const V1_3::Request& request)
{
#if !defined(ARMNN_ANDROID_S)
    if (!setRunTimePoolInfosFromMemoryPools(&memPools, request.pools))
#else
    if (!setRunTimePoolInfosFromMemoryPools(&memPools, uncheckedConvert(request.pools)))
#endif
    {
        return {V1_3::ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
    }

    // add the inputs and outputs with their data
    try
    {
        if (PrepareMemoryForInputs(inputs, request, memPools) != V1_3::ErrorStatus::NONE)
        {
            return {V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
        }

        std::vector<V1_2::OutputShape> outputShapes(request.outputs.size());

        auto errorStatus = PrepareMemoryForOutputs(outputs, outputShapes, request, memPools);
        if (errorStatus != V1_3::ErrorStatus::NONE)
        {
            return {errorStatus, outputShapes, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught while preparing for EnqueueWorkload: %s", e.what());
        return {V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught while preparing for EnqueueWorkload: %s", e.what());
        return {V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
    }

    return {V1_3::ErrorStatus::NONE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute"};
}

template<typename HalVersion>
template<typename CallbackContext>
Return<void> ArmnnPreparedModel_1_3<HalVersion>::ExecuteSynchronously(const V1_3::Request& request,
                                                                      CallbackContext cbCtx)
{
    if (cbCtx.ctx.measureTimings == V1_2::MeasureTiming::YES)
    {
        cbCtx.ctx.driverStart = Now();
    }

    if (!android::nn::validateRequest(convertToV1_3(request), m_Model))
    {
        ALOGE("ArmnnPreparedModel_1_3::ExecuteSynchronously invalid request model");
        cbCtx.callback(V1_3::ErrorStatus::INVALID_ARGUMENT,
                       {},
                       g_NoTiming,
                       "ArmnnPreparedModel_1_3::ExecuteSynchronously invalid request model");
        return Void();
    }

    if (!android::nn::validateRequest(request, m_Model))
    {
        ALOGE("ArmnnPreparedModel_1_3::ExecuteSynchronously invalid request model");
        cbCtx.callback(V1_3::ErrorStatus::INVALID_ARGUMENT,
                       {},
                       g_NoTiming,
                       "ArmnnPreparedModel_1_3::ExecuteSynchronously invalid request model");
        return Void();
    }


    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputs = std::make_shared<armnn::InputTensors>();
    auto outputs = std::make_shared<armnn::OutputTensors>();

    auto [status, outputShapes, timing, message] = PrepareMemoryForIO(*inputs, *outputs, *memPools, request);
    if (status != V1_3::ErrorStatus::NONE)
    {
        cbCtx.callback(status, outputShapes, timing, message);
        return Void();
    }

    ALOGV("ArmnnPreparedModel_1_3::ExecuteSynchronously() before Execution");

    ExecuteGraph(memPools, *inputs, *outputs, cbCtx);
    return Void();
}

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_3<HalVersion>::executeSynchronously(const V1_0::Request& request,
                                                                      V1_2::MeasureTiming measureTiming,
                                                                      executeSynchronously_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_3::executeSynchronously(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (cb == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::executeSynchronously invalid callback passed");
        return Void();
    }

    auto cbWrapper = [cb](V1_3::ErrorStatus errorStatus,
                          std::vector<V1_2::OutputShape> outputShapes,
                          const V1_2::Timing& timing,
                          std::string)
    {
        cb(convertToV1_0(errorStatus), outputShapes, timing);
    };

    CallbackContext_1_3 cbCtx;
    cbCtx.callback = cbWrapper;
    cbCtx.ctx.measureTimings = measureTiming;

    ExecuteSynchronously(convertToV1_3(request), cbCtx);
    return Void();
}

template<typename HalVersion>
Return<void>  ArmnnPreparedModel_1_3<HalVersion>::executeSynchronously_1_3(
        const V1_3::Request& request,
        V1_2::MeasureTiming measureTiming,
        const V1_3::OptionalTimePoint& deadline,
        const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
        executeSynchronously_1_3_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_3::executeSynchronously_1_3(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (cb == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_3::executeSynchronously_1_3 invalid callback passed");
        return Void();
    }

    if (deadline.getDiscriminator() != V1_3::OptionalTimePoint::hidl_discriminator::none)
    {
        ALOGW("ArmnnPreparedModel_1_3::executeSynchronously_1_3 parameter deadline is set but not supported.");
    }

    if (loopTimeoutDuration.getDiscriminator() != V1_3::OptionalTimeoutDuration::hidl_discriminator::none)
    {
        ALOGW(
           "ArmnnPreparedModel_1_3::executeSynchronously_1_3 parameter loopTimeoutDuration is set but not supported.");
    }

    auto cbWrapper = [cb](V1_3::ErrorStatus errorStatus,
                          std::vector<V1_2::OutputShape> outputShapes,
                          const V1_2::Timing& timing,
                          std::string)
    {
        cb(errorStatus, outputShapes, timing);
    };

    CallbackContext_1_3 cbCtx;
    cbCtx.callback = cbWrapper;
    cbCtx.ctx.measureTimings = measureTiming;

    ExecuteSynchronously(request, cbCtx);
    return Void();
}

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_3<HalVersion>::configureExecutionBurst(
        const sp<V1_2::IBurstCallback>& callback,
        const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
        const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
        V1_3::IPreparedModel::configureExecutionBurst_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_3::configureExecutionBurst");
    const sp<V1_2::IBurstContext> burst = ExecutionBurstServer::create(callback,
                                                                       requestChannel,
                                                                       resultChannel,
                                                                       this);

    if (burst == nullptr)
    {
        cb(V1_0::ErrorStatus::GENERAL_FAILURE, {});
    }
    else
    {
        cb(V1_0::ErrorStatus::NONE, burst);
    }
    return Void();
}

template<typename HalVersion>
template<typename CallbackContext>
Return <V1_3::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::ExecuteGraph(
    std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
    armnn::InputTensors& inputTensors,
    armnn::OutputTensors& outputTensors,
    CallbackContext cb)
{
    ALOGV("ArmnnPreparedModel_1_3::ExecuteGraph(...)");

    DumpTensorsIfRequired("Input", inputTensors);

    std::vector<V1_2::OutputShape> outputShapes(outputTensors.size());
    for (unsigned int i = 0; i < outputTensors.size(); i++)
    {
        std::pair<int, armnn::Tensor> outputTensorPair = outputTensors[i];
        const armnn::Tensor outputTensor = outputTensorPair.second;
        const armnn::TensorInfo outputTensorInfo = outputTensor.GetInfo();

        outputShapes[i] = ComputeShape(outputTensorInfo);
    }

    // run it
    try
    {
        if (cb.ctx.measureTimings == V1_2::MeasureTiming::YES)
        {
            cb.ctx.deviceStart = Now();
        }

        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors);

        if (cb.ctx.measureTimings == V1_2::MeasureTiming::YES)
        {
            cb.ctx.deviceEnd = Now();
        }
        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb.callback(V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::ExecuteGraph");
            return V1_3::ErrorStatus::GENERAL_FAILURE;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn:Exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::ExecuteGraph");
        return V1_3::ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_3::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_3::ExecuteGraph");
        return V1_3::ErrorStatus::GENERAL_FAILURE;
    }

    CommitPools(*pMemPools);

    DumpTensorsIfRequired("Output", outputTensors);

    if (cb.ctx.measureTimings == V1_2::MeasureTiming::YES)
    {
        cb.ctx.driverEnd = Now();
        V1_2::Timing timing;
        timing.timeOnDevice = MicrosecondsDuration(cb.ctx.deviceEnd, cb.ctx.deviceStart);
        timing.timeInDriver = MicrosecondsDuration(cb.ctx.driverEnd, cb.ctx.driverStart);
        ALOGV("ArmnnPreparedModel_1_3::execute timing - Device = %lu Driver = %lu", timing.timeOnDevice,
              timing.timeInDriver);
        cb.callback(V1_3::ErrorStatus::NONE, outputShapes, timing, "ArmnnPreparedModel_1_3::ExecuteGraph");
    } else
    {
        cb.callback(V1_3::ErrorStatus::NONE, outputShapes, g_NoTiming, "ArmnnPreparedModel_1_3::ExecuteGraph");
    }
    return V1_3::ErrorStatus::NONE;
}

template<typename HalVersion>
bool ArmnnPreparedModel_1_3<HalVersion>::ExecuteWithDummyInputs()
{
    std::vector<std::vector<char>> storage;
    armnn::InputTensors inputTensors;
    for (unsigned int i = 0; i < getMainModel(m_Model).inputIndexes.size(); i++)
    {
        const armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        storage.emplace_back(inputTensorInfo.GetNumBytes());
        const armnn::ConstTensor inputTensor(inputTensorInfo, storage.back().data());

        inputTensors.emplace_back(i, inputTensor);
    }

    armnn::OutputTensors outputTensors;
    for (unsigned int i = 0; i < getMainModel(m_Model).outputIndexes.size(); i++)
    {
        const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        storage.emplace_back(outputTensorInfo.GetNumBytes());
        const armnn::Tensor outputTensor(outputTensorInfo, storage.back().data());

        outputTensors.emplace_back(i, outputTensor);
    }

    auto nullCallback = [](V1_3::ErrorStatus, std::vector<V1_2::OutputShape>, const V1_2::Timing&, std::string) {};
    CallbackContext_1_3 callbackContext;
    callbackContext.callback = nullCallback;
    callbackContext.ctx.measureTimings = V1_2::MeasureTiming::NO;
    auto memPools = std::make_shared<std::vector<::android::nn::RunTimePoolInfo>>();

    auto errorStatus = ExecuteGraph(memPools,
                                    inputTensors,
                                    outputTensors,
                                    callbackContext);
    return errorStatus == V1_3::ErrorStatus::NONE;
}

template<typename HalVersion>
Return <V1_3::ErrorStatus> ArmnnPreparedModel_1_3<HalVersion>::Execute(const V1_3::Request& request,
                                                                       V1_2::MeasureTiming measureTiming,
                                                                       CallbackAsync_1_3 callback)
{
    ExecutionContext_1_3 ctx;
    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        ctx.measureTimings = measureTiming;
        ctx.driverStart = Now();
    }

    ALOGV("ArmnnPreparedModel_1_3::execute(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (!android::nn::validateRequest(request, m_Model))
    {
        callback(V1_3::ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming, "ArmnnPreparedModel_1_3::execute");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        ALOGD("Dumping inputs and outputs for request %" PRIuPTR, reinterpret_cast<std::uintptr_t>(&callback));
    }

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputTensors = std::make_shared<armnn::InputTensors>();
    auto outputTensors = std::make_shared<armnn::OutputTensors>();

    auto [status, outShapes, timing, message] = PrepareMemoryForIO(*inputTensors, *outputTensors,
                                                                   *memPools, request);
    if (status != V1_3::ErrorStatus::NONE)
    {
        callback(status, outShapes, timing, message);
    }

    switch(status)
    {
        case V1_3::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE:
            return V1_3::ErrorStatus::NONE;
        case V1_3::ErrorStatus::GENERAL_FAILURE:
            return V1_3::ErrorStatus::GENERAL_FAILURE;
        default:
        {}
    }

    ALOGV("ArmnnPreparedModel_1_3::execute(...) before PostMsg");

    // post the request for asynchronous execution
    CallbackContext_1_3 cb;
    cb.callback = callback;
    cb.ctx = ctx;
    m_RequestThread.PostMsg(this, memPools, inputTensors, outputTensors, cb);
    ALOGV("ArmnnPreparedModel_1_3::execute(...) after PostMsg");
    return V1_3::ErrorStatus::NONE;
}

template<typename HalVersion>
V1_3::Priority ArmnnPreparedModel_1_3<HalVersion>::GetModelPriority()
{
    return m_ModelPriority;
}

#ifdef ARMNN_ANDROID_NN_V1_3
template class ArmnnPreparedModel_1_3<hal_1_3::HalPolicy>;
template Return <V1_3::ErrorStatus> ArmnnPreparedModel_1_3<hal_1_3::HalPolicy>::ExecuteGraph<CallbackContext_1_3>(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        armnn::InputTensors& pInputTensors,
        armnn::OutputTensors& pOutputTensors,
        CallbackContext_1_3 cb);
#endif

} // namespace armnn_driver
