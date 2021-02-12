//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel_1_2.hpp"
#include "Utils.hpp"

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
                            V1_0::ErrorStatus errorStatus,
                            std::vector<V1_2::OutputShape>,
                            const V1_2::Timing,
                            std::string callingFunction)
{
    Return<void> returned = callback->notify(errorStatus);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriver::%s: hidl callback failed to return properly: %s",
              callingFunction.c_str(), returned.description().c_str());
    }
}

void NotifyCallbackAndCheck(const ::android::sp<V1_2::IExecutionCallback>& callback,
                            V1_0::ErrorStatus errorStatus,
                            std::vector<V1_2::OutputShape> outputShapes,
                            const V1_2::Timing timing,
                            std::string callingFunction)
{
    Return<void> returned = callback->notify_1_2(errorStatus, outputShapes, timing);
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
RequestThread<ArmnnPreparedModel_1_2, HalVersion, CallbackContext_1_2>
        ArmnnPreparedModel_1_2<HalVersion>::m_RequestThread;

template<typename HalVersion>
template<typename TensorBindingCollection>
void ArmnnPreparedModel_1_2<HalVersion>::DumpTensorsIfRequired(char const* tensorNamePrefix,
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
ArmnnPreparedModel_1_2<HalVersion>::ArmnnPreparedModel_1_2(armnn::NetworkId networkId,
                                                           armnn::IRuntime* runtime,
                                                           const V1_2::Model& model,
                                                           const std::string& requestInputsAndOutputsDumpDir,
                                                           const bool gpuProfilingEnabled)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_Model(model)
    , m_RequestCount(0)
    , m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir)
    , m_GpuProfilingEnabled(gpuProfilingEnabled)
{
    // Enable profiling if required.
    m_Runtime->GetProfiler(m_NetworkId)->EnableProfiling(m_GpuProfilingEnabled);
}

template<typename HalVersion>
ArmnnPreparedModel_1_2<HalVersion>::~ArmnnPreparedModel_1_2()
{
    // Get a hold of the profiler used by this model.
    std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);

    // Unload the network associated with this model.
    m_Runtime->UnloadNetwork(m_NetworkId);

    // Dump the profiling info to a file if required.
    DumpJsonProfilingIfRequired(m_GpuProfilingEnabled, m_RequestInputsAndOutputsDumpDir, m_NetworkId, profiler.get());
}

template<typename HalVersion>
Return <V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::execute(const V1_0::Request& request,
        const ::android::sp<V1_0::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::execute invalid callback passed");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](V1_0::ErrorStatus errorStatus,
                         std::vector<V1_2::OutputShape> outputShapes,
                         const V1_2::Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return Execute(request, V1_2::MeasureTiming::NO, cb);
}

template<typename HalVersion>
Return <V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::execute_1_2(
        const V1_0::Request& request,
        V1_2::MeasureTiming measureTiming,
        const sp<V1_2::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::execute_1_2 invalid callback passed");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](V1_0::ErrorStatus errorStatus,
                         std::vector<V1_2::OutputShape> outputShapes,
                         const V1_2::Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return Execute(request, measureTiming, cb);
}

template<typename HalVersion>
Return<V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::PrepareMemoryForInputs(
    armnn::InputTensors& inputs,
    const V1_0::Request& request,
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
            return V1_0::ErrorStatus::GENERAL_FAILURE;
        }

        inputs.emplace_back(i, inputTensor);
    }

    return V1_0::ErrorStatus::NONE;
}

template<typename HalVersion>
Return<V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::PrepareMemoryForOutputs(
    armnn::OutputTensors& outputs,
    std::vector<V1_2::OutputShape> &outputShapes,
    const V1_0::Request& request,
    const std::vector<android::nn::RunTimePoolInfo>& memPools)
{
    outputs.reserve(request.outputs.size());
    for (unsigned int i = 0; i < request.outputs.size(); i++)
    {
        const auto& outputArg = request.outputs[i];

        const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        const armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, memPools);
        if (outputTensor.GetMemoryArea() == nullptr)
        {
            ALOGE("Cannot execute request. Error converting request output %u to tensor", i);
            return V1_0::ErrorStatus::GENERAL_FAILURE;
        }

        const size_t outputSize = outputTensorInfo.GetNumBytes();

        if (outputArg.location.length < outputSize)
        {
            ALOGW("ArmnnPreparedModel_1_2::Execute failed: outputArg.location.length < outputSize");
            return V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }

#if !defined(ARMNN_ANDROID_S)
        const size_t bufferSize = memPools.at(outputArg.location.poolIndex).getHidlMemory().size();
        if (bufferSize < outputSize)
        {
            ALOGW("ArmnnPreparedModel_1_2::Execute failed: bufferSize < outputSize");
            return V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }
#else
        const size_t bufferSize = memPools.at(outputArg.location.poolIndex).getMemory().size;
        if (bufferSize < outputSize)
        {
            ALOGW("ArmnnPreparedModel_1_2::Execute failed bufferSize (%s) < outputSize (%s)",
                  std::to_string(bufferSize).c_str(), std::to_string(outputSize).c_str());
            outputShapes[i].isSufficient = false;
            return V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }
#endif
        outputs.emplace_back(i, outputTensor);
        outputShapes[i] = ComputeShape(outputTensorInfo);
    }

    return V1_0::ErrorStatus::NONE;
}

template<typename HalVersion>
Return<V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::PrepareMemoryForIO(
                                         armnn::InputTensors& inputs,
                                         armnn::OutputTensors& outputs,
                                         std::vector<android::nn::RunTimePoolInfo>& memPools,
                                         const V1_0::Request& request,
                                         CallbackAsync_1_2 callback)
{
#if !defined(ARMNN_ANDROID_S)
    if (!setRunTimePoolInfosFromHidlMemories(&memPools, request.pools))
#else
    if (!setRunTimePoolInfosFromCanonicalMemories(&memPools, uncheckedConvert(request.pools)))
#endif
    {
        callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }
    // add the inputs and outputs with their data
    try
    {
        if (PrepareMemoryForInputs(inputs, request, memPools) != V1_0::ErrorStatus::NONE)
        {
            callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
            return V1_0::ErrorStatus::GENERAL_FAILURE;
        }

        std::vector<V1_2::OutputShape> outputShapes(request.outputs.size());

        auto errorStatus = PrepareMemoryForOutputs(outputs, outputShapes, request, memPools);
        if (errorStatus != V1_0::ErrorStatus::NONE)
        {
            callback(errorStatus,
                     outputShapes,
                     g_NoTiming,
                     "ArmnnPreparedModel_1_2::Execute");
            return errorStatus;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught while preparing for EnqueueWorkload: %s", e.what());
        callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught while preparing for EnqueueWorkload: %s", e.what());
        callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    return V1_0::ErrorStatus::NONE;
}

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_2<HalVersion>::executeSynchronously(const V1_0::Request& request,
                                                                      V1_2::MeasureTiming measureTiming,
                                                                      executeSynchronously_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_2::executeSynchronously(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (cb == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::executeSynchronously invalid callback passed");
        return Void();
    }

    TimePoint driverStart;

    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        driverStart = Now();
    }

    if (!android::nn::validateRequest(request, m_Model))
    {
        ALOGE("ArmnnPreparedModel_1_2::executeSynchronously invalid request model");
        cb(V1_0::ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming);
        return Void();
    }

    auto cbWrapper = [cb](V1_0::ErrorStatus errorStatus,
                          std::vector<V1_2::OutputShape> outputShapes,
                          const V1_2::Timing& timing,
                          std::string)
        {
            cb(errorStatus, outputShapes, timing);
        };

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputs = std::make_shared<armnn::InputTensors>();
    auto outputs = std::make_shared<armnn::OutputTensors>();

    auto prepareStatus = PrepareMemoryForIO(*inputs, *outputs, *memPools, request, cbWrapper);
    if (prepareStatus != V1_0::ErrorStatus::NONE)
    {
        return Void();
    }

    ALOGV("ArmnnPreparedModel_1_2::executeSynchronously() before Execution");

    CallbackContext_1_2 cbCtx;
    cbCtx.callback = cbWrapper;
    cbCtx.ctx.measureTimings = measureTiming;
    cbCtx.ctx.driverStart = driverStart;
    ExecuteGraph(memPools, *inputs, *outputs, cbCtx);

    return Void();
}

template<typename HalVersion>
template<typename CallbackContext>
bool ArmnnPreparedModel_1_2<HalVersion>::ExecuteGraph(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        armnn::InputTensors& inputTensors,
        armnn::OutputTensors& outputTensors,
        CallbackContext cb)
{
    ALOGV("ArmnnPreparedModel_1_2::ExecuteGraph(...)");

    TimePoint driverEnd, deviceStart, deviceEnd;

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
            deviceStart = Now();
        }

        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors);

        if (cb.ctx.measureTimings == V1_2::MeasureTiming::YES)
        {
            deviceEnd = Now();
        }
        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming,
                    "ArmnnPreparedModel_1_2::ExecuteGraph");
            return false;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn:Exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::ExecuteGraph");
        return false;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::ExecuteGraph");
        return false;
    }

    CommitPools(*pMemPools);

    DumpTensorsIfRequired("Output", outputTensors);

    if (cb.ctx.measureTimings == V1_2::MeasureTiming::YES)
    {
        driverEnd = Now();
        V1_2::Timing timing;
        timing.timeOnDevice = MicrosecondsDuration(deviceEnd, deviceStart);
        timing.timeInDriver = MicrosecondsDuration(driverEnd, cb.ctx.driverStart);
        ALOGV("ArmnnPreparedModel_1_2::execute timing - Device = %lu Driver = %lu", timing.timeOnDevice,
              timing.timeInDriver);
        cb.callback(V1_0::ErrorStatus::NONE, outputShapes, timing, "ArmnnPreparedModel_1_2::ExecuteGraph");
    } else {
        cb.callback(V1_0::ErrorStatus::NONE, outputShapes, g_NoTiming, "ArmnnPreparedModel_1_2::ExecuteGraph");
    }

    return true;
}

template<typename HalVersion>
bool ArmnnPreparedModel_1_2<HalVersion>::ExecuteWithDummyInputs()
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

    auto nullCallback = [](V1_0::ErrorStatus, std::vector<V1_2::OutputShape>, const V1_2::Timing&, std::string) {};
    CallbackContext_1_2 callbackContext;
    callbackContext.callback = nullCallback;
    callbackContext.ctx.measureTimings = V1_2::MeasureTiming::NO;
    auto memPools = std::make_shared<std::vector<::android::nn::RunTimePoolInfo>>();
    return ExecuteGraph(memPools,
                        inputTensors,
                        outputTensors,
                        callbackContext);
}

template<typename HalVersion>
Return <V1_0::ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::Execute(const V1_0::Request& request,
                                                                       V1_2::MeasureTiming measureTiming,
                                                                       CallbackAsync_1_2 callback)
{
    ExecutionContext_1_2 ctx;
    if (measureTiming == V1_2::MeasureTiming::YES)
    {
        ctx.measureTimings = measureTiming;
        ctx.driverStart = Now();
    }

    ALOGV("ArmnnPreparedModel_1_2::execute(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (!android::nn::validateRequest(request, m_Model))
    {
        callback(V1_0::ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
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

    auto prepareStatus = PrepareMemoryForIO(*inputTensors, *outputTensors, *memPools, request, callback);
    switch(prepareStatus)
    {
        case V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE:
            return V1_0::ErrorStatus::NONE;
        case V1_0::ErrorStatus::GENERAL_FAILURE:
            return V1_0::ErrorStatus::GENERAL_FAILURE;
        default:
        {}
    }

    ALOGV("ArmnnPreparedModel_1_2::execute(...) before PostMsg");

    // post the request for asynchronous execution
    CallbackContext_1_2 cb;
    cb.callback = callback;
    cb.ctx = ctx;
    m_RequestThread.PostMsg(this, memPools, inputTensors, outputTensors, cb);
    ALOGV("ArmnnPreparedModel_1_2::execute(...) after PostMsg");
    return V1_0::ErrorStatus::NONE;
}

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_2<HalVersion>::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
    V1_2::IPreparedModel::configureExecutionBurst_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_2::configureExecutionBurst");
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

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)
template class ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>;
template bool ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>::ExecuteGraph<CallbackContext_1_2>(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        armnn::InputTensors& pInputTensors,
        armnn::OutputTensors& pOutputTensors,
        CallbackContext_1_2 cb);
#endif

} // namespace armnn_driver
