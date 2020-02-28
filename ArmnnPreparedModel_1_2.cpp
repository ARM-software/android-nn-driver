//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Note: the ArmnnBurstExecutorWithCache in this file is based on Android code
//       under the Apache 2.0 license. See comment below for details.
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel_1_2.hpp"
#include "Utils.hpp"

#include <boost/format.hpp>
#include <log/log.h>
#include <OperationsUtils.h>
#include <ExecutionBurstServer.h>
#include <ValidateHal.h>

#include <cassert>
#include <cinttypes>

using namespace android;
using namespace android::hardware;

namespace {

static const Timing g_NoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};
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
                            ErrorStatus errorStatus,
                            std::vector<OutputShape>,
                            const Timing,
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
                            ErrorStatus errorStatus,
                            std::vector<OutputShape> outputShapes,
                            const Timing timing,
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

bool ValidateRequestArgument(const RequestArgument& requestArg, const armnn::TensorInfo& tensorInfo)
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
            if (requestArg.dimensions[d] != tensorInfo.GetShape()[d])
            {
                ALOGE("Mismatched size for dimension %d (request argument: %u, expected %u)",
                      d, requestArg.dimensions[d], tensorInfo.GetShape()[d]);
                return false;
            }
        }
    }

    return true;
}

armnn::Tensor GetTensorForRequestArgument(const RequestArgument& requestArg,
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
RequestThread<ArmnnPreparedModel_1_2, HalVersion, ArmnnCallback_1_2>
        ArmnnPreparedModel_1_2<HalVersion>::m_RequestThread;

template<typename HalVersion>
template<typename TensorBindingCollection>
void ArmnnPreparedModel_1_2<HalVersion>::DumpTensorsIfRequired(char const* tensorNamePrefix,
                                                               const TensorBindingCollection& tensorBindings)
{
    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        const std::string requestName = boost::str(boost::format("%1%_%2%.dump") % m_NetworkId % m_RequestCount);
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
Return <ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::execute(const Request& request,
        const ::android::sp<V1_0::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::execute invalid callback passed");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](ErrorStatus errorStatus,
                         std::vector<OutputShape> outputShapes,
                         const Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return Execute(request, MeasureTiming::NO, cb);
}

template<typename HalVersion>
Return <ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::execute_1_2(const Request& request,
                                                                     MeasureTiming measureTiming,
                                                                     const sp<V1_2::IExecutionCallback>& callback)
{
    if (callback.get() == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::execute_1_2 invalid callback passed");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    auto cb = [callback](ErrorStatus errorStatus,
                         std::vector<OutputShape> outputShapes,
                         const Timing& timing,
                         std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, outputShapes, timing, callingFunction);
    };

    return Execute(request, measureTiming, cb);
}

template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_2<HalVersion>::executeSynchronously(const Request& request,
                                                                      MeasureTiming measureTiming,
                                                                      executeSynchronously_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_2::executeSynchronously(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (cb == nullptr)
    {
        ALOGE("ArmnnPreparedModel_1_2::executeSynchronously invalid callback passed");
        return Void();
    }

    TimePoint driverStart, driverEnd, deviceStart, deviceEnd;

    if (measureTiming == MeasureTiming::YES)
    {
        driverStart = Now();
    }

    if (!android::nn::validateRequest(request, m_Model))
    {
        ALOGE("ArmnnPreparedModel_1_2::executeSynchronously invalid request model");
        cb(ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming);
        return Void();
    }

    // allocate the tensors on the heap, as they are passed to the request thread
    auto pInputTensors = std::make_shared<armnn::InputTensors>();
    auto pOutputTensors = std::make_shared<armnn::OutputTensors>();

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto pMemPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    if (!setRunTimePoolInfosFromHidlMemories(pMemPools.get(), request.pools))
    {
        cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
        return Void();
    }
    std::vector<OutputShape> outputShapes(request.outputs.size());

    try
    {
        pInputTensors->reserve(request.inputs.size());
        for (unsigned int i = 0; i < request.inputs.size(); i++)
        {
            const auto& inputArg = request.inputs[i];

            const armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
            const armnn::Tensor inputTensor = GetTensorForRequestArgument(inputArg, inputTensorInfo, *pMemPools);

            if (inputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request input %u to tensor", i);
                cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
                return Void();
            }

            pInputTensors->emplace_back(i, inputTensor);
        }
        pOutputTensors->reserve(request.outputs.size());

        for (unsigned int i = 0; i < request.outputs.size(); i++)
        {
            const auto& outputArg = request.outputs[i];

            const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
            const armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, *pMemPools);

            if (outputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request output %u to tensor", i);
                cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
                return Void();
            }
            const size_t outputSize = outputTensorInfo.GetNumBytes();
            const size_t bufferSize = pMemPools->at(outputArg.location.poolIndex).getHidlMemory().size();

            hidl_vec<uint32_t> dimensions;

            armnn::TensorShape tensorShape = outputTensorInfo.GetShape();
            const unsigned int numDims = tensorShape.GetNumDimensions();
            dimensions.resize(numDims);

            for (unsigned int outputIdx = 0u; outputIdx < numDims; ++outputIdx)
            {
                dimensions[outputIdx] = tensorShape[outputIdx];
            }
            outputShapes[i].dimensions = dimensions;
            outputShapes[i].isSufficient = bufferSize >= outputSize;

            if (bufferSize < outputSize)
            {
                ALOGW("ArmnnPreparedModel_1_2::Execute failed");
                cb(ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, outputShapes, g_NoTiming);
                return Void();
            }

            pOutputTensors->emplace_back(i, outputTensor);
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught while preparing for EnqueueWorkload: %s", e.what());
        cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
        return Void();
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught while preparing for EnqueueWorkload: %s", e.what());
        cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
        return Void();
    }

    ALOGV("ArmnnPreparedModel_1_2::executeSynchronously() before Execution");

    DumpTensorsIfRequired("Input", *pInputTensors);
    // run it
    try
    {
        if (measureTiming == MeasureTiming::YES)
        {
            deviceStart = Now();
        }

        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, *pInputTensors, *pOutputTensors);

        if (measureTiming == MeasureTiming::YES)
        {
            deviceEnd = Now();
        }

        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
            return Void();
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught from EnqueueWorkload: %s", e.what());
        cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
        return Void();
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught from EnqueueWorkload: %s", e.what());
        cb(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming);
        return Void();
    }

    DumpTensorsIfRequired("Output", *pOutputTensors);

    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (android::nn::RunTimePoolInfo& pool : *pMemPools)
    {
        pool.update();
    }
    ALOGV("ArmnnPreparedModel_1_2::executeSynchronously() after Execution");

    if (measureTiming == MeasureTiming::YES)
    {
        driverEnd = Now();
        Timing timing;
        timing.timeOnDevice = MicrosecondsDuration(deviceEnd, deviceStart);
        timing.timeInDriver = MicrosecondsDuration(driverEnd, driverStart);
        ALOGV("ArmnnPreparedModel_1_2::executeSynchronously timing Device = %lu Driver = %lu", timing.timeOnDevice,
                timing.timeInDriver);
        cb(ErrorStatus::NONE, outputShapes, timing);
    }
    else
    {
        cb(ErrorStatus::NONE, outputShapes, g_NoTiming);
    }
    return Void();
}

/// This class is strongly inspired by the default implementation in Android named DefaultBurstExecutorWithCache.
/// The original code is licensed under Apache-2.0 and can be found at the following link:
/// https://android.googlesource.com/platform/frameworks/
///         ml/+/refs/tags/android-10.0.0_r20/nn/common/ExecutionBurstServer.cpp
class ArmnnBurstExecutorWithCache : public ExecutionBurstServer::IBurstExecutorWithCache {
public:
    ArmnnBurstExecutorWithCache(IPreparedModel* preparedModel)
        : m_PreparedModel(preparedModel)
    {}

    bool isCacheEntryPresent(int32_t slot) const override
    {
        const auto it = m_MemoryCache.find(slot);
        return (it != m_MemoryCache.end()) && it->second.valid();
    }

    void addCacheEntry(const hidl_memory& memory, int32_t slot) override
    {
        m_MemoryCache[slot] = memory;
    }

    void removeCacheEntry(int32_t slot) override
    {
        m_MemoryCache.erase(slot);
    }

    std::tuple<ErrorStatus, hidl_vec<OutputShape>, Timing> execute(
            const Request& request, const std::vector<int32_t>& slots,
            MeasureTiming measure) override
    {
        ALOGV("ArmnnPreparedModel_1_2::BurstExecutorWithCache::execute");
        hidl_vec<hidl_memory> pools(slots.size());

        std::transform(slots.begin(), slots.end(), pools.begin(), [this](int32_t slot)
        {
            return m_MemoryCache[slot];
        });

        Request fullRequest = request;
        fullRequest.pools = std::move(pools);

        // Setup Callback
        ErrorStatus returnedStatus = ErrorStatus::GENERAL_FAILURE;
        hidl_vec<OutputShape> returnedOutputShapes;
        Timing returnedTiming;
        auto cb = [&returnedStatus, &returnedOutputShapes, &returnedTiming](ErrorStatus status,
                                                                            const hidl_vec<OutputShape>& outputShapes,
                                                                            const Timing& timing)
        {
            returnedStatus = status;
            returnedOutputShapes = outputShapes;
            returnedTiming = timing;
        };

        // Execute
        ALOGV("ArmnnPreparedModel_1_2::BurstExecutorWithCache executing");
        const Return<void> ret = m_PreparedModel->executeSynchronously(fullRequest, measure, cb);

        if (!ret.isOk() || returnedStatus != ErrorStatus::NONE)
        {
            ALOGE("ArmnnPreparedModel_1_2::BurstExecutorWithCache::error executing");
        }
        return std::make_tuple(returnedStatus, std::move(returnedOutputShapes), returnedTiming);
    }

private:
    IPreparedModel* const m_PreparedModel;
    std::map<int, hidl_memory> m_MemoryCache;
};


template<typename HalVersion>
Return<void> ArmnnPreparedModel_1_2<HalVersion>::configureExecutionBurst(
        const sp<V1_2::IBurstCallback>& callback,
        const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
        const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
        V1_2::IPreparedModel::configureExecutionBurst_cb cb)
{
    ALOGV("ArmnnPreparedModel_1_2::configureExecutionBurst");
    const std::shared_ptr<ArmnnBurstExecutorWithCache> executorWithCache =
            std::make_shared<ArmnnBurstExecutorWithCache>(this);
    const sp<V1_2::IBurstContext> burst = ExecutionBurstServer::create(callback,
                                                                       requestChannel,
                                                                       resultChannel,
                                                                       executorWithCache);

    if (burst == nullptr)
    {
        cb(ErrorStatus::GENERAL_FAILURE, {});
    }
    else
    {
        cb(ErrorStatus::NONE, burst);
    }
    return Void();
}

template<typename HalVersion>
void ArmnnPreparedModel_1_2<HalVersion>::ExecuteGraph(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        std::shared_ptr<armnn::InputTensors>& pInputTensors,
        std::shared_ptr<armnn::OutputTensors>& pOutputTensors,
        ArmnnCallback_1_2 cb)
{
    ALOGV("ArmnnPreparedModel_1_2::ExecuteGraph(...)");

    TimePoint driverEnd, deviceStart, deviceEnd;

    DumpTensorsIfRequired("Input", *pInputTensors);

    std::vector<std::pair<int, armnn::Tensor> > outputTensors = *pOutputTensors.get();
    std::vector<OutputShape> outputShapes(outputTensors.size());

    for (unsigned int i = 0; i < outputTensors.size(); i++)
    {
        std::pair<int, armnn::Tensor> outputTensorPair = outputTensors[i];
        const armnn::Tensor outputTensor = outputTensorPair.second;
        const armnn::TensorInfo outputTensorInfo = outputTensor.GetInfo();

        hidl_vec<uint32_t> dimensions;

        armnn::TensorShape tensorShape = outputTensorInfo.GetShape();
        const unsigned int numDims = tensorShape.GetNumDimensions();
        dimensions.resize(numDims);

        for (unsigned int outputIdx = 0u; outputIdx < numDims; ++outputIdx)
        {
            dimensions[outputIdx] = tensorShape[outputIdx];
        }
        outputShapes[i].dimensions = dimensions;
        outputShapes[i].isSufficient = true;
    }

    // run it
    try
    {
        if (cb.measureTiming == MeasureTiming::YES)
        {
            deviceStart = Now();
        }

        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, *pInputTensors, *pOutputTensors);

        if (cb.measureTiming == MeasureTiming::YES)
        {
            deviceEnd = Now();
        }
        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb.callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming,
                    "ArmnnPreparedModel_1_2::ExecuteGraph");
            return;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn:Exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::ExecuteGraph");
        return;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::ExecuteGraph");
        return;
    }

    DumpTensorsIfRequired("Output", *pOutputTensors);

    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (android::nn::RunTimePoolInfo& pool : *pMemPools)
    {
        pool.update();
    }

    if (cb.measureTiming == MeasureTiming::YES)
    {
        driverEnd = Now();
        Timing timing;
        timing.timeOnDevice = MicrosecondsDuration(deviceEnd, deviceStart);
        timing.timeInDriver = MicrosecondsDuration(driverEnd, cb.driverStart);
        cb.callback(ErrorStatus::NONE, outputShapes, timing, "ExecuteGraph");
    } else {
        cb.callback(ErrorStatus::NONE, outputShapes, g_NoTiming, "ExecuteGraph");
    }
}

template<typename HalVersion>
bool ArmnnPreparedModel_1_2<HalVersion>::ExecuteWithDummyInputs()
{
    std::vector<std::vector<char>> storage;
    armnn::InputTensors inputTensors;
    for (unsigned int i = 0; i < m_Model.inputIndexes.size(); i++)
    {
        const armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        storage.emplace_back(inputTensorInfo.GetNumBytes());
        const armnn::ConstTensor inputTensor(inputTensorInfo, storage.back().data());

        inputTensors.emplace_back(i, inputTensor);
    }

    armnn::OutputTensors outputTensors;
    for (unsigned int i = 0; i < m_Model.outputIndexes.size(); i++)
    {
        const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        storage.emplace_back(outputTensorInfo.GetNumBytes());
        const armnn::Tensor outputTensor(outputTensorInfo, storage.back().data());

        outputTensors.emplace_back(i, outputTensor);
    }

    try
    {
        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors);
        if (status != armnn::Status::Success)
        {
            ALOGW("ExecuteWithDummyInputs: EnqueueWorkload failed");
            return false;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("ExecuteWithDummyInputs: armnn::Exception caught from EnqueueWorkload: %s", e.what());
        return false;
    }
    catch (std::exception& e)
    {
        ALOGE("ExecuteWithDummyInputs: std::exception caught from EnqueueWorkload: %s", e.what());
        return false;
    }
    return true;
}

template<typename HalVersion>
Return <ErrorStatus> ArmnnPreparedModel_1_2<HalVersion>::Execute(const Request& request,
                                                                 MeasureTiming measureTiming,
                                                                 armnnExecuteCallback_1_2 callback)
{
    TimePoint driverStart;

    if (measureTiming == MeasureTiming::YES)
    {
        driverStart = Now();
    }

    ALOGV("ArmnnPreparedModel_1_2::execute(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (!android::nn::validateRequest(request, m_Model))
    {
        callback(ErrorStatus::INVALID_ARGUMENT, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        ALOGD("Dumping inputs and outputs for request %" PRIuPTR, reinterpret_cast<std::uintptr_t>(&callback));
    }

    // allocate the tensors on the heap, as they are passed to the request thread
    auto pInputTensors = std::make_shared<armnn::InputTensors>();
    auto pOutputTensors = std::make_shared<armnn::OutputTensors>();

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto pMemPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    if (!setRunTimePoolInfosFromHidlMemories(pMemPools.get(), request.pools))
    {
        callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return ErrorStatus::GENERAL_FAILURE;
    }

    // add the inputs and outputs with their data
    try
    {
        pInputTensors->reserve(request.inputs.size());
        for (unsigned int i = 0; i < request.inputs.size(); i++)
        {
            const auto& inputArg = request.inputs[i];

            const armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
            const armnn::Tensor inputTensor = GetTensorForRequestArgument(inputArg, inputTensorInfo, *pMemPools);

            if (inputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request input %u to tensor", i);
                callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
                return ErrorStatus::GENERAL_FAILURE;
            }

            pInputTensors->emplace_back(i, inputTensor);
        }

        pOutputTensors->reserve(request.outputs.size());
        std::vector<OutputShape> outputShapes(request.outputs.size());

        for (unsigned int i = 0; i < request.outputs.size(); i++)
        {
            const auto& outputArg = request.outputs[i];

            const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
            const armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, *pMemPools);
            if (outputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request output %u to tensor", i);
                callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
                return ErrorStatus::GENERAL_FAILURE;
            }

            const size_t outputSize = outputTensorInfo.GetNumBytes();
            const size_t bufferSize = pMemPools->at(outputArg.location.poolIndex).getHidlMemory().size();
            pOutputTensors->emplace_back(i, outputTensor);

            hidl_vec<uint32_t> dimensions;

            armnn::TensorShape tensorShape = outputTensorInfo.GetShape();
            const unsigned int numDims = tensorShape.GetNumDimensions();
            dimensions.resize(numDims);

            for (unsigned int outputIdx = 0u; outputIdx < numDims; ++outputIdx)
            {
                dimensions[outputIdx] = tensorShape[outputIdx];
            }
            outputShapes[i].dimensions = dimensions;
            outputShapes[i].isSufficient = bufferSize >= outputSize;

            if (bufferSize < outputSize)
            {
                ALOGW("ArmnnPreparedModel_1_2::Execute failed");
                callback(ErrorStatus::OUTPUT_INSUFFICIENT_SIZE,
                         outputShapes,
                         g_NoTiming,
                         "ArmnnPreparedModel_1_2::Execute");
                return ErrorStatus::NONE;
            }
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught while preparing for EnqueueWorkload: %s", e.what());
        callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught while preparing for EnqueueWorkload: %s", e.what());
        callback(ErrorStatus::GENERAL_FAILURE, {}, g_NoTiming, "ArmnnPreparedModel_1_2::execute");
        return ErrorStatus::GENERAL_FAILURE;
    }

    ALOGV("ArmnnPreparedModel_1_2::execute(...) before PostMsg");
    // post the request for asynchronous execution
    ArmnnCallback_1_2 armnnCb;
    armnnCb.callback = callback;
    armnnCb.measureTiming = measureTiming;
    armnnCb.driverStart = driverStart;
    m_RequestThread.PostMsg(this, pMemPools, pInputTensors, pOutputTensors, armnnCb);
    ALOGV("ArmnnPreparedModel_1_2::execute(...) after PostMsg");
    return ErrorStatus::NONE;
}

#ifdef ARMNN_ANDROID_NN_V1_2
template class ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>;
#endif

} // namespace armnn_driver
