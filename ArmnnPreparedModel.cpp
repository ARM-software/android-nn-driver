//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel.hpp"
#include "Utils.hpp"

#include <armnn/Types.hpp>

#include <log/log.h>
#include <OperationsUtils.h>
#include <ValidateHal.h>

#include <chrono>
#include <cinttypes>

#ifdef ARMNN_ANDROID_S
#include <LegacyUtils.h>
#endif

using namespace android;

namespace
{
using namespace armnn_driver;

void NotifyCallbackAndCheck(const ::android::sp<V1_0::IExecutionCallback>& callback, V1_0::ErrorStatus errorStatus,
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
RequestThread<ArmnnPreparedModel, HalVersion, CallbackContext_1_0>
    ArmnnPreparedModel<HalVersion>::m_RequestThread;

template<typename HalVersion>
std::unique_ptr<armnn::Threadpool> ArmnnPreparedModel<HalVersion>::m_Threadpool(nullptr);

template<typename HalVersion>
template <typename TensorBindingCollection>
void ArmnnPreparedModel<HalVersion>::DumpTensorsIfRequired(char const* tensorNamePrefix,
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
ArmnnPreparedModel<HalVersion>::ArmnnPreparedModel(armnn::NetworkId networkId,
                                                   armnn::IRuntime* runtime,
                                                   const HalModel& model,
                                                   const std::string& requestInputsAndOutputsDumpDir,
                                                   const bool gpuProfilingEnabled,
                                                   const bool asyncModelExecutionEnabled,
                                                   const unsigned int numberOfThreads,
                                                   const bool importEnabled,
                                                   const bool exportEnabled)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_Model(model)
    , m_RequestCount(0)
    , m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir)
    , m_GpuProfilingEnabled(gpuProfilingEnabled)
    , m_AsyncModelExecutionEnabled(asyncModelExecutionEnabled)
    , m_EnableImport(importEnabled)
    , m_EnableExport(exportEnabled)
{
    // Enable profiling if required.
    m_Runtime->GetProfiler(m_NetworkId)->EnableProfiling(m_GpuProfilingEnabled);

    if (m_AsyncModelExecutionEnabled)
    {
        std::vector<std::shared_ptr<armnn::IWorkingMemHandle>> memHandles;
        for (unsigned int i=0; i < numberOfThreads; ++i)
        {
            memHandles.emplace_back(m_Runtime->CreateWorkingMemHandle(networkId));
        }

        if (!m_Threadpool)
        {
            m_Threadpool = std::make_unique<armnn::Threadpool>(numberOfThreads, runtime, memHandles);
        }
        else
        {
            m_Threadpool->LoadMemHandles(memHandles);
        }

        m_WorkingMemHandle = memHandles.back();
    }
}

template<typename HalVersion>
ArmnnPreparedModel<HalVersion>::~ArmnnPreparedModel()
{
    // Get a hold of the profiler used by this model.
    std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);
    if (profiler && m_GpuProfilingEnabled)
    {
        // Dump the profiling info to a file if required.
        DumpJsonProfilingIfRequired(m_GpuProfilingEnabled, m_RequestInputsAndOutputsDumpDir, m_NetworkId,
                                    profiler.get());
    }

    // Unload the network associated with this model.
    m_Runtime->UnloadNetwork(m_NetworkId);

    // Unload the network memhandles from the threadpool
    if (m_AsyncModelExecutionEnabled)
    {
        m_Threadpool->UnloadMemHandles(m_NetworkId);
    }
}

template<typename HalVersion>
Return<V1_0::ErrorStatus> ArmnnPreparedModel<HalVersion>::execute(
    const V1_0::Request& request,
    const ::android::sp<V1_0::IExecutionCallback>& callback)
{
    ALOGV("ArmnnPreparedModel::execute(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (callback.get() == nullptr) {
        ALOGE("ArmnnPreparedModel::execute invalid callback passed");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!android::nn::validateRequest(request, m_Model))
    {
        NotifyCallbackAndCheck(callback, V1_0::ErrorStatus::INVALID_ARGUMENT, "ArmnnPreparedModel::execute");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        ALOGD("Dumping inputs and outputs for request %" PRIuPTR, reinterpret_cast<std::uintptr_t>(callback.get()));
    }

    // allocate the tensors on the heap, as they are passed to the request thread
    auto pInputTensors = std::make_shared<armnn::InputTensors>();
    auto pOutputTensors = std::make_shared<armnn::OutputTensors>();

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto pMemPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();
#if !defined(ARMNN_ANDROID_S)
    if (!setRunTimePoolInfosFromHidlMemories(pMemPools.get(), request.pools))
#else
    if (!setRunTimePoolInfosFromCanonicalMemories(pMemPools.get(), uncheckedConvert(request.pools)))
#endif
    {
        NotifyCallbackAndCheck(callback, V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    // add the inputs and outputs with their data
    try
    {
        pInputTensors->reserve(request.inputs.size());
        for (unsigned int i = 0; i < request.inputs.size(); i++)
        {
            const auto& inputArg = request.inputs[i];
            armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
            // pInputTensors (of type InputTensors) is composed of a vector of ConstTensors.
            // Therefore, set all TensorInfo isConstant parameters of input Tensors to true.
            inputTensorInfo.SetConstant();
            auto result = ValidateRequestArgument<V1_0::ErrorStatus, V1_0::Request>(request,
                                                                                    inputTensorInfo,
                                                                                    inputArg,
                                                                                    "input");
            if (result != V1_0::ErrorStatus::NONE)
            {
                return result;
            }

            const armnn::Tensor inputTensor = GetTensorForRequestArgument(inputArg, inputTensorInfo, *pMemPools);
            if (inputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request input %u to tensor", i);
                return V1_0::ErrorStatus::GENERAL_FAILURE;
            }

            pInputTensors->emplace_back(i, inputTensor);
        }

        pOutputTensors->reserve(request.outputs.size());
        for (unsigned int i = 0; i < request.outputs.size(); i++)
        {
            const auto& outputArg = request.outputs[i];
            const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
            auto result = ValidateRequestArgument<V1_0::ErrorStatus, V1_0::Request>(request,
                                                                                    outputTensorInfo,
                                                                                    outputArg,
                                                                                    "output");

            if (result != V1_0::ErrorStatus::NONE)
            {
                return result;
            }

            const armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, *pMemPools);
            if (outputTensor.GetMemoryArea() == nullptr)
            {
                ALOGE("Cannot execute request. Error converting request output %u to tensor", i);
                return V1_0::ErrorStatus::GENERAL_FAILURE;
            }

            pOutputTensors->emplace_back(i, outputTensor);
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught while preparing for EnqueueWorkload: %s", e.what());
        NotifyCallbackAndCheck(callback, V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught while preparing for EnqueueWorkload: %s", e.what());
        NotifyCallbackAndCheck(callback, V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::execute");
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    auto cb = [callback](V1_0::ErrorStatus errorStatus, std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, callingFunction);
    };

    CallbackContext_1_0 armnnCb;
    armnnCb.callback = cb;

    if (m_AsyncModelExecutionEnabled)
    {
        ALOGV("ArmnnPreparedModel::execute(...) before ScheduleGraphForExecution");
        ScheduleGraphForExecution(pMemPools, pInputTensors, pOutputTensors, armnnCb);
        ALOGV("ArmnnPreparedModel::execute(...) after ScheduleGraphForExecution");
        return V1_0::ErrorStatus::NONE;
    }

    // post the request for asynchronous execution
    ALOGV("ArmnnPreparedModel::execute(...) before PostMsg");
    m_RequestThread.PostMsg(this, pMemPools, pInputTensors, pOutputTensors, armnnCb);
    ALOGV("ArmnnPreparedModel::execute(...) after PostMsg");
    return V1_0::ErrorStatus::NONE; // successfully queued
}

template<typename HalVersion>
void ArmnnPreparedModel<HalVersion>::ExecuteGraph(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        armnn::InputTensors& inputTensors,
        armnn::OutputTensors& outputTensors,
        CallbackContext_1_0 cb)
{
    ALOGV("ArmnnPreparedModel::ExecuteGraph(...)");
    // Capture the graph execution start time.
    std::chrono::time_point<std::chrono::system_clock> graphExecutionStart = std::chrono::system_clock::now();

    DumpTensorsIfRequired("Input", inputTensors);

    // run it
    try
    {
        armnn::Status status;
        if (m_AsyncModelExecutionEnabled)
        {
            ALOGW("ArmnnPreparedModel::ExecuteGraph m_AsyncModelExecutionEnabled true");
            status = m_Runtime->Execute(*m_WorkingMemHandle, inputTensors, outputTensors);
        }
        else
        {
            ALOGW("ArmnnPreparedModel::ExecuteGraph m_AsyncModelExecutionEnabled false");
            // Create a vector of Input and Output Ids which can be imported. An empty vector means all will be copied.
            std::vector<armnn::ImportedInputId> importedInputIds;
            if (m_EnableImport)
            {
                importedInputIds =  m_Runtime->ImportInputs(m_NetworkId, inputTensors, armnn::MemorySource::Malloc);
            }
            std::vector<armnn::ImportedOutputId> importedOutputIds;
            if (m_EnableExport)
            {
                importedOutputIds = m_Runtime->ImportOutputs(m_NetworkId, outputTensors, armnn::MemorySource::Malloc);
            }
            status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors,
                                                importedInputIds, importedOutputIds);
        }
        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::ExecuteGraph");
            return;
        }
    }
    catch (armnn::Exception& e)
    {
        ALOGW("armnn::Exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::ExecuteGraph");
        return;
    }
    catch (std::exception& e)
    {
        ALOGE("std::exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(V1_0::ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::ExecuteGraph");
        return;
    }

    DumpTensorsIfRequired("Output", outputTensors);

    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (android::nn::RunTimePoolInfo& pool : *pMemPools)
    {
        // Type android::nn::RunTimePoolInfo has changed between Android P & Q and Android R, where
        // update() has been removed and flush() added.
        #if defined(ARMNN_ANDROID_R) || defined(ARMNN_ANDROID_S) // Use the new Android implementation.
            pool.flush();
        #else
            pool.update();
        #endif
    }

    // Log the total time in this call. This is a good number to compare to that printed out by
    // RuntimeImpl::EnqueueWorkload. The difference should be the execution overhead of the driver.
    ALOGI("ArmnnPreparedModel::ExecuteGraph Execution time = %lld µs",
           std::chrono::duration_cast<std::chrono::microseconds>
          (std::chrono::system_clock::now() - graphExecutionStart).count());

    cb.callback(V1_0::ErrorStatus::NONE, "ExecuteGraph");
}

template<typename HalVersion>
bool ArmnnPreparedModel<HalVersion>::ExecuteWithDummyInputs()
{
    std::vector<std::vector<char>> storage;
    armnn::InputTensors inputTensors;
    for (unsigned int i = 0; i < getMainModel(m_Model).inputIndexes.size(); i++)
    {
        armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        // pInputTensors (of type InputTensors) is composed of a vector of ConstTensors.
        // Therefore, set all TensorInfo isConstant parameters of input Tensors to true.
        inputTensorInfo.SetConstant();

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

    try
    {
        armnn::Status status;
        if (m_AsyncModelExecutionEnabled)
        {
            ALOGW("ArmnnPreparedModel::ExecuteGraph m_AsyncModelExecutionEnabled true");
            status = m_Runtime->Execute(*m_WorkingMemHandle, inputTensors, outputTensors);
        }
        else
        {
            ALOGW("ArmnnPreparedModel::ExecuteGraph m_AsyncModelExecutionEnabled false");
            // Create a vector of Input and Output Ids which can be imported. An empty vector means all will be copied.
            std::vector<armnn::ImportedInputId> importedInputIds;
            if (m_EnableImport)
            {
                importedInputIds =  m_Runtime->ImportInputs(m_NetworkId, inputTensors, armnn::MemorySource::Malloc);
            }
            std::vector<armnn::ImportedOutputId> importedOutputIds;
            if (m_EnableExport)
            {
                importedOutputIds = m_Runtime->ImportOutputs(m_NetworkId, outputTensors, armnn::MemorySource::Malloc);
            }
            status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors,
                                                importedInputIds, importedOutputIds);
        }
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

/// Schedule the graph prepared from the request for execution
template<typename HalVersion>
template<typename CallbackContext>
void ArmnnPreparedModel<HalVersion>::ScheduleGraphForExecution(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        std::shared_ptr<armnn::InputTensors>& inputTensors,
        std::shared_ptr<armnn::OutputTensors>& outputTensors,
        CallbackContext callbackContext)
{
    ALOGV("ArmnnPreparedModel::ScheduleGraphForExecution(...)");

    DumpTensorsIfRequired("Input", *inputTensors);


    auto tpCb = std::make_shared<
                ArmnnThreadPoolCallback<CallbackContext_1_0>>(this,
                                                              pMemPools,
                                                              inputTensors,
                                                              outputTensors,
                                                              callbackContext);

    m_Threadpool->Schedule(m_NetworkId,
                           *tpCb->m_InputTensors,
                           *tpCb->m_OutputTensors,
                           armnn::QosExecPriority::Medium,
                           tpCb);
    ALOGV("ArmnnPreparedModel::ScheduleGraphForExecution end");
}

template<typename HalVersion>
template <typename CallbackContext>
void ArmnnPreparedModel<HalVersion>::ArmnnThreadPoolCallback<CallbackContext>::Notify(
        armnn::Status status, armnn::InferenceTimingPair timeTaken)
{
    armnn::IgnoreUnused(status, timeTaken);
    ALOGV("ArmnnPreparedModel::ArmnnThreadPoolCallback_1_2 Notify");

    m_Model->DumpTensorsIfRequired("Output", *m_OutputTensors);

    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (android::nn::RunTimePoolInfo& pool : *m_MemPools)
    {
        // Type android::nn::RunTimePoolInfo has changed between Android P & Q and Android R, where
        // update() has been removed and flush() added.
        #if defined(ARMNN_ANDROID_R) || defined(ARMNN_ANDROID_S) // Use the new Android implementation.
            pool.flush();
        #else
            pool.update();
        #endif
    }

    m_CallbackContext.callback(V1_0::ErrorStatus::NONE, "ArmnnPreparedModel::ArmnnThreadPoolCallback_1_2 Notify");
    return;
}

///
/// Class template specializations
///

template class ArmnnPreparedModel<hal_1_0::HalPolicy>;
template void ArmnnPreparedModel<hal_1_0::HalPolicy>::ScheduleGraphForExecution<CallbackContext_1_0>(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        std::shared_ptr<armnn::InputTensors>& inputTensors,
        std::shared_ptr<armnn::OutputTensors>& outputTensors,
        CallbackContext_1_0 callbackContext);

#ifdef ARMNN_ANDROID_NN_V1_1
template class ArmnnPreparedModel<hal_1_1::HalPolicy>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_2
template class ArmnnPreparedModel<hal_1_1::HalPolicy>;
template class ArmnnPreparedModel<hal_1_2::HalPolicy>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
template class ArmnnPreparedModel<hal_1_1::HalPolicy>;
template class ArmnnPreparedModel<hal_1_2::HalPolicy>;
template class ArmnnPreparedModel<hal_1_3::HalPolicy>;
#endif
} // namespace armnn_driver
