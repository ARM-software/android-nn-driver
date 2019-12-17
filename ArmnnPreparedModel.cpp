//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel.hpp"
#include "Utils.hpp"

#include <boost/format.hpp>
#include <log/log.h>
#include <OperationsUtils.h>

#if defined(ARMNN_ANDROID_P) || defined(ARMNN_ANDROID_Q)
// The headers of the ML framework have changed between Android O and Android P.
// The validation functions have been moved into their own header, ValidateHal.h.
#include <ValidateHal.h>
#endif

#include <cassert>
#include <cinttypes>

using namespace android;

namespace
{
using namespace armnn_driver;

void NotifyCallbackAndCheck(const ::android::sp<V1_0::IExecutionCallback>& callback, ErrorStatus errorStatus,
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
RequestThread<ArmnnPreparedModel, HalVersion, ArmnnCallback_1_0> ArmnnPreparedModel<HalVersion>::m_RequestThread;

template<typename HalVersion>
template <typename TensorBindingCollection>
void ArmnnPreparedModel<HalVersion>::DumpTensorsIfRequired(char const* tensorNamePrefix,
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
ArmnnPreparedModel<HalVersion>::ArmnnPreparedModel(armnn::NetworkId networkId,
                                                   armnn::IRuntime* runtime,
                                                   const HalModel& model,
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
ArmnnPreparedModel<HalVersion>::~ArmnnPreparedModel()
{
    // Get a hold of the profiler used by this model.
    std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);

    // Unload the network associated with this model.
    m_Runtime->UnloadNetwork(m_NetworkId);

    // Dump the profiling info to a file if required.
    DumpJsonProfilingIfRequired(m_GpuProfilingEnabled, m_RequestInputsAndOutputsDumpDir, m_NetworkId, profiler.get());
}

template<typename HalVersion>
Return<ErrorStatus> ArmnnPreparedModel<HalVersion>::execute(const Request& request,
                                                            const ::android::sp<V1_0::IExecutionCallback>& callback)
{
    ALOGV("ArmnnPreparedModel::execute(): %s", GetModelSummary(m_Model).c_str());
    m_RequestCount++;

    if (callback.get() == nullptr) {
        ALOGE("ArmnnPreparedModel::execute invalid callback passed");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!android::nn::validateRequest(request, m_Model))
    {
        NotifyCallbackAndCheck(callback, ErrorStatus::INVALID_ARGUMENT, "ArmnnPreparedModel::execute");
        return ErrorStatus::INVALID_ARGUMENT;
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
    if (!setRunTimePoolInfosFromHidlMemories(pMemPools.get(), request.pools))
    {
        NotifyCallbackAndCheck(callback, ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::execute");
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
                return ErrorStatus::GENERAL_FAILURE;
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
                return ErrorStatus::GENERAL_FAILURE;
            }

            pOutputTensors->emplace_back(i, outputTensor);
        }
    }
    catch (std::exception& e)
    {
        ALOGW("Exception caught while preparing for EnqueueWorkload: %s", e.what());
        NotifyCallbackAndCheck(callback, ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::execute");
        return ErrorStatus::GENERAL_FAILURE;
    }

    ALOGV("ArmnnPreparedModel::execute(...) before PostMsg");

    auto cb = [callback](ErrorStatus errorStatus, std::string callingFunction)
    {
        NotifyCallbackAndCheck(callback, errorStatus, callingFunction);
    };

    ArmnnCallback_1_0 armnnCb;
    armnnCb.callback = cb;
    // post the request for asynchronous execution
    m_RequestThread.PostMsg(this, pMemPools, pInputTensors, pOutputTensors, armnnCb);
    ALOGV("ArmnnPreparedModel::execute(...) after PostMsg");
    return ErrorStatus::NONE; // successfully queued
}

template<typename HalVersion>
void ArmnnPreparedModel<HalVersion>::ExecuteGraph(
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
        std::shared_ptr<armnn::InputTensors>& pInputTensors,
        std::shared_ptr<armnn::OutputTensors>& pOutputTensors,
        ArmnnCallback_1_0 cb)
{
    ALOGV("ArmnnPreparedModel::ExecuteGraph(...)");

    DumpTensorsIfRequired("Input", *pInputTensors);

    // run it
    try
    {
        armnn::Status status = m_Runtime->EnqueueWorkload(m_NetworkId, *pInputTensors, *pOutputTensors);
        if (status != armnn::Status::Success)
        {
            ALOGW("EnqueueWorkload failed");
            cb.callback(ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::ExecuteGraph");
            return;
        }
    }
    catch (std::exception& e)
    {
        ALOGW("Exception caught from EnqueueWorkload: %s", e.what());
        cb.callback(ErrorStatus::GENERAL_FAILURE, "ArmnnPreparedModel::ExecuteGraph");
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

    cb.callback(ErrorStatus::NONE, "ExecuteGraph");
}

template<typename HalVersion>
bool ArmnnPreparedModel<HalVersion>::ExecuteWithDummyInputs()
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
    catch (std::exception& e)
    {
        ALOGW("ExecuteWithDummyInputs: Exception caught from EnqueueWorkload: %s", e.what());
        return false;
    }
    return true;
}

///
/// Class template specializations
///

template class ArmnnPreparedModel<hal_1_0::HalPolicy>;

#ifdef ARMNN_ANDROID_NN_V1_1
template class ArmnnPreparedModel<hal_1_1::HalPolicy>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_2
template class ArmnnPreparedModel<hal_1_1::HalPolicy>;
template class ArmnnPreparedModel<hal_1_2::HalPolicy>;
#endif
} // namespace armnn_driver
