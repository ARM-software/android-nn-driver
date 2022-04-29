//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"
#include "RequestThread.hpp"
#include "ModelToINetworkConverter.hpp"

#include <NeuralNetworks.h>
#include <armnn/ArmNN.hpp>
#include <armnn/Threadpool.hpp>

#include <string>
#include <vector>

namespace armnn_driver
{

using CallbackAsync_1_2 = std::function<
                                void(V1_0::ErrorStatus errorStatus,
                                     std::vector<::android::hardware::neuralnetworks::V1_2::OutputShape> outputShapes,
                                     const ::android::hardware::neuralnetworks::V1_2::Timing& timing,
                                     std::string callingFunction)>;

struct ExecutionContext_1_2
{
    ::android::hardware::neuralnetworks::V1_2::MeasureTiming    measureTimings =
        ::android::hardware::neuralnetworks::V1_2::MeasureTiming::NO;
    TimePoint driverStart;
};

using CallbackContext_1_2 = CallbackContext<CallbackAsync_1_2, ExecutionContext_1_2>;

template <typename HalVersion>
class ArmnnPreparedModel_1_2 : public V1_2::IPreparedModel
{
public:
    using HalModel = typename V1_2::Model;

    ArmnnPreparedModel_1_2(armnn::NetworkId networkId,
                           armnn::IRuntime* runtime,
                           const HalModel& model,
                           const std::string& requestInputsAndOutputsDumpDir,
                           const bool gpuProfilingEnabled,
                           const bool asyncModelExecutionEnabled = false,
                           const unsigned int numberOfThreads = 1,
                           const bool importEnabled = false,
                           const bool exportEnabled = false);

    ArmnnPreparedModel_1_2(armnn::NetworkId networkId,
                           armnn::IRuntime* runtime,
                           const std::string& requestInputsAndOutputsDumpDir,
                           const bool gpuProfilingEnabled,
                           const bool asyncModelExecutionEnabled = false,
                           const unsigned int numberOfThreads = 1,
                           const bool importEnabled = false,
                           const bool exportEnabled = false,
                           const bool preparedFromCache = false);

    virtual ~ArmnnPreparedModel_1_2();

    virtual Return<V1_0::ErrorStatus> execute(const V1_0::Request& request,
                                              const ::android::sp<V1_0::IExecutionCallback>& callback) override;

    virtual Return<V1_0::ErrorStatus> execute_1_2(const V1_0::Request& request, V1_2::MeasureTiming measure,
                                                  const ::android::sp<V1_2::IExecutionCallback>& callback) override;

    virtual Return<void> executeSynchronously(const V1_0::Request &request,
                                              V1_2::MeasureTiming measure,
                                              V1_2::IPreparedModel::executeSynchronously_cb cb) override;

    virtual Return<void> configureExecutionBurst(
            const ::android::sp<V1_2::IBurstCallback>& callback,
            const android::hardware::MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
            const android::hardware::MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
            configureExecutionBurst_cb cb) override;

    /// execute the graph prepared from the request
    template<typename CallbackContext>
    bool ExecuteGraph(std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
                      armnn::InputTensors& inputTensors,
                      armnn::OutputTensors& outputTensors,
                      CallbackContext callback);

    /// Executes this model with dummy inputs (e.g. all zeroes).
    /// \return false on failure, otherwise true
    bool ExecuteWithDummyInputs(unsigned int numInputs, unsigned int numOutputs);

private:

    template<typename CallbackContext>
    class ArmnnThreadPoolCallback_1_2 : public armnn::IAsyncExecutionCallback
    {
    public:
        ArmnnThreadPoolCallback_1_2(ArmnnPreparedModel_1_2<HalVersion>* model,
                                    std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
                                    std::vector<V1_2::OutputShape> outputShapes,
                                    std::shared_ptr<armnn::InputTensors>& inputTensors,
                                    std::shared_ptr<armnn::OutputTensors>& outputTensors,
                                    CallbackContext callbackContext) :
                m_Model(model),
                m_MemPools(pMemPools),
                m_OutputShapes(outputShapes),
                m_InputTensors(inputTensors),
                m_OutputTensors(outputTensors),
                m_CallbackContext(callbackContext)
        {}

        void Notify(armnn::Status status, armnn::InferenceTimingPair timeTaken) override;

        ArmnnPreparedModel_1_2<HalVersion>* m_Model;
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>> m_MemPools;
        std::vector<V1_2::OutputShape> m_OutputShapes;
        std::shared_ptr<armnn::InputTensors> m_InputTensors;
        std::shared_ptr<armnn::OutputTensors> m_OutputTensors;
        CallbackContext m_CallbackContext;
    };

    Return<V1_0::ErrorStatus> Execute(const V1_0::Request& request,
                                      V1_2::MeasureTiming measureTiming,
                                      CallbackAsync_1_2 callback);

    Return<V1_0::ErrorStatus> PrepareMemoryForInputs(
            armnn::InputTensors& inputs,
            const V1_0::Request& request,
            const std::vector<android::nn::RunTimePoolInfo>& memPools);

    Return<V1_0::ErrorStatus> PrepareMemoryForOutputs(
            armnn::OutputTensors& outputs,
            std::vector<V1_2::OutputShape> &outputShapes,
            const V1_0::Request& request,
            const std::vector<android::nn::RunTimePoolInfo>& memPools);

    Return <V1_0::ErrorStatus> PrepareMemoryForIO(
            armnn::InputTensors& inputs,
            armnn::OutputTensors& outputs,
            std::vector<android::nn::RunTimePoolInfo>& memPools,
            const V1_0::Request& request,
            CallbackAsync_1_2 callback);

    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings);

    /// schedule the graph prepared from the request for execution
    template<typename CallbackContext>
    void ScheduleGraphForExecution(
            std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
            std::shared_ptr<armnn::InputTensors>& inputTensors,
            std::shared_ptr<armnn::OutputTensors>& outputTensors,
            CallbackContext m_CallbackContext);

    armnn::NetworkId                          m_NetworkId;
    armnn::IRuntime*                          m_Runtime;
    V1_2::Model                               m_Model;
    // There must be a single RequestThread for all ArmnnPreparedModel objects to ensure serial execution of workloads
    // It is specific to this class, so it is declared as static here
    static RequestThread<ArmnnPreparedModel_1_2,
                         HalVersion,
                         CallbackContext_1_2> m_RequestThread;
    uint32_t                                  m_RequestCount;
    const std::string&                        m_RequestInputsAndOutputsDumpDir;
    const bool                                m_GpuProfilingEnabled;
    // Static to allow sharing of threadpool between ArmnnPreparedModel instances
    static std::unique_ptr<armnn::Threadpool> m_Threadpool;
    std::shared_ptr<IWorkingMemHandle>        m_WorkingMemHandle;
    const bool                                m_AsyncModelExecutionEnabled;
    const bool                                m_EnableImport;
    const bool                                m_EnableExport;
    const bool                                m_PreparedFromCache;
};

}
