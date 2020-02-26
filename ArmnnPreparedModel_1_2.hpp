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

#include <string>
#include <vector>

namespace armnn_driver
{

typedef std::function<void(::android::hardware::neuralnetworks::V1_0::ErrorStatus status,
        std::vector<::android::hardware::neuralnetworks::V1_2::OutputShape> outputShapes,
        const ::android::hardware::neuralnetworks::V1_2::Timing& timing,
        std::string callingFunction)> armnnExecuteCallback_1_2;

struct ArmnnCallback_1_2
{
    armnnExecuteCallback_1_2 callback;
    TimePoint driverStart;
    MeasureTiming measureTiming;
};

template <typename HalVersion>
class ArmnnPreparedModel_1_2 : public V1_2::IPreparedModel
{
public:
    using HalModel = typename V1_2::Model;

    ArmnnPreparedModel_1_2(armnn::NetworkId networkId,
                           armnn::IRuntime* runtime,
                           const HalModel& model,
                           const std::string& requestInputsAndOutputsDumpDir,
                           const bool gpuProfilingEnabled);

    virtual ~ArmnnPreparedModel_1_2();

    virtual Return<V1_0::ErrorStatus> execute(const V1_0::Request& request,
                                              const sp<V1_0::IExecutionCallback>& callback) override;

    virtual Return<V1_0::ErrorStatus> execute_1_2(const V1_0::Request& request, MeasureTiming measure,
                                                  const sp<V1_2::IExecutionCallback>& callback) override;

    virtual Return<void> executeSynchronously(const V1_0::Request &request,
                                              MeasureTiming measure,
                                              V1_2::IPreparedModel::executeSynchronously_cb cb) override;

    virtual Return<void> configureExecutionBurst(
            const sp<V1_2::IBurstCallback>& callback,
            const android::hardware::MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
            const android::hardware::MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
            configureExecutionBurst_cb cb) override;

    /// execute the graph prepared from the request
    void ExecuteGraph(std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
                      std::shared_ptr<armnn::InputTensors>& pInputTensors,
                      std::shared_ptr<armnn::OutputTensors>& pOutputTensors,
                      ArmnnCallback_1_2 callbackDescriptor);

    /// Executes this model with dummy inputs (e.g. all zeroes).
    /// \return false on failure, otherwise true
    bool ExecuteWithDummyInputs();

private:
    Return <V1_0::ErrorStatus> Execute(const V1_0::Request& request,
                                       MeasureTiming measureTiming,
                                       armnnExecuteCallback_1_2 callback);

    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings);

    armnn::NetworkId                                                            m_NetworkId;
    armnn::IRuntime*                                                            m_Runtime;
    V1_2::Model                                                                 m_Model;
    // There must be a single RequestThread for all ArmnnPreparedModel objects to ensure serial execution of workloads
    // It is specific to this class, so it is declared as static here
    static RequestThread<ArmnnPreparedModel_1_2, HalVersion, ArmnnCallback_1_2> m_RequestThread;
    uint32_t                                                                    m_RequestCount;
    const std::string&                                                          m_RequestInputsAndOutputsDumpDir;
    const bool                                                                  m_GpuProfilingEnabled;
};

}
