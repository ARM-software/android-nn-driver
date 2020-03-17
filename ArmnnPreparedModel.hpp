//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"
#include "RequestThread.hpp"

#include <NeuralNetworks.h>
#include <armnn/ArmNN.hpp>

#include <string>
#include <vector>

namespace armnn_driver
{
using armnnExecuteCallback_1_0 = std::function<void(V1_0::ErrorStatus status, std::string callingFunction)>;

struct ArmnnCallback_1_0
{
    armnnExecuteCallback_1_0 callback;
};

struct ExecutionContext_1_0 {};

using CallbackContext_1_0 = CallbackContext<armnnExecuteCallback_1_0, ExecutionContext_1_0>;

template <typename HalVersion>
class ArmnnPreparedModel : public V1_0::IPreparedModel
{
public:
    using HalModel = typename HalVersion::Model;

    ArmnnPreparedModel(armnn::NetworkId networkId,
                       armnn::IRuntime* runtime,
                       const HalModel& model,
                       const std::string& requestInputsAndOutputsDumpDir,
                       const bool gpuProfilingEnabled);

    virtual ~ArmnnPreparedModel();

    virtual Return<V1_0::ErrorStatus> execute(const V1_0::Request& request,
                                              const ::android::sp<V1_0::IExecutionCallback>& callback) override;

    /// execute the graph prepared from the request
    void ExecuteGraph(std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
                      armnn::InputTensors& inputTensors,
                      armnn::OutputTensors& outputTensors,
                      CallbackContext_1_0 callback);

    /// Executes this model with dummy inputs (e.g. all zeroes).
    /// \return false on failure, otherwise true
    bool ExecuteWithDummyInputs();

private:
    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings);

    armnn::NetworkId                                                        m_NetworkId;
    armnn::IRuntime*                                                        m_Runtime;
    HalModel                                                                m_Model;
    // There must be a single RequestThread for all ArmnnPreparedModel objects to ensure serial execution of workloads
    // It is specific to this class, so it is declared as static here
    static RequestThread<ArmnnPreparedModel, HalVersion, CallbackContext_1_0> m_RequestThread;
    uint32_t                                                                m_RequestCount;
    const std::string&                                                      m_RequestInputsAndOutputsDumpDir;
    const bool                                                              m_GpuProfilingEnabled;
};

}
