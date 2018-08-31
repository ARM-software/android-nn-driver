//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "RequestThread.hpp"

#include "ArmnnDriver.hpp"

#include <NeuralNetworks.h>
#include <armnn/ArmNN.hpp>

#include <string>
#include <vector>

namespace armnn_driver
{

class ArmnnPreparedModel : public IPreparedModel
{
public:
    ArmnnPreparedModel(armnn::NetworkId networkId,
                       armnn::IRuntime* runtime,
                       const ::android::hardware::neuralnetworks::V1_0::Model& model,
                       const std::string& requestInputsAndOutputsDumpDir,
                       const bool gpuProfilingEnabled);

    virtual ~ArmnnPreparedModel();

    virtual Return<ErrorStatus> execute(const Request& request,
                                        const ::android::sp<IExecutionCallback>& callback) override;

    /// execute the graph prepared from the request
    void ExecuteGraph(std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
                      std::shared_ptr<armnn::InputTensors>& pInputTensors,
                      std::shared_ptr<armnn::OutputTensors>& pOutputTensors,
                      const ::android::sp<IExecutionCallback>& callback);

    /// Executes this model with dummy inputs (e.g. all zeroes).
    void ExecuteWithDummyInputs();

private:

    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings);

    armnn::NetworkId                                 m_NetworkId;
    armnn::IRuntime*                                 m_Runtime;
    ::android::hardware::neuralnetworks::V1_0::Model m_Model;
    // There must be a single RequestThread for all ArmnnPreparedModel objects to ensure serial execution of workloads
    // It is specific to this class, so it is declared as static here
    static RequestThread                             m_RequestThread;
    uint32_t                                         m_RequestCount;
    const std::string&                               m_RequestInputsAndOutputsDumpDir;
    const bool                                       m_GpuProfilingEnabled;
};

}
