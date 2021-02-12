//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"
#include "RequestThread_1_3.hpp"
#include "ModelToINetworkConverter.hpp"

#include <NeuralNetworks.h>
#include <armnn/ArmNN.hpp>

#include <string>
#include <vector>

namespace armnn_driver
{
using CallbackAsync_1_3 = std::function<
                                void(V1_3::ErrorStatus errorStatus,
                                std::vector<::android::hardware::neuralnetworks::V1_2::OutputShape> outputShapes,
                                const ::android::hardware::neuralnetworks::V1_2::Timing& timing,
                                std::string callingFunction)>;

struct ExecutionContext_1_3
{
    ::android::hardware::neuralnetworks::V1_2::MeasureTiming    measureTimings =
        ::android::hardware::neuralnetworks::V1_2::MeasureTiming::NO;
    TimePoint driverStart;
    TimePoint driverEnd;
    TimePoint deviceStart;
    TimePoint deviceEnd;
};

using CallbackContext_1_3 = CallbackContext<CallbackAsync_1_3, ExecutionContext_1_3>;

using executeFenced_cb = std::function<void(::android::hardware::neuralnetworks::V1_3::ErrorStatus status,
    const ::android::hardware::hidl_handle& syncFence,
    const ::android::sp<::android::hardware::neuralnetworks::V1_3::IFencedExecutionCallback>& callback)>;

template <typename HalVersion>
class ArmnnPreparedModel_1_3 : public V1_3::IPreparedModel
{
public:
    using HalModel = typename V1_3::Model;

    ArmnnPreparedModel_1_3(armnn::NetworkId networkId,
                           armnn::IRuntime* runtime,
                           const HalModel& model,
                           const std::string& requestInputsAndOutputsDumpDir,
                           const bool gpuProfilingEnabled,
                           V1_3::Priority priority = V1_3::Priority::MEDIUM);

    virtual ~ArmnnPreparedModel_1_3();

    Return<V1_0::ErrorStatus> execute(const V1_0::Request& request,
                                      const ::android::sp<V1_0::IExecutionCallback>& callback) override;

    Return<V1_0::ErrorStatus> execute_1_2(const V1_0::Request& request, V1_2::MeasureTiming measure,
                                          const ::android::sp<V1_2::IExecutionCallback>& callback) override;

    Return<V1_3::ErrorStatus> execute_1_3(const V1_3::Request& request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint&,
                                          const V1_3::OptionalTimeoutDuration&,
                                          const ::android::sp<V1_3::IExecutionCallback>& callback) override;

    Return<void> executeSynchronously(const V1_0::Request &request,
                                      V1_2::MeasureTiming measure,
                                      V1_3::IPreparedModel::executeSynchronously_cb cb) override;

    Return<void> executeSynchronously_1_3(const V1_3::Request &request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          V1_3::IPreparedModel::executeSynchronously_1_3_cb cb) override;

    Return<void> executeFenced(const V1_3::Request& request,
                               const android::hardware::hidl_vec<android::hardware::hidl_handle>& fenceWaitFor,
                               V1_2::MeasureTiming measure,
                               const V1_3::OptionalTimePoint& deadline,
                               const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                               const V1_3::OptionalTimeoutDuration& duration,
                               executeFenced_cb callback) override;

    Return<void> configureExecutionBurst(
            const ::android::sp<V1_2::IBurstCallback>& callback,
            const android::hardware::MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
            const android::hardware::MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
            configureExecutionBurst_cb cb) override;

    template<typename CallbackContext>
    Return<void> ExecuteSynchronously(const V1_3::Request& request, CallbackContext cbCtx);

    /// execute the graph prepared from the request
    template<typename CallbackContext>
    Return <V1_3::ErrorStatus> ExecuteGraph(
              std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& pMemPools,
              armnn::InputTensors& inputTensors,
              armnn::OutputTensors& outputTensors,
              CallbackContext callback);

    /// Executes this model with dummy inputs (e.g. all zeroes).
    /// \return false on failure, otherwise true
    bool ExecuteWithDummyInputs();

    V1_3::Priority GetModelPriority();

private:
    Return <V1_3::ErrorStatus> Execute(const V1_3::Request& request,
                                       V1_2::MeasureTiming measureTiming,
                                       CallbackAsync_1_3 callback);

    Return<V1_3::ErrorStatus> PrepareMemoryForInputs(
        armnn::InputTensors& inputs,
        const V1_3::Request& request,
        const std::vector<android::nn::RunTimePoolInfo>& memPools);

    Return<V1_3::ErrorStatus> PrepareMemoryForOutputs(
        armnn::OutputTensors& outputs,
        std::vector<V1_2::OutputShape> &outputShapes,
        const V1_3::Request& request,
        const std::vector<android::nn::RunTimePoolInfo>& memPools);

    std::tuple<V1_3::ErrorStatus, hidl_vec<V1_2::OutputShape>, V1_2::Timing, std::string> PrepareMemoryForIO(
        armnn::InputTensors& inputs,
        armnn::OutputTensors& outputs,
        std::vector<android::nn::RunTimePoolInfo>& memPools,
        const V1_3::Request& request);

    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings);

    armnn::NetworkId                                                            m_NetworkId;
    armnn::IRuntime*                                                            m_Runtime;
    V1_3::Model                                                                 m_Model;
    // There must be a single RequestThread for all ArmnnPreparedModel objects to ensure serial execution of workloads
    // It is specific to this class, so it is declared as static here
    static RequestThread_1_3<ArmnnPreparedModel_1_3, HalVersion, CallbackContext_1_3> m_RequestThread;
    uint32_t                                                                    m_RequestCount;
    const std::string&                                                          m_RequestInputsAndOutputsDumpDir;
    const bool                                                                  m_GpuProfilingEnabled;
    V1_3::Priority                                                              m_ModelPriority;
};

}
