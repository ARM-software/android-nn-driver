//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"

#include <NeuralNetworks.h>
#include <ActivationFunctor.h>

#include <armnn/ArmNN.hpp>
#include <armnn/INetwork.hpp>
#include <CpuExecutor.h>

#include "Utils.hpp"

#include <memory>
#include <vector>
#include <set>

namespace armnn_driver
{

class ConstTensorPin;
class LayerInputHandle;

enum class ConversionResult
{
    Success,
    ErrorMappingPools,
    UnsupportedFeature
};

// A helper performing the conversion from an AndroidNN driver Model representation,
// to an armnn::INetwork object
template<typename HalVersion>
class ModelToINetworkConverter
{
public:
    using HalModel = typename HalVersion::Model;

    ModelToINetworkConverter(armnn::Compute compute,
                             const HalModel& model,
                             const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

#if defined(ARMNN_ANDROID_NN_V1_1)
    bool ConvertOperation(const ::android::hardware::neuralnetworks::V1_1::Operation& operation);

    bool ConvertDiv(const ::android::hardware::neuralnetworks::V1_1::Operation& operation);
#endif

    bool ConvertOperation(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertAdd(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertAveragePool2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertConcatenation(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertConv2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertDepthwiseConv2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertFloor(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertFullyConnected(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertLogistic(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertLocalResponseNormalization(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertL2Normalization(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertL2Pool2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertMaxPool2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertMul(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertReLu(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertReLu1(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertReLu6(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertSoftmax(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertTanH(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertReshape(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertResizeBilinear(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertLstm(const ::android::hardware::neuralnetworks::V1_0::Operation& operation);

    bool ConvertToActivation(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                             const char* operationName,
                             const armnn::ActivationDescriptor& activationDesc);

    bool ConvertPooling2d(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                          const char* name, armnn::PoolingAlgorithm poolType);

    const void* GetOperandValueReadOnlyAddress(const Operand& operand) const;

    template<typename HalOperation>
    const Operand* GetInputOperand(const HalOperation& operation, uint32_t inputIndex) const;

    template<typename HalOperation>
    const Operand* GetOutputOperand(const HalOperation& operation, uint32_t outputIndex) const;

    template<typename HalOperation, typename T>
    bool GetInputScalar(const HalOperation& operation, uint32_t inputIndex, OperandType type, T& outValue) const;

    template<typename HalOperation>
    bool GetInputInt32(const HalOperation& operation, uint32_t inputIndex, int32_t& outValue) const;

    template<typename HalOperation>
    bool GetInputFloat32(const HalOperation& operation, uint32_t inputIndex, float& outValue) const;

    template<typename HalOperation>
    bool GetInputActivationFunctionImpl(const HalOperation& operation,
                                        uint32_t inputIndex,
                                        OperandType type,
                                        ActivationFn& outActivationFunction) const;

    template<typename HalOperation>
    bool GetInputActivationFunction(const HalOperation& operation,
                                    uint32_t inputIndex,
                                    ActivationFn& outActivationFunction) const;

    template<typename HalOperation>
    bool GetInputActivationFunctionFromTensor(const HalOperation& operation,
                                              uint32_t inputIndex,
                                              ActivationFn& outActivationFunction) const;

    template<typename HalOperation>
    bool GetOptionalInputActivation(const HalOperation& operation,
                                    uint32_t inputIndex,
                                    ActivationFn& activationFunction) const;

    template<typename HalOperation>
    bool GetInputPaddingScheme(const HalOperation& operation,
                               uint32_t inputIndex,
                               android::nn::PaddingScheme& outPaddingScheme) const;

    template<typename HalOperation>
    LayerInputHandle ConvertToLayerInputHandle(const HalOperation& operation, uint32_t inputIndex);

    template<typename HalOperation>
    ConstTensorPin ConvertOperationInputToConstTensorPin(
        const HalOperation& operation,
        uint32_t inputIndex,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr,
        bool optional = false);

    ConstTensorPin ConvertOperandToConstTensorPin(
        const Operand& operand,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr,
        bool optional = false);

    bool GetTensorInt32Values(const Operand& operand, std::vector<int32_t>& outValues) const;

    armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo,
                                                ActivationFn activation,
                                                armnn::IConnectableLayer* prevLayer);

    template<typename HalOperation>
    bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                      uint32_t operationOutputIndex,
                                      armnn::IConnectableLayer& layer,
                                      uint32_t layerOutputIndex);

    template<typename HalOperation>
    bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                      uint32_t outputIndex,
                                      armnn::IConnectableLayer& layer);

    // Input data
    armnn::Compute                m_Compute;
    const HalModel&               m_Model;
    const std::set<unsigned int>& m_ForcedUnsupportedOperations;

    // Output data
    armnn::INetworkPtr       m_Network;
    ConversionResult         m_ConversionResult;
    std::map<uint32_t, bool> m_OperationSupported;

    // Working/intermediate data
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForOperand;
    std::vector<android::nn::RunTimePoolInfo> m_MemPools;
};

} // armnn_driver