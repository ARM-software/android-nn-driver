//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ArmnnDriver.hpp"

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
class ModelToINetworkConverter
{
public:
    ModelToINetworkConverter(armnn::Compute compute,
        const ::android::hardware::neuralnetworks::V1_0::Model& model,
        const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

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

    const Operand* GetInputOperand(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                   uint32_t inputIndex) const;

    const Operand* GetOutputOperand(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                    uint32_t outputIndex) const;

    template<typename T>
    bool GetInputScalar(const ::android::hardware::neuralnetworks::V1_0::Operation& operation, uint32_t inputIndex,
                        OperandType type, T& outValue) const;

    bool GetInputInt32(const ::android::hardware::neuralnetworks::V1_0::Operation& operation, uint32_t inputIndex,
                       int32_t& outValue) const;

    bool GetInputFloat32(const ::android::hardware::neuralnetworks::V1_0::Operation& operation, uint32_t inputIndex,
                         float& outValue) const;

    bool GetInputActivationFunctionImpl(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                        uint32_t inputIndex,
                                        OperandType type,
                                        ActivationFn& outActivationFunction) const;

    bool GetInputActivationFunction(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                    uint32_t inputIndex,
                                    ActivationFn& outActivationFunction) const;

    bool GetInputActivationFunctionFromTensor(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                              uint32_t inputIndex,
                                              ActivationFn& outActivationFunction) const;

    bool GetOptionalInputActivation(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                    uint32_t inputIndex,
                                    ActivationFn& activationFunction) const;

    bool GetInputPaddingScheme(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                               uint32_t inputIndex,
                               android::nn::PaddingScheme& outPaddingScheme) const;

    LayerInputHandle ConvertToLayerInputHandle(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                               uint32_t inputIndex);

    ConstTensorPin ConvertOperationInputToConstTensorPin(
            const ::android::hardware::neuralnetworks::V1_0::Operation& operation, uint32_t inputIndex,
            const armnn::PermutationVector& dimensionMappings = g_DontPermute,
            const armnn::TensorShape* overrideTensorShape = nullptr, bool optional = false);

    ConstTensorPin ConvertOperandToConstTensorPin(const Operand& operand,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr, bool optional = false);

    bool GetTensorInt32Values(const Operand& operand, std::vector<int32_t>& outValues) const;


    armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo, ActivationFn activation,
                                                armnn::IConnectableLayer* prevLayer);

    bool SetupAndTrackLayerOutputSlot(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                      uint32_t operationOutputIndex,
                                      armnn::IConnectableLayer& layer,
                                      uint32_t layerOutputIndex);

    bool SetupAndTrackLayerOutputSlot(const ::android::hardware::neuralnetworks::V1_0::Operation& operation,
                                      uint32_t outputIndex,
                                      armnn::IConnectableLayer& layer);


    // Input data
    armnn::Compute                                          m_Compute;
    const ::android::hardware::neuralnetworks::V1_0::Model& m_Model;
    const std::set<unsigned int>&                           m_ForcedUnsupportedOperations;

    // Output data
    armnn::INetworkPtr       m_Network;
    ConversionResult         m_ConversionResult;
    std::map<uint32_t, bool> m_OperationSupported;

    // Working/intermediate data
    std::vector<armnn::IOutputSlot*>  m_OutputSlotForOperand;
    std::vector<android::nn::RunTimePoolInfo> m_MemPools;
};

} // armnn_driver
