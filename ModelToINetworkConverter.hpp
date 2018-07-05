//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "ActivationFunctor.h"

#include "ArmnnDriver.hpp"

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
    ModelToINetworkConverter(armnn::Compute compute, const V1_0::Model& model,
        const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

    bool ConvertOperation(const V1_0::Operation& operation);

    bool ConvertAdd(const V1_0::Operation& operation);

    bool ConvertAveragePool2d(const V1_0::Operation& operation);

    bool ConvertConcatenation(const V1_0::Operation& operation);

    bool ConvertConv2d(const V1_0::Operation& operation);

    bool ConvertDepthwiseConv2d(const V1_0::Operation& operation);

    bool ConvertFloor(const V1_0::Operation& operation);

    bool ConvertFullyConnected(const V1_0::Operation& operation);

    bool ConvertLogistic(const V1_0::Operation& operation);

    bool ConvertLocalResponseNormalization(const V1_0::Operation& operation);

    bool ConvertL2Normalization(const V1_0::Operation& operation);

    bool ConvertL2Pool2d(const V1_0::Operation& operation);

    bool ConvertMaxPool2d(const V1_0::Operation& operation);

    bool ConvertMul(const V1_0::Operation& operation);

    bool ConvertReLu(const V1_0::Operation& operation);

    bool ConvertReLu1(const V1_0::Operation& operation);

    bool ConvertReLu6(const V1_0::Operation& operation);

    bool ConvertSoftmax(const V1_0::Operation& operation);

    bool ConvertTanH(const V1_0::Operation& operation);

    bool ConvertReshape(const V1_0::Operation& operation);

    bool ConvertResizeBilinear(const V1_0::Operation& operation);

    bool ConvertToActivation(const V1_0::Operation& operation, const char* operationName,
        const armnn::ActivationDescriptor& activationDesc);

    bool ConvertPooling2d(const V1_0::Operation& operation, const char* name, armnn::PoolingAlgorithm poolType);


    const void* GetOperandValueReadOnlyAddress(const Operand& operand) const;

    const Operand* GetInputOperand(const V1_0::Operation& operation, uint32_t inputIndex) const;

    const Operand* GetOutputOperand(const V1_0::Operation& operation, uint32_t outputIndex) const;

    template<typename T>
    bool GetInputScalar(const V1_0::Operation& operation, uint32_t inputIndex, OperandType type, T& outValue) const;

    bool GetInputInt32(const V1_0::Operation& operation, uint32_t inputIndex, int32_t& outValue) const;

    bool GetInputFloat32(const V1_0::Operation& operation, uint32_t inputIndex, float& outValue) const;

    bool GetInputActivationFunction(const V1_0::Operation& operation, uint32_t inputIndex,
        ActivationFn& outActivationFunction) const;

    bool GetInputPaddingScheme(const V1_0::Operation& operation, uint32_t inputIndex,
        android::nn::PaddingScheme& outPaddingScheme) const;

    LayerInputHandle ConvertToLayerInputHandle(const V1_0::Operation& operation, uint32_t inputIndex);

    ConstTensorPin ConvertOperationInputToConstTensorPin(const V1_0::Operation& operation, uint32_t inputIndex,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr);

    ConstTensorPin ConvertOperandToConstTensorPin(const Operand& operand,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr);

    bool GetTensorInt32Values(const Operand& operand, std::vector<int32_t>& outValues) const;


    armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo, ActivationFn activation,
                                                armnn::IConnectableLayer* prevLayer);


    bool SetupAndTrackLayerOutputSlot(const V1_0::Operation& operation, uint32_t outputIndex,
                                      armnn::IConnectableLayer& layer);


    // Input data
    armnn::Compute                    m_Compute;
    const V1_0::Model&                m_Model;
    const std::set<unsigned int>&     m_ForcedUnsupportedOperations;

    // Output data
    armnn::INetworkPtr                m_Network;
    ConversionResult                  m_ConversionResult;
    std::map<uint32_t, bool>          m_OperationSupported;

    // Working/intermediate data
    std::vector<armnn::IOutputSlot*>  m_OutputSlotForOperand;
    std::vector<android::nn::RunTimePoolInfo> m_MemPools;
};

} // armnn_driver
