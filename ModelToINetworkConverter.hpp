//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "ActivationFunctor.h"

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
    ModelToINetworkConverter(armnn::Compute compute, const Model& model,
        const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

    bool ConvertOperation(const Operation& operation);

    bool ConvertAdd(const Operation& operation);

    bool ConvertAveragePool2d(const Operation& operation);

    bool ConvertConcatenation(const Operation& operation);

    bool ConvertConv2d(const Operation& operation);

    bool ConvertDepthwiseConv2d(const Operation& operation);

    bool ConvertFloor(const Operation& operation);

    bool ConvertFullyConnected(const Operation& operation);

    bool ConvertLogistic(const Operation& operation);

    bool ConvertLocalResponseNormalization(const Operation& operation);

    bool ConvertL2Normalization(const Operation& operation);

    bool ConvertL2Pool2d(const Operation& operation);

    bool ConvertMaxPool2d(const Operation& operation);

    bool ConvertMul(const Operation& operation);

    bool ConvertReLu(const Operation& operation);

    bool ConvertReLu1(const Operation& operation);

    bool ConvertReLu6(const Operation& operation);

    bool ConvertSoftmax(const Operation& operation);

    bool ConvertTanH(const Operation& operation);

    bool ConvertReshape(const Operation& operation);

    bool ConvertResizeBilinear(const Operation& operation);

    bool ConvertToActivation(const Operation& operation, const char* operationName,
        const armnn::ActivationDescriptor& activationDesc);

    bool ConvertPooling2d(const Operation& operation, const char* name, armnn::PoolingAlgorithm poolType);


    const void* GetOperandValueReadOnlyAddress(const Operand& operand) const;

    const Operand* GetInputOperand(const Operation& operation, uint32_t inputIndex) const;

    const Operand* GetOutputOperand(const Operation& operation, uint32_t outputIndex) const;

    template<typename T>
    bool GetInputScalar(const Operation& operation, uint32_t inputIndex, OperandType type, T& outValue) const;

    bool GetInputInt32(const Operation& operation, uint32_t inputIndex, int32_t& outValue) const;

    bool GetInputFloat32(const Operation& operation, uint32_t inputIndex, float& outValue) const;

    bool GetInputActivationFunction(const Operation& operation, uint32_t inputIndex,
        ActivationFn& outActivationFunction) const;

    bool GetInputPaddingScheme(const Operation& operation, uint32_t inputIndex,
        android::nn::PaddingScheme& outPaddingScheme) const;

    LayerInputHandle ConvertToLayerInputHandle(const Operation& operation, uint32_t inputIndex);

    ConstTensorPin ConvertOperationInputToConstTensorPin(const Operation& operation, uint32_t inputIndex,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr);

    ConstTensorPin ConvertOperandToConstTensorPin(const Operand& operand,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr);

    bool GetTensorInt32Values(const Operand& operand, std::vector<int32_t>& outValues) const;


    armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo, ActivationFn activation,
                                                armnn::IConnectableLayer* prevLayer);


    bool SetupAndTrackLayerOutputSlot(const Operation& operation, uint32_t outputIndex,
                                      armnn::IConnectableLayer& layer);


    // Input data
    armnn::Compute                    m_Compute;
    const Model&                      m_Model;
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
