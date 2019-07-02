//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include <armnn/Optional.hpp>

#include "FullyConnected.hpp"

namespace armnn_driver
{
namespace hal_1_0
{

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case V1_0::OperationType::ADD:
            return ConvertAdd(operation, model, data);
        case V1_0::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_0::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_0::OperationType::CONV_2D:
            return ValidateConv2dParameters(operation) &&
                   ConvertConv2d<hal_1_0::HalPolicy>(operation, model, data);
        case V1_0::OperationType::DEPTHWISE_CONV_2D:
            return ValidateDepthwiseConv2dParameters(operation) &&
                   ConvertDepthwiseConv2d<hal_1_0::HalPolicy>(operation, model, data);
        case V1_0::OperationType::DEQUANTIZE:
            return ConvertDequantize(operation, model, data);
        case V1_0::OperationType::FLOOR:
            return ConvertFloor(operation, model, data);
        case V1_0::OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation, model, data);
        case V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation, model, data);
        case V1_0::OperationType::LOGISTIC:
            return ConvertLogistic(operation, model, data);
        case V1_0::OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        case V1_0::OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation, model, data);
        case V1_0::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case V1_0::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case V1_0::OperationType::MUL:
            return ConvertMul(operation, model, data);
        case V1_0::OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case V1_0::OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case V1_0::OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case V1_0::OperationType::SOFTMAX:
            return ConvertSoftmax(operation, model, data);
        case V1_0::OperationType::SPACE_TO_DEPTH:
            return ConvertSpaceToDepth(operation, model, data);
        case V1_0::OperationType::TANH:
            return ConvertTanH(operation, model, data);
        case V1_0::OperationType::RESHAPE:
            return ConvertReshape(operation, model, data);
        case V1_0::OperationType::RESIZE_BILINEAR:
            return ConvertResizeBilinear(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ValidateConv2dParameters(const Operation &operation)
{
    if (operation.inputs.size() != 10 && operation.inputs.size() != 7)
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }
    return true;
}

bool HalPolicy::ValidateDepthwiseConv2dParameters(const Operation &operation)
{
    if (operation.inputs.size() != 11 && operation.inputs.size() != 8)
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }
    return true;
}

bool HalPolicy::ConvertAdd(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<hal_1_0::HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsAdditionSupported,
                                       data.m_Backends,
                                       input0.GetTensorInfo(),
                                       input1.GetTensorInfo(),
                                       outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddAdditionLayer();
    armnn::IConnectableLayer* const endLayer   = ProcessActivation(outInfo, activationFunction, startLayer, data);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer != nullptr)
    {
        BroadcastTensor(input0, input1, startLayer, *data.m_Network);
        return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *endLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    // The first N (0..N-1) inputs are tensors. The Nth input is the concatenation axis.
    if (operation.inputs.size() <= 1)
    {
        return Fail("%s: Operation has insufficient arguments", __func__);
    }

    // Get inputs and outputs
    const std::size_t numInputTensors = operation.inputs.size() - 1;

    int32_t concatDim;
    if (!GetInputScalar<hal_1_0::HalPolicy>(operation, numInputTensors, OperandType::INT32, concatDim, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }


    armnn::TensorInfo  outputInfo  = GetTensorInfoForOperand(*outputOperand);
    armnn::TensorShape outputShape = outputInfo.GetShape();

    //
    // handle negative concat dims along the lines of tensorflow as described here:
    //    https://www.tensorflow.org/api_docs/python/tf/concat
    // "negative axis refers to axis + rank(values)-th dimension"
    //
    if (concatDim < 0)
    {
        concatDim += outputShape.GetNumDimensions();
    }

    if (concatDim >= static_cast<int32_t>(outputShape.GetNumDimensions()) || concatDim < 0)
    {
        return Fail("%s: Operation has invalid concat axis: %d", __func__, concatDim);
    }

    std::vector<LayerInputHandle>   inputHandles;
    std::vector<armnn::TensorShape> inputShapes;

    inputHandles.reserve(numInputTensors);
    inputShapes.reserve(numInputTensors);

    bool inputsHaveBeenReshaped        = false;
    unsigned int tensorDimensionsAdded = 0;

    for (uint32_t i = 0; i < numInputTensors; ++i)
    {
        const Operand* const operand = GetInputOperand<hal_1_0::HalPolicy>(operation, i, model);
        if (!operand)
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        armnn::TensorShape operandShape     = GetTensorShapeForOperand(*operand);
        LayerInputHandle operandInputHandle =
            ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, i, model, data);

        if (operandShape.GetNumDimensions() == 0)
        {
            return Fail("%s: Operands with rank 0 are not supported", __func__);
        }

        if (RequiresReshape(operandShape))
        {
            inputsHaveBeenReshaped = true;

            armnn::TensorInfo reshapeInfo = operandInputHandle.GetTensorInfo();

            // Expand the tensor to three dimensions
            if (operandShape.GetNumDimensions() == 2)
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, operandShape[0], operandShape[1]}));
                tensorDimensionsAdded = 1;
            }
            else
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, 1, operandShape[0]}));
                tensorDimensionsAdded = 2;
            }

            armnn::IConnectableLayer& newReshape = AddReshapeLayer(
                    *data.m_Network,
                    operandInputHandle,
                    reshapeInfo
            );

            // Point to the reshape operation rather then the input operation
            operandShape = reshapeInfo.GetShape();
            operandInputHandle = LayerInputHandle(true, &newReshape.GetOutputSlot(0), reshapeInfo);
        }

        inputShapes.emplace_back(operandShape);
        inputHandles.emplace_back(operandInputHandle);

        if (!inputHandles.back().IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }

    BOOST_ASSERT(inputShapes.size() == inputHandles.size());

    if (inputsHaveBeenReshaped)
    {
        // Adjust the concatenation dimension by the amount of dimensions added (if any)
        concatDim += tensorDimensionsAdded;

        // Add extra dimensions to the output shape to reflect the addition of the reshape layers
        if (tensorDimensionsAdded == 1)
        {
            outputShape = armnn::TensorShape({1, outputShape[0], outputShape[1]});
        }
        else if (tensorDimensionsAdded == 2)
        {
            outputShape = armnn::TensorShape({1, 1, outputShape[0]});
        }
    }

    // Check if permutations is required and get the pair of permutations required for the concatenation.
    // Permutation is required when the concat dimension is 2 for a 4D tensor or 1 for a 3D tensor.
    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutationPair =
            std::make_pair(IdentityPermutation4D, IdentityPermutation4D);

    bool needPermute =
        CreateConcatPermutationParameters(inputShapes[0].GetNumDimensions(), concatDim, permutationPair);

    if (needPermute)
    {
        outputShape = armnnUtils::Permuted(outputShape, permutationPair.first);
    }

    outputInfo.SetShape(outputShape);

    // this is no-op for identity swizzles, otherwise it replaces both
    // the handles and shapes with the swizzled layer output handles and shapes
    SwizzleInputs(*data.m_Network, inputHandles, inputShapes, permutationPair.first);

    // Create an armnn concat layer descriptor - this will also perform validation on the input shapes
    armnn::OriginsDescriptor concatDescriptor;

    try
    {
        // The concat descriptor is always created across the only supported concat dimension
        // which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
        concatDescriptor =
            armnn::CreateDescriptorForConcatenation(inputShapes.begin(), inputShapes.end(), concatDim);
    }
    catch (const armnn::Exception& error)
    {
        return Fail("%s: Error preparing concat descriptor. %s", __func__, error.what());
    }

    // Validate the output shape is correct given the input shapes based on the
    // only valid concat dimension which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
    if (!ValidateConcatOutputShape(inputShapes, outputShape, concatDim))
    {
        return Fail("%s: Error validating the output shape for concat", __func__);
    }

    std::vector<const armnn::TensorInfo*> inputTensorInfos;
    std::transform(inputHandles.begin(), inputHandles.end(), std::back_inserter(inputTensorInfos),
        [](const LayerInputHandle& h) -> const armnn::TensorInfo*{ return &h.GetTensorInfo(); });
    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsConcatSupported,
                                       data.m_Backends,
                                       inputTensorInfos,
                                       outputInfo,
                                       concatDescriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddConcatLayer(concatDescriptor);
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Connect inputs to the layer
    const int numInputSlots = layer->GetNumInputSlots();
    assert(static_cast<std::size_t>(numInputSlots) == inputHandles.size());
    for (int i = 0; i < numInputSlots; ++i)
    {
        // connect the input directly to the merge (concat) layer
        inputHandles[static_cast<unsigned int>(i)].Connect(layer->GetInputSlot(i));
    }

    if (needPermute)
    {
        // Add permutation layer and connect the output to it, the permutation becomes the output layer
        armnn::IConnectableLayer& deswizzleLayer = AddPermuteLayer(*data.m_Network,
                                                                   layer->GetOutputSlot(0),
                                                                   permutationPair.second);
        layer = &deswizzleLayer;
    }

    if (inputsHaveBeenReshaped)
    {
        armnn::TensorInfo afterConcatInfo = layer->GetOutputSlot(0).GetTensorInfo();

        // Undo the reshape knowing the amount of dimensions added
        if (tensorDimensionsAdded == 1)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({ afterConcatInfo.GetShape()[1],
                                                          afterConcatInfo.GetShape()[2] }));
        }
        else if (tensorDimensionsAdded == 2)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({ afterConcatInfo.GetShape()[2] }));
        }

        layer = &AddReshapeLayer(
                *data.m_Network,
                layer->GetOutputSlot(0),
                afterConcatInfo
        );
    }

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsDequantizeSupported,
                                       data.m_Backends,
                                       input.GetTensorInfo(),
                                       GetTensorInfoForOperand(*outputOperand)))
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddDequantizeLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsFloorSupported,
                                       data.m_Backends,
                                       input.GetTensorInfo(),
                                       GetTensorInfoForOperand(*outputOperand)))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddFloorLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    ConstTensorPin weightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 1, model, data); // 2D
    ConstTensorPin biasPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 2, model, data); // 1D

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias    = biasPin.GetConstTensor();
    armnn::TensorInfo reshapedInfo = inputInfo;

    try
    {
        reshapedInfo.SetShape(FlattenFullyConnectedInput(inputInfo.GetShape(), weights.GetInfo().GetShape()));
    } catch (const std::exception &e) {
        return Fail("%s: %s", __func__, e.what());
    }

    // ensuring that the bias value is within 1% of the weights input (small float differences can exist)
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), reshapedInfo);

    ActivationFn activationFunction;
    if (!GetInputActivationFunction<hal_1_0::HalPolicy>(operation, 3, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::FullyConnectedDescriptor desc;
    desc.m_TransposeWeightMatrix = true;
    desc.m_BiasEnabled           = true;

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsFullyConnectedSupported,
                                       data.m_Backends,
                                       reshapedInfo,
                                       outputInfo,
                                       weights.GetInfo(),
                                       bias.GetInfo(),
                                       desc))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddFullyConnectedLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));
    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activationFunction, startLayer, data);

    if (endLayer != nullptr)
    {
        if (inputInfo.GetNumDimensions() > 2U)
        {
            armnn::ReshapeDescriptor reshapeDescriptor;
            reshapeDescriptor.m_TargetShape = reshapedInfo.GetShape();

            armnn::IConnectableLayer* reshapeLayer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
            assert(reshapeLayer != nullptr);
            input.Connect(reshapeLayer->GetInputSlot(0));
            reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);
            reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        }
        else
        {
            input.Connect(startLayer->GetInputSlot(0));
        }

        return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *endLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::NormalizationDescriptor descriptor;

    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Across;
    descriptor.m_NormMethodType = armnn::NormalizationAlgorithmMethod::LocalBrightness;

    if (!input.IsValid() ||
        !GetInputScalar<hal_1_0::HalPolicy>(operation, 1, OperandType::INT32, descriptor.m_NormSize, model, data) ||
        !GetInputFloat32<hal_1_0::HalPolicy>(operation, 2, descriptor.m_K, model, data) ||
        !GetInputFloat32<hal_1_0::HalPolicy>(operation, 3, descriptor.m_Alpha, model, data) ||
        !GetInputFloat32<hal_1_0::HalPolicy>(operation, 4, descriptor.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in AndroidNN.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsNormalizationSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       descriptor))
    {
        return false;
    }


    armnn::IConnectableLayer* layer = data.m_Network->AddNormalizationLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::Sigmoid;

    return ConvertToActivation<hal_1_0::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 2, model, data);
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 3, model, data);
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 4, model, data);
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 6, model, data);
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 7, model, data);
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 8, model, data);
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation, 15, model, data);

    if (!inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input tensors:
    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  1,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  5,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  9,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  10,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  11,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  12,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                                  16,
                                                                  model,
                                                                  data,
                                                                  g_DontPermute,
                                                                  nullptr,
                                                                  true);

    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
    ConvertOperationInputToConstTensorPin<hal_1_0::HalPolicy>(operation,
                                                              17,
                                                              model,
                                                              data,
                                                              g_DontPermute,
                                                              nullptr,
                                                              true);

    if ((!inputToInputWeightsPin.IsValid() && !inputToInputWeightsPin.IsOptional()) ||
        (!recurrentToInputWeightsPin.IsValid() && !recurrentToInputWeightsPin.IsOptional()) ||
        (!cellToInputWeightsPin.IsValid() && !cellToInputWeightsPin.IsOptional()) ||
        (!cellToForgetWeightsPin.IsValid() && !cellToForgetWeightsPin.IsOptional()) ||
        (!cellToOutputWeightsPin.IsValid() && !cellToOutputWeightsPin.IsOptional()) ||
        (!inputGateBiasPin.IsValid() && !inputGateBiasPin.IsOptional()) ||
        (!projectionWeightsPin.IsValid() && !projectionWeightsPin.IsOptional()) ||
        (!projectionBiasPin.IsValid() && !projectionBiasPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the mandatory input scalars (actually 1-D tensors of size 1):
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    ActivationFn activation;
    float cellClip;
    float projClip;
    if (!GetInputActivationFunctionFromTensor<hal_1_0::HalPolicy>(operation, 20, activation, model, data) ||
        !GetInputScalar<hal_1_0::HalPolicy>(operation, 21, OperandType::FLOAT32, cellClip, model, data) ||
        !GetInputScalar<hal_1_0::HalPolicy>(operation, 22, OperandType::FLOAT32, projClip, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4]
    // with CIFG, or [batch_size, num_units * 3] without CIFG.
    const Operand* scratchBuffer = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const Operand* outputStateOut = GetOutputOperand<hal_1_0::HalPolicy>(operation, 1, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand<hal_1_0::HalPolicy>(operation, 2, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 3, model);
    if (!output)
    {
        return Fail("%s: Could not read output 3: output", __func__);
    }

    // set the params structure for the AddLstmLayer call
    armnn::LstmInputParams params;
    params.m_InputToInputWeights = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_CellToInputWeights = cellToInputWeightsPin.GetConstTensorPtr();
    params.m_CellToForgetWeights = cellToForgetWeightsPin.GetConstTensorPtr();
    params.m_CellToOutputWeights = cellToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias = outputGateBiasPin.GetConstTensorPtr();
    params.m_ProjectionWeights = projectionWeightsPin.GetConstTensorPtr();
    params.m_ProjectionBias = projectionBiasPin.GetConstTensorPtr();

    // set the layer descriptor
    armnn::LstmDescriptor desc;
    desc.m_ActivationFunc = activation;
    desc.m_ClippingThresCell = cellClip;
    desc.m_ClippingThresProj = projClip;
    desc.m_CifgEnabled = (params.m_InputToInputWeights == nullptr ||
                          params.m_RecurrentToInputWeights == nullptr ||
                          params.m_InputGateBias == nullptr);
    desc.m_PeepholeEnabled = (params.m_CellToForgetWeights != nullptr ||
                              params.m_CellToOutputWeights != nullptr);
    desc.m_ProjectionEnabled = (params.m_ProjectionWeights != nullptr);

    // validate the optional input groups
    if (desc.m_CifgEnabled &&
        (params.m_InputToInputWeights != nullptr ||
         params.m_RecurrentToInputWeights != nullptr ||
         params.m_InputGateBias != nullptr))
    {
        return Fail("%s: All, or none, of input-to-input weights, recurrent-to-input weights,"
                    " and input gate bias must be provided", __func__);
    }

    if (!desc.m_ProjectionEnabled && params.m_ProjectionBias != nullptr)
    {
        return Fail("%s: projection bias should not be provided without projection weights", __func__);
    }

    if (desc.m_PeepholeEnabled &&
        (params.m_CellToForgetWeights == nullptr ||
         params.m_CellToOutputWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_CellToInputWeights == nullptr)))
    {
        return Fail("%s: All, or none, of cell-to-forget weights and cell-to-output weights must be provided"
                    " and, if CIFG is not enabled, cell-to-input weights must also be provided", __func__);
    }

    // Check if the layer is supported
    // Inputs
    const armnn::TensorInfo& inputInfo         = input.GetTensorInfo();
    const armnn::TensorInfo& outputStateInInfo = outputStateIn.GetTensorInfo();
    const armnn::TensorInfo& cellStateInInfo   = cellStateIn.GetTensorInfo();

    // Outputs
    const armnn::TensorInfo& scratchBufferInfo  = GetTensorInfoForOperand(*scratchBuffer);
    const armnn::TensorInfo& outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    const armnn::TensorInfo& cellStateOutInfo   = GetTensorInfoForOperand(*cellStateOut);
    const armnn::TensorInfo& outputInfo         = GetTensorInfoForOperand(*output);

    // Basic parameters
    armnn::LstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    if(!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if(desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if(desc.m_PeepholeEnabled)
    {
        paramsInfo.m_CellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        paramsInfo.m_CellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsLstmSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputStateInInfo,
                                       cellStateInInfo,
                                       scratchBufferInfo,
                                       outputStateOutInfo,
                                       cellStateOutInfo,
                                       outputInfo,
                                       desc,
                                       paramsInfo))
    {
        return false;
    }

    // Add the layer
    armnn::IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, 0, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 1, *layer, 1, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 2, *layer, 2, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 3, *layer, 3, model, data));
}

bool HalPolicy::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsL2NormalizationSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddL2NormalizationLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertMul(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<hal_1_0::HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);

    if (outputOperand == nullptr)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsMultiplicationSupported,
                                       data.m_Backends,
                                       input0.GetTensorInfo(),
                                       input1.GetTensorInfo(),
                                       outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddMultiplicationLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer, data);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer != nullptr)
    {
        BroadcastTensor(input0, input1, startLayer, *data.m_Network);
        return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *endLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::ReLu;

    return ConvertToActivation<hal_1_0::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 1.0f;
    desc.m_B        = -1.0f;

    return ConvertToActivation<hal_1_0::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 6.0f;

    return ConvertToActivation<hal_1_0::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    armnn::SoftmaxDescriptor desc;
    if (!GetInputFloat32<hal_1_0::HalPolicy>(operation, 1, desc.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsSoftmaxSupported,
                                       data.m_Backends,
                                       input.GetTensorInfo(),
                                       outInfo,
                                       desc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);

    if (!input.IsValid() )
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    if (rank != 4)
    {
        return Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    armnn::SpaceToDepthDescriptor desc;
    bool dataLayoutCheck;

    GetInputScalar<hal_1_0::HalPolicy>(operation, 1, OperandType::INT32, desc.m_BlockSize, model, data);

    if (desc.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsSpaceToDepthSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc))
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::TanH;
    desc.m_A = 1.0f; // android nn does not support tanH parameters
    desc.m_B = 1.0f; // set to 1.0f for unity scaling

    return ConvertToActivation<hal_1_0::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    const Operand* inputOperand = GetInputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    const Operand* requestedShapeOperand = GetInputOperand<hal_1_0::HalPolicy>(operation, 1, model);
    const Operand* outputOperand = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);

    if (inputOperand == nullptr
        || requestedShapeOperand == nullptr
        || outputOperand == nullptr)
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }


    if (requestedShapeOperand->dimensions.size() != 1)
    {
        return Fail("%s: Input 1 expected to be one-dimensional (found %i dimensions)",
            __func__, requestedShapeOperand->dimensions.size());
    }

    std::vector<int32_t> targetDimensions;
    if (!GetTensorInt32Values<hal_1_0::HalPolicy>(*requestedShapeOperand, targetDimensions, model, data))
    {
        return Fail("%s: Could not read values of input 1", __func__);
    }

    const Shape inputOperandShape = GetOperandShape(*inputOperand);

    Shape requestedShape;
    // targetDimensions may contain special values (e.g. -1). reshapePrepare() is an AndroidNN provided utility
    // function that resolves these values into a fully specified tensor shape.
    if (!reshapePrepare(inputOperandShape, targetDimensions.data(), targetDimensions.size(), &requestedShape))
    {
        return Fail("%s: Failed to resolve the requested shape", __func__);
    }

    const Shape outputOperandShape = GetOperandShape(*outputOperand);
    if (!SameShape(requestedShape, outputOperandShape))
    {
        return Fail("%s: Shape of output operand does not match resolved requested shape", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = armnn::TensorShape(requestedShape.dimensions.size(),
                                                         requestedShape.dimensions.data());

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsReshapeSupported,
                                       data.m_Backends,
                                       input.GetTensorInfo(),
                                       reshapeDescriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertResizeBilinear(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_0::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_0::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::ResizeDescriptor desc;
    desc.m_Method     = armnn::ResizeMethod::Bilinear;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsResizeSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc))
    {
        return false;
    }

    if (!GetInputScalar<hal_1_0::HalPolicy>(operation, 1, OperandType::INT32, desc.m_TargetWidth, model, data) ||
        !GetInputScalar<hal_1_0::HalPolicy>(operation, 2, OperandType::INT32, desc.m_TargetHeight, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddResizeLayer(desc);

    assert(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);

}

} // namespace hal_1_0
} // namespace armnn_driver
