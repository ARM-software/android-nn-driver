//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "armnn/Optional.hpp"

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
            return ConvertConv2d(operation, model, data);
        case V1_0::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
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

bool HalPolicy::ConvertAdd(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsAdditionSupported,
                          data.m_Compute,
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
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::Average, model, data);
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
    if (!GetInputScalar(operation, numInputTensors, OperandType::INT32, concatDim, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0, model);
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
        const Operand* const operand = GetInputOperand(operation, i, model);
        if (!operand)
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        armnn::TensorShape operandShape     = GetTensorShapeForOperand(*operand);
        LayerInputHandle operandInputHandle = ConvertToLayerInputHandle(operation, i, model, data);

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
            outputShape = armnn::TensorShape({1, 1, outputShape[0], outputShape[1]});
        }
    }

    // Get the pair of permutations required for the concatenation
    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutationPair =
            std::make_pair(IdentityPermutation4D, IdentityPermutation4D);

    CreatePermutationParameters(inputShapes[0].GetNumDimensions(), concatDim, permutationPair);

    outputShape = armnnUtils::Permuted(outputShape, permutationPair.first);
    outputInfo.SetShape(outputShape);

    // this is no-op for identity swizzles, otherwise it replaces both
    // the handles and shapes with the swizzled layer output handles and shapes
    SwizzleInputs(*data.m_Network, inputHandles, inputShapes, permutationPair.first);

    // Create an armnn merger layer descriptor - this will also perform validation on the input shapes
    armnn::OriginsDescriptor mergerDescriptor;
    try
    {
        // The merger descriptor is always created across the only supported concat
        // dimension, which is 0 or 1
        mergerDescriptor =
            armnn::CreateMergerDescriptorForConcatenation(
                inputShapes.begin(), inputShapes.end(), concatDim);
    }
    catch (const armnn::Exception& error)
    {
        return Fail("%s: Error preparing merger descriptor. %s", __func__, error.what());
    }

    // Validate the output shape is correct given the input shapes based on the
    // only valid concat dimension which is 0 or 1
    if (!ValidateConcatOutputShape(inputShapes, outputShape, concatDim))
    {
        return Fail("%s: Error validating the output shape for concat", __func__);
    }

    std::vector<const armnn::TensorInfo*> inputTensorInfos;
    std::transform(inputHandles.begin(), inputHandles.end(), std::back_inserter(inputTensorInfos),
        [](const LayerInputHandle& h) -> const armnn::TensorInfo*{ return &h.GetTensorInfo(); });
    if (!IsLayerSupported(__func__,
                          armnn::IsMergerSupported,
                          data.m_Compute,
                          inputTensorInfos,
                          mergerDescriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddMergerLayer(mergerDescriptor);
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

    // Add permutation layer and connect the output to it, the permutation becomes the output layer
    armnn::IConnectableLayer& deswizzleLayer = AddPermuteLayer(*data.m_Network,
                                                               layer->GetOutputSlot(0),
                                                               permutationPair.second);
    layer = &deswizzleLayer;

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
            afterConcatInfo.SetShape(armnn::TensorShape({ afterConcatInfo.GetShape()[2],
                                                          afterConcatInfo.GetShape()[3] }));
        }

        layer = &AddReshapeLayer(
                *data.m_Network,
                layer->GetOutputSlot(0),
                afterConcatInfo
        );
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo  = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    // ArmNN does not currently support non-fixed weights or bias
    const ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin(operation, 1, model, data, NHWCToArmNN);
    const ConstTensorPin biasPin    = ConvertOperationInputToConstTensorPin(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), swizzledInputInfo);

    armnn::Convolution2dDescriptor desc;
    ActivationFn activation;

    if (operation.inputs.size() == 10)
    {
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data)   ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data)  ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data)    ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data)   ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data)   ||
            !GetInputActivationFunction(operation, 9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 7)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 3, paddingScheme, model, data)               ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction(operation, 6, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = swizzledInputInfo.GetShape()[3];
        const uint32_t inputY  = swizzledInputInfo.GetShape()[2];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupported(__func__,
                          armnn::IsConvolution2dSupported,
                          data.m_Compute,
                          swizzledInputInfo,
                          swizzledOutputInfo,
                          desc,
                          weights.GetInfo(),
                          biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddConvolution2dLayer(desc, weights, bias);
    armnn::IConnectableLayer* endLayer = ProcessActivation(swizzledOutputInfo, activation, startLayer, data);

    if (endLayer != nullptr)
    {
        armnn::IConnectableLayer& outSwizzleLayer =
                SwizzleInDeswizzleOut(*data.m_Network, input, *startLayer, *endLayer);
        return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo  = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    // ArmNN does not currently support non-fixed weights or bias

    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    // but in ArmNN it needs to be [ M, I, H, W ]
    const Operand* weightsOperand = GetInputOperand(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }

    // Reinterpret weight data as [ H, W, I, M ]
    armnn::TensorShape weightsShape({ weightsOperand->dimensions[1], weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[3],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[3] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const armnn::PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };
    ConstTensorPin weightsPin =
            ConvertOperationInputToConstTensorPin(operation, 1, model, data, HWIMToMIHW, &weightsShape);

    // Bias is a 1D tensor
    ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), swizzledInputInfo);

    armnn::DepthwiseConvolution2dDescriptor desc;
    ActivationFn activation;

    if (operation.inputs.size() == 11)
    {
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data)         ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data)        ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data)          ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data)       ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data)         ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data)         ||
            !GetInputActivationFunction(operation,  10, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 8)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 3, paddingScheme, model, data)                       ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX, model, data)         ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY, model, data)         ||
            !GetInputActivationFunction(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = swizzledInputInfo.GetShape()[3];
        const uint32_t inputY  = swizzledInputInfo.GetShape()[2];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupported(__func__,
                          armnn::IsDepthwiseConvolutionSupported,
                          data.m_Compute,
                          swizzledInputInfo,
                          swizzledOutputInfo,
                          desc,
                          weights.GetInfo(),
                          biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddDepthwiseConvolution2dLayer(desc, weights, bias);
    armnn::IConnectableLayer* endLayer   = ProcessActivation(swizzledOutputInfo, activation, startLayer, data);

    if (endLayer != nullptr)
    {
        armnn::IConnectableLayer& outSwizzleLayer =
                SwizzleInDeswizzleOut(*data.m_Network, input, *startLayer, *endLayer);
        return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer, model, data);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsFloorSupported,
                          data.m_Compute,
                          input.GetTensorInfo(),
                          GetTensorInfoForOperand(*outputOperand)))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddFloorLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin(operation, 1, model, data); // 2D
    ConstTensorPin biasPin    = ConvertOperationInputToConstTensorPin(operation, 2, model, data);    // 1D

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias    = biasPin.GetConstTensor();

    armnn::TensorInfo reshapedInfo = inputInfo;
    if (inputInfo.GetNumDimensions() > 2U)
    {
        unsigned int dim0 = inputInfo.GetShape()[0];
        unsigned int dim1 = inputInfo.GetShape()[1];

        for (unsigned int i = 2U; i < inputInfo.GetNumDimensions(); ++i)
        {
            dim1 *= inputInfo.GetShape()[i];
        }

        unsigned int divisor = weights.GetInfo().GetShape()[1] / dim1;
        if(dim0 % divisor != 0)
        {
            return Fail("%s: Failed to deduce tensor shape", __func__);
        }

        reshapedInfo.SetShape(armnn::TensorShape({dim0 / divisor, dim1 * divisor}));
    }

    // ensuring that the bias value is within 1% of the weights input (small float differences can exist)
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), reshapedInfo);

    ActivationFn activationFunction;
    if (!GetInputActivationFunction(operation, 3, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::FullyConnectedDescriptor desc;
    desc.m_TransposeWeightMatrix = true;
    desc.m_BiasEnabled           = true;

    if (!IsLayerSupported(__func__,
                          armnn::IsFullyConnectedSupported,
                          data.m_Compute,
                          inputInfo,
                          outputInfo,
                          weights.GetInfo(),
                          bias.GetInfo(),
                          desc))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddFullyConnectedLayer(desc, weights, bias);
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

        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer, model, data);
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
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
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
        !GetInputScalar(operation, 1, OperandType::INT32, descriptor.m_NormSize, model, data) ||
        !GetInputFloat32(operation, 2, descriptor.m_K, model, data) ||
        !GetInputFloat32(operation, 3, descriptor.m_Alpha, model, data) ||
        !GetInputFloat32(operation, 4, descriptor.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in AndroidNN.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    if (!IsLayerSupported(__func__,
                        armnn::IsNormalizationSupported,
                        data.m_Compute,
                        inputInfo,
                        outputInfo,
                        descriptor))
    {
        return false;
    }


    armnn::IConnectableLayer* layer = data.m_Network->AddNormalizationLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::Sigmoid;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin = ConvertOperationInputToConstTensorPin(operation, 2, model, data);
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin = ConvertOperationInputToConstTensorPin(operation, 3, model, data);
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 4, model, data);
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 6, model, data);
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin = ConvertOperationInputToConstTensorPin(operation, 7, model, data);
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 8, model, data);
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin = ConvertOperationInputToConstTensorPin(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 15, model, data);

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
    const ConstTensorPin inputToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 1, model, data);
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 5, model, data);
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 9, model, data);
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin = ConvertOperationInputToConstTensorPin(operation, 10, model, data);
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 11, model, data);
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 12, model, data);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin = ConvertOperationInputToConstTensorPin(operation, 16, model, data);
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin = ConvertOperationInputToConstTensorPin(operation, 17, model, data);

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
    if (!GetInputActivationFunctionFromTensor(operation, 20, activation, model, data) ||
        !GetInputScalar(operation, 21, OperandType::FLOAT32, cellClip, model, data) ||
        !GetInputScalar(operation, 22, OperandType::FLOAT32, projClip, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    const Operand* scratchBuffer = GetOutputOperand(operation, 0, model);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const Operand* outputStateOut = GetOutputOperand(operation, 1, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand(operation, 2, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand(operation, 3, model);
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
    const armnn::TensorInfo& inputToForgetWeights = params.m_InputToForgetWeights->GetInfo();
    const armnn::TensorInfo& inputToCellWeights   = params.m_InputToCellWeights->GetInfo();
    const armnn::TensorInfo& inputToOutputWeights = params.m_InputToOutputWeights->GetInfo();
    const armnn::TensorInfo& recurrentToForgetWeights = params.m_RecurrentToForgetWeights->GetInfo();
    const armnn::TensorInfo& recurrentToCellWeights = params.m_RecurrentToCellWeights->GetInfo();
    const armnn::TensorInfo& recurrentToOutputWeights = params.m_RecurrentToOutputWeights->GetInfo();
    const armnn::TensorInfo& forgetGateBias = params.m_ForgetGateBias->GetInfo();
    const armnn::TensorInfo& cellBias = params.m_CellBias->GetInfo();
    const armnn::TensorInfo& outputGateBias = params.m_OutputGateBias->GetInfo();

    //Optional parameters
    const armnn::TensorInfo* inputToInputWeights = nullptr;
    const armnn::TensorInfo* recurrentToInputWeights = nullptr;
    const armnn::TensorInfo* cellToInputWeights = nullptr;
    const armnn::TensorInfo* inputGateBias = nullptr;
    const armnn::TensorInfo* projectionWeights = nullptr;
    const armnn::TensorInfo* projectionBias    = nullptr;
    const armnn::TensorInfo* cellToForgetWeights = nullptr;
    const armnn::TensorInfo* cellToOutputWeights = nullptr;

    if(!desc.m_CifgEnabled)
    {
        inputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        recurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            cellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        inputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if(desc.m_ProjectionEnabled)
    {
        projectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            projectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if(desc.m_PeepholeEnabled)
    {
        cellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        cellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsLstmSupported,
                          data.m_Compute,
                          inputInfo,
                          outputStateInInfo,
                          cellStateInInfo,
                          scratchBufferInfo,
                          outputStateOutInfo,
                          cellStateOutInfo,
                          outputInfo,
                          desc,
                          inputToForgetWeights,
                          inputToCellWeights,
                          inputToOutputWeights,
                          recurrentToForgetWeights,
                          recurrentToCellWeights,
                          recurrentToOutputWeights,
                          forgetGateBias,
                          cellBias,
                          outputGateBias,
                          inputToInputWeights,
                          recurrentToInputWeights,
                          cellToInputWeights,
                          inputGateBias,
                          projectionWeights,
                          projectionBias,
                          cellToForgetWeights,
                          cellToOutputWeights))
    {
        return false;
    }

    // Add the layer
    armnn::IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0, model, data) &&
            SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1, model, data) &&
            SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2, model, data) &&
            SetupAndTrackLayerOutputSlot(operation, 3, *layer, 3, model, data));
}

bool HalPolicy::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    if (!IsLayerSupported(__func__,
                          armnn::IsL2NormalizationSupported,
                          data.m_Compute,
                          inputInfo,
                          outputInfo,
                          desc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddL2NormalizationLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertMul(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);

    if (outputOperand == nullptr)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsMultiplicationSupported,
                          data.m_Compute,
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
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer, model, data);
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

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 1.0f;
    desc.m_B        = -1.0f;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 6.0f;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    armnn::SoftmaxDescriptor desc;
    if (!GetInputFloat32(operation, 1, desc.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsSoftmaxSupported,
                          data.m_Compute,
                          input.GetTensorInfo(),
                          outInfo,
                          desc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::TanH;
    desc.m_A = 1.0f; // android nn does not support tanH parameters
    desc.m_B = 1.0f; // set to 1.0f for unity scaling

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    const Operand* inputOperand = GetInputOperand(operation, 0, model);
    const Operand* requestedShapeOperand = GetInputOperand(operation, 1, model);
    const Operand* outputOperand = GetOutputOperand(operation, 0, model);

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
    if (!GetTensorInt32Values(*requestedShapeOperand, targetDimensions, model, data))
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

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsReshapeSupported,
                          data.m_Compute,
                          input.GetTensorInfo()))
    {
        return false;
    }


    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = armnn::TensorShape(requestedShape.dimensions.size(),
                                                         requestedShape.dimensions.data());

    armnn::IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertResizeBilinear(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::ResizeBilinearDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    if (!IsLayerSupported(__func__,
                          armnn::IsResizeBilinearSupported,
                          data.m_Compute,
                          inputInfo))
    {
        return false;
    }


    if (   !GetInputScalar(operation, 1, OperandType::INT32, desc.m_TargetHeight, model, data)
        || !GetInputScalar(operation, 2, OperandType::INT32, desc.m_TargetWidth, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddResizeBilinearLayer(desc);

    assert(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);

}

} // namespace hal_1_0
} // namespace armnn_driver
