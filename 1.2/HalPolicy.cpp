//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "../1.0/HalPolicy.hpp"
#include "../1.1/HalPolicy.hpp"

#include <DataLayoutIndexed.hpp>

#include <cmath>

namespace armnn_driver
{
namespace hal_1_2
{

bool HandledByV1_0(V1_2::OperationType operationType)
{
    switch (static_cast<V1_0::OperationType>(operationType))
    {
        case V1_0::OperationType::ADD:
        case V1_0::OperationType::AVERAGE_POOL_2D:
        case V1_0::OperationType::CONCATENATION:
        case V1_0::OperationType::DEPTH_TO_SPACE:
        case V1_0::OperationType::DEQUANTIZE:
        case V1_0::OperationType::EMBEDDING_LOOKUP:
        case V1_0::OperationType::FLOOR:
        case V1_0::OperationType::FULLY_CONNECTED:
        case V1_0::OperationType::HASHTABLE_LOOKUP:
        case V1_0::OperationType::L2_NORMALIZATION:
        case V1_0::OperationType::L2_POOL_2D:
        case V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case V1_0::OperationType::LOGISTIC:
        case V1_0::OperationType::LSH_PROJECTION:
        case V1_0::OperationType::LSTM:
        case V1_0::OperationType::MAX_POOL_2D:
        case V1_0::OperationType::MUL:
        case V1_0::OperationType::RELU:
        case V1_0::OperationType::RELU1:
        case V1_0::OperationType::RELU6:
        case V1_0::OperationType::RESHAPE:
        case V1_0::OperationType::RESIZE_BILINEAR:
        case V1_0::OperationType::RNN:
        case V1_0::OperationType::SOFTMAX:
        case V1_0::OperationType::SPACE_TO_DEPTH:
        case V1_0::OperationType::SVDF:
        case V1_0::OperationType::TANH:
        case V1_0::OperationType::OEM_OPERATION:
            return true;
        default:
            return false;
    }
}

bool HandledByV1_1(V1_2::OperationType operationType)
{
    if (HandledByV1_0(operationType))
    {
        return true;
    }
    switch (static_cast<V1_1::OperationType>(operationType))
    {
        case V1_1::OperationType::BATCH_TO_SPACE_ND:
        case V1_1::OperationType::DIV:
        case V1_1::OperationType::MEAN:
        case V1_1::OperationType::PAD:
        case V1_1::OperationType::SPACE_TO_BATCH_ND:
        case V1_1::OperationType::SQUEEZE:
        case V1_1::OperationType::STRIDED_SLICE:
        case V1_1::OperationType::SUB:
        case V1_1::OperationType::TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool HandledByV1_0(const V1_2::Operation& operation)
{
    return HandledByV1_0(operation.type);
}

bool HandledByV1_1(const V1_2::Operation& operation)
{
    return HandledByV1_1(operation.type);
}

V1_0::OperationType CastToV1_0(V1_2::OperationType type)
{
    return static_cast<V1_0::OperationType>(type);
}

V1_1::OperationType CastToV1_1(V1_2::OperationType type)
{
    return static_cast<V1_1::OperationType>(type);
}

V1_0::Operation ConvertToV1_0(const V1_2::Operation& operation)
{
    V1_0::Operation op;
    op.type = CastToV1_0(operation.type);
    op.inputs = operation.inputs;
    op.outputs = operation.outputs;
    return op;
}

V1_1::Operation ConvertToV1_1(const V1_2::Operation& operation)
{
    V1_1::Operation op;
    op.type = CastToV1_1(operation.type);
    op.inputs = operation.inputs;
    op.outputs = operation.outputs;
    return op;
}

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    if (HandledByV1_0(operation) && compliantWithV1_0(model))
    {
        hal_1_0::HalPolicy::Operation v10Operation = ConvertToV1_0(operation);
        hal_1_0::HalPolicy::Model v10Model = convertToV1_0(model);

        return hal_1_0::HalPolicy::ConvertOperation(v10Operation, v10Model, data);
    }

    if (HandledByV1_1(operation) && compliantWithV1_1(model))
    {
        hal_1_1::HalPolicy::Operation v11Operation = ConvertToV1_1(operation);
        hal_1_1::HalPolicy::Model v11Model = convertToV1_1(model);

        return hal_1_1::HalPolicy::ConvertOperation(v11Operation, v11Model, data);
    }

    switch (operation.type)
    {
        case V1_2::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_2::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case V1_2::OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case V1_2::OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResizeNearestNeighbor(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    const ConstTensorPin weightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data);
    const ConstTensorPin biasPin    =
        ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 2, model, data);

    if (!weightsPin.IsValid())
    {
        return Fail("%s: Operation has invalid weights", __func__);
    }

    if (!biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid biases", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    armnn::Convolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    ActivationFn activation;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 7 ||
        (operation.inputs.size() >= 8 &&
        GetInputOperand<hal_1_2::HalPolicy>(operation, 7, model)->type == OperandType::BOOL);

    if (implicitPadding)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<hal_1_2::HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 6, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<hal_1_2::HalPolicy>(operation, 8, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[2];
        const uint32_t kernelY = weights.GetShape()[1];
        const uint32_t inputX  = inputInfo.GetShape()[2];
        const uint32_t inputY  = inputInfo.GetShape()[1];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);

        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 7, model, data);
    }
    else if (operation.inputs.size() >= 10)
    {
        // explicit padding
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 9, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<hal_1_2::HalPolicy>(operation, 11, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 10, model, data);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsConvolution2dSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc,
                                       weights.GetInfo(),
                                       biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);

    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

bool HalPolicy::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const Operand* weightsOperand = GetInputOperand<hal_1_2::HalPolicy>(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }
    armnn::DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 8 ||
        (operation.inputs.size() >= 9 &&
        GetInputOperand<hal_1_2::HalPolicy>(operation, 8, model)->type == OperandType::BOOL);

    // Look ahead to find the optional DataLayout, if present
    const uint32_t dataLayoutFlagIndex = implicitPadding ? 8 : 11;
    desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, dataLayoutFlagIndex, model, data);

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    // Reinterpret weight data as [ H, W, I, M ]
    armnn::TensorShape weightsShape({ weightsOperand->dimensions[1],
                                      weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[channelsIndex],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[channelsIndex] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const armnn::PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };

    const ConstTensorPin weightsPin =
        ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                  1,
                                                                  model,
                                                                  data,
                                                                  HWIMToMIHW,
                                                                  &weightsShape);

    // Bias is a 1D tensor
    const ConstTensorPin biasPin =
        ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 2, model, data);

    if (!weightsPin.IsValid())
    {
        return Fail("%s: Operation has invalid weights", __func__);
    }

    if (!biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid biases", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    ActivationFn activation;

    if (implicitPadding)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<hal_1_2::HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 7, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<hal_1_2::HalPolicy>(operation, 9, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else if (operation.inputs.size() >= 11)
    {
        // explicit padding
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation,  10, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<hal_1_2::HalPolicy>(operation, 12, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsDepthwiseConvolutionSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc,
                                       weights.GetInfo(),
                                       biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
        data.m_Network->AddDepthwiseConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));
    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

bool HalPolicy::ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    LayerInputHandle alpha = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 1, model, data);

    if (!input.IsValid() || !alpha.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& alphaInfo  = alpha.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsPreluSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       alphaInfo,
                                       outputInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddPreluLayer();

    if (!layer)
    {
        return Fail("%s: AddPreluLayer failed", __func__);
    }

    input.Connect(layer->GetInputSlot(0));
    alpha.Connect(layer->GetInputSlot(1));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertResizeNearestNeighbor(const Operation& operation, const Model& model, ConversionData& data)
{
        LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::ResizeDescriptor descriptor;
    descriptor.m_Method     = armnn::ResizeMethod::NearestNeighbor;
    descriptor.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 3, model, data);

    OperandType operandType1;
    OperandType operandType2;

    if (!GetOperandType<hal_1_2::HalPolicy>(operation, 1, model, operandType1) ||
        !GetOperandType<hal_1_2::HalPolicy>(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (operandType1 != operandType2)
    {
        return Fail("%s: Operation has invalid inputs. Type of input 1 and 2 should be the same", __func__);
    }

    if (operandType1 == OperandType::INT32)
    {
        // Case 1: resizing by shape
        int32_t targetWidth  = 0;
        int32_t targetHeight = 0;

        if (!GetInputInt32<hal_1_2::HalPolicy>(operation, 1, targetWidth, model, data) ||
            !GetInputInt32<hal_1_2::HalPolicy>(operation, 2, targetHeight, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by shape", __func__);
        }

        if (targetWidth < 0 || targetHeight < 0)
        {
            return Fail("%s: Operation has invalid inputs for resizing by shape. "
                        "Target width/height cannot be < 0", __func__);
        }

        descriptor.m_TargetWidth = static_cast<uint32_t>(targetWidth);
        descriptor.m_TargetWidth = static_cast<uint32_t>(targetHeight);
    }
    else if (operandType1 == OperandType::FLOAT32)
    {
        // Case 2: resizing by scale
        float widthScale  = 1.0f;
        float heightScale = 1.0f;

        if (!GetInputFloat32<hal_1_2::HalPolicy>(operation, 1, widthScale, model, data) ||
            !GetInputFloat32<hal_1_2::HalPolicy>(operation, 2, heightScale, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by scale", __func__);
        }

        const armnn::TensorShape& inputShape = inputInfo.GetShape();
        armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);

        float width  = inputShape[dataLayoutIndexed.GetWidthIndex()];
        float height = inputShape[dataLayoutIndexed.GetHeightIndex()];

        descriptor.m_TargetWidth  = std::floor(width  * widthScale);
        descriptor.m_TargetHeight = std::floor(height * heightScale);
    }
    else
    {
        // NOTE: FLOAT16 scales are not supported
        return false;
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsResizeSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       descriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddResizeLayer(descriptor);

    assert(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

} // namespace hal_1_2
} // namespace armnn_driver
