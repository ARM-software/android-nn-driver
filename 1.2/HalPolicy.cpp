//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "Utils.hpp"

#include "../1.0/HalPolicy.hpp"
#include "../1.1/HalPolicy.hpp"

#include <DataLayoutIndexed.hpp>
#include <Half.hpp>

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
        case V1_0::OperationType::DEPTH_TO_SPACE:
        case V1_0::OperationType::DEQUANTIZE:
        case V1_0::OperationType::EMBEDDING_LOOKUP:
        case V1_0::OperationType::FLOOR:
        case V1_0::OperationType::FULLY_CONNECTED:
        case V1_0::OperationType::HASHTABLE_LOOKUP:
        case V1_0::OperationType::L2_NORMALIZATION:
        case V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case V1_0::OperationType::LOGISTIC:
        case V1_0::OperationType::LSH_PROJECTION:
        case V1_0::OperationType::MUL:
        case V1_0::OperationType::RESHAPE:
        case V1_0::OperationType::RNN:
        case V1_0::OperationType::SVDF:
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
        case V1_1::OperationType::DIV:
        case V1_1::OperationType::MEAN:
        case V1_1::OperationType::SQUEEZE:
        case V1_1::OperationType::STRIDED_SLICE:
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
        case V1_2::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_2::OperationType::BATCH_TO_SPACE_ND:
            return ConvertBatchToSpaceNd(operation, model, data);
        case V1_2::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_2::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_2::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case V1_2::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case V1_2::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case V1_2::OperationType::MAXIMUM:
            return ConvertMaximum(operation, model, data);
        case V1_2::OperationType::MINIMUM:
            return ConvertMinimum(operation, model, data);
        case V1_2::OperationType::PAD:
            return ConvertPad(operation, model, data);
        case V1_2::OperationType::PAD_V2:
            return ConvertPadV2(operation, model, data);
        case V1_2::OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case V1_2::OperationType::QUANTIZE:
            return ConvertQuantize(operation, model, data);
        case V1_2::OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case V1_2::OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case V1_2::OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case V1_2::OperationType::RESIZE_BILINEAR:
            return ConvertResize(operation, model, data, armnn::ResizeMethod::Bilinear);
        case V1_2::OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResize(operation, model, data, armnn::ResizeMethod::NearestNeighbor);
        case V1_2::OperationType::TRANSPOSE_CONV_2D:
            return ConvertTransposeConv2d(operation, model, data);
        case V1_2::OperationType::SOFTMAX:
            return ConvertSoftmax(operation, model, data);
        case V1_2::OperationType::SPACE_TO_BATCH_ND  :
            return ConvertSpaceToBatchNd(operation, model, data);
        case V1_2::OperationType::SPACE_TO_DEPTH:
            return ConvertSpaceToDepth(operation, model, data);
        case V1_2::OperationType::SUB:
            return ConvertSub(operation, model, data);
        case V1_2::OperationType::TANH:
            return ConvertTanH(operation, model, data);
        case V1_2::OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertAveragePool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertBatchToSpaceNd()");
    return ::ConvertBatchToSpaceNd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertConcatenation()");
    return ::ConvertConcatenation<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertConv2d()");

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

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::Convolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 7 ||
                           (operation.inputs.size() >= 8 &&
                            GetInputOperand<hal_1_2::HalPolicy>(operation, 7, model)->type == OperandType::BOOL);

    if (implicitPadding)
    {
        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 7, model, data);
    }
    else if (operation.inputs.size() >= 10)
    {
        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 10, model, data);
    }

    const armnn::PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // ArmNN does not currently support non-fixed weights or bias
    // The NNAPI filter is always OHWI [depth_out, filter_height, filter_width, depth_in] but ArmNN expects the
    // filter's height and width indices to match the input's height and width indices so we permute it to OIHW if
    // the DataLayout is NCHW
    const ConstTensorPin weightsPin = (desc.m_DataLayout == armnn::DataLayout::NCHW) ?
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data, OHWIToOIHW) :
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

    ActivationFn activation;

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

        armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
        unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
        unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();
        const uint32_t kernelX = weights.GetShape()[widthIndex];
        const uint32_t kernelY = weights.GetShape()[heightIndex];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);

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
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsConvolution2dSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               desc,
                               weights.GetInfo(),
                               biases);

    if (!isSupported)
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
    ALOGV("hal_1_2::HalPolicy::ConvertDepthwiseConv2d()");

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

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

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

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
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

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsDepthwiseConvolutionSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               desc,
                               weights.GetInfo(),
                               biases);

    if (!isSupported)
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

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertL2Pool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMaxPool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertMaximum(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMaximum()");

    LayerInputHandle input0 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsMaximumSupported,
                               data.m_Backends,
                               isSupported,
                               input0.GetTensorInfo(),
                               input1.GetTensorInfo(),
                               outInfo);

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddMaximumLayer();
    assert(layer != nullptr);
    BroadcastTensor(input0, input1, layer, *data.m_Network);

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertMinimum(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMinimum()");

    LayerInputHandle input0 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
         return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsMinimumSupported,
                               data.m_Backends,
                               isSupported,
                               input0.GetTensorInfo(),
                               input1.GetTensorInfo(),
                               outputInfo);

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddMinimumLayer();
    assert(layer != nullptr);
    BroadcastTensor(input0, input1, layer, *data.m_Network);

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertPad(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPad()");
    return ::ConvertPad<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPadV2()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    armnn::PadDescriptor descriptor;
    if (!ConvertPaddings<hal_1_2::HalPolicy>(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Determine type of padding value
    OperandType operandType0;
    OperandType operandType2;

    if (!GetOperandType<hal_1_2::HalPolicy>(operation, 0, model, operandType0) ||
        !GetOperandType<hal_1_2::HalPolicy>(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Read value to use for padding
    if (operandType0 == OperandType::TENSOR_FLOAT16 && operandType2 == OperandType::FLOAT16)
    {
        armnn::Half f16PadValue;
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 2, operandType2, f16PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT16)", __func__);
        }

        descriptor.m_PadValue = f16PadValue;
    }
    else if (operandType0 == OperandType::TENSOR_FLOAT32 && operandType2 == OperandType::FLOAT32)
    {
        if (!GetInputFloat32<hal_1_2::HalPolicy>(operation, 2, descriptor.m_PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT32)", __func__);
        }
    }
    else if (operandType0 == OperandType::TENSOR_QUANT8_ASYMM && operandType2 == OperandType::INT32)
    {
        int32_t intPadValue = 0;
        if (!GetInputInt32<hal_1_2::HalPolicy>(operation, 2, intPadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (INT32)", __func__);
        }
        descriptor.m_PadValue = intPadValue;
    }
    else
    {
        return Fail("%s: Operation has invalid inputs: type mismatch", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsPadSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               descriptor);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPrelu()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    LayerInputHandle alpha = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 1, model, data);

    if (!input.IsValid() || !alpha.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& alphaInfo  = alpha.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsPreluSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               alphaInfo,
                               outputInfo);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddPreluLayer();

    if (!layer)
    {
        return Fail("%s: AddPreluLayer failed", __func__);
    }

    BroadcastTensor(input, alpha, layer, *data.m_Network);

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertQuantize()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsQuantizeSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               outputInfo);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddQuantizeLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu()");
    return ::ConvertReLu<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu1()");
    return ::ConvertReLu1<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu6()");
    return ::ConvertReLu6<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              armnn::ResizeMethod resizeMethod)
{
    ALOGV("hal_1_2::HalPolicy::ConvertResize()");

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

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::ResizeDescriptor descriptor;
    descriptor.m_Method     = resizeMethod;
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
        descriptor.m_TargetHeight = static_cast<uint32_t>(targetHeight);
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

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsResizeSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               descriptor);

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddResizeLayer(descriptor);

    assert(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSpaceToBatchNd()");
    return ::ConvertSpaceToBatchNd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSpaceToDepth()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
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

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::SpaceToDepthDescriptor desc;

    GetInputScalar<hal_1_2::HalPolicy>(operation, 1, OperandType::INT32, desc.m_BlockSize, model, data);

    if (desc.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 2, model, data);

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSpaceToDepthSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               desc);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSoftmax()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::SoftmaxDescriptor desc;
    if (!GetInputFloat32<hal_1_2::HalPolicy>(operation, 1, desc.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (operation.inputs.size() > 2 && !GetInputScalar<hal_1_2::HalPolicy>(operation,
                                                                           2,
                                                                           HalPolicy::OperandType::INT32,
                                                                           desc.m_Axis,
                                                                           model,
                                                                           data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSoftmaxSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               outputInfo,
                               desc);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertSub(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSub()");
    return ::ConvertSub<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertTanH()");
    return ::ConvertTanH<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 2, model, data);
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 3, model, data);
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 4, model, data);
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 6, model, data);
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 7, model, data);
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 8, model, data);
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 15, model, data);

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
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
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
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      5,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      9,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      10,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      11,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      12,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      16,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
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
    if (!GetInputActivationFunctionFromTensor<hal_1_2::HalPolicy>(operation, 20, activation, model, data) ||
        !GetInputScalar<hal_1_2::HalPolicy>(operation, 21, OperandType::FLOAT32, cellClip, model, data) ||
        !GetInputScalar<hal_1_2::HalPolicy>(operation, 22, OperandType::FLOAT32, projClip, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Get the normalization tensors
    // 23: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      23,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 24: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      24,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 25: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      25,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // 26: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation,
                                                                      26,
                                                                      model,
                                                                      data,
                                                                      g_DontPermute,
                                                                      nullptr,
                                                                      true);

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4]
    // with CIFG, or [batch_size, num_units * 3] without CIFG.
    const Operand* scratchBuffer = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const Operand* outputStateOut = GetOutputOperand<hal_1_2::HalPolicy>(operation, 1, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand<hal_1_2::HalPolicy>(operation, 2, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 3, model);
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
    params.m_InputLayerNormWeights = inputLayerNormWeightsPin.GetConstTensorPtr();
    params.m_ForgetLayerNormWeights = forgetLayerNormWeightsPin.GetConstTensorPtr();
    params.m_CellLayerNormWeights = cellLayerNormWeightsPin.GetConstTensorPtr();
    params.m_OutputLayerNormWeights = outputLayerNormWeightsPin.GetConstTensorPtr();

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
    desc.m_LayerNormEnabled = (params.m_InputLayerNormWeights != nullptr ||
                               params.m_ForgetLayerNormWeights != nullptr ||
                               params.m_CellLayerNormWeights != nullptr ||
                               params.m_OutputLayerNormWeights != nullptr);

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

    if (desc.m_LayerNormEnabled &&
        (params.m_ForgetLayerNormWeights == nullptr ||
         params.m_CellLayerNormWeights == nullptr ||
         params.m_OutputLayerNormWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_InputLayerNormWeights == nullptr)))
    {
        return Fail("%s: All, or none, of forget-norm weights, cell-norm weights and output-norm weights must be"
                    " provided and, if CIFG is not enabled, input-norm weights must also be provided", __func__);
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

    if (IsDynamicTensor(scratchBufferInfo)  ||
        IsDynamicTensor(outputStateOutInfo) ||
        IsDynamicTensor(cellStateOutInfo)   ||
        IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

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

    // Optional parameters
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

    if (desc.m_LayerNormEnabled)
    {
        if(!desc.m_CifgEnabled)
        {
            paramsInfo.m_InputLayerNormWeights = &(params.m_InputLayerNormWeights->GetInfo());
        }
        paramsInfo.m_ForgetLayerNormWeights = &(params.m_ForgetLayerNormWeights->GetInfo());
        paramsInfo.m_CellLayerNormWeights = &(params.m_CellLayerNormWeights->GetInfo());
        paramsInfo.m_OutputLayerNormWeights = &(params.m_OutputLayerNormWeights->GetInfo());
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsLstmSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputStateInInfo,
                               cellStateInInfo,
                               scratchBufferInfo,
                               outputStateOutInfo,
                               cellStateOutInfo,
                               outputInfo,
                               desc,
                               paramsInfo);
    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    armnn::IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, 0, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 1, *layer, 1, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 2, *layer, 2, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 3, *layer, 3, model, data));
}

bool HalPolicy::ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data)
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
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const Operand* weightsOperand = GetInputOperand<hal_1_2::HalPolicy>(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }
    armnn::TransposeConvolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 9;

    if (implicitPadding )
    {
        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 8, model, data);
    }
    else
    {
        desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 10, model, data);
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    const armnn::PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // The shape of the weight is [depth_out, filter_height, filter_width, depth_in].
    // We have to permute it to OIHW if the data layout is NCHW.
    const ConstTensorPin weightsPin = (desc.m_DataLayout == armnn::DataLayout::NCHW) ?
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data, OHWIToOIHW) :
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data);

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
        if (!GetInputPaddingScheme<hal_1_2::HalPolicy>(operation, 4, paddingScheme, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[widthIndex];
        const uint32_t kernelY = weights.GetShape()[heightIndex];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else if (operation.inputs.size() == 11)
    {
        // explicit padding
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation,  9, activation, model, data))
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

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsTransposeConvolution2dSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               desc,
                               weights.GetInfo(),
                               biases);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
        data.m_Network->AddTransposeConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));
    if (!startLayer)
    {
        return Fail("%s: AddTransposeConvolution2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

} // namespace hal_1_2
} // namespace armnn_driver
