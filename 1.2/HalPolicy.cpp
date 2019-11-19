//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "Utils.hpp"

#include <DataLayoutIndexed.hpp>
#include <Half.hpp>
#include <TensorUtils.hpp>

#include <armnn/TypesUtils.hpp>

#include <cmath>
#include <string>

namespace armnn_driver
{
namespace hal_1_2
{

using namespace armnn;

namespace
{

bool IsQSymmDequantizeForWeights(const Operation& operation, const Model& model)
{
    const Operand* operand = GetInputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!operand)
    {
        return false;
    }

    if(!IsQSymm8(*operand))
    {
        // Only QSymm8 weights are dequantized on the fly by the driver
        return false;
    }

    if (!IsOperandConstant<hal_1_2::HalPolicy>(*operand))
    {
        // Non-const input is not accepted for weights
        return false;
    }

    // Iterate through all the operations and find the operation feeding from the Dequantize output
    const size_t outputIndex = operation.outputs[0];
    for (uint32_t operationIdx = 0; operationIdx < model.operations.size(); ++operationIdx)
    {
        const auto& operationIt = model.operations[operationIdx];
        switch (operationIt.type)
        {
            case HalPolicy::OperationType::FULLY_CONNECTED:
                if (outputIndex == operationIt.inputs[1]) // Weights are bound to slot 1
                {
                    // If the output is going into the FC weights return true
                    return true;
                }
                break;
            case HalPolicy::OperationType::LSTM:
                for (size_t k = 0; k < operationIt.inputs.size(); ++k)
                {
                    if (outputIndex == operationIt.inputs[k])
                    {
                        // If the output is going into the LSTM weights return true
                        return true;
                    }
                }
                break;
            default:
                break;
        }
    }

    return false;
}

} // anonymous namespace

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case V1_2::OperationType::ABS:
            return ConvertAbs(operation, model, data);
        case V1_2::OperationType::ADD:
            return ConvertAdd(operation, model, data);
        case V1_2::OperationType::ARGMAX:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Max);
        case V1_2::OperationType::ARGMIN:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Min);
        case V1_2::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_2::OperationType::BATCH_TO_SPACE_ND:
            return ConvertBatchToSpaceNd(operation, model, data);
        case V1_2::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_2::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_2::OperationType::DEPTH_TO_SPACE:
            return ConvertDepthToSpace(operation, model, data);
        case V1_2::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case V1_2::OperationType::DEQUANTIZE:
            return ConvertDequantize(operation, model, data);
        case V1_2::OperationType::DIV:
            return ConvertDiv(operation, model, data);
        case V1_2::OperationType::EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::Equal);
        case V1_2::OperationType::EXPAND_DIMS:
            return ConvertExpandDims(operation, model, data);
        case V1_2::OperationType::FLOOR:
            return ConvertFloor(operation, model, data);
        case V1_2::OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation, model, data);
        case V1_2::OperationType::GREATER:
            return ConvertComparison(operation, model, data, ComparisonOperation::Greater);
        case V1_2::OperationType::GREATER_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::GreaterOrEqual);
        case V1_2::OperationType::GROUPED_CONV_2D:
            return ConvertGroupedConv2d(operation, model, data);
        case V1_2::OperationType::INSTANCE_NORMALIZATION:
            return ConvertInstanceNormalization(operation, model, data);
        case V1_2::OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation, model, data);
        case V1_2::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case V1_2::OperationType::LESS:
            return ConvertComparison(operation, model, data, ComparisonOperation::Less);
        case V1_2::OperationType::LESS_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::LessOrEqual);
        case V1_2::OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation, model, data);
        case V1_2::OperationType::LOGISTIC:
            return ConvertLogistic(operation, model, data);
        case V1_2::OperationType::LOG_SOFTMAX:
            return ConvertLogSoftmax(operation, model, data);
        case V1_2::OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        case V1_2::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case V1_2::OperationType::MAXIMUM:
            return ConvertMaximum(operation, model, data);
        case V1_2::OperationType::MEAN:
            return ConvertMean(operation, model, data);
        case V1_2::OperationType::MINIMUM:
            return ConvertMinimum(operation, model, data);
        case V1_2::OperationType::MUL:
            return ConvertMul(operation, model, data);
        case V1_2::OperationType::NOT_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::NotEqual);
        case V1_2::OperationType::PAD:
            return ConvertPad(operation, model, data);
        case V1_2::OperationType::PAD_V2:
            return ConvertPadV2(operation, model, data);
        case V1_2::OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case V1_2::OperationType::QUANTIZE:
            return ConvertQuantize(operation, model, data);
        case V1_2::OperationType::QUANTIZED_16BIT_LSTM:
            return ConvertQuantizedLstm(operation, model, data);
        case V1_2::OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case V1_2::OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case V1_2::OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case V1_2::OperationType::RESHAPE:
            return ConvertReshape(operation, model, data);
        case V1_2::OperationType::RESIZE_BILINEAR:
            return ConvertResize(operation, model, data, ResizeMethod::Bilinear);
        case V1_2::OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResize(operation, model, data, ResizeMethod::NearestNeighbor);
        case V1_2::OperationType::RSQRT:
            return ConvertRsqrt(operation, model, data);
        case V1_2::OperationType::SQRT:
            return ConvertSqrt(operation, model, data);
        case V1_2::OperationType::SQUEEZE:
            return ConvertSqueeze(operation, model, data);
        case V1_2::OperationType::STRIDED_SLICE:
            return ConvertStridedSlice(operation, model, data);
        case V1_2::OperationType::TRANSPOSE:
            return ConvertTranspose(operation, model, data);
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
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ConvertAbs(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertAbs()");
    return ::ConvertAbs<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertAdd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertAdd()");
    return ::ConvertAdd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertArgMinMax(const V1_2::Operation& operation,
                                 const V1_2::Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction)
{
    ALOGV("hal_1_2::HalPolicy::ConvertArgMinMax()");
    return ::ConvertArgMinMax<hal_1_2::HalPolicy>(operation, model, data, argMinMaxFunction);
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertAveragePool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertBatchToSpaceNd()");
    return ::ConvertBatchToSpaceNd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  ComparisonOperation comparisonOperation)
{
    ALOGV("hal_1_2::HalPolicy::ConvertComparison()");
    ALOGV("comparisonOperation = %s", GetComparisonOperationAsCString(comparisonOperation));

    LayerInputHandle input0 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 1, model, data);

    if (!(input0.IsValid() && input1.IsValid()))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const TensorInfo& inputInfo1 = input1.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    ComparisonDescriptor descriptor(comparisonOperation);

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsComparisonSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo0,
                               inputInfo1,
                               outputInfo,
                               descriptor);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddComparisonLayer(descriptor);
    assert(layer != nullptr);

    input0.Connect(layer->GetInputSlot(0));
    input1.Connect(layer->GetInputSlot(1));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    Convolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

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

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // ArmNN does not currently support non-fixed weights or bias
    // The NNAPI filter is always OHWI [depth_out, filter_height, filter_width, depth_in] but ArmNN expects the
    // filter's height and width indices to match the input's height and width indices so we permute it to OIHW if
    // the DataLayout is NCHW
    const ConstTensorPin weightsPin = (desc.m_DataLayout == DataLayout::NCHW) ?
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

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor bias = biasPin.GetConstTensor();
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
    Optional<TensorInfo> biases(bias.GetInfo());

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

    IConnectableLayer* startLayer =
            data.m_Network->AddConvolution2dLayer(desc, weights, Optional<ConstTensor>(bias));

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);

    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

bool HalPolicy::ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDepthToSpace()");
    return ::ConvertDepthToSpace<hal_1_2::HalPolicy>(operation, model, data);
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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

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
    if ( weightsOperand->dimensions[0] != 1)
    {
        return Fail("%s: Invalid weights; for depthwise convolution, dimension 0 must be 1 but it is %i",
                    __func__, weightsOperand->dimensions[0] );
    }

    DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

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
    TensorShape weightsShape({ weightsOperand->dimensions[1],
                                      weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[channelsIndex],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[channelsIndex] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };

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

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor bias = biasPin.GetConstTensor();
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
    Optional<TensorInfo> biases(bias.GetInfo());

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

    IConnectableLayer* startLayer =
        data.m_Network->AddDepthwiseConvolution2dLayer(desc, weights, Optional<ConstTensor>(bias));

    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

bool HalPolicy::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDequantize()");

    if (IsQSymmDequantizeForWeights(operation, model))
    {
        // NOTE: QSymm8 weights are dequantized internally by the driver,
        // therefore this type of Dequantize is implicitly supported
        return true;
    }

    return ::ConvertDequantize<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDiv(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDiv()");
    return ::ConvertDiv<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertExpandDims()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    int32_t axis;
    if (!GetInputScalar<HalPolicy>(operation, 1, OperandType::INT32, axis, model, data))
    {
        return Fail("%s: failed to get axis input value", __func__);
    }

    TensorShape targetShape;

    try
    {
        targetShape = armnnUtils::ExpandDims(input.GetTensorInfo().GetShape(), axis);
    }
    catch (const std::exception &e)
    {
        return Fail("%s: %s", __func__, e.what());
    }

    if (targetShape != outputInfo.GetShape())
    {
        return Fail("%s: Shape of the output operand does not match the resolved expanded shape", __func__);
    }

    ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = targetShape;

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               reshapeDescriptor);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertFloor()");
    return ::ConvertFloor<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertFullyConnected()");
    return ::ConvertFullyConnected<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertGroupedConv2d()");

    //
    // Parse data
    //
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    const TensorInfo& inputInfo  = input.GetTensorInfo();

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Look ahead to determine data layout
    DataLayout dataLayout = DataLayout::NHWC;
    if (operation.inputs.size() == 12)
    {
        dataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 11, model, data);
    }
    else
    {
        dataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 8, model, data);
    }

    // NOTE:
    // NNAPI weights are always OHWI, i.e. [depth_out, filter_height, filter_width, depth_group],
    // but Arm NN expects the filter's height and width indices to match the input's height and
    // width indices so when the DataLayout is NCHW, we need to permute the weights to OIHW
    const PermutationVector ohwiToOihw = { 0u, 2u, 3u, 1u };
    const ConstTensorPin weightsPin = (dataLayout == DataLayout::NCHW) ?
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data, ohwiToOihw) :
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data);
    const ConstTensorPin biasesPin  =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 2, model, data);
    if (!weightsPin.IsValid() || !biasesPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor biases  = biasesPin.GetConstTensor();
    SanitizeBiasQuantizationScale(biases.GetInfo(), weights.GetInfo(), inputInfo);

    const TensorShape& inputShape   = inputInfo.GetShape();
    const TensorShape& outputShape  = outputInfo.GetShape();
    const TensorShape& weightsShape = weights.GetShape();
    const TensorShape& biasesShape  = biases.GetShape();

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    Convolution2dDescriptor desc;
    desc.m_DataLayout  = dataLayout;
    desc.m_BiasEnabled = true;

    int numGroups;
    ActivationFn activation;

    if (operation.inputs.size() == 12)
    {
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 9, OperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 10, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }

    }
    else if (operation.inputs.size() == 9)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<hal_1_2::HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 4, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t inputX = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY = inputInfo.GetShape()[heightIndex];

        const uint32_t kernelX = weightsShape[widthIndex];
        const uint32_t kernelY = weightsShape[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    const unsigned int outputChannels = outputShape[channelsIndex];

    const unsigned int channelsPerGroup  = weightsShape[channelsIndex];
    const unsigned int channelMultiplier = outputChannels / numGroups;

    //
    // Validate all relevant inputs
    //
    if (numGroups <= 0)
    {
        return Fail("%s: Number of groups must be greater than 0. Got: %d", __func__, numGroups);
    }

    if (outputChannels % numGroups != 0u)
    {
        return Fail("%s: Output channels must be divisible by the number of groups", __func__);
    }

    //
    // Set up Splitter layer
    //
    unsigned int splitterDimSizes[4] = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    splitterDimSizes[channelsIndex] /= numGroups; // split in depth

    TensorInfo splitterOutputInfo(4,
                                  splitterDimSizes,
                                  inputInfo.GetDataType(),
                                  inputInfo.GetQuantizationScale(),
                                  inputInfo.GetQuantizationOffset());

    std::vector<std::reference_wrapper<TensorInfo>> splitterOutputInfos(numGroups, std::ref(splitterOutputInfo));

    ViewsDescriptor splitterDesc(numGroups);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        splitterDesc.SetViewOriginCoord(group, channelsIndex, splitterDimSizes[channelsIndex] * group);
        for (unsigned int dimIdx = 0u; dimIdx < 4u; dimIdx++)
        {
            splitterDesc.SetViewSize(group, dimIdx, splitterDimSizes[dimIdx]);
        }
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSplitterSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               splitterOutputInfos,
                               splitterDesc);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* splitterLayer = data.m_Network->AddSplitterLayer(splitterDesc);
    if (!splitterLayer)
    {
        return Fail("%s: Failed to add SplitterLayer", __func__);
    }

    input.Connect(splitterLayer->GetInputSlot(0));
    for (unsigned int group = 0u; group < splitterLayer->GetNumOutputSlots(); ++group)
    {
        splitterLayer->GetOutputSlot(group).SetTensorInfo(splitterOutputInfo);
    }

    //
    // Set up Convolution2d layers for each group
    //

    // Set up group tensor shapes
    TensorShape groupInputShape(inputShape);
    groupInputShape[channelsIndex] = channelsPerGroup;

    TensorShape groupOutputShape(outputShape);
    groupOutputShape[channelsIndex] = 1;

    TensorShape groupWeightsShape(weightsShape);
    groupWeightsShape[0] /= channelMultiplier * numGroups;

    TensorShape groupBiasesShape({ 1 });

    // Set up group tensor infos
    TensorInfo groupInputInfo(inputInfo);
    groupInputInfo.SetShape(groupInputShape);

    const TensorInfo& weightsInfo = weights.GetInfo();
    TensorInfo groupWeightsInfo(weightsInfo);
    groupWeightsInfo.SetShape(groupWeightsShape);

    const TensorInfo& biasesInfo = biases.GetInfo();
    TensorInfo groupBiasesInfo(biasesInfo);
    groupBiasesInfo.SetShape(groupBiasesShape);

    TensorInfo groupOutputInfo(outputInfo);
    groupOutputInfo.SetShape(groupOutputShape);

    const unsigned int weightsDataTypeSize = GetDataTypeSize(groupWeightsInfo.GetDataType());
    const unsigned int biasesDataTypeSize  = GetDataTypeSize(groupBiasesInfo.GetDataType());

    std::vector<IConnectableLayer*> convLayers(numGroups * channelMultiplier, nullptr);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;

            const unsigned int weightsDataOffset = groupWeightsShape.GetNumElements() * index * weightsDataTypeSize;
            const unsigned int biasesDataOffset = groupBiasesShape.GetNumElements() * index * biasesDataTypeSize;

            if (weightsInfo.HasPerAxisQuantization())
            {
                // Extract per-axis quantization scales for group weights
                const std::vector<float>& weightsQuantScales = weightsInfo.GetQuantizationScales();
                groupWeightsInfo.SetQuantizationScales(
                    std::vector<float>(weightsQuantScales.begin() + index,
                                       weightsQuantScales.begin() + index + groupWeightsShape[0]));

                // Extract per-axis quantization scales for group biases
                const std::vector<float>& biasesQuantScales  = biasesInfo.GetQuantizationScales();
                groupBiasesInfo.SetQuantizationScales(
                    std::vector<float>(biasesQuantScales.begin() + index,
                                       biasesQuantScales.begin() + index + groupWeightsShape[0]));
            }

            // Extract weights and biases data for current group convolution
            ConstTensor groupWeights(groupWeightsInfo,
                                     static_cast<const void *>(reinterpret_cast<const char *>(weights.GetMemoryArea()) +
                                                               weightsDataOffset));
            ConstTensor groupBiases(groupBiasesInfo,
                                    static_cast<const void *>(reinterpret_cast<const char *>(biases.GetMemoryArea()) +
                                                              biasesDataOffset));

            isSupported = false;
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsConvolution2dSupported,
                                       data.m_Backends,
                                       isSupported,
                                       groupInputInfo,
                                       groupOutputInfo,
                                       desc,
                                       groupWeightsInfo,
                                       Optional<TensorInfo>(groupBiasesInfo));
            if (!isSupported)
            {
                return false;
            }

            IConnectableLayer *convLayer =
                    data.m_Network->AddConvolution2dLayer(desc, groupWeights, Optional<ConstTensor>(groupBiases));
            if (!convLayer)
            {
                return Fail("%s: AddConvolution2dLayer failed", __func__);
            }

            splitterLayer->GetOutputSlot(group).Connect(convLayer->GetInputSlot(0));
            convLayer->GetOutputSlot(0).SetTensorInfo(groupOutputInfo);

            convLayers[index] = convLayer;
        }
    }

    //
    // Set up Concat layer
    //
    ConcatDescriptor concatDescriptor(outputInfo.GetShape()[channelsIndex]);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;
            concatDescriptor.SetViewOriginCoord(index, channelsIndex, index);
            concatDescriptor.SetConcatAxis(channelsIndex);
        }
    }

    isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsConcatSupported,
                               data.m_Backends,
                               isSupported,
                               std::vector<const TensorInfo*>(numGroups * channelMultiplier, &groupOutputInfo),
                               outputInfo,
                               concatDescriptor);
    if (!isSupported)
    {
       return false;
    }

    IConnectableLayer* concatLayer = data.m_Network->AddConcatLayer(concatDescriptor);
    if (!concatLayer)
    {
        return Fail("%s: AddConcatLayer failed", __func__);
    }

    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;
            convLayers[index]->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(index));
        }
    }
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    //
    // Set up Activation layer (if it is set)
    //
    IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, concatLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *endLayer, model, data);
}

bool HalPolicy::ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertInstanceNormalization()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has an invalid input 0", __func__);
    }

    const Operand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has an invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Determine data type of input tensor
    OperandType inputType;
    if (!GetOperandType<hal_1_2::HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    InstanceNormalizationDescriptor desc;

    // Read gamma, beta & epsilon
    if (inputType == OperandType::TENSOR_FLOAT16)
    {
        Half fp16Gamma;
        Half fp16Beta;
        Half fp16Epsilon;

        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 1, OperandType::FLOAT16, fp16Gamma, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 2, OperandType::FLOAT16, fp16Beta, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::FLOAT16, fp16Epsilon, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_Gamma = static_cast<float>(fp16Gamma);
        desc.m_Beta  = static_cast<float>(fp16Beta);
        desc.m_Eps   = static_cast<float>(fp16Epsilon);
    }
    else if (inputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 1, OperandType::FLOAT32, desc.m_Gamma, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 2, OperandType::FLOAT32, desc.m_Beta, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 3, OperandType::FLOAT32, desc.m_Eps, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    desc.m_DataLayout = OptionalDataLayout<hal_1_2::HalPolicy>(operation, 4, model, data);

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsInstanceNormalizationSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               outputInfo,
                               desc);
    if (!isSupported)
    {
       return false;
    }

    IConnectableLayer* layer = data.m_Network->AddInstanceNormalizationLayer(desc);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertL2Normalization()");
    return ::ConvertL2Normalization<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertL2Pool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLocalResponseNormalization()");
    return ::ConvertLocalResponseNormalization<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLogistic()");
    return ::ConvertLogistic<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLogSoftmax()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Failed to read input 0", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Failed to read output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Determine data type of input tensor
    OperandType inputType;
    if (!GetOperandType<hal_1_2::HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LogSoftmaxDescriptor descriptor;

    // Read beta
    if (inputType == OperandType::TENSOR_FLOAT16)
    {
        Half fp16Beta;
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 1, OperandType::FLOAT16, fp16Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT16)", __func__);
        }

        descriptor.m_Beta  = static_cast<float>(fp16Beta);
    }
    else if (inputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<hal_1_2::HalPolicy>(operation, 1, OperandType::FLOAT32, descriptor.m_Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    // Read axis
    if (!GetInputInt32<hal_1_2::HalPolicy>(operation, 2, descriptor.m_Axis, model, data))
    {
        return Fail("%s: Failed to read input 2", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsLogSoftmaxSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               outputInfo,
                               descriptor);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddLogSoftmaxLayer(descriptor);
    if (!layer)
    {
        return Fail("%s: AddLogSoftmaxLayer() returned nullptr", __func__);
    }

    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMaxPool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::Max, model, data);
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

    const TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);
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

    IConnectableLayer* layer = data.m_Network->AddMaximumLayer();
    assert(layer != nullptr);
    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMean()");
    return ::ConvertMean<hal_1_2::HalPolicy>(operation, model, data);
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

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
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

    IConnectableLayer* const layer = data.m_Network->AddMinimumLayer();
    assert(layer != nullptr);
    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertMul(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMul()");
    return ::ConvertMul<hal_1_2::HalPolicy>(operation, model, data);
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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    PadDescriptor descriptor;
    if (!ConvertPaddings<hal_1_2::HalPolicy>(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
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
        Half f16PadValue;
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

    IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& alphaInfo  = alpha.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

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

    IConnectableLayer* const layer = data.m_Network->AddPreluLayer();

    if (!layer)
    {
        return Fail("%s: AddPreluLayer failed", __func__);
    }

    bool isReshapeSupported = BroadcastTensor(input, alpha, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

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

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);
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

    IConnectableLayer* const layer = data.m_Network->AddQuantizeLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertQuantizedLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertQuantizedLstm()");

    //Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }

    //13: The previous cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape
    //    [numBatches, outputSize] specifying the cell state from the previous time step of the LSTM cell.
    //    It is quantized using a quantization range of -2^4, 2^4 * 32767/32768.
    LayerInputHandle previousCellStateIn = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 13, model, data);
    if (!previousCellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 13: previousCellStateIn", __func__);
    }

    // 14: The previous output state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //     [numBathes, outputSize] specifying the output of the LSTM cell from previous time-step. Tensor
    //     is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle previousOutputIn = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 14, model, data);
    if (!previousOutputIn.IsValid())
    {
        return Fail("%s: Could not read input 14: previousOutputIn", __func__);
    }

    // Get the input tensors:
    // 1: The input-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-input part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToInputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 1, model, data);

    // 2: The input-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-forget part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 2, model, data);

    // 3: The input-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-cell part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToCellWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 3, model, data);

    // 4: The input-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-output part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 4, model, data);

    // 5: The recurrent-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-input part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToInputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 5, model, data);

    // 6: The recurrent-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-forget part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 6, model, data);

    // 7: The recurrent-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-cell part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToCellWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 7, model, data);

    // 8: The recurrent-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-output part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 8, model, data);

    // 9: The input gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the
    //    bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin inputGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 9, model, data);

    // 10: The forget gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin forgetGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 10, model, data);

    // 11:The cell bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the bias
    //    for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product of input
    //    and weights scales and zeroPoint equal to 0.
    const ConstTensorPin cellBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 11, model, data);

    // 12:The output gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //    the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin outputGateBiasPin =
            ConvertOperationInputToConstTensorPin<hal_1_2::HalPolicy>(operation, 12, model, data);

    if (!inputToInputWeightsPin.IsValid() ||
        !inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToInputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !inputGateBiasPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Outputs:
    // 0: The cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape [numBatches, outputSize]
    //    which contains a cell state from the current time step. Tensor is quantized using a quantization range
    //    of -2^4, 2^4 * 32767/32768.
    const Operand* cellStateOut = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 0: cellStateOut", __func__);
    }

    // 1: The output: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBathes, outputSize] which
    //      contains the output value. Tensor is quantized with a fixed quantization range of -1, 127/128.
    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 1, model);
    if (!output)
    {
        return Fail("%s: Could not read output 1: output", __func__);
    }

    // Inputs
    const TensorInfo& inputInfo               = input.GetTensorInfo();
    const TensorInfo& previousCellStateInInfo = previousCellStateIn.GetTensorInfo();
    const TensorInfo& previousOutputInInfo    = previousOutputIn.GetTensorInfo();

    // Outputs
    const TensorInfo& cellStateOutInfo = GetTensorInfoForOperand(*cellStateOut);
    const TensorInfo& outputInfo       = GetTensorInfoForOperand(*output);

    // Dynamic tensors currently not supported
    if (IsDynamicTensor(cellStateOutInfo) || IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    QuantizedLstmInputParams params;

    params.m_InputToInputWeights      = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights     = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights       = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights     = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights  = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights   = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias            = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias           = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias                 = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias           = outputGateBiasPin.GetConstTensorPtr();

    QuantizedLstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToInputWeights      = &(params.m_InputToInputWeights->GetInfo());
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToInputWeights  = &(params.m_RecurrentToInputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_InputGateBias            = &(params.m_InputGateBias->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsQuantizedLstmSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               previousCellStateInInfo,
                               previousOutputInInfo,
                               cellStateOutInfo,
                               outputInfo,
                               paramsInfo);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddQuantizedLstmLayer(params, "QuantizedLstm");
    input.Connect(layer->GetInputSlot(0));
    previousCellStateIn.Connect(layer->GetInputSlot(1));
    previousOutputIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, 0, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 1, *layer, 1, model, data));
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

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReshape()");
    return ::ConvertReshape<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ResizeMethod resizeMethod)
{
    ALOGV("hal_1_2::HalPolicy::ConvertResize()");
    ALOGV("resizeMethod = %s", GetResizeMethodAsCString(resizeMethod));

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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    ResizeDescriptor descriptor;
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

        const TensorShape& inputShape = inputInfo.GetShape();
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

    IConnectableLayer* layer = data.m_Network->AddResizeLayer(descriptor);

    assert(layer != nullptr);

    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertRsqrt(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertRsqrt()");

    LayerInputHandle input = ConvertToLayerInputHandle<hal_1_2::HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* output = GetOutputOperand<hal_1_2::HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsRsqrtSupported,
                               data.m_Backends,
                               isSupported,
                               input.GetTensorInfo(),
                               outputInfo);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddRsqrtLayer();
    assert(layer != nullptr);
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

    const TensorInfo& inputInfo = input.GetTensorInfo();
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

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    SpaceToDepthDescriptor desc;

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

    IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
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

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    SoftmaxDescriptor desc;
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

    if (input.GetTensorInfo().GetNumDimensions() > 2 ||
        !(desc.m_Axis == 1 ||
          (desc.m_Axis < 0 && static_cast<int>(input.GetTensorInfo().GetNumDimensions()) + desc.m_Axis == 1)))
    {
        return Fail("%s: Unsupported input greater than 2D or axis != 1", __func__);
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

    IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
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
    ALOGV("hal_1_2::HalPolicy::ConvertLstm()");

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
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 2));
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 3));
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
           (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 4));
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
           (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 6));
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
           (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 7));
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 8));
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
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 1, true));
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 5, true));
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 9, true));
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 10, true));
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 11, true));
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
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 16, true));
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
    const ConstTensorPin inputLayerNormWeightsPin
            (DequantizeAndMakeConstTensorPin<hal_1_2::HalPolicy>(operation, model, data, 23, true));

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
    LstmInputParams params;
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
    LstmDescriptor desc;
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
    const TensorInfo& inputInfo         = input.GetTensorInfo();
    const TensorInfo& outputStateInInfo = outputStateIn.GetTensorInfo();
    const TensorInfo& cellStateInInfo   = cellStateIn.GetTensorInfo();

    // Outputs
    const TensorInfo& scratchBufferInfo  = GetTensorInfoForOperand(*scratchBuffer);
    const TensorInfo& outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    const TensorInfo& cellStateOutInfo   = GetTensorInfoForOperand(*cellStateOut);
    const TensorInfo& outputInfo         = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(scratchBufferInfo)  ||
        IsDynamicTensor(outputStateOutInfo) ||
        IsDynamicTensor(cellStateOutInfo)   ||
        IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported %d %d %d %d", __func__,
                    IsDynamicTensor(scratchBufferInfo), IsDynamicTensor(outputStateOutInfo),
                    IsDynamicTensor(cellStateOutInfo), IsDynamicTensor(outputInfo));
    }

    // Basic parameters
    LstmInputParamsInfo paramsInfo;
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
    IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *layer, 0, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 1, *layer, 1, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 2, *layer, 2, model, data) &&
            SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 3, *layer, 3, model, data));
}

bool HalPolicy::ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSqrt()");
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Sqrt;

    return ::ConvertToActivation<hal_1_2::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSqueeze()");
    return ::ConvertSqueeze<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertStridedSlice()");
    return ::ConvertStridedSlice<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertTranspose()");
    return ::ConvertTranspose<hal_1_2::HalPolicy>(operation, model, data);
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

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
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
    TransposeConvolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

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

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // The shape of the weight is [depth_out, filter_height, filter_width, depth_in].
    // We have to permute it to OIHW if the data layout is NCHW.
    const ConstTensorPin weightsPin = (desc.m_DataLayout == DataLayout::NCHW) ?
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

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    ActivationFn activation;

    if (implicitPadding)
    {
        int32_t strideX{0};
        int32_t strideY{0};
        int32_t padLeft{0};
        int32_t padRight{0};
        int32_t padTop{0};
        int32_t padBottom{0};

        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<hal_1_2::HalPolicy>(operation, 4, paddingScheme, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 5, OperandType::INT32, strideX, model, data) ||
            !GetInputScalar<hal_1_2::HalPolicy>(operation, 6, OperandType::INT32, strideY, model, data) ||
            !GetInputActivationFunction<hal_1_2::HalPolicy>(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[widthIndex];
        const uint32_t kernelY = weights.GetShape()[heightIndex];
        const uint32_t outputX  = outputInfo.GetShape()[widthIndex];
        const uint32_t outputY  = outputInfo.GetShape()[heightIndex];

        CalcPaddingTransposeConv(outputX, kernelX, desc.m_StrideX, padLeft, padRight, paddingScheme);
        CalcPaddingTransposeConv(outputY, kernelY, desc.m_StrideY, padTop, padBottom, paddingScheme);

        // NOTE: The Android NN API allows for negative padding values in TransposeConv2d,
        // but Arm NN only supports values >= 0
        if (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0)
        {
            return Fail("%s: Negative padding values are not supported", __func__);
        }

        desc.m_StrideX   = boost::numeric_cast<uint32_t>(strideX);
        desc.m_StrideY   = boost::numeric_cast<uint32_t>(strideY);
        desc.m_PadLeft   = boost::numeric_cast<uint32_t>(padLeft);
        desc.m_PadRight  = boost::numeric_cast<uint32_t>(padRight);
        desc.m_PadTop    = boost::numeric_cast<uint32_t>(padTop);
        desc.m_PadBottom = boost::numeric_cast<uint32_t>(padBottom);
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
    Optional<TensorInfo> biases(bias.GetInfo());

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

    IConnectableLayer* startLayer =
        data.m_Network->AddTransposeConvolution2dLayer(desc, weights, Optional<ConstTensor>(bias));
    if (!startLayer)
    {
        return Fail("%s: AddTransposeConvolution2dLayer failed", __func__);
    }

    IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_2::HalPolicy>(operation, 0, *endLayer, model, data);
}

} // namespace hal_1_2
} // namespace armnn_driver
