//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Utils.hpp"

#include "ConversionUtils.hpp"

#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <half/half.hpp>

using Half = half_float::half;

namespace armnn_driver
{

using namespace armnn;
using namespace android::nn;

template<typename HalPolicy,
        typename HalOperation = typename HalPolicy::Operation,
        typename HalModel     = typename HalPolicy::Model>
bool IsWeightsValid(const HalOperation& operation,
                    uint32_t inputIndex,
                    const HalModel& model)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;
    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand %i", __func__, inputIndex);
        return false;
    }

    if (operand->lifetime    != HalOperandLifeTime::CONSTANT_COPY
        && operand->lifetime != HalOperandLifeTime::CONSTANT_REFERENCE
        && operand->lifetime != HalOperandLifeTime::NO_VALUE)
    {
        return false;
    }
    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool IsQSymmDequantizeForWeights(const HalOperation& operation, const HalModel& model)
{
    using HalOperand       = typename HalPolicy::Operand;
    using HalOperationType = typename HalPolicy::OperationType;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, 0, model);
    if (!operand)
    {
        return false;
    }

    if(!IsQSymm8(*operand))
    {
        // Only QSymm8 weights are dequantized on the fly by the driver
        return false;
    }

    if (!IsOperandConstant<HalPolicy>(*operand))
    {
        // Non-const input is not accepted for weights
        return false;
    }

    // Iterate through all the operations and find the operation feeding from the Dequantize output
    const size_t outputIndex = operation.outputs[0];
    for (uint32_t operationIdx = 0; operationIdx < getMainModel(model).operations.size(); ++operationIdx)
    {
        const auto& operationIt = getMainModel(model).operations[operationIdx];
        switch (operationIt.type)
        {
            case HalOperationType::FULLY_CONNECTED:
                if (outputIndex == operationIt.inputs[1]) // Weights are bound to slot 1
                {
                    // If the output is going into the FC weights return true
                    return true;
                }
                break;
            case HalOperationType::LSTM:
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

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool SetupAndTrackLayerOutputSlotAndOverrideTensorInfo(const HalOperation& operation,
                                                       uint32_t operationOutputIndex,
                                                       armnn::IConnectableLayer& layer,
                                                       uint32_t layerOutputIndex,
                                                       const HalModel& model,
                                                       ConversionData& data,
                                                       const armnn::TensorInfo tensor_info)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, operationOutputIndex, model);
    if ((outputOperand == nullptr) || (operationOutputIndex >= layer.GetNumOutputSlots()))
    {
        return false;
    }

    armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(layerOutputIndex);

    const uint32_t operandIndex = operation.outputs[operationOutputIndex];
    data.m_OutputSlotForOperand[operandIndex] = &outputSlot;

    outputSlot.SetTensorInfo(tensor_info);

    return true;
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertCast(const HalOperation& operation,
                 const HalModel& model,
                 ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertCast()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsCastSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddCastLayer();
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the CastLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertChannelShuffle(const HalOperation& operation,
                           const HalModel& model,
                           ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertChannelShuffle()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    auto inputDimensions = static_cast<int32_t>(input.GetTensorInfo().GetNumDimensions());

    ChannelShuffleDescriptor descriptor;

    int32_t groups;
    if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, groups, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported number of groups operand", __func__);
    }
    descriptor.m_NumGroups = static_cast<uint32_t>(groups);

    int32_t axis;
    if (!GetInputScalar<HalPolicy>(operation, 2, HalOperandType::INT32, axis, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported dimension channel shuffle operand", __func__);
    }
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        return Fail("%s: Operation has invalid dimension: %d. It is out of bounds [-%d, %d))", __func__, axis,
                    inputDimensions, inputDimensions);
    }
    int positiveAxis = (axis < 0) ? inputDimensions + axis : axis;
    descriptor.m_Axis = static_cast<uint32_t>(positiveAxis);

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsChannelShuffleSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddChannelShuffleLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertComparison_1_2(const HalOperation& operation,
                           const HalModel& model,
                           ConversionData& data,
                           ComparisonOperation comparisonOperation)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertComparison()");
    ALOGV("comparisonOperation = %s", GetComparisonOperationAsCString(comparisonOperation));

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!(input0.IsValid() && input1.IsValid()))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const TensorInfo& inputInfo1 = input1.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ComparisonDescriptor descriptor(comparisonOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsComparisonSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo,
                                   descriptor);

    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddComparisonLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ComparisonLayer", __func__);
    }

    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertConv2d_1_2(const HalOperation& operation, const HalModel& model, ConversionData& data)
{

    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertConv2d_1_2()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    Convolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 7 ||
                           (operation.inputs.size() >= 8 &&
                            GetInputOperand<HalPolicy>(operation, 7, model)->type == HalOperandType::BOOL);

    if (implicitPadding)
    {
        desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 7, model, data);
    }
    else if (operation.inputs.size() >= 10)
    {
        desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 10, model, data);
    }

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // ArmNN does not currently support non-fixed weights or bias
    // The NNAPI filter is always OHWI [depth_out, filter_height, filter_width, depth_in] but ArmNN expects the
    // filter's height and width indices to match the input's height and width indices so we permute it to OIHW if
    // the DataLayout is NCHW


    if (!IsWeightsValid<HalPolicy>(operation, 1, model) && desc.m_DataLayout == DataLayout::NCHW)
    {
        return Fail("%s: Operation has unsupported weights HalOperandLifeTime", __func__);
    }

    LayerInputHandle weightsInput = (desc.m_DataLayout == DataLayout::NCHW) ?
                                     ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data, OHWIToOIHW) :
                                     ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!weightsInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LayerInputHandle biasInput = ConvertToLayerInputHandle<HalPolicy>(operation, 2, model, data); // 1D
    if (!biasInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    biasInput.SanitizeQuantizationScale(weightsInput, input);
    armnn::TensorInfo weightsInfo = weightsInput.GetTensorInfo();
    armnn::TensorInfo biasInfo = biasInput.GetTensorInfo();

    ActivationFn activation;

    if (implicitPadding)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 6, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<HalPolicy>(operation, 8, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
        unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
        unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();
        const uint32_t kernelX = weightsInfo.GetShape()[widthIndex];
        const uint32_t kernelY = weightsInfo.GetShape()[heightIndex];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);

    }
    else if (operation.inputs.size() >= 10)
    {
        // explicit padding
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 9, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<HalPolicy>(operation, 11, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    Optional<TensorInfo> biases(biasInfo);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsConvolution2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weightsInfo,
                                   biases);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddConvolution2dLayer(desc);
    startLayer->SetBackendId(setBackend);

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));
    weightsInput.Connect(startLayer->GetInputSlot(1));
    biasInput.Connect(startLayer->GetInputSlot(2));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertDepthwiseConv2d_1_2(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertDepthwiseConv2d_1_2()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const HalOperand* weightsOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    if (!weightsOperand)
    {
        return Fail("%s: Could not read weights", __func__);
    }
    if (weightsOperand->dimensions[0] != 1)
    {
        return Fail("%s: Invalid weights; for depthwise convolution, dimension 0 must be 1 but it is %i",
                    __func__, weightsOperand->dimensions[0] );
    }

    DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 8 ||
                           (operation.inputs.size() >= 9 &&
                            GetInputOperand<HalPolicy>(operation, 8, model)->type == HalOperandType::BOOL);

    // Look ahead to find the optional DataLayout, if present
    const uint32_t dataLayoutFlagIndex = implicitPadding ? 8 : 11;
    desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, dataLayoutFlagIndex, model, data);

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    LayerInputHandle weightsInput = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);
    if (!weightsInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* biasOperand = GetInputOperand<HalPolicy>(operation, 2, model);
    if (!biasOperand)
    {
        return Fail("%s: Could not read bias", __func__);
    }

    LayerInputHandle biasInput = ConvertToLayerInputHandle<HalPolicy>(operation, 2, model, data); // 1D
    if (!biasInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    biasInput.SanitizeQuantizationScale(weightsInput, input);
    armnn::TensorInfo weightsInfo = weightsInput.GetTensorInfo();
    armnn::TensorInfo biasInfo = biasInput.GetTensorInfo();

    ActivationFn activation;

    if (implicitPadding)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 7, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<HalPolicy>(operation, 9, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weightsInfo.GetShape()[2];
        const uint32_t kernelY = weightsInfo.GetShape()[1];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else if (operation.inputs.size() >= 11)
    {
        // explicit padding
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation,  10, activation, model, data) ||
            !GetOptionalConvolutionDilationParams<HalPolicy>(operation, 12, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    Optional<TensorInfo> biases(biasInfo);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDepthwiseConvolutionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weightsInfo,
                                   biases);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddDepthwiseConvolution2dLayer(desc);
    startLayer->SetBackendId(setBackend);

    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    // Connect weights and bias inputs
    weightsInput.Connect(startLayer->GetInputSlot(1));
    biasInput.Connect(startLayer->GetInputSlot(2));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertDequantize_1_2(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    ALOGV("HalPolicy::ConvertDequantize()");

    if (IsQSymmDequantizeForWeights<HalPolicy>(operation, model))
    {
        // NOTE: QSymm8 weights are dequantized internally by the driver,
        // therefore this type of Dequantize is implicitly supported
        return true;
    }

    return ::ConvertDequantize<HalPolicy>(operation, model, data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertElementwiseUnary(const HalOperation& operation,
                             const HalModel& model,
                             ConversionData& data,
                             UnaryOperation unaryOperation)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertElementwiseUnary()");
    ALOGV("unaryOperation = %s", GetUnaryOperationAsCString(unaryOperation));

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ElementwiseUnaryDescriptor descriptor(unaryOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsElementwiseUnarySupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddElementwiseUnaryLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ElementwiseUnaryLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertExpandDims(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertExpandDims()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    int32_t axis;
    if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, axis, model, data))
    {
        return Fail("%s: failed to get axis input value", __func__);
    }

    TensorShape targetShape;

    try
    {
        targetShape = armnnUtils::ExpandDims(input.GetTensorInfo().GetShape(), axis);
    }
    catch (const std::exception& e)
    {
        return Fail("%s: %s", __func__, e.what());
    }

    ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = targetShape;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReshapeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   reshapeDescriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        if (targetShape != outputInfo.GetShape())
        {
            return Fail("%s: Shape of the output operand does not match the resolved expanded shape", __func__);
        }
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ReshapeLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
        typename HalOperation = typename HalPolicy::Operation,
        typename HalModel     = typename HalPolicy::Model>
bool ConvertGather(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertGather()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }
    auto inputDimensions = input.GetTensorInfo().GetNumDimensions();

    LayerInputHandle indices = ConvertToLayerInputHandle<HalPolicy>(operation, 2, model, data);
    if (!indices.IsValid())
    {
        return Fail("%s: Operation has invalid indices", __func__);
    }
    auto indicesDimensions = indices.GetTensorInfo().GetNumDimensions();

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has invalid output", __func__);
    }
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    auto outputDimensions = outputInfo.GetNumDimensions();
    if (outputDimensions != inputDimensions + indicesDimensions - 1)
    {
        return Fail("%s: Operation has invalid output dimensions: %d. Output must be an (%d + %d - 1)-D tensor",
                     __func__, outputDimensions, inputDimensions, indicesDimensions);
    }

    int32_t axis;
    if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, axis, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported axis operand", __func__);
    }
    int32_t inputDimensions_int = static_cast<int32_t>(inputDimensions);
    if ((axis < -inputDimensions_int) || (inputDimensions_int <= axis))
    {
        return Fail("%s: Operation has invalid axis: %d. It is out of bounds [-%d, %d))", __func__, axis,
                    inputDimensions, inputDimensions);
    }

    GatherDescriptor desc;
    desc.m_Axis = axis;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsGatherSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   indices.GetTensorInfo(),
                                   outputInfo,
                                   desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddGatherLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the GatherLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));
    indices.Connect(layer->GetInputSlot(1));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertGroupedConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertGroupedConv2d()");

    //
    // Parse data
    //
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    const TensorInfo& inputInfo  = input.GetTensorInfo();

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }
    TensorInfo outputInfo = GetTensorInfoForOperand(*output);

    // Look ahead to determine data layout
    DataLayout dataLayout = DataLayout::NHWC;
    if (operation.inputs.size() == 12)
    {
        dataLayout = OptionalDataLayout<HalPolicy>(operation, 11, model, data);
    }
    else
    {
        dataLayout = OptionalDataLayout<HalPolicy>(operation, 8, model, data);
    }

    // NOTE:
    // NNAPI weights are always OHWI, i.e. [depth_out, filter_height, filter_width, depth_group],
    // but Arm NN expects the filter's height and width indices to match the input's height and
    // width indices so when the DataLayout is NCHW, we need to permute the weights to OIHW
    const PermutationVector ohwiToOihw = { 0u, 2u, 3u, 1u };
    const ConstTensorPin weightsPin = (dataLayout == DataLayout::NCHW) ?
                                      ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1,
                                                                                       model, data, ohwiToOihw) :
                                      ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1, model, data);
    const ConstTensorPin biasesPin  =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);
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

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    Convolution2dDescriptor desc;
    desc.m_DataLayout  = dataLayout;
    desc.m_BiasEnabled = true;

    unsigned int numGroups;
    ActivationFn activation;

    if (operation.inputs.size() == 12)
    {
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 9, HalOperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 10, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }

    }
    else if (operation.inputs.size() == 9)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 7, activation, model, data))
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

    // Equivalent to outputShape[channelsIndex], but we can't know the outputShape in the case of dynamic tensors
    const unsigned int outputChannels = weightsShape[0];

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
    armnn::BackendId setBackendSplit;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSplitterSupported,
                               data.m_Backends,
                               isSupported,
                               setBackendSplit,
                               inputInfo,
                               splitterOutputInfos,
                               splitterDesc);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* splitterLayer = data.m_Network->AddSplitterLayer(splitterDesc);
    splitterLayer->SetBackendId(setBackendSplit);
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

    TensorShape groupOutputShape(outputShape);
    const bool isDynamic = IsDynamicTensor(outputInfo);
    if (!isDynamic)
    {
        groupOutputShape[channelsIndex] = 1;
    }
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
            armnn::BackendId setBackendConv;
            auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
            {
                FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                           IsConvolution2dSupported,
                                           data.m_Backends,
                                           isSupported,
                                           setBackendConv,
                                           groupInputInfo,
                                           outputInfo,
                                           desc,
                                           groupWeightsInfo,
                                           Optional<TensorInfo>(groupBiasesInfo));
            };

            if(!isDynamic)
            {
                validateFunc(groupOutputInfo, isSupported);
            }
            else
            {
                isSupported = AreDynamicTensorsSupported();
            }

            if (!isSupported)
            {
                return false;
            }

            IConnectableLayer* weightsLayer = data.m_Network->AddConstantLayer(groupWeights);
            IConnectableLayer* biasLayer = data.m_Network->AddConstantLayer(groupBiases);
            IConnectableLayer* convLayer = data.m_Network->AddConvolution2dLayer(desc);
            convLayer->SetBackendId(setBackendConv);

            if (!convLayer)
            {
                return Fail("%s: AddConvolution2dLayer failed", __func__);
            }

            splitterLayer->GetOutputSlot(group).Connect(convLayer->GetInputSlot(0));
            weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
            biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));

            weightsLayer->GetOutputSlot(0).SetTensorInfo(groupWeightsInfo);
            biasLayer->GetOutputSlot(0).SetTensorInfo(groupBiasesInfo);
            convLayer->GetOutputSlot(0).SetTensorInfo(groupOutputInfo);

            if(isDynamic)
            {
                convLayer->GetOutputSlot(0).IsTensorInfoSet();

                validateFunc(convLayer->GetOutputSlot(0).GetTensorInfo(), isSupported);

                outputInfo = convLayer->GetOutputSlot(0).GetTensorInfo();

                if (!isSupported)
                {
                    return false;
                }
            }

            convLayers[index] = convLayer;
        }
    }

    //
    // Set up Concat layer
    //
    ConcatDescriptor concatDescriptor;
    // Equivalent to outputShape[channelsIndex], but we can't know the outputShape in the case of dynamic tensors
    concatDescriptor = ConcatDescriptor(weightsShape[0]);
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
    armnn::BackendId setBackendConcat;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsConcatSupported,
                               data.m_Backends,
                               isSupported,
                               setBackendConcat,
                               std::vector<const TensorInfo*>(numGroups * channelMultiplier, &groupOutputInfo),
                               outputInfo,
                               concatDescriptor);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* concatLayer = data.m_Network->AddConcatLayer(concatDescriptor);
    concatLayer->SetBackendId(setBackendConcat);
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

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *concatLayer, model,
                                                   data, nullptr, nullptr, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertInstanceNormalization(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertInstanceNormalization()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has an invalid input 0", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has an invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine data type of input tensor
    HalOperandType inputType;
    if (!GetOperandType<HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    InstanceNormalizationDescriptor desc;

    // Read gamma, beta & epsilon
    if (inputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half fp16Gamma;
        Half fp16Beta;
        Half fp16Epsilon;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, fp16Gamma, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 2, HalOperandType::FLOAT16, fp16Beta, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 3, HalOperandType::FLOAT16, fp16Epsilon, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_Gamma = static_cast<float>(fp16Gamma);
        desc.m_Beta  = static_cast<float>(fp16Beta);
        desc.m_Eps   = static_cast<float>(fp16Epsilon);
    }
    else if (inputType == HalOperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT32, desc.m_Gamma, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 2, HalOperandType::FLOAT32, desc.m_Beta, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 3, HalOperandType::FLOAT32, desc.m_Eps, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 4, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsInstanceNormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   desc);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddInstanceNormalizationLayer(desc);
    layer->SetBackendId(setBackend);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertLogSoftmax(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertLogSoftmax()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Failed to read input 0", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Failed to read output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine data type of input tensor
    HalOperandType inputType;
    if (!GetOperandType<HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LogSoftmaxDescriptor descriptor;

    // Read beta
    if (inputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half fp16Beta;
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, fp16Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT16)", __func__);
        }

        descriptor.m_Beta  = static_cast<float>(fp16Beta);
    }
    else if (inputType == HalOperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT32, descriptor.m_Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    // Read axis
    if (!GetInputInt32<HalPolicy>(operation, 2, descriptor.m_Axis, model, data))
    {
        return Fail("%s: Failed to read input 2", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLogSoftmaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddLogSoftmaxLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the LogSoftmaxLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertPadV2(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertPadV2()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    PadDescriptor descriptor;
    if (!ConvertPaddings<HalPolicy>(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine type of padding value
    HalOperandType operandType0;
    HalOperandType operandType2;

    if (!GetOperandType<HalPolicy>(operation, 0, model, operandType0) ||
        !GetOperandType<HalPolicy>(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Read value to use for padding
    if (operandType0 == HalOperandType::TENSOR_FLOAT16 && operandType2 == HalOperandType::FLOAT16)
    {
        Half f16PadValue;
        if (!GetInputScalar<HalPolicy>(operation, 2, operandType2, f16PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT16)", __func__);
        }

        descriptor.m_PadValue = f16PadValue;
    }
    else if (operandType0 == HalOperandType::TENSOR_FLOAT32 && operandType2 == HalOperandType::FLOAT32)
    {
        if (!GetInputFloat32<HalPolicy>(operation, 2, descriptor.m_PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT32)", __func__);
        }
    }
    else if (isQuantizedOperand(operandType0) && operandType2 == HalOperandType::INT32)
    {
        int32_t intPadValue = 0;
        if (!GetInputInt32<HalPolicy>(operation, 2, intPadValue, model, data))
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
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPadSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the PadLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertPrelu(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertPrelu()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle alpha = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!input.IsValid() || !alpha.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& alphaInfo  = alpha.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPreluSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   alphaInfo,
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddPreluLayer();
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the PreluLayer", __func__);
    }

    bool isReshapeSupported = BroadcastTensor(input, alpha, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertQuantize(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertQuantize()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const HalOperand* const outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQuantizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddQuantizeLayer();
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the QuantizeLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertQuantized16BitLstm(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertQuantized16BitLstm()");

    //Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }

    //13: The previous cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape
    //    [numBatches, outputSize] specifying the cell state from the previous time step of the LSTM cell.
    //    It is quantized using a quantization range of -2^4, 2^4 * 32767/32768.
    LayerInputHandle previousCellStateIn = ConvertToLayerInputHandle<HalPolicy>(operation, 13, model, data);
    if (!previousCellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 13: previousCellStateIn", __func__);
    }

    // 14: The previous output state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //     [numBathes, outputSize] specifying the output of the LSTM cell from previous time-step. Tensor
    //     is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle previousOutputIn = ConvertToLayerInputHandle<HalPolicy>(operation, 14, model, data);
    if (!previousOutputIn.IsValid())
    {
        return Fail("%s: Could not read input 14: previousOutputIn", __func__);
    }

    // Get the input tensors:
    // 1: The input-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-input part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1, model, data);

    // 2: The input-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-forget part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);

    // 3: The input-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-cell part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 3, model, data);

    // 4: The input-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-output part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 4, model, data);

    // 5: The recurrent-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-input part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 5, model, data);

    // 6: The recurrent-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-forget part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 6, model, data);

    // 7: The recurrent-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-cell part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 7, model, data);

    // 8: The recurrent-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-output part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 8, model, data);

    // 9: The input gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the
    //    bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 9, model, data);

    // 10: The forget gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 10, model, data);

    // 11:The cell bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the bias
    //    for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product of input
    //    and weights scales and zeroPoint equal to 0.
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 11, model, data);

    // 12:The output gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //    the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin outputGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 12, model, data);

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
    const HalOperand* cellStateOut = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 0: cellStateOut", __func__);
    }

    // 1: The output: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBathes, outputSize] which
    //      contains the output value. Tensor is quantized with a fixed quantization range of -1, 127/128.
    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 1, model);
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
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQuantizedLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   previousCellStateInInfo,
                                   previousOutputInInfo,
                                   cellStateOutInfo,
                                   outputInfo,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(cellStateOutInfo) &&
        !IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddQuantizedLstmLayer(params, "QuantizedLstm");
    layer->SetBackendId(setBackend);
    input.Connect(layer->GetInputSlot(0));
    previousCellStateIn.Connect(layer->GetInputSlot(1));
    previousOutputIn.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return (SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 0, model, data) &&
                SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 1, *layer, 1, model, data));
    }
    else
    {
        return (SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 0, model, data) &&
                SetupAndTrackLayerOutputSlot<HalPolicy>(
                    operation, 1, *layer, 1, model, data, nullptr, validateFunc, ActivationFn::kActivationNone, true));
    }

}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertReduce(const HalOperation& operation,
                   const HalModel& model,
                   ConversionData& data,
                   ReduceOperation reduceOperation)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    armnn::ReduceDescriptor descriptor;
    descriptor.m_ReduceOperation = reduceOperation;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const HalOperand* axisOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    if (!axisOperand)
    {
        return Fail("%s: Could not read input 1", __func__);
    }
    std::vector<int32_t> axis;
    if (!GetTensorInt32Values<HalPolicy>(*axisOperand, axis, model, data))
    {
        return Fail("%s: Input 1 has invalid values", __func__);
    }

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(), axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i) -> unsigned int { return (i + rank) % rank; });
    descriptor.m_vAxis.assign(uniqueAxis.begin(), uniqueAxis.end());

    // Get the "keep dims" flag.
    if (!GetInputScalar<HalPolicy>(operation, 2, HalOperandType::BOOL, descriptor.m_KeepDims, model, data))
    {
        return Fail("%s: Could not read input 2", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReduceSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddReduceLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ReduceLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertResize(const HalOperation& operation,
                   const HalModel& model,
                   ConversionData& data,
                   ResizeMethod resizeMethod)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;
    ALOGV("HalPolicy::ConvertResize()");
    ALOGV("resizeMethod = %s", GetResizeMethodAsCString(resizeMethod));

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ResizeDescriptor descriptor;
    descriptor.m_Method     = resizeMethod;
    descriptor.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 3, model, data);

    HalOperandType operandType1;
    HalOperandType operandType2;

    if (!GetOperandType<HalPolicy>(operation, 1, model, operandType1) ||
        !GetOperandType<HalPolicy>(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (operandType1 != operandType2)
    {
        return Fail("%s: Operation has invalid inputs. Type of input 1 and 2 should be the same", __func__);
    }

    if (operandType1 == HalOperandType::INT32)
    {
        // Case 1: resizing by shape
        int32_t targetWidth  = 0;
        int32_t targetHeight = 0;

        if (!GetInputInt32<HalPolicy>(operation, 1, targetWidth, model, data) ||
            !GetInputInt32<HalPolicy>(operation, 2, targetHeight, model, data))
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
    else if (operandType1 == HalOperandType::FLOAT32)
    {
        // Case 2: resizing by scale
        float widthScale  = 1.0f;
        float heightScale = 1.0f;

        if (!GetInputFloat32<HalPolicy>(operation, 1, widthScale, model, data) ||
            !GetInputFloat32<HalPolicy>(operation, 2, heightScale, model, data))
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
    else if (operandType1 == HalOperandType::FLOAT16)
    {
        Half widthScale;
        Half heightScale;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, widthScale, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 2, HalOperandType::FLOAT16, heightScale, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by scale", __func__);
        }

        const TensorShape& inputShape = inputInfo.GetShape();
        armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);

        Half width  = static_cast<Half>(inputShape[dataLayoutIndexed.GetWidthIndex()]);
        Half height = static_cast<Half>(inputShape[dataLayoutIndexed.GetHeightIndex()]);

        descriptor.m_TargetWidth  = std::floor(width  * widthScale);
        descriptor.m_TargetHeight = std::floor(height * heightScale);
    }
    else
    {
        return Fail("%s: Operand has invalid data type for resizing by scale", __func__);
    }

    descriptor.m_AlignCorners     = GetOptionalBool<HalPolicy>(operation, 4, model, data);
    descriptor.m_HalfPixelCenters = GetOptionalBool<HalPolicy>(operation, 5, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsResizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
        };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddResizeLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ResizeLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertSpaceToDepth(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertSpaceToDepth()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
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

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    SpaceToDepthDescriptor desc;

    GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, desc.m_BlockSize, model, data);

    if (desc.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 2, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSpaceToDepthSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the SpaceToDepthLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertSoftmax(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertSoftmax()");

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    SoftmaxDescriptor desc;
    HalOperandType outputType = outputOperand->type;

    // Read beta value
    if (outputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half value;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        desc.m_Beta = static_cast<float>(value);
    }
    else
    {
        if (!GetInputFloat32<HalPolicy>(operation, 1, desc.m_Beta, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }
    }

    if (operation.inputs.size() > 2 && !GetInputScalar<HalPolicy>(operation,
                                                                  2,
                                                                  HalOperandType::INT32,
                                                                  desc.m_Axis,
                                                                  model,
                                                                  data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSoftmaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   desc);
        };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the SoftmaxLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertLstm(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertLstm()");

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle<HalPolicy>(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle<HalPolicy>(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 2));
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 3));
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 4));
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 6));
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 7));
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 8));
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 15, model, data);

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
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 1, true));
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 5, true));
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 9, true));
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 10, true));
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 11, true));
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         12,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 16, true));
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
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
    ActivationFn activation = ActivationFn::kActivationNone;
    float cellClip;
    float projClip;
    if (!GetInputActivationFunctionFromTensor<HalPolicy>(operation, 20, activation, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 21, HalOperandType::FLOAT32, cellClip, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 22, HalOperandType::FLOAT32, projClip, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Get the normalization tensors
    // 23: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin
        (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 23, true));

    // 24: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                        24,
                                                        model,
                                                        data,
                                                        g_DontPermute,
                                                        nullptr,
                                                        true);

    // 25: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         25,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 26: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         26,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4]
    // with CIFG, or [batch_size, num_units * 3] without CIFG.
    const HalOperand* scratchBuffer = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const HalOperand* outputStateOut = GetOutputOperand<HalPolicy>(operation, 1, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const HalOperand* cellStateOut = GetOutputOperand<HalPolicy>(operation, 2, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 3, model);
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
    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if (desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if (desc.m_PeepholeEnabled)
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
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputStateInInfo,
                                   cellStateInInfo,
                                   scratchBufferInfo,
                                   outputStateOutInfo,
                                   cellStateOutInfo,
                                   outputInfo,
                                   desc,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(outputStateOutInfo) &&
        !IsDynamicTensor(scratchBufferInfo)  &&
        !IsDynamicTensor(cellStateOutInfo)   &&
        !IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");
    layer->SetBackendId(setBackend);

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return (
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 0, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 1, *layer, 1, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 2, *layer, 2, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 3, *layer, 3, model, data));
    }
    else
    {
        return (
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 0, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 1, *layer, 1, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 2, *layer, 2, model, data) &&
             SetupAndTrackLayerOutputSlot<HalPolicy>(
                 operation, 3, *layer, 3, model, data, nullptr, validateFunc, ActivationFn::kActivationNone, true));
    }

}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertTransposeConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const HalOperand* weightsOperand = GetInputOperand<HalPolicy>(operation, 1, model);

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
        desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 8, model, data);
    }
    else
    {
        desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 10, model, data);
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // The shape of the weight is [depth_out, filter_height, filter_width, depth_in].
    // We have to permute it to OIHW if the data layout is NCHW.
    const ConstTensorPin weightsPin = (desc.m_DataLayout == DataLayout::NCHW) ?
                                      ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1,
                                                                                       model, data, OHWIToOIHW) :
                                      ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1, model, data);

    // Bias is a 1D tensor
    const ConstTensorPin biasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);

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
        if (!GetInputPaddingScheme<HalPolicy>(operation, 4, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, strideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, strideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[widthIndex];
        const uint32_t kernelY = weights.GetShape()[heightIndex];

        // If output shape has been specified as a parameter then extract it and make it available.
        const HalOperand* outputShapeOperand = GetInputOperand<HalPolicy>(operation, 3, model, false);
        std::vector<int32_t> outputShape;
        if ((outputShapeOperand) && (GetTensorInt32Values<HalPolicy>(*outputShapeOperand, outputShape, model, data)))
        {
            // Change from signed to unsigned int to store in TransposeConvolution2dDescriptor.
            for (int dimension : outputShape)
            {
                desc.m_OutputShape.push_back(static_cast<unsigned int>(dimension));
            }
            desc.m_OutputShapeEnabled = true;
        }

        uint32_t outputX;
        uint32_t outputY;

        if (IsDynamicTensor(outputInfo))
        {
            if (outputShape.size() == 0)
            {
                return Fail("%s: Padding sizes cannot be inferred", __func__);
            }

            outputX = outputShape[widthIndex];
            outputY = outputShape[heightIndex];
        }
        else
        {
            outputX = outputInfo.GetShape()[widthIndex];
            outputY = outputInfo.GetShape()[heightIndex];
        }

        CalcPaddingTransposeConv(outputX, kernelX, strideX, padLeft, padRight, paddingScheme);
        CalcPaddingTransposeConv(outputY, kernelY, strideY, padTop, padBottom, paddingScheme);

        // NOTE: The Android NN API allows for negative padding values in TransposeConv2d,
        // but Arm NN only supports values >= 0
        if (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0)
        {
            return Fail("%s: Negative padding values are not supported", __func__);
        }

        desc.m_StrideX   = armnn::numeric_cast<uint32_t>(strideX);
        desc.m_StrideY   = armnn::numeric_cast<uint32_t>(strideY);
        desc.m_PadLeft   = armnn::numeric_cast<uint32_t>(padLeft);
        desc.m_PadRight  = armnn::numeric_cast<uint32_t>(padRight);
        desc.m_PadTop    = armnn::numeric_cast<uint32_t>(padTop);
        desc.m_PadBottom = armnn::numeric_cast<uint32_t>(padBottom);
    }
    else if (operation.inputs.size() == 11)
    {
        // explicit padding
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation,  9, activation, model, data))
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
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeConvolution2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weights.GetInfo(),
                                   biases);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* startLayer =
        data.m_Network->AddTransposeConvolution2dLayer(desc, weights, Optional<ConstTensor>(bias));
    startLayer->SetBackendId(setBackend);
    if (!startLayer)
    {
        return Fail("%s: AddTransposeConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertUnidirectionalSequenceLstm(const HalOperation& operation,
                                       const HalModel& model,
                                       ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertUnidirectionalSequenceLstm()");

    // Determine if input OperandType is ANEURALNETWORKS_TENSOR_FLOAT 32 or 16
    HalOperandType inputType;
    if (!GetOperandType<HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Inputs:
    // 0: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    // [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length), “batch_size”
    // corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle<HalPolicy>(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle<HalPolicy>(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 2));
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 3));
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 4));
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 6));
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 7));
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 8));
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin cellBiasPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 15, model, data);

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
    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 1, true));
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 5, true));
    // 09: The cell-to-input weights: Optional.
    // A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 9, true));
    // 10: The cell-to-forget weights: Optional.
    // A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 10, true));
    // 11: The cell-to-output weights: Optional.
    // A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 11, true));
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                                              12,
                                                                              model,
                                                                              data,
                                                                              g_DontPermute,
                                                                              nullptr,
                                                                              true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 16, true));
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
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
    // Determine data type of input tensor
    ActivationFn activation = ActivationFn::kActivationNone;
    LstmDescriptor desc;

    if (inputType == HalOperandType::TENSOR_FLOAT32)
    {
        float cellClip;
        float projClip;

        if (!GetInputActivationFunctionFromTensor<HalPolicy>(operation, 20, activation, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 21, HalOperandType::FLOAT32, cellClip, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 22, HalOperandType::FLOAT32, projClip, model, data))
        {
            return Fail("%s: Operation has invalid scalar inputs", __func__);
        }

        desc.m_ClippingThresCell = cellClip;
        desc.m_ClippingThresProj = projClip;
    }

    if (inputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half cellClip;
        Half projClip;

        if (!GetInputActivationFunctionFromTensor<HalPolicy>(operation, 20, activation, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 21, HalOperandType::FLOAT16, cellClip, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 22, HalOperandType::FLOAT16, projClip, model, data))
        {
            return Fail("%s: Operation has invalid scalar inputs", __func__);
        }

        desc.m_ClippingThresCell = cellClip;
        desc.m_ClippingThresProj = projClip;
    }

    // Determine if time-major or batch-major.
    // 23: Time-major if true, batch-major if false.
    bool isTimeMajor = GetOptionalBool<HalPolicy>(operation, 23, model, data);

    // Get the normalization tensors
    // 24: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin
                             (DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 24, true));

    // 25: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                                              25,
                                                                              model,
                                                                              data,
                                                                              g_DontPermute,
                                                                              nullptr,
                                                                              true);

    // 26: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                                              26,
                                                                              model,
                                                                              data,
                                                                              g_DontPermute,
                                                                              nullptr,
                                                                              true);

    // 27: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
                             ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                                              27,
                                                                              model,
                                                                              data,
                                                                              g_DontPermute,
                                                                              nullptr,
                                                                              true);

    // Outputs:
    // 00: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    // [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output: ", __func__);
    }

    //
    // 01 & 02: 
    // hiddenStateOut and cellStateOut are not currently supported by our android versioning.
    //

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
    desc.m_ActivationFunc = activation;
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
    desc.m_TimeMajor = isTimeMajor;

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
    const TensorInfo& outputInfo         = GetTensorInfoForOperand(*output);

    unsigned int batchSize               = inputInfo.GetShape()[0];
    unsigned int outputSize              = outputInfo.GetShape()[2];
    unsigned int numUnits                = cellStateInInfo.GetShape()[1];

    armnn::DataType dataType             = inputInfo.GetDataType();
    float qScale                         = inputInfo.GetQuantizationScale();
    int qOffset                          = inputInfo.GetQuantizationOffset();

    armnn::TensorInfo cellStateOutInfo({batchSize, numUnits}, cellStateInInfo.GetDataType(),
                                       cellStateInInfo.GetQuantizationScale(), cellStateInInfo.GetQuantizationOffset());
    armnn::TensorInfo outputStateOutInfo({batchSize, outputSize}, dataType, qScale, qOffset);

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
    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if (desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if (desc.m_PeepholeEnabled)
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
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsUnidirectionalSequenceLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputStateInInfo,
                                   cellStateInInfo,
                                   outputStateOutInfo,
                                   cellStateOutInfo,
                                   outputInfo,
                                   desc,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    IConnectableLayer* layer = data.m_Network->AddUnidirectionalSequenceLstmLayer(desc,
                                                                                  params,
                                                                                  "UnidirectionalSequenceLstm");
    layer->SetBackendId(setBackend);

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return (SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 2, model, data));
    }
    else
    {
        return (SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, 2, model, data, nullptr,
                                                        validateFunc, ActivationFn::kActivationNone, true));
    }
}

} // armnn_driver namespace