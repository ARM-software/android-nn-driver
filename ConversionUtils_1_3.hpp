//
// Copyright © 2020,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ConversionUtils_1_2.hpp"

using Half = half_float::half;

namespace armnn_driver
{

using namespace armnn;
using namespace android::nn;

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertElu(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input0.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Determine data type of input tensor
    HalOperandType inputType;
    if (!GetOperandType<HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Elu;

    // Read alpha
    if (inputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half alpha;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, alpha, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_A = static_cast<float>(alpha);
    }
    else if (inputType == HalOperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT32, desc.m_A, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    return ::ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertFill(const HalOperation& operation, const HalModel& model, ConversionData& data)
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
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Determine data type of output tensor
    HalOperandType outputType = output->type;
    FillDescriptor descriptor;
    // Read the scalar fill value
    if (outputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half value;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        descriptor.m_Value = static_cast<float>(value);
    }
    else if (outputType == HalOperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT32, descriptor.m_Value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }
    }
    else if (outputType == HalOperandType::TENSOR_INT32)
    {
        int32_t value;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        descriptor.m_Value = static_cast<float>(value);
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, outputType);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsFillSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               inputInfo,
                               outputInfo,
                               descriptor);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddFillLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the FillLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertLogicalBinary(const HalOperation& operation,
                          const HalModel& model,
                          ConversionData& data,
                          LogicalBinaryOperation logicalOperation)
{
    using HalOperand = typename HalPolicy::Operand;

    ALOGV("HalPolicy::ConvertLogicalBinary()");
    ALOGV("logicalOperation = %s", GetLogicalBinaryOperationAsCString(logicalOperation));

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

    LogicalBinaryDescriptor descriptor(logicalOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLogicalBinarySupported,
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

    IConnectableLayer* layer = data.m_Network->AddLogicalBinaryLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the LogicalBinaryLayer", __func__);
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
bool ConvertQuantizedLstm(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    ALOGV("HalPolicy::ConvertQuantizedLstm()");

    //Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, of shape [batch_size, output_size].
    LayerInputHandle outputStatePrevTimeStep = ConvertToLayerInputHandle<HalPolicy>(operation, 18, model, data);
    if (!outputStatePrevTimeStep.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStatePrevTimeStep", __func__);
    }

    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape [batch_size, num_units].
    LayerInputHandle cellStatePrevTimeStep = ConvertToLayerInputHandle<HalPolicy>(operation, 19, model, data);
    if (!cellStatePrevTimeStep.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStatePrevTimeStep", __func__);
    }

    // Get the mandatory input tensors:

    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);

    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 3, model, data);

    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 4, model, data);

    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 6, model, data);

    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 7, model, data);

    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 8, model, data);

    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 13, model, data);

    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 14, model, data);

    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
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

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         1,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         5,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToInputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         9,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         10,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         11,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         12,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         16,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         17,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    if ((!inputToInputWeightsPin.IsValid() && !inputToInputWeightsPin.IsOptional())
        || (!recurrentToInputWeightsPin.IsValid() && !recurrentToInputWeightsPin.IsOptional())
        || (!cellToInputWeightsPin.IsValid() && !cellToInputWeightsPin.IsOptional())
        || (!cellToForgetWeightsPin.IsValid() && !cellToForgetWeightsPin.IsOptional())
        || (!cellToOutputWeightsPin.IsValid() && !cellToOutputWeightsPin.IsOptional())
        || (!inputGateBiasPin.IsValid() && !inputGateBiasPin.IsOptional())
        || (!projectionWeightsPin.IsValid() && !projectionWeightsPin.IsOptional())
        || (!projectionBiasPin.IsValid() && !projectionBiasPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }


    // Get the optional normalization tensors

    // 20: The input layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM.
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         20,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 21: The forget layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         21,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 22: The cell layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM.
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         22,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 23: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         23,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    if ((!inputLayerNormWeightsPin.IsValid() && !inputLayerNormWeightsPin.IsOptional())
        || (!forgetLayerNormWeightsPin.IsValid() && !forgetLayerNormWeightsPin.IsOptional())
        || (!cellLayerNormWeightsPin.IsValid() && !cellLayerNormWeightsPin.IsOptional())
        || (!outputLayerNormWeightsPin.IsValid() && !outputLayerNormWeightsPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input scalars:
    // 24: The cell clip:  If provided the cell state is clipped by this value prior to the cell output activation.
    // 25: The projection clip: If provided and projection is enabled, this is used for clipping the projected values.

    // Get the mandatory input scalars:
    // 26: The scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
    // 27: The scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
    // 28: The scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
    // 29: The scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
    // 30: The zero point of the hidden state, i.e. input to projection.
    // 31: The scale of the hidden state, i.e. input to projection.
    float cellClip, projClip, matMulInputGate, matMulForgetGate, matMulCellGate, matMulOutputGate, projInputScale;
    int projInputZeroPoint;

    if (!GetInputScalar<HalPolicy>(operation, 24, HalOperandType::FLOAT32, cellClip, model, data, true) ||
        !GetInputScalar<HalPolicy>(operation, 25, HalOperandType::FLOAT32, projClip, model, data, true) ||
        !GetInputScalar<HalPolicy>(operation, 26, HalOperandType::FLOAT32, matMulInputGate, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 27, HalOperandType::FLOAT32, matMulForgetGate, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 28, HalOperandType::FLOAT32, matMulCellGate, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 29, HalOperandType::FLOAT32, matMulOutputGate, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 30, HalOperandType::INT32, projInputZeroPoint, model, data) ||
        !GetInputScalar<HalPolicy>(operation, 31, HalOperandType::FLOAT32, projInputScale, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Outputs:
    // 0: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, of shape [batch_size,
    // output_size].
    const HalOperand* outputStateOut = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 0: outputStateOut", __func__);
    }

    // 1: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape [batch_size, num_units].
    const HalOperand* cellStateOut = GetOutputOperand<HalPolicy>(operation, 1, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 1: cellStateOut", __func__);
    }

    // 2: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, of shape [batch_size, output_size].
    // This is effectively the same as the current “output state (out)” value.
    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 2, model);
    if (!output)
    {
        return Fail("%s: Could not read output 2: output", __func__);
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
    QLstmDescriptor desc;
    desc.m_CellClip = cellClip;
    desc.m_ProjectionClip = projClip;
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
    desc.m_InputIntermediateScale = matMulInputGate;
    desc.m_ForgetIntermediateScale = matMulForgetGate;
    desc.m_CellIntermediateScale = matMulCellGate;
    desc.m_OutputIntermediateScale = matMulOutputGate;
    desc.m_HiddenStateScale = projInputScale;
    desc.m_HiddenStateZeroPoint = projInputZeroPoint;

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

    // Inputs
    const TensorInfo& inputInfo = input.GetTensorInfo();
    const TensorInfo& outputStatePrevTimeStepInfo = outputStatePrevTimeStep.GetTensorInfo();
    const TensorInfo& cellStatePrevTimeStepInfo = cellStatePrevTimeStep.GetTensorInfo();

    // Outputs
    TensorInfo outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    TensorInfo outputInfo = GetTensorInfoForOperand(*output);
    const TensorInfo& cellStateOutInfo = GetTensorInfoForOperand(*cellStateOut);

    // Optional parameters
    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (desc.m_PeepholeEnabled)
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
    else
    {
        // If Projection is disabled, override non-const outputs to change the quant info with hidden params, then
        // create a new const TensorInfo based on this
        outputStateOutInfo.SetQuantizationScale(projInputScale);
        outputStateOutInfo.SetQuantizationOffset(projInputZeroPoint);
        outputInfo.SetQuantizationScale(projInputScale);
        outputInfo.SetQuantizationOffset(projInputZeroPoint);
    }

    const TensorInfo constOutputStateOutInfo(outputStateOutInfo);
    const TensorInfo constOutputInfo(outputInfo);

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

    // Check if the layer is supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& cellStateOutInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputStatePrevTimeStepInfo,
                                   cellStatePrevTimeStepInfo,
                                   constOutputStateOutInfo,
                                   cellStateOutInfo,
                                   constOutputInfo,
                                   desc,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(constOutputStateOutInfo) &&
        !IsDynamicTensor(cellStateOutInfo)  &&
        !IsDynamicTensor(constOutputInfo))
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
    IConnectableLayer* layer = data.m_Network->AddQLstmLayer(desc, params, "QLstm");
    layer->SetBackendId(setBackend);

    input.Connect(layer->GetInputSlot(0));
    outputStatePrevTimeStep.Connect(layer->GetInputSlot(1));
    cellStatePrevTimeStep.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return ( SetupAndTrackLayerOutputSlot<HalPolicy>(
                       operation, 0, *layer, 0, model, data, &constOutputStateOutInfo) &&
                 SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 1, *layer, 1, model, data) &&
                 SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 2, *layer, 2, model, data, &constOutputInfo));
    }
    else
    {
        return ( SetupAndTrackLayerOutputSlot<HalPolicy>(
                       operation, 0, *layer, 0, model, data, &constOutputStateOutInfo) &&
                 SetupAndTrackLayerOutputSlot<HalPolicy>(
                       operation, 1, *layer, 1, model, data, nullptr, validateFunc,
                       ActivationFn::kActivationNone, true) &&
                 SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 2, *layer, 2, model, data, &constOutputInfo));
    }
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertRank(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* inputOperand = GetInputOperand<HalPolicy>(operation, 0, model);
    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (inputOperand == nullptr || outputOperand == nullptr)
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Shape inputOperandShape = GetOperandShape(*inputOperand);
    const Shape outputOperandShape = GetOperandShape(*outputOperand);

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsRankSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               input.GetTensorInfo(),
                               outInfo);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddRankLayer();
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the RankLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, &outInfo);
}

} // armnn_driver namespace
