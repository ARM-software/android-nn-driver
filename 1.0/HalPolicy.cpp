//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include <armnn/Optional.hpp>

#include "FullyConnected.hpp"
#include "Utils.hpp"

namespace armnn_driver
{
namespace hal_1_0
{

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case V1_0::OperationType::ADD:
            return ConvertElementwiseBinary(operation, model, data, armnn::BinaryOperation::Add);
        case V1_0::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_0::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_0::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_0::OperationType::DEPTH_TO_SPACE:
            return ConvertDepthToSpace(operation, model, data);
        case V1_0::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
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
            return ConvertElementwiseBinary(operation, model, data, armnn::BinaryOperation::Mul);
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

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertAveragePool2d()");
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertConcatenation()");
    return ::ConvertConcatenation<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertConv2d()");
    return ::ConvertConv2d<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertDepthToSpace()");
    return ::ConvertDepthToSpace<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertDepthwiseConv2d()");
    return ::ConvertDepthwiseConv2d<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertDequantize()");
    return ::ConvertDequantize<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation)
{
    ALOGV("hal_1_0::HalPolicy::ConvertElementwiseBinary()");
    return ::ConvertElementwiseBinary<hal_1_0::HalPolicy>(operation, model, data, binaryOperation);
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertFloor()");
    return ::ConvertFloor<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertFullyConnected()");
    return ::ConvertFullyConnected<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertLocalResponseNormalization()");
    return ::ConvertLocalResponseNormalization<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertLogistic()");
    return ::ConvertLogistic<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertLstm()");

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

    bool isSupported = false;
    armnn::BackendId setBackend;
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
    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    armnn::IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");
    layer->SetBackendId(setBackend);

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
    ALOGV("hal_1_0::HalPolicy::ConvertL2Normalization()");
    return ::ConvertL2Normalization<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertL2Pool2d()");
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertMaxPool2d()");
    return ConvertPooling2d<hal_1_0::HalPolicy>(operation, __func__, armnn::PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertReLu()");
    return ::ConvertReLu<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertReLu1()");
    return ::ConvertReLu1<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertReLu6()");
    return ::ConvertReLu6<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertSoftmax()");

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

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::SoftmaxDescriptor desc;
    if (!GetInputFloat32<hal_1_0::HalPolicy>(operation, 1, desc.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSoftmaxSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               input.GetTensorInfo(),
                               outputInfo,
                               desc);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the SoftmaxLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertSpaceToDepth()");

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
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSpaceToDepthSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               inputInfo,
                               outputInfo,
                               desc);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the SpaceToDepthLayer", __func__);
    }
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertTanH()");
    return ::ConvertTanH<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertReshape()");
    return ::ConvertReshape<hal_1_0::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertResizeBilinear(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_0::HalPolicy::ConvertResizeBilinear()");

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

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    armnn::ResizeDescriptor desc;
    desc.m_Method     = armnn::ResizeMethod::Bilinear;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsResizeSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               inputInfo,
                               outputInfo,
                               desc);
    if (!isSupported)
    {
        return false;
    }

    if (!GetInputScalar<hal_1_0::HalPolicy>(operation, 1, OperandType::INT32, desc.m_TargetWidth, model, data) ||
        !GetInputScalar<hal_1_0::HalPolicy>(operation, 2, OperandType::INT32, desc.m_TargetHeight, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddResizeLayer(desc);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: Could not add the ResizeLayer", __func__);
    }
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<hal_1_0::HalPolicy>(operation, 0, *layer, model, data);

}

} // namespace hal_1_0
} // namespace armnn_driver
