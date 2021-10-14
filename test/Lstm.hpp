//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverTestHelpers.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

#include <array>

using ArmnnDriver   = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using RequestArgument = V1_0::RequestArgument;

#ifdef ARMNN_ANDROID_S
#include <nnapi/Types.h>
#endif

using namespace driverTestHelpers;
using namespace android::hardware;

namespace
{

template<typename T>
RequestArgument CreateRequestArgument(const std::vector<T>& value, unsigned int poolIndex)
{
    V1_0::DataLocation inputInloc = {};
    inputInloc.poolIndex = poolIndex;
    inputInloc.offset = 0;
    inputInloc.length = value.size() * sizeof(T);
    RequestArgument inputRequestArgument = {};
    inputRequestArgument.location = inputInloc;
    inputRequestArgument.dimensions = hidl_vec<uint32_t>{};
    return inputRequestArgument;
}

// Helper function to create an OperandLifeTime::NO_VALUE for testing.
// To be used on optional input operands that have no values - these are valid and should be tested.
V1_0::OperandLifeTime CreateNoValueLifeTime(const hidl_vec<uint32_t>& dimensions)
{
    // Only create a NO_VALUE for optional operands that have no elements
    if (dimensions.size() == 0 || dimensions[0] == 0)
    {
        return V1_0::OperandLifeTime::NO_VALUE;
    }
    return V1_0::OperandLifeTime::CONSTANT_COPY;
}

template<typename HalModel>
void ExecuteModel(const HalModel& model, armnn_driver::ArmnnDriver& driver, const V1_0::Request& request)
{
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, driver);
    if (preparedModel.get() != nullptr)
    {
        Execute(preparedModel, request);
    }
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

template<>
void ExecuteModel<armnn_driver::hal_1_2::HalPolicy::Model>(const armnn_driver::hal_1_2::HalPolicy::Model& model,
                                                           armnn_driver::ArmnnDriver& driver,
                                                           const V1_0::Request& request)
{
    android::sp<V1_2::IPreparedModel> preparedModel = PrepareModel_1_2(model, driver);
    if (preparedModel.get() != nullptr)
    {
        Execute(preparedModel, request);
    }
}

#endif

} // anonymous namespace

// Add our own tests here since we fail the lstm tests which Google supplies (because of non-const weights)
template <typename HalPolicy>
void LstmTestImpl(const hidl_vec<uint32_t>&   inputDimensions,
                  const std::vector<float>&   inputValue,
                  const hidl_vec<uint32_t>&   inputToInputWeightsDimensions,
                  const std::vector<float>&   inputToInputWeightsValue,
                  const hidl_vec<uint32_t>&   inputToForgetWeightsDimensions,
                  const std::vector<float>&   inputToForgetWeightsValue,
                  const hidl_vec<uint32_t>&   inputToCellWeightsDimensions,
                  const std::vector<float>&   inputToCellWeightsValue,
                  const hidl_vec<uint32_t>&   inputToOutputWeightsDimensions,
                  const std::vector<float>&   inputToOutputWeightsValue,
                  const hidl_vec<uint32_t>&   recurrentToInputWeightsDimensions,
                  const std::vector<float>&   recurrentToInputWeightsValue,
                  const hidl_vec<uint32_t>&   recurrentToForgetWeightsDimensions,
                  const std::vector<float>&   recurrentToForgetWeightsValue,
                  const hidl_vec<uint32_t>&   recurrentToCellWeightsDimensions,
                  const std::vector<float>&   recurrentToCellWeightsValue,
                  const hidl_vec<uint32_t>&   recurrentToOutputWeightsDimensions,
                  const std::vector<float>&   recurrentToOutputWeightsValue,
                  const hidl_vec<uint32_t>&   cellToInputWeightsDimensions,
                  const std::vector<float>&   cellToInputWeightsValue,
                  const hidl_vec<uint32_t>&   cellToForgetWeightsDimensions,
                  const std::vector<float>&   cellToForgetWeightsValue,
                  const hidl_vec<uint32_t>&   cellToOutputWeightsDimensions,
                  const std::vector<float>&   cellToOutputWeightsValue,
                  const hidl_vec<uint32_t>&   inputGateBiasDimensions,
                  const std::vector<float>&   inputGateBiasValue,
                  const hidl_vec<uint32_t>&   forgetGateBiasDimensions,
                  const std::vector<float>&   forgetGateBiasValue,
                  const hidl_vec<uint32_t>&   cellBiasDimensions,
                  const std::vector<float>&   cellBiasValue,
                  const hidl_vec<uint32_t>&   outputGateBiasDimensions,
                  const std::vector<float>&   outputGateBiasValue,
                  const hidl_vec<uint32_t>&   projectionWeightsDimensions,
                  const std::vector<float>&   projectionWeightsValue,
                  const hidl_vec<uint32_t>&   projectionBiasDimensions,
                  const std::vector<float>&   projectionBiasValue,
                  const hidl_vec<uint32_t>&   outputStateInDimensions,
                  const std::vector<float>&   outputStateInValue,
                  const hidl_vec<uint32_t>&   cellStateInDimensions,
                  const std::vector<float>&   cellStateInValue,
                  const hidl_vec<uint32_t>&   activationFunctionDimensions,
                  const std::vector<int32_t>& activationFunctionValue,
                  const hidl_vec<uint32_t>&   cellClippingThresholdDimensions,
                  const std::vector<float>&   cellClippingThresholdValue,
                  const hidl_vec<uint32_t>&   projectionClippingThresholdDimensions,
                  const std::vector<float>&   projectionClippingThresholdValue,
                  const hidl_vec<uint32_t>&   inputLayerNormWeightsDimensions,
                  const std::vector<float>&   inputLayerNormWeightsValue,
                  const hidl_vec<uint32_t>&   forgetLayerNormWeightsDimensions,
                  const std::vector<float>&   forgetLayerNormWeightsValue,
                  const hidl_vec<uint32_t>&   cellLayerNormWeightsDimensions,
                  const std::vector<float>&   cellLayerNormWeightsValue,
                  const hidl_vec<uint32_t>&   outputLayerNormWeightsDimensions,
                  const std::vector<float>&   outputLayerNormWeightsValue,
                  const hidl_vec<uint32_t>&   scratchBufferDimensions,
                  const std::vector<float>&   scratchBufferValue,
                  const hidl_vec<uint32_t>&   outputStateOutDimensions,
                  const std::vector<float>&   outputStateOutValue,
                  const hidl_vec<uint32_t>&   cellStateOutDimensions,
                  const std::vector<float>&   cellStateOutValue,
                  const hidl_vec<uint32_t>&   outputDimensions,
                  const std::vector<float>&   outputValue,
                  armnn::Compute              compute)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(compute));
    using Model = typename HalPolicy::Model;
    Model model = {};

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    AddInputOperand<HalPolicy>(model, inputDimensions);

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    AddTensorOperand<HalPolicy>(model,
                                inputToInputWeightsDimensions,
                                inputToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(inputToInputWeightsDimensions));
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    AddTensorOperand<HalPolicy>(model, inputToForgetWeightsDimensions, inputToForgetWeightsValue);
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    AddTensorOperand<HalPolicy>(model, inputToCellWeightsDimensions, inputToCellWeightsValue);
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    AddTensorOperand<HalPolicy>(model, inputToOutputWeightsDimensions, inputToOutputWeightsValue);
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    AddTensorOperand<HalPolicy>(model,
                                recurrentToInputWeightsDimensions,
                                recurrentToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(recurrentToInputWeightsDimensions));
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    AddTensorOperand<HalPolicy>(model, recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue);
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    AddTensorOperand<HalPolicy>(model, recurrentToCellWeightsDimensions, recurrentToCellWeightsValue);
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    AddTensorOperand<HalPolicy>(model, recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue);
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model,
                                cellToInputWeightsDimensions,
                                cellToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(cellToInputWeightsDimensions));
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model,
                                cellToForgetWeightsDimensions,
                                cellToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(cellToForgetWeightsDimensions));
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model,
                                cellToOutputWeightsDimensions,
                                cellToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(cellToOutputWeightsDimensions));
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model,
                                inputGateBiasDimensions,
                                inputGateBiasValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(inputGateBiasDimensions));
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model, forgetGateBiasDimensions, forgetGateBiasValue);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model, cellBiasDimensions, cellBiasValue);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    AddTensorOperand<HalPolicy>(model, outputGateBiasDimensions, outputGateBiasValue);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    AddTensorOperand<HalPolicy>(model,
                                projectionWeightsDimensions,
                                projectionWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(projectionWeightsDimensions));
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    AddTensorOperand<HalPolicy>(model,
                                projectionBiasDimensions,
                                projectionBiasValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(projectionBiasDimensions));

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    AddInputOperand<HalPolicy>(model, outputStateInDimensions);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    AddInputOperand<HalPolicy>(model, cellStateInDimensions);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    AddTensorOperand<HalPolicy>(model,
                                activationFunctionDimensions,
                                activationFunctionValue,
                                HalPolicy::OperandType::INT32);
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    AddTensorOperand<HalPolicy>(model,
                                cellClippingThresholdDimensions,
                                cellClippingThresholdValue,
                                HalPolicy::OperandType::FLOAT32);
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    AddTensorOperand<HalPolicy>(model,
                                projectionClippingThresholdDimensions,
                                projectionClippingThresholdValue,
                                HalPolicy::OperandType::FLOAT32);

    bool normalizationEnabled = false;

    // If any of the tensors have a value all normalization tensors are set
    if (!inputLayerNormWeightsValue.empty()  ||
        !forgetLayerNormWeightsValue.empty() ||
        !cellLayerNormWeightsValue.empty()   ||
        !outputLayerNormWeightsValue.empty())
    {
        // Normalization:
        // 23:The input layer normalization weights. A 1-D tensor of shape [num_units].
        //    Used to rescale normalized inputs to activation at input gate.
        AddTensorOperand<HalPolicy>(model,
                                    inputLayerNormWeightsDimensions,
                                    inputLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_FLOAT32,
                                    CreateNoValueLifeTime(inputLayerNormWeightsDimensions));
        // 24:The forget layer normalization weights. A 1-D tensor of shape [num_units].
        //    Used to rescale normalized inputs to activation at forget gate.
        AddTensorOperand<HalPolicy>(model,
                                    forgetLayerNormWeightsDimensions,
                                    forgetLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_FLOAT32,
                                    CreateNoValueLifeTime(forgetLayerNormWeightsDimensions));
        // 25:The cell layer normalization weights. A 1-D tensor of shape [num_units].
        //    Used to rescale normalized inputs to activation at cell gate.
        AddTensorOperand<HalPolicy>(model,
                                    cellLayerNormWeightsDimensions,
                                    cellLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_FLOAT32,
                                    CreateNoValueLifeTime(cellLayerNormWeightsDimensions));
        // 26:The output layer normalization weights. A 1-D tensor of shape [num_units].
        //    Used to rescale normalized inputs to activation at output gate.
        AddTensorOperand<HalPolicy>(model,
                                    outputLayerNormWeightsDimensions,
                                    outputLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_FLOAT32,
                                    CreateNoValueLifeTime(outputLayerNormWeightsDimensions));

        normalizationEnabled = true;
    }

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    AddOutputOperand<HalPolicy>(model, scratchBufferDimensions);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    AddOutputOperand<HalPolicy>(model, outputStateOutDimensions);
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    AddOutputOperand<HalPolicy>(model, cellStateOutDimensions);
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    AddOutputOperand<HalPolicy>(model, outputDimensions);

    // make the lstm operation
    model.operations.resize(1);
    model.operations[0].type = HalPolicy::OperationType::LSTM;

    if (normalizationEnabled)
    {
        model.operations[0].inputs = hidl_vec<uint32_t> { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                                                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
        model.operations[0].outputs = hidl_vec<uint32_t> {27, 28, 29, 30};
    }
    else
    {
        model.operations[0].inputs = hidl_vec<uint32_t> { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                                                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
        model.operations[0].outputs = hidl_vec<uint32_t> {23, 24, 25, 26};
    }

    // define the input values
    hidl_vec<RequestArgument> inputArguments;
    inputArguments.resize(3);

    inputArguments[0] = CreateRequestArgument<float>(inputValue, 0);
    inputArguments[1] = CreateRequestArgument<float>(outputStateInValue, 1);
    inputArguments[2] = CreateRequestArgument<float>(cellStateInValue, 2);

    // define the expected output values
    hidl_vec<RequestArgument> outputArguments;
    outputArguments.resize(4);

    outputArguments[0] = CreateRequestArgument<float>(scratchBufferValue, 3);
    outputArguments[1] = CreateRequestArgument<float>(outputStateOutValue, 4);
    outputArguments[2] = CreateRequestArgument<float>(cellStateOutValue, 5);
    outputArguments[3] = CreateRequestArgument<float>(outputValue, 6);

    V1_0::Request request = {};
    request.inputs  = inputArguments;
    request.outputs = outputArguments;

    // set the input data
    AddPoolAndSetData(inputValue.size(), request, inputValue.data());
    AddPoolAndSetData(outputStateInValue.size(), request, outputStateInValue.data());
    AddPoolAndSetData(cellStateInValue.size(), request, cellStateInValue.data());

    // add memory for the outputs
    AddPoolAndGetData<float>(scratchBufferValue.size(), request);
    android::sp<IMemory> outputStateOutMemory = AddPoolAndGetData<float>(outputStateOutValue.size(), request);
    float* outputStateOutData = static_cast<float*>(static_cast<void*>(outputStateOutMemory->getPointer()));
    android::sp<IMemory> cellStateOutMemory = AddPoolAndGetData<float>(cellStateOutValue.size(), request);
    float* cellStateOutData = static_cast<float*>(static_cast<void*>(cellStateOutMemory->getPointer()));
    android::sp<IMemory> outputMemory = AddPoolAndGetData<float>(outputValue.size(), request);
    float* outputData = static_cast<float*>(static_cast<void*>(outputMemory->getPointer()));

    // make the prepared model and run the execution
    ExecuteModel(model, *driver, request);

    // check the results
    for (size_t i = 0; i < outputStateOutValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputStateOutValue[i] == doctest::Approx( outputStateOutData[i] ),
                              "outputStateOut[" << i << "]: " << outputStateOutValue[i] << " != "
                              << outputStateOutData[i]);
    }
    for (size_t i = 0; i < cellStateOutValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(cellStateOutValue[i] == doctest::Approx( cellStateOutData[i] ),
                              "cellStateOutValue[" << i << "]: " << cellStateOutValue[i] << " != "
                              << cellStateOutData[i]);
    }
    for (size_t i = 0; i < outputValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputValue[i] == doctest::Approx( outputData[i] ),
                              "outputValue[" << i << "]: " << outputValue[i] << " != " << outputData[i]);
    }
}

template <typename HalPolicy>
void QuantizedLstmTestImpl(const hidl_vec<uint32_t>&    inputDimensions,
                           const std::vector<uint8_t>&  inputValue,
                           const hidl_vec<uint32_t>&    inputToInputWeightsDimensions,
                           const std::vector<uint8_t>&  inputToInputWeightsValue,
                           const hidl_vec<uint32_t>&    inputToForgetWeightsDimensions,
                           const std::vector<uint8_t>&  inputToForgetWeightsValue,
                           const hidl_vec<uint32_t>&    inputToCellWeightsDimensions,
                           const std::vector<uint8_t>&  inputToCellWeightsValue,
                           const hidl_vec<uint32_t>&    inputToOutputWeightsDimensions,
                           const std::vector<uint8_t>&  inputToOutputWeightsValue,
                           const hidl_vec<uint32_t>&    recurrentToInputWeightsDimensions,
                           const std::vector<uint8_t>&  recurrentToInputWeightsValue,
                           const hidl_vec<uint32_t>&    recurrentToForgetWeightsDimensions,
                           const std::vector<uint8_t>&  recurrentToForgetWeightsValue,
                           const hidl_vec<uint32_t>&    recurrentToCellWeightsDimensions,
                           const std::vector<uint8_t>&  recurrentToCellWeightsValue,
                           const hidl_vec<uint32_t>&    recurrentToOutputWeightsDimensions,
                           const std::vector<uint8_t>&  recurrentToOutputWeightsValue,
                           const hidl_vec<uint32_t>&    inputGateBiasDimensions,
                           const std::vector<int32_t>&  inputGateBiasValue,
                           const hidl_vec<uint32_t>&    forgetGateBiasDimensions,
                           const std::vector<int32_t>&  forgetGateBiasValue,
                           const hidl_vec<uint32_t>&    cellBiasDimensions,
                           const std::vector<int32_t>&  cellBiasValue,
                           const hidl_vec<uint32_t>&    outputGateBiasDimensions,
                           const std::vector<int32_t>&  outputGateBiasValue,
                           const hidl_vec<uint32_t>&    previousOutputInDimensions,
                           const std::vector<uint8_t>&  previousOutputInValue,
                           const hidl_vec<uint32_t>&    previousCellStateInDimensions,
                           const std::vector<int16_t>&  previousCellStateInValue,
                           const hidl_vec<uint32_t>&    cellStateOutDimensions,
                           const std::vector<int16_t>&  cellStateOutValue,
                           const hidl_vec<uint32_t>&    outputDimensions,
                           const std::vector<uint8_t>&  outputValue)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::GpuAcc));
    using Model = typename HalPolicy::Model;
    Model model = {};

    float inputOutputScale = 0.0078125f;
    int32_t inputOutputOffset = 128;

    float cellStateScale = 0.00048828125f;
    int32_t cellStateOffset = 0;

    float weightsScale = 0.00408021f;
    int32_t weightsOffset = 100;

    float biasScale = 3.1876640625e-05f;
    int32_t biasOffset = 0;

    // Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    AddInputOperand<HalPolicy>(model,
                               inputDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                               inputOutputScale,
                               inputOutputOffset);

    // 1: The input-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-input part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                inputToInputWeightsDimensions,
                                inputToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(inputToInputWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 2: The input-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-forget part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                inputToForgetWeightsDimensions,
                                inputToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(inputToForgetWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 3: The input-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-cell part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                inputToCellWeightsDimensions,
                                inputToCellWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(inputToCellWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 4: The input-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-output part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                inputToOutputWeightsDimensions,
                                inputToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(inputToOutputWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 5: The recurrent-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-input part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                recurrentToInputWeightsDimensions,
                                recurrentToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(recurrentToInputWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 6: The recurrent-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-forget part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                recurrentToForgetWeightsDimensions,
                                recurrentToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(recurrentToForgetWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 7: The recurrent-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-cell part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                recurrentToCellWeightsDimensions,
                                recurrentToCellWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(recurrentToCellWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 8: The recurrent-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-output part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    AddTensorOperand<HalPolicy>(model,
                                recurrentToOutputWeightsDimensions,
                                recurrentToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                CreateNoValueLifeTime(recurrentToOutputWeightsDimensions),
                                weightsScale,
                                weightsOffset);
    // 9: The input gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the
    //    bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    AddTensorOperand<HalPolicy>(model,
                                inputGateBiasDimensions,
                                inputGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(inputGateBiasDimensions),
                                biasScale,
                                biasOffset);
    // 10: The forget gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    AddTensorOperand<HalPolicy>(model,
                                forgetGateBiasDimensions,
                                forgetGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(forgetGateBiasDimensions),
                                biasScale,
                                biasOffset);
    // 11: The cell bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the bias
    //     for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product of input
    //     and weights scales and zeroPoint equal to 0.
    AddTensorOperand<HalPolicy>(model,
                                cellBiasDimensions,
                                cellBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(cellBiasDimensions),
                                biasScale,
                                biasOffset);
    // 12: The output gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    AddTensorOperand<HalPolicy>(model,
                                outputGateBiasDimensions,
                                outputGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(outputGateBiasDimensions),
                                biasScale,
                                biasOffset);

    //13: The previous cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape
    //    [numBatches, outputSize] specifying the cell state from the previous time step of the LSTM cell.
    //    It is quantized using a quantization range of -2^4, 2^4 * 32767/32768.
    AddInputOperand<HalPolicy>(model,
                               previousCellStateInDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                               cellStateScale,
                               cellStateOffset);
    // 14: The previous output state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //     [numBathes, outputSize] specifying the output of the LSTM cell from previous time-step. Tensor
    //     is quantized with a fixed quantization range of -1, 127/128.
    AddInputOperand<HalPolicy>(model,
                               previousOutputInDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                               inputOutputScale,
                               inputOutputOffset);

    // Outputs:
    // 0: The cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape [numBatches, outputSize]
    //    which contains a cell state from the current time step. Tensor is quantized using a quantization range
    //    of -2^4, 2^4 * 32767/32768.
    AddOutputOperand<HalPolicy>(model,
                                cellStateOutDimensions,
                                HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                cellStateScale,
                                cellStateOffset);
    // 1: The output: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBathes, outputSize] which
    //      contains the output value. Tensor is quantized with a fixed quantization range of -1, 127/128.
    AddOutputOperand<HalPolicy>(model,
                                outputDimensions,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                inputOutputScale,
                                inputOutputOffset);

    // make the lstm operation
    model.operations.resize(1);
    model.operations[0].type = HalPolicy::OperationType::QUANTIZED_16BIT_LSTM;

    model.operations[0].inputs = hidl_vec<uint32_t> { 0,  1,  2,  3,  4,  5,  6,  7,
                                                      8,  9,  10, 11, 12, 13, 14};
    model.operations[0].outputs = hidl_vec<uint32_t> {15, 16};

    // define the input values
    hidl_vec<RequestArgument> inputArguments;
    inputArguments.resize(3);

    inputArguments[0] = CreateRequestArgument<uint8_t>(inputValue, 0);
    inputArguments[1] = CreateRequestArgument<int16_t>(previousCellStateInValue, 1);
    inputArguments[2] = CreateRequestArgument<uint8_t>(previousOutputInValue, 2);

    // define the expected output values
    hidl_vec<RequestArgument> outputArguments;
    outputArguments.resize(2);

    outputArguments[0] = CreateRequestArgument<int16_t>(cellStateOutValue, 3);
    outputArguments[1] = CreateRequestArgument<uint8_t>(outputValue, 4);

    V1_0::Request request = {};
    request.inputs  = inputArguments;
    request.outputs = outputArguments;

    // set the input data
    AddPoolAndSetData(inputValue.size(), request, inputValue.data());
    AddPoolAndSetData(previousCellStateInValue.size(), request, previousCellStateInValue.data());
    AddPoolAndSetData(previousOutputInValue.size(), request, previousOutputInValue.data());

    // add memory for the outputs
    android::sp<IMemory> cellStateOutMemory = AddPoolAndGetData<int16_t>(cellStateOutValue.size(), request);
    int16_t* cellStateOutData = static_cast<int16_t*>(static_cast<void*>(cellStateOutMemory->getPointer()));
    android::sp<IMemory> outputMemory = AddPoolAndGetData<uint8_t>(outputValue.size(), request);
    uint8_t* outputData = static_cast<uint8_t*>(static_cast<void*>(outputMemory->getPointer()));

    // make the prepared model and run the execution
    ExecuteModel(model, *driver, request);

    // check the results
    for (size_t i = 0; i < cellStateOutValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(cellStateOutValue[i] == doctest::Approx( cellStateOutData[i] ),
                              "cellStateOutValue[" << i << "]: " << cellStateOutValue[i] << " != "
                              << cellStateOutData[i]);
    }
    for (size_t i = 0; i < outputValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputValue[i] == doctest::Approx( outputData[i] ),
                              "outputValue[" << i << "]: " << outputValue[i] << " != " << outputData[i]);
    }
}

template <typename HalPolicy>
void LstmNoCifgNoPeepholeNoProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/lstm.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/lstm.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 1;
    uint32_t inputSize = 2;
    uint32_t numUnits = 4;
    uint32_t outputSize = numUnits;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{2.0f, 3.0f};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{-0.45018822f, -0.02338299f,
                                                -0.08705890f, -0.34550029f,
                                                 0.04266912f, -0.15680569f,
                                                -0.34856534f,  0.43890524f};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{ 0.09701663f,  0.20334584f,
                                                 -0.50592935f, -0.31343272f,
                                                 -0.40032279f,  0.44781327f,
                                                  0.01387155f, -0.35593212f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.50013041f,  0.13702840f,
                                                0.11810488f,  0.20131630f,
                                               -0.20583314f,  0.44344562f,
                                                0.22077113f, -0.29909778f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{-0.25065863f, -0.28290087f,
                                                  0.04613829f,  0.40525138f,
                                                  0.44272184f,  0.03897077f,
                                                 -0.15568960f,  0.19487578f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{-0.00635350f, -0.20423880f,  0.31454784f, -0.35746509f,
                                                     0.28902304f,  0.08183324f, -0.16555229f,  0.02286911f,
                                                    -0.13566875f,  0.03034258f,  0.48091322f, -0.12528998f,
                                                     0.24077177f, -0.51332325f, -0.33502164f,  0.10629296f};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.48684245f, -0.06655136f,  0.42224967f,  0.21126390f,
                                                      0.27654213f,  0.20864892f, -0.07646349f,  0.45877004f,
                                                      0.00141793f, -0.14609534f,  0.36447752f,  0.09196436f,
                                                      0.28053468f,  0.01560611f, -0.20127171f, -0.01140004f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.34074140f,  0.24443203f, -0.20785320f,  0.26320225f,
                                                    0.05695659f, -0.00123841f, -0.47447860f, -0.35869038f,
                                                   -0.06418842f, -0.13502428f, -0.50176400f,  0.22830659f,
                                                   -0.46367589f,  0.26016325f, -0.03894562f, -0.16368064f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{ 0.43385774f, -0.17194885f,  0.27182370f,  0.09215671f,
                                                      0.24107647f, -0.39835793f,  0.18212086f,  0.01301402f,
                                                      0.48572797f, -0.50656658f,  0.20047462f, -0.20607421f,
                                                     -0.51818722f, -0.15390486f,  0.04681480f,  0.39922136f};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<float> cellToInputWeightsValue;
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{0};
    std::vector<float> cellToForgetWeightsValue;
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{0};
    std::vector<float> cellToOutputWeightsValue;
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<float> inputGateBiasValue(numUnits, 0.0f);
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue(numUnits, 1.0f);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue(numUnits, 0.0f);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue(numUnits, 0.0f);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{0};
    std::vector<float> projectionWeightsValue;
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<float> projectionBiasValue;

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 24:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 25:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 26:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 4};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 4, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue {-0.0297319f, 0.122947f, 0.208851f, -0.153588f};
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue {-0.145439f, 0.157475f, 0.293663f, -0.277353f};
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue {-0.02973187f, 0.1229473f, 0.20885126f, -0.15358765f};

    LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                            inputToInputWeightsDimensions,         inputToInputWeightsValue,
                            inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                            inputToCellWeightsDimensions,          inputToCellWeightsValue,
                            inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                            recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                            recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                            recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                            recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                            cellToInputWeightsDimensions,          cellToInputWeightsValue,
                            cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                            cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                            inputGateBiasDimensions,               inputGateBiasValue,
                            forgetGateBiasDimensions,              forgetGateBiasValue,
                            cellBiasDimensions,                    cellBiasValue,
                            outputGateBiasDimensions,              outputGateBiasValue,
                            projectionWeightsDimensions,           projectionWeightsValue,
                            projectionBiasDimensions,              projectionBiasValue,
                            outputStateInDimensions,               outputStateInValue,
                            cellStateInDimensions,                 cellStateInValue,
                            activationFunctionDimensions,          activationFunctionValue,
                            cellClippingThresholdDimensions,       cellClippingThresholdValue,
                            projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                            inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                            forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                            cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                            outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                            scratchBufferDimensions,               scratchBufferValue,
                            outputStateOutDimensions,              outputStateOutValue,
                            cellStateOutDimensions,                cellStateOutValue,
                            outputDimensions,                      outputValue,
                            compute);
}

template <typename HalPolicy>
void LstmCifgPeepholeNoProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/lstm2.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/lstm2.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 1;
    uint32_t inputSize = 2;
    uint32_t numUnits = 4;
    uint32_t outputSize = numUnits;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{2.0f, 3.0f};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{0};
    std::vector<float> inputToInputWeightsValue;
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.55291498f, -0.42866567f,
                                                  0.13056988f, -0.36333650f,
                                                 -0.22755712f,  0.28253698f,
                                                  0.24407166f,  0.33826375f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.49770179f, -0.27711356f,
                                               -0.09624726f,  0.05100781f,
                                                0.04717243f,  0.48944736f,
                                               -0.38535351f, -0.17212132f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{ 0.10725588f, -0.02335852f,
                                                 -0.55932593f, -0.09426838f,
                                                 -0.44257352f,  0.54939759f,
                                                  0.01533556f,  0.42751634f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0}; // VTS was {4, 4} -> {0} ?
    std::vector<float> recurrentToInputWeightsValue;
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.13832897f, -0.05151010f, -0.23590070f, -0.16661474f,
                                                     -0.14340827f,  0.36986142f,  0.23414481f,  0.55899000f,
                                                      0.10798943f, -0.41174671f,  0.17751795f, -0.34484994f,
                                                     -0.35874045f, -0.11352962f,  0.27268326f,  0.54058349f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{ 0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f,
                                                    0.42957711f,  0.01841056f, -0.32764608f, -0.33027974f,
                                                   -0.10826075f,  0.20675004f,  0.19069612f, -0.03026325f,
                                                   -0.54532051f,  0.33003211f,  0.44901288f,  0.21193194f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{0.41613156f,  0.42610586f, -0.16495961f, -0.56638730f,
                                                     0.30579174f, -0.05115908f, -0.33941799f,  0.23364776f,
                                                     0.11178309f,  0.09481031f, -0.26424935f,  0.46261835f,
                                                     0.50248802f,  0.26114327f, -0.43736315f,  0.33149987f};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<float> cellToInputWeightsValue;
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{4};
    std::vector<float> cellToForgetWeightsValue{0.47485286f, -0.51955009f, -0.24458408f, 0.31544167f};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{4};
    std::vector<float> cellToOutputWeightsValue{-0.17135078f, 0.82760304f, 0.85573703f, -0.77109635f};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{0}; // VTS was {4} -> {0} ?
    std::vector<float> inputGateBiasValue;
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue(numUnits, 1.0f);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue(numUnits, 0.0f);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue(numUnits, 0.0f);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{0};
    std::vector<float> projectionWeightsValue;
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<float> projectionBiasValue;

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 24:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 25:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 26:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 3};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 3, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue{-0.364445f, -0.00352185f, 0.128866f, -0.0516365f};
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue{-0.760444f, -0.0180416f, 0.182264f, -0.0649371f};
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue{-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f};

    LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                            inputToInputWeightsDimensions,         inputToInputWeightsValue,
                            inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                            inputToCellWeightsDimensions,          inputToCellWeightsValue,
                            inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                            recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                            recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                            recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                            recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                            cellToInputWeightsDimensions,          cellToInputWeightsValue,
                            cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                            cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                            inputGateBiasDimensions,               inputGateBiasValue,
                            forgetGateBiasDimensions,              forgetGateBiasValue,
                            cellBiasDimensions,                    cellBiasValue,
                            outputGateBiasDimensions,              outputGateBiasValue,
                            projectionWeightsDimensions,           projectionWeightsValue,
                            projectionBiasDimensions,              projectionBiasValue,
                            outputStateInDimensions,               outputStateInValue,
                            cellStateInDimensions,                 cellStateInValue,
                            activationFunctionDimensions,          activationFunctionValue,
                            cellClippingThresholdDimensions,       cellClippingThresholdValue,
                            projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                            inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                            forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                            cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                            outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                            scratchBufferDimensions,               scratchBufferValue,
                            outputStateOutDimensions,              outputStateOutValue,
                            cellStateOutDimensions,                cellStateOutValue,
                            outputDimensions,                      outputValue,
                            compute);
}

template <typename HalPolicy>
void LstmNoCifgPeepholeProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/lstm3.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/lstm3.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 2;
    uint32_t inputSize = 5;
    uint32_t numUnits = 20;
    uint32_t outputSize = 16;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{0.787926f, 0.151646f, 0.071352f, 0.118426f, 0.458058f,
                                  0.295743f, 0.544053f, 0.690064f, 0.858138f, 0.497181f};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue
    {
         0.0213936830f,  0.0612455100f,  0.0469051670f, -0.0146576770f, -0.0314946300f,
         0.0917180300f,  0.1464780100f,  0.1079719300f, -0.0057968358f,  0.0019193048f,
        -0.2726754000f,  0.1015402900f, -0.0185398850f,  0.0803498850f, -0.1026238500f,
        -0.0225997870f, -0.0912115500f, -0.0086759670f, -0.0452061030f, -0.0821282000f,
        -0.0080459520f,  0.0154780810f,  0.0552172470f,  0.0387195870f,  0.0441536270f,
        -0.0645324300f,  0.0503182500f, -0.0469351080f, -0.0081644309f,  0.0145742260f,
        -0.1671009000f, -0.1551955200f, -0.1681979700f, -0.1397126900f, -0.1195305900f,
         0.2500548700f, -0.2279098300f,  0.0098550870f, -0.0281409580f, -0.1120069800f,
         0.1129540800f, -0.0035217577f,  0.0544850750f,  0.0518469500f,  0.0647112060f,
         0.1098919300f,  0.1167478600f,  0.0349060700f,  0.0772735700f,  0.1139058500f,
        -0.1863375000f, -0.1034451000f, -0.1394518900f, -0.0494012270f, -0.1876706300f,
         0.0424839030f,  0.1423355200f,  0.1383258100f,  0.1835016500f,  0.1454560300f,
        -0.0285457040f,  0.0249395310f,  0.0509297180f,  0.0076203286f, -0.0029723682f,
        -0.0424842240f, -0.1182759600f, -0.0917110400f, -0.1080862800f, -0.1632798800f,
        -0.2273378000f, -0.0993647000f, -0.0171551070f,  0.0023917493f,  0.0492727640f,
         0.0038534778f,  0.0547645050f,  0.0897537840f,  0.0694723400f,  0.0801447600f,
        -0.0454423400f, -0.0497073000f, -0.0713563100f, -0.0489291060f, -0.0040420120f,
        -0.0092840260f,  0.0180420540f,  0.0036860977f, -0.0742730200f, -0.1143460400f,
        -0.0189954560f,  0.0314875430f,  0.0128349080f,  0.0199777540f,  0.0442566540f,
        -0.3929261300f, -0.1851933400f, -0.1165128100f, -0.0680989200f,  0.0113736770f
    };
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue
    {
        -0.0018401089f, -0.0048522370f,  0.0369842400f,  0.0141817040f,  0.0282732360f,
        -0.0167261940f, -0.0524975900f, -0.1020426100f,  0.0086106600f, -0.0409795050f,
        -0.0098991870f,  0.0192389200f, -0.0281772690f, -0.0853510300f, -0.1458549500f,
         0.1066256700f, -0.0190973100f, -0.0178835340f, -0.0047269356f, -0.0451033230f,
         0.0030784295f,  0.0767847750f,  0.0746369600f,  0.0945313950f,  0.0814421000f,
        -0.1225789900f, -0.0339457580f, -0.0313034650f,  0.0456306260f,  0.0684388700f,
        -0.1349294500f, -0.0124800070f, -0.0811829000f, -0.0722449900f, -0.0962879100f,
         0.0451009460f,  0.0012300825f,  0.0139646620f,  0.0993723940f,  0.0254305900f,
         0.0695832400f,  0.0342572960f,  0.0482646000f,  0.0626799700f,  0.0526250680f,
         0.1278466600f,  0.0707789700f,  0.0257259350f,  0.0416500900f,  0.0724190500f,
         0.0186686440f, -0.0373772940f, -0.0627778300f, -0.0883363600f, -0.0401206050f,
        -0.0114055860f, -0.0078083350f, -0.0103013860f, -0.0051021670f,  0.0277174640f,
         0.0548342300f,  0.1144911100f,  0.1128965200f,  0.1093983900f,  0.1339650600f,
        -0.0840216600f, -0.0190146200f, -0.0446783040f, -0.0772056500f,  0.0143500630f,
        -0.1175795800f, -0.0652038000f, -0.0818573300f, -0.0767543240f, -0.0926143750f,
         0.1040549100f,  0.0529603360f,  0.0357558950f,  0.0358393860f, -0.0125405530f,
         0.0368812980f,  0.0291337600f,  0.0342015900f,  0.0544844700f, -0.0545233530f,
         0.0258271500f,  0.0232735500f, -0.0118571790f, -0.0011980024f, -0.0346417170f,
        -0.0261250940f, -0.1758261500f, -0.1592365700f, -0.2748677400f, -0.0006143371f,
         0.0001771948f, -8.470171e-05f,  0.0265180700f,  0.0457907650f,  0.069564960f
    };
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue
    {
        -0.0458028300f, -0.0954946200f, -0.0324189850f, -0.0645463300f, -0.0435284530f,
         0.0430185870f, -0.0491523440f, -0.1241814400f, -0.0789854750f, -0.0759688900f,
         0.0194843620f, -0.1143496200f, -0.0074034138f, -0.0631484400f, -0.0929814950f,
         0.0062155537f, -0.0250343380f, -0.0028890965f,  0.0489295270f,  0.0623507500f,
         0.1066591800f, -0.0320367920f, -0.0850591600f, -0.1084335800f, -0.1300243300f,
        -0.0368164370f, -0.0213013400f, -0.0165182390f,  0.0047691227f, -0.0025825808f,
         0.0660178660f,  0.0299915340f, -0.1065283600f, -0.1037554000f, -0.1305607100f,
        -0.0326664300f, -0.0337024140f, -0.0064734240f, -0.0461169200f,  0.0144193390f,
        -0.0251743230f,  0.0396852000f,  0.0817775060f,  0.0615746800f,  0.1021009500f,
        -0.0096581940f,  0.0465117170f,  0.0360390600f,  0.0069369148f,  0.0159600950f,
        -0.0650766600f,  0.0955159800f,  0.0535688360f,  0.0640871400f,  0.1283566700f,
        -0.0087143290f, -0.2021196600f, -0.1209367400f,  0.0294504720f,  0.2849013000f,
        -0.0292279010f,  0.1164364000f, -0.0856026300f,  0.0994178600f, -0.0369995650f,
        -0.0288426260f, -0.0033637602f, -0.0170129020f, -0.0972086500f, -0.1119335100f,
        -0.0291551170f, -0.0179360340f, -0.0097689360f, -0.0422332400f, -0.0361596350f,
         0.0650511200f, -0.0217428920f, -0.0233772120f, -0.0722136400f, -0.0643055200f,
         0.0545386500f,  0.0911498140f,  0.0638733100f,  0.0075183930f,  0.0559609530f,
         0.0697793440f,  0.0464111680f,  0.1050991100f,  0.0746389400f,  0.0075130584f,
         0.0128509820f,  0.0455543100f,  0.0569556880f,  0.0655528500f,  0.0508014560f,
        -0.0098626830f,  0.0082677200f, -0.0265556090f, -0.0073611983f, -0.0014897042f
    };
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue
    {
        -0.0998932000f, -0.0720195600f, -0.0528037730f, -0.1562959300f, -0.1500191800f,
        -0.0765075100f,  0.0235985500f, -0.0751553550f, -0.0803770900f, -0.1509353400f,
         0.0295175520f, -0.0475139300f,  0.0103505310f, -0.0266485100f, -0.0168397220f,
        -0.0231211630f,  0.0077019283f,  0.0128512570f, -0.0504064900f, -0.0129761000f,
        -0.0217377470f, -0.0383057930f, -0.0687058600f, -0.0148124700f, -0.0012853940f,
         0.1012423600f,  0.0831228350f,  0.0533130060f, -0.0622356460f, -0.0756371540f,
        -0.0278339030f,  0.0297749710f,  0.1130802000f,  0.0921890600f,  0.0950613500f,
        -0.0866657640f, -0.0371627060f, -0.0388809140f, -0.0358328450f, -0.0144815640f,
        -0.0982500300f, -0.1204856900f, -0.0976655860f, -0.0528763300f, -0.0964047000f,
        -0.1136642900f,  0.0357775050f,  0.1356881900f,  0.0524513830f,  0.0506493040f,
         0.0579895100f, -0.0218523350f, -0.0998488440f,  0.0147404750f, -0.0788979460f,
         0.0497469900f,  0.0141604730f,  0.0697393200f,  0.0496494200f,  0.0333646460f,
         0.0819012400f,  0.0255353670f,  0.0508931650f,  0.0485142540f,  0.0694581300f,
        -0.0789075640f, -0.0670761600f, -0.1184450800f, -0.0998668800f, -0.0750940300f,
         0.0626322600f,  0.1492558700f,  0.2018843600f,  0.1209845100f,  0.1463941500f,
         0.0015017595f, -0.0142673820f, -0.0341725700f,  0.0127114680f,  0.0028300495f,
        -0.0247584820f, -0.0509854800f, -0.0821182000f,  0.0142256720f,  0.0215441580f,
         0.0894972500f,  0.0750526800f, -0.0020780868f,  0.0490825800f,  0.0647629500f,
        -0.0229070630f,  0.0275624560f,  0.0401857350f,  0.0195675770f, -0.0155987390f,
        -0.0490973030f, -0.0171218660f, -0.0833682340f, -0.0233200200f, -0.084095600f
    };
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue
    {
        -0.001374326f, -0.078856036f, 0.10672688f, 0.029162422f,        // 00
        -0.11585556f, 0.02557986f, -0.13446963f, -0.035785314f,
        -0.01244275f, 0.025961924f, -0.02337298f, -0.044228926f,
        -0.055839065f, -0.046598054f, -0.010546039f, -0.06900766f,
         0.027239809f, 0.022582639f, -0.013296484f, -0.05459212f,        // 01
         0.08981f, -0.045407712f, 0.08682226f, -0.06867011f,
        -0.14390695f, -0.02916037f, 0.000996957f, 0.091420636f,
         0.14283475f, -0.07390571f, -0.06402044f, 0.062524505f,
        -0.093129106f, 0.04860203f, -0.08364217f, -0.08119002f,         // 02
         0.009352075f, 0.22920375f, 0.0016303885f, 0.11583097f,
        -0.13732095f, 0.012405723f, -0.07551853f, 0.06343048f,
         0.12162708f, -0.031923793f, -0.014335606f, 0.01790974f,
        -0.10650317f, -0.0724401f, 0.08554849f, -0.05727212f,           // 03
         0.06556731f, -0.042729504f, -0.043227166f, 0.011683251f,
        -0.013082158f, -0.029302018f, -0.010899579f, -0.062036745f,
        -0.022509435f, -0.00964907f, -0.01567329f, 0.04260106f,
        -0.07787477f, -0.11576462f, 0.017356863f, 0.048673786f,         // 04
        -0.017577527f, -0.05527947f, -0.082487635f, -0.040137455f,
        -0.10820036f, -0.04666372f, 0.022746278f, -0.07851417f,
         0.01068115f, 0.032956902f, 0.022433773f, 0.0026891115f,
         0.08944216f, -0.0685835f, 0.010513544f, 0.07228705f,            // 05
         0.02032331f, -0.059686817f, -0.0005566496f, -0.086984694f,
         0.040414046f, -0.1380399f, 0.094208956f, -0.05722982f,
         0.012092817f, -0.04989123f, -0.086576f, -0.003399834f,
        -0.04696032f, -0.045747425f, 0.10091314f, 0.048676282f,         // 06
        -0.029037097f, 0.031399418f, -0.0040285117f, 0.047237843f,
         0.09504992f, 0.041799378f, -0.049185462f, -0.031518843f,
        -0.10516937f, 0.026374253f, 0.10058866f, -0.0033195973f,
        -0.041975245f, 0.0073591834f, 0.0033782164f, -0.004325073f,     // 07
        -0.10167381f, 0.042500053f, -0.01447153f, 0.06464186f,
        -0.017142897f, 0.03312627f, 0.009205989f, 0.024138335f,
        -0.011337001f, 0.035530265f, -0.010912711f, 0.0706555f,
        -0.005894094f, 0.051841937f, -0.1401738f, -0.02351249f,         // 08
         0.0365468f, 0.07590991f, 0.08838724f, 0.021681072f,
        -0.10086113f, 0.019608743f, -0.06195883f, 0.077335775f,
         0.023646897f, -0.095322326f, 0.02233014f, 0.09756986f,
        -0.048691444f, -0.009579111f, 0.07595467f, 0.11480546f,         // 09
        -0.09801813f, 0.019894179f, 0.08502348f, 0.004032281f,
         0.037211012f, 0.068537936f, -0.048005626f, -0.091520436f,
        -0.028379958f, -0.01556313f, 0.06554592f, -0.045599163f,
        -0.01672207f, -0.020169014f, -0.011877351f, -0.20212261f,       // 10
         0.010889619f, 0.0047078193f, 0.038385306f, 0.08540671f,
        -0.017140968f, -0.0035865551f, 0.016678626f, 0.005633034f,
         0.015963363f, 0.00871737f, 0.060130805f, 0.028611384f,
         0.10109069f, -0.015060172f, -0.07894427f, 0.06401885f,          // 11
         0.011584063f, -0.024466386f, 0.0047652307f, -0.09041358f,
         0.030737216f, -0.0046374933f, 0.14215417f, -0.11823516f,
         0.019899689f, 0.006106124f, -0.027092824f, 0.0786356f,
         0.05052217f, -0.058925f, -0.011402121f, -0.024987547f,          // 12
        -0.0013661642f, -0.06832946f, -0.015667673f, -0.1083353f,
        -0.00096863037f, -0.06988685f, -0.053350925f, -0.027275559f,
        -0.033664223f, -0.07978348f, -0.025200296f, -0.017207067f,
        -0.058403496f, -0.055697463f, 0.005798788f, 0.12965427f,        // 13
        -0.062582195f, 0.0013350133f, -0.10482091f, 0.0379771f,
         0.072521195f, -0.0029455067f, -0.13797039f, -0.03628521f,
         0.013806405f, -0.017858358f, -0.01008298f, -0.07700066f,
        -0.017081132f, 0.019358726f, 0.0027079724f, 0.004635139f,       // 14
         0.062634714f, -0.02338735f, -0.039547626f, -0.02050681f,
         0.03385117f, -0.083611414f, 0.002862572f, -0.09421313f,
         0.058618143f, -0.08598433f, 0.00972939f, 0.023867095f,
        -0.053934585f, -0.023203006f, 0.07452513f, -0.048767887f,       // 15
        -0.07314807f, -0.056307215f, -0.10433547f, -0.06440842f,
         0.04328182f, 0.04389765f, -0.020006588f, -0.09076438f,
        -0.11652589f, -0.021705797f, 0.03345259f, -0.010329105f,
        -0.025767034f, 0.013057034f, -0.07316461f, -0.10145612f,        // 16
         0.06358255f, 0.18531723f, 0.07759293f, 0.12006465f,
         0.1305557f, 0.058638252f, -0.03393652f, 0.09622831f,
        -0.16253184f, -2.4580743e-06f, 0.079869635f, -0.070196845f,
        -0.005644518f, 0.06857898f, -0.12598175f, -0.035084512f,        // 17
         0.03156317f, -0.12794146f, -0.031963028f, 0.04692781f,
         0.030070418f, 0.0071660685f, -0.095516115f, -0.004643372f,
         0.040170413f, -0.062104587f, -0.0037324072f, 0.0554317f,
         0.08184801f, -0.019164372f, 0.06791302f, 0.034257166f,          // 18
        -0.10307039f, 0.021943003f, 0.046745934f, 0.0790918f,
        -0.0265588f, -0.007824208f, 0.042546265f, -0.00977924f,
        -0.0002440307f, -0.017384544f, -0.017990116f, 0.12252321f,
        -0.014512694f, -0.08251313f, 0.08861942f, 0.13589665f,          // 19
         0.026351685f, 0.012641483f, 0.07466548f, 0.044301085f,
        -0.045414884f, -0.051112458f, 0.03444247f, -0.08502782f,
        -0.04106223f, -0.028126027f, 0.028473156f, 0.10467447f
    };
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue
    {
        -0.057784554f, -0.026057621f, -0.068447545f, -0.022581743f,     // 00
         0.14811787f, 0.10826372f, 0.09471067f, 0.03987225f,
        -0.0039523416f, 0.00030638507f, 0.053185795f, 0.10572994f,
         0.08414449f, -0.022036452f, -0.00066928595f, -0.09203576f,
         0.032950465f, -0.10985798f, -0.023809856f, 0.0021431844f,       // 01
        -0.02196096f, -0.00326074f, 0.00058621005f, -0.074678116f,
        -0.06193199f, 0.055729095f, 0.03736828f, 0.020123724f,
         0.061878487f, -0.04729229f, 0.034919553f, -0.07585433f,
        -0.04421272f, -0.044019096f, 0.085488975f, 0.04058006f,         // 02
        -0.06890133f, -0.030951202f, -0.024628663f, -0.07672815f,
         0.034293607f, 0.08556707f, -0.05293577f, -0.033561368f,
        -0.04899627f, 0.0241671f, 0.015736353f, -0.095442444f,
        -0.029564252f, 0.016493602f, -0.035026584f, 0.022337519f,       // 03
        -0.026871363f, 0.004780428f, 0.0077918363f, -0.03601621f,
         0.016435321f, -0.03263031f, -0.09543275f, -0.047392778f,
         0.013454138f, 0.028934088f, 0.01685226f, -0.086110644f,
        -0.046250615f, -0.01847454f, 0.047608484f, 0.07339695f,         // 04
         0.034546845f, -0.04881143f, 0.009128804f, -0.08802852f,
         0.03761666f, 0.008096139f, -0.014454086f, 0.014361001f,
        -0.023502491f, -0.0011840804f, -0.07607001f, 0.001856849f,
        -0.06509276f, -0.006021153f, -0.08570962f, -0.1451793f,         // 05
         0.060212336f, 0.055259194f, 0.06974018f, 0.049454916f,
        -0.027794661f, -0.08077226f, -0.016179763f, 0.1169753f,
         0.17213494f, -0.0056326236f, -0.053934924f, -0.0124349f,
        -0.11520337f, 0.05409887f, 0.088759385f, 0.0019655675f,         // 06
         0.0042065294f, 0.03881498f, 0.019844765f, 0.041858196f,
        -0.05695512f, 0.047233116f, 0.038937137f, -0.06542224f,
         0.014429736f, -0.09719407f, 0.13908425f, -0.05379757f,
         0.012321099f, 0.082840554f, -0.029899208f, 0.044217527f,        // 07
         0.059855383f, 0.07711018f, -0.045319796f, 0.0948846f,
        -0.011724666f, -0.0033288454f, -0.033542685f, -0.04764985f,
        -0.13873616f, 0.040668588f, 0.034832682f, -0.015319203f,
        -0.018715994f, 0.046002675f, 0.0599172f, -0.043107376f,         // 08
         0.0294216f, -0.002314414f, -0.022424703f, 0.0030315618f,
         0.0014641669f, 0.0029166266f, -0.11878115f, 0.013738511f,
         0.12375372f, -0.0006038222f, 0.029104086f, 0.087442465f,
         0.052958444f, 0.07558703f, 0.04817258f, 0.044462286f,           // 09
        -0.015213451f, -0.08783778f, -0.0561384f, -0.003008196f,
         0.047060397f, -0.002058388f, 0.03429439f, -0.018839769f,
         0.024734668f, 0.024614193f, -0.042046934f, 0.09597743f,
        -0.0043254104f, 0.04320769f, 0.0064070094f, -0.0019131786f,     // 10
        -0.02558259f, -0.022822596f, -0.023273505f, -0.02464396f,
        -0.10991725f, -0.006240552f, 0.0074488563f, 0.024044557f,
         0.04383914f, -0.046476185f, 0.028658995f, 0.060410924f,
         0.050786525f, 0.009452605f, -0.0073054377f, -0.024810238f,      // 11
         0.0052906186f, 0.0066939713f, -0.0020913032f, 0.014515517f,
         0.015898481f, 0.021362653f, -0.030262267f, 0.016587038f,
        -0.011442813f, 0.041154444f, -0.007631438f, -0.03423484f,
        -0.010977775f, 0.036152758f, 0.0066366293f, 0.11915515f,        // 12
         0.02318443f, -0.041350313f, 0.021485701f, -0.10906167f,
        -0.028218046f, -0.00954771f, 0.020531068f, -0.11995105f,
        -0.03672871f, 0.024019798f, 0.014255957f, -0.05221243f,
        -0.00661567f, -0.04630967f, 0.033188973f, 0.10107534f,          // 13
        -0.014027541f, 0.030796422f, -0.10270911f, -0.035999842f,
         0.15443139f, 0.07684145f, 0.036571592f, -0.035900835f,
        -0.0034699554f, 0.06209149f, 0.015920248f, -0.031122351f,
        -0.03858649f, 0.01849943f, 0.13872518f, 0.01503974f,            // 14
         0.069941424f, -0.06948533f, -0.0088794185f, 0.061282158f,
        -0.047401894f, 0.03100163f, -0.041533746f, -0.10430945f,
         0.044574402f, -0.01425562f, -0.024290353f, 0.034563623f,
         0.05866852f, 0.023947537f, -0.09445152f, 0.035450947f,          // 15
         0.02247216f, -0.0042998926f, 0.061146557f, -0.10250651f,
         0.020881841f, -0.06747029f, 0.10062043f, -0.0023941975f,
         0.03532124f, -0.016341697f, 0.09685456f, -0.016764693f,
         0.051808182f, 0.05875331f, -0.04536488f, 0.001626336f,          // 16
        -0.028892258f, -0.01048663f, -0.009793449f, -0.017093895f,
         0.010987891f, 0.02357273f, -0.00010856845f, 0.0099760275f,
        -0.001845119f, -0.03551521f, 0.0018358806f, 0.05763657f,
        -0.01769146f, 0.040995963f, 0.02235177f, -0.060430344f,         // 17
         0.11475477f, -0.023854522f, 0.10071741f, 0.0686208f,
        -0.014250481f, 0.034261297f, 0.047418304f, 0.08562733f,
        -0.030519066f, 0.0060542435f, 0.014653856f, -0.038836084f,
         0.04096551f, 0.032249358f, -0.08355519f, -0.026823482f,         // 18
         0.056386515f, -0.010401743f, -0.028396193f, 0.08507674f,
         0.014410365f, 0.020995233f, 0.17040324f, 0.11511526f,
         0.02459721f, 0.0066619175f, 0.025853224f, -0.023133837f,
        -0.081302024f, 0.017264642f, -0.009585969f, 0.09491168f,        // 19
        -0.051313367f, 0.054532815f, -0.014298593f, 0.10657464f,
         0.007076659f, 0.10964551f, 0.0409152f, 0.008275321f,
        -0.07283536f, 0.07937492f, 0.04192024f, -0.1075027f
    };
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue
    {
        -0.037322544f, 0.018592842f, 0.0056175636f, -0.06253426f,
         0.055647098f, -0.05713207f, -0.05626563f, 0.005559383f,
         0.03375411f, -0.025757805f, -0.088049285f, 0.06017052f,
        -0.06570978f, 0.007384076f, 0.035123326f, -0.07920549f,
         0.053676967f, 0.044480428f, -0.07663568f, 0.0071805613f,
         0.08089997f, 0.05143358f, 0.038261272f, 0.03339287f,
        -0.027673481f, 0.044746667f, 0.028349208f, 0.020090483f,
        -0.019443132f, -0.030755889f, -0.0040000007f, 0.04465846f,
        -0.021585021f, 0.0031670958f, 0.0053199246f, -0.056117613f,
        -0.10893326f, 0.076739706f, -0.08509834f, -0.027997585f,
         0.037871376f, 0.01449768f, -0.09002357f, -0.06111149f,
        -0.046195522f, 0.0422062f, -0.005683705f, -0.1253618f,
        -0.012925729f, -0.04890792f, 0.06985068f, 0.037654128f,
         0.03398274f, -0.004781977f, 0.007032333f, -0.031787455f,
         0.010868644f, -0.031489216f, 0.09525667f, 0.013939797f,
         0.0058680447f, 0.0167067f, 0.02668468f, -0.04797466f,
        -0.048885044f, -0.12722108f, 0.035304096f, 0.06554885f,
         0.00972396f, -0.039238118f, -0.05159735f, -0.11329045f,
         0.1613692f, -0.03750952f, 0.06529313f, -0.071974665f,
        -0.11769596f, 0.015524369f, -0.0013754242f, -0.12446318f,
         0.02786344f, -0.014179351f, 0.005264273f, 0.14376344f,
         0.015983658f, 0.03406988f, -0.06939408f, 0.040699873f,
         0.02111075f, 0.09669095f, 0.041345075f, -0.08316494f,
        -0.07684199f, -0.045768797f, 0.032298047f, -0.041805092f,
         0.0119405f, 0.0061010392f, 0.12652606f, 0.0064572375f,
        -0.024950314f, 0.11574242f, 0.04508852f, -0.04335324f,
         0.06760663f, -0.027437469f, 0.07216407f, 0.06977076f,
        -0.05438599f, 0.034033038f, -0.028602652f, 0.05346137f,
         0.043184172f, -0.037189785f, 0.10420091f, 0.00882477f,
        -0.054019816f, -0.074273005f, -0.030617684f, -0.0028467078f,
         0.024302477f, -0.0038869337f, 0.005332455f, 0.0013399826f,
         0.04361412f, -0.007001822f, 0.09631092f, -0.06702025f,
        -0.042049985f, -0.035070654f, -0.04103342f, -0.10273396f,
         0.0544271f, 0.037184782f, -0.13150354f, -0.0058036847f,
        -0.008264958f, 0.042035464f, 0.05891794f, 0.029673764f,
         0.0063542654f, 0.044788733f, 0.054816857f, 0.062257513f,
        -0.00093483756f, 0.048938446f, -0.004952862f, -0.007730018f,
        -0.04043371f, -0.017094059f, 0.07229206f, -0.023670016f,
        -0.052195564f, -0.025616996f, -0.01520939f, 0.045104615f,
        -0.007376126f, 0.003533447f, 0.006570588f, 0.056037236f,
         0.12436656f, 0.051817212f, 0.028532185f, -0.08686856f,
         0.11868599f, 0.07663395f, -0.07323171f, 0.03463402f,
        -0.050708205f, -0.04458982f, -0.11590894f, 0.021273347f,
         0.1251325f, -0.15313013f, -0.12224372f, 0.17228661f,
         0.023029093f, 0.086124025f, 0.006445803f, -0.03496501f,
         0.028332196f, 0.04449512f, -0.042436164f, -0.026587414f,
        -0.006041347f, -0.09292539f, -0.05678812f, 0.03897832f,
         0.09465633f, 0.008115513f, -0.02171956f, 0.08304309f,
         0.071401566f, 0.019622514f, 0.032163795f, -0.004167056f,
         0.02295182f, 0.030739572f, 0.056506045f, 0.004612461f,
         0.06524936f, 0.059999723f, 0.046395954f, -0.0045512207f,
        -0.1335546f, -0.030136576f, 0.11584653f, -0.014678886f,
         0.0020118146f, -0.09688814f, -0.0790206f, 0.039770417f,
        -0.0329582f, 0.07922767f, 0.029322514f, 0.026405897f,
         0.04207835f, -0.07073373f, 0.063781224f, 0.0859677f,
        -0.10925287f, -0.07011058f, 0.048005477f, 0.03438226f,
        -0.09606514f, -0.006669445f, -0.043381985f, 0.04240257f,
        -0.06955775f, -0.06769346f, 0.043903265f, -0.026784198f,
        -0.017840602f, 0.024307009f, -0.040079936f, -0.019946516f,
         0.045318738f, -0.12233574f, 0.026170589f, 0.0074471775f,
         0.15978073f, 0.10185836f, 0.10298046f, -0.015476589f,
        -0.039390966f, -0.072174534f, 0.0739445f, -0.1211869f,
        -0.0347889f, -0.07943156f, 0.014809798f, -0.12412325f,
        -0.0030663363f, 0.039695457f, 0.0647603f, -0.08291318f,
        -0.018529687f, -0.004423833f, 0.0037507233f, 0.084633216f,
        -0.01514876f, -0.056505352f, -0.012800942f, -0.06994386f,
         0.012962922f, -0.031234352f, 0.07029052f, 0.016418684f,
         0.03618972f, 0.055686004f, -0.08663945f, -0.017404709f,
        -0.054761406f, 0.029065743f, 0.052404847f, 0.020238016f,
         0.0048197987f, -0.0214882f, 0.07078733f, 0.013016777f,
         0.06262858f, 0.009184685f, 0.020785125f, -0.043904778f,
        -0.0270329f, -0.03299152f, -0.060088247f, -0.015162964f,
        -0.001828936f, 0.12642565f, -0.056757294f, 0.013586685f,
         0.09232601f, -0.035886683f, 0.06000002f, 0.05229691f,
        -0.052580316f, -0.082029596f, -0.010794592f, 0.012947712f,
        -0.036429964f, -0.085508935f, -0.13127148f, -0.017744139f,
         0.031502828f, 0.036232427f, -0.031581745f, 0.023051167f,
        -0.05325106f, -0.03421577f, 0.028793324f, -0.034633752f,
        -0.009881397f, -0.043551125f, -0.018609839f, 0.0019097115f,
        -0.008799762f, 0.056595087f, 0.0022273948f, 0.055752404f
    };
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue
    {
         0.025825322f, -0.05813119f, 0.09495884f, -0.045984812f,
        -0.01255415f, -0.0026479573f, -0.08196161f, -0.054914974f,
        -0.0046604523f, -0.029587349f, -0.044576716f, -0.07480124f,
        -0.082868785f, 0.023254942f, 0.027502948f, -0.0039728214f,
        -0.08683098f, -0.08116779f, -0.014675607f, -0.037924774f,
        -0.023314456f, -0.007401714f, -0.09255757f, 0.029460307f,
        -0.08829125f, -0.005139627f, -0.08989442f, -0.0555066f,
         0.13596267f, -0.025062224f, -0.048351806f, -0.03850004f,
         0.07266485f, -0.022414139f, 0.05940088f, 0.075114764f,
         0.09597592f, -0.010211725f, -0.0049794707f, -0.011523867f,
        -0.025980417f, 0.072999895f, 0.11091378f, -0.081685916f,
         0.014416728f, 0.043229222f, 0.034178585f, -0.07530371f,
         0.035837382f, -0.085607f, -0.007721233f, -0.03287832f,
        -0.043848954f, -0.06404588f, -0.06632928f, -0.073643476f,
         0.008214239f, -0.045984086f, 0.039764922f, 0.03474462f,
         0.060612556f, -0.080590084f, 0.049127717f, 0.04151091f,
        -0.030063879f, 0.008801774f, -0.023021035f, -0.019558564f,
         0.05158114f, -0.010947698f, -0.011825728f, 0.0075720972f,
         0.0699727f, -0.0039981045f, 0.069350146f, 0.08799282f,
         0.016156472f, 0.035502106f, 0.11695009f, 0.006217345f,
         0.13392477f, -0.037875112f, 0.025745004f, 0.08940699f,
        -0.00924166f, 0.0046702605f, -0.036598757f, -0.08811812f,
         0.10522024f, -0.032441203f, 0.008176899f, -0.04454919f,
         0.07058152f, 0.0067963637f, 0.039206743f, 0.03259838f,
         0.03725492f, -0.09515802f, 0.013326398f, -0.052055415f,
        -0.025676316f, 0.03198509f, -0.015951829f, -0.058556724f,
         0.036879618f, 0.043357447f, 0.028362012f, -0.05908629f,
         0.0059240665f, -0.04995891f, -0.019187413f, 0.0276265f,
        -0.01628143f, 0.0025863599f, 0.08800015f, 0.035250366f,
        -0.022165963f, -0.07328642f, -0.009415526f, -0.07455109f,
         0.11690406f, 0.0363299f, 0.07411125f, 0.042103454f,
        -0.009660886f, 0.019076364f, 0.018299393f, -0.046004917f,
         0.08891175f, 0.0431396f, -0.026327137f, -0.051502608f,
         0.08979574f, -0.051670972f, 0.04940282f, -0.07491107f,
        -0.021240504f, 0.022596184f, -0.034280192f, 0.060163025f,
        -0.058211457f, -0.051837247f, -0.01349775f, -0.04639988f,
        -0.035936575f, -0.011681591f, 0.064818054f, 0.0073146066f,
        -0.021745546f, -0.043124277f, -0.06471268f, -0.07053354f,
        -0.029321948f, -0.05330136f, 0.016933719f, -0.053782392f,
         0.13747959f, -0.1361751f, -0.11569455f, 0.0033329215f,
         0.05693899f, -0.053219706f, 0.063698f, 0.07977434f,
        -0.07924483f, 0.06936997f, 0.0034815092f, -0.007305279f,
        -0.037325785f, -0.07251102f, -0.033633437f, -0.08677009f,
         0.091591336f, -0.14165086f, 0.021752775f, 0.019683983f,
         0.0011612234f, -0.058154266f, 0.049996935f, 0.0288841f,
        -0.0024567875f, -0.14345716f, 0.010955264f, -0.10234828f,
         0.1183656f, -0.0010731248f, -0.023590032f, -0.072285876f,
        -0.0724771f, -0.026382286f, -0.0014920527f, 0.042667855f,
         0.0018776858f, 0.02986552f, 0.009814309f, 0.0733756f,
         0.12289186f, 0.018043943f, -0.0458958f, 0.049412545f,
         0.033632483f, 0.05495232f, 0.036686596f, -0.013781798f,
        -0.010036754f, 0.02576849f, -0.08307328f, 0.010112348f,
         0.042521734f, -0.05869831f, -0.071689695f, 0.03876447f,
        -0.13275425f, -0.0352966f, -0.023077697f, 0.10285965f,
         0.084736146f, 0.15568255f, -0.00040734606f, 0.027835453f,
        -0.10292561f, -0.032401145f, 0.10053256f, -0.026142767f,
        -0.08271222f, -0.0030240538f, -0.016368777f, 0.1070414f,
         0.042672627f, 0.013456989f, -0.0437609f, -0.022309763f,
         0.11576483f, 0.04108048f, 0.061026827f, -0.0190714f,
        -0.0869359f, 0.037901703f, 0.0610107f, 0.07202949f,
         0.01675338f, 0.086139716f, -0.08795751f, -0.014898893f,
        -0.023771819f, -0.01965048f, 0.007955471f, -0.043740474f,
         0.03346837f, -0.10549954f, 0.090567775f, 0.042013682f,
        -0.03176985f, 0.12569028f, -0.02421228f, -0.029526481f,
         0.023851605f, 0.031539805f, 0.05292009f, -0.02344001f,
        -0.07811758f, -0.08834428f, 0.10094801f, 0.16594367f,
        -0.06861939f, -0.021256343f, -0.041093912f, -0.06669611f,
         0.035498552f, 0.021757556f, -0.09302526f, -0.015403468f,
        -0.06614931f, -0.051798206f, -0.013874718f, 0.03630673f,
         0.010412845f, -0.08077351f, 0.046185967f, 0.0035662893f,
         0.03541868f, -0.094149634f, -0.034814864f, 0.003128424f,
        -0.020674974f, -0.03944324f, -0.008110165f, -0.11113267f,
         0.08484226f, 0.043586485f, 0.040582247f, 0.0968012f,
        -0.065249965f, -0.028036479f, 0.0050708856f, 0.0017462453f,
         0.0326779f, 0.041296225f, 0.09164146f, -0.047743853f,
        -0.015952192f, -0.034451712f, 0.084197424f, -0.05347844f,
        -0.11768019f, 0.085926116f, -0.08251791f, -0.045081906f,
         0.0948852f, 0.068401024f, 0.024856757f, 0.06978981f,
        -0.057309967f, -0.012775832f, -0.0032452994f, 0.01977615f,
        -0.041040014f, -0.024264973f, 0.063464895f, 0.05431621f
    };
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{numUnits};
    std::vector<float> cellToInputWeightsValue
    {
         0.040369894f, 0.030746894f, 0.24704495f, 0.018586371f, -0.037586458f,
        -0.15312155f, -0.11812848f, -0.11465643f, 0.20259799f, 0.11418174f,
        -0.10116027f, -0.011334949f, 0.12411352f, -0.076769054f, -0.052169047f,
         0.21198851f, -0.38871562f, -0.09061183f, -0.09683246f, -0.21929175f
    };
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue
    {
        -0.01998659f, -0.15568835f, -0.24248174f, -0.012770197f, 0.041331276f,
        -0.072311886f, -0.052123554f, -0.0066330447f, -0.043891653f, 0.036225766f,
        -0.047248036f, 0.021479502f, 0.033189066f, 0.11952997f, -0.020432774f,
         0.64658105f, -0.06650122f, -0.03467612f, 0.095340036f, 0.23647355f
    };
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue
    {
         0.08286371f, -0.08261836f, -0.51210177f, 0.002913762f, 0.17764764f,
        -0.5495371f, -0.08460716f, -0.24552552f, 0.030037103f, 0.04123544f,
        -0.11940523f, 0.007358328f, 0.1890978f, 0.4833202f, -0.34441817f,
         0.36312827f, -0.26375428f, 0.1457655f, -0.19724406f, 0.15548733f
    };
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<float> inputGateBiasValue
    {
         0.02234832f, 0.14757581f, 0.18176508f, 0.10380666f, 0.053110216f,
        -0.06928846f, -0.13942584f, -0.11816189f, 0.19483899f, 0.03652339f,
        -0.10250295f, 0.036714908f, -0.18426876f, 0.036065217f, 0.21810818f,
         0.02383196f, -0.043370757f, 0.08690144f, -0.04444982f, 0.00030581196f
    };
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue
    {
        0.035185695f, -0.042891346f, -0.03032477f, 0.23027696f, 0.11098921f,
        0.15378423f, 0.09263801f, 0.09790885f, 0.09508917f, 0.061199076f,
        0.07665568f, -0.015443159f, -0.03499149f, 0.046190713f, 0.08895977f,
        0.10899629f, 0.40694186f, 0.06030037f, 0.012413437f, -0.06108739f
    };
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue
    {
        -0.024379363f, 0.0055531194f, 0.23377132f, 0.033463873f, -0.1483596f,
        -0.10639995f, -0.091433935f, 0.058573797f, -0.06809782f, -0.07889636f,
        -0.043246906f, -0.09829136f, -0.4279842f, 0.034901652f, 0.18797937f,
         0.0075234566f, 0.016178843f, 0.1749513f, 0.13975595f, 0.92058027f
    };
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue
    {
         0.046159424f, -0.0012809046f, 0.03563469f, 0.12648113f, 0.027195795f,
         0.35373217f, -0.018957434f, 0.008907322f, -0.0762701f, 0.12018895f,
         0.04216877f, 0.0022856654f, 0.040952638f, 0.3147856f, 0.08225149f,
        -0.057416286f, -0.14995944f, -0.008040261f, 0.13208859f, 0.029760877f
    };
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{outputSize, numUnits};
    std::vector<float> projectionWeightsValue
    {
        -0.009802181f, 0.09401916f, 0.0717386f, -0.13895074f, 0.09641832f,
         0.060420845f, 0.08539281f, 0.054285463f, 0.061395317f, 0.034448683f,
        -0.042991187f, 0.019801661f, -0.16840284f, -0.015726732f, -0.23041931f,
        -0.024478018f, -0.10959692f, -0.013875541f, 0.18600968f, -0.061274476f,
         0.0138165f, -0.08160894f, -0.07661644f, 0.032372914f, 0.16169067f,
         0.22465782f, -0.03993472f, -0.004017731f, 0.08633481f, -0.28869787f,
         0.08682067f, 0.17240396f, 0.014975425f, 0.056431185f, 0.031037588f,
         0.16702051f, 0.0077946745f, 0.15140012f, 0.29405436f, 0.120285f,
        -0.188994f, -0.027265169f, 0.043389652f, -0.022061434f, 0.014777949f,
        -0.20203483f, 0.094781205f, 0.19100232f, 0.13987629f, -0.036132768f,
        -0.06426278f, -0.05108664f, 0.13221376f, 0.009441198f, -0.16715929f,
         0.15859416f, -0.040437475f, 0.050779544f, -0.022187516f, 0.012166504f,
         0.027685808f, -0.07675938f, -0.0055694645f, -0.09444123f, 0.0046453946f,
         0.050794356f, 0.10770313f, -0.20790008f, -0.07149004f, -0.11425117f,
         0.008225835f, -0.035802525f, 0.14374903f, 0.15262283f, 0.048710253f,
         0.1847461f, -0.007487823f, 0.11000021f, -0.09542012f, 0.22619456f,
        -0.029149994f, 0.08527916f, 0.009043713f, 0.0042746216f, 0.016261552f,
         0.022461696f, 0.12689082f, -0.043589946f, -0.12035478f, -0.08361797f,
        -0.050666027f, -0.1248618f, -0.1275799f, -0.071875185f, 0.07377272f,
         0.09944291f, -0.18897448f, -0.1593054f, -0.06526116f, -0.040107165f,
        -0.004618631f, -0.067624845f, -0.007576253f, 0.10727444f, 0.041546922f,
        -0.20424393f, 0.06907816f, 0.050412357f, 0.00724631f, 0.039827548f,
         0.12449835f, 0.10747581f, 0.13708383f, 0.09134148f, -0.12617786f,
        -0.06428341f, 0.09956831f, 0.1208086f, -0.14676677f, -0.0727722f,
         0.1126304f, 0.010139365f, 0.015571211f, -0.038128063f, 0.022913318f,
        -0.042050496f, 0.16842307f, -0.060597885f, 0.10531834f, -0.06411776f,
        -0.07451711f, -0.03410368f, -0.13393489f, 0.06534304f, 0.003620307f,
         0.04490757f, 0.05970546f, 0.05197996f, 0.02839995f, 0.10434969f,
        -0.013699693f, -0.028353551f, -0.07260381f, 0.047201227f, -0.024575593f,
        -0.036445823f, 0.07155557f, 0.009672501f, -0.02328883f, 0.009533515f,
        -0.03606021f, -0.07421458f, -0.028082801f, -0.2678904f, -0.13221288f,
         0.18419984f, -0.13012612f, -0.014588381f, -0.035059117f, -0.04824723f,
         0.07830115f, -0.056184657f, 0.03277091f, 0.025466874f, 0.14494097f,
        -0.12522776f, -0.098633975f, -0.10766018f, -0.08317623f, 0.08594209f,
         0.07749552f, 0.039474737f, 0.1776665f, -0.07409566f, -0.0477268f,
         0.29323658f, 0.10801441f, 0.1154011f, 0.013952499f, 0.10739139f,
         0.10708251f, -0.051456142f, 0.0074137426f, -0.10430189f, 0.10034707f,
         0.045594677f, 0.0635285f, -0.0715442f, -0.089667566f, -0.10811871f,
         0.00026344223f, 0.08298446f, -0.009525053f, 0.006585689f, -0.24567553f,
        -0.09450807f, 0.09648481f, 0.026996298f, -0.06419476f, -0.04752702f,
        -0.11063944f, -0.23441927f, -0.17608605f, -0.052156363f, 0.067035615f,
         0.19271925f, -0.0032889997f, -0.043264326f, 0.09663576f, -0.057112187f,
        -0.10100678f, 0.0628376f, 0.04447668f, 0.017961001f, -0.10094388f,
        -0.10190601f, 0.18335468f, 0.10494553f, -0.052095775f, -0.0026118709f,
         0.10539724f, -0.04383912f, -0.042349473f, 0.08438151f, -0.1947263f,
         0.02251204f, 0.11216432f, -0.10307853f, 0.17351969f, -0.039091777f,
         0.08066188f, -0.00561982f, 0.12633002f, 0.11335965f, -0.0088127935f,
        -0.019777594f, 0.06864014f, -0.059751723f, 0.016233567f, -0.06894641f,
        -0.28651384f, -0.004228674f, 0.019708522f, -0.16305895f, -0.07468996f,
        -0.0855457f, 0.099339016f, -0.07580735f, -0.13775392f, 0.08434318f,
         0.08330512f, -0.12131499f, 0.031935584f, 0.09180414f, -0.08876437f,
        -0.08049874f, 0.008753825f, 0.03498998f, 0.030215185f, 0.03907079f,
         0.089751154f, 0.029194152f, -0.03337423f, -0.019092513f, 0.04331237f,
         0.04299654f, -0.036394123f, -0.12915532f, 0.09793732f, 0.07512415f,
        -0.11319543f, -0.032502122f, 0.15661901f, 0.07671967f, -0.005491124f,
        -0.19379048f, -0.218606f, 0.21448623f, 0.017840758f, 0.1416943f,
        -0.07051762f, 0.19488361f, 0.02664691f, -0.18104725f, -0.09334311f,
         0.15026465f, -0.15493552f, -0.057762887f, -0.11604192f, -0.262013f,
        -0.01391798f, 0.012185008f, 0.11156489f, -0.07483202f, 0.06693364f,
        -0.26151478f, 0.046425626f, 0.036540434f, -0.16435726f, 0.17338543f,
        -0.21401681f, -0.11385144f, -0.08283257f, -0.069031075f, 0.030635102f,
         0.010969227f, 0.11109743f, 0.010919218f, 0.027526086f, 0.13519906f,
         0.01891392f, -0.046839405f, -0.040167913f, 0.017953383f, -0.09700955f,
         0.0061885654f, -0.07000971f, 0.026893595f, -0.038844477f, 0.14543656f
    };
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<float> projectionBiasValue(outputSize, 0.0f);

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 24:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 25:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 26:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 4};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 4, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue
    {
        -0.00396806f, 0.029352f, -0.00279226f, 0.0159977f, -0.00835577f, -0.0211779f, 0.0283512f, -0.0114597f,
         0.00907307f, -0.0244004f, -0.0152191f, -0.0259063f, 0.00914318f, 0.00415119f, 0.017147f, 0.0134203f,
        -0.013869f, 0.0287268f, -0.00334694f, 0.00733397f, -0.0287926f, -0.0186926f, 0.0193662f, -0.0115437f,
         0.00422612f, -0.0345232f, 0.00223253f, -0.00957321f, 0.0210624f, 0.013331f, 0.0150954f, 0.0216801f
    };
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue
    {
        -0.0531632f, -0.0118138f, 0.0870833f, 0.0347929f, -0.076144f,
        -0.0659219f, -0.0463811f, 0.0141307f, -0.0127706f, -0.03782f,
        -0.00402401f, -0.00571876f, -0.187957f, -0.0247127f, 0.0711425f,
         0.008244f, 0.0492649f, 0.126972f, 0.0933097f, 0.29848f,
        -0.0966178f, -0.114417f, 0.0387229f, 0.0453255f, -0.181286f,
        -0.0651251f, -0.0996879f, -0.00276995f, 0.0617558f, -0.0100728f,
         0.056304f, -0.077416f, -0.162858f, -0.0541251f, 0.0571202f,
        -0.0525331f, 0.0724297f, 0.171029f, 0.141738f, 0.295483f
    };
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue
    {
        -0.00396806f, 0.029352f, -0.00279226f, 0.0159977f, -0.00835576f, -0.0211779f, 0.0283512f, -0.0114597f,
         0.00907307f, -0.0244004f, -0.0152191f, -0.0259063f, 0.00914318f, 0.00415118f, 0.017147f, 0.0134203f,
        -0.013869f, 0.0287268f, -0.00334693f, 0.00733398f, -0.0287926f, -0.0186926f, 0.0193662f, -0.0115437f,
         0.00422612f, -0.0345232f, 0.00223253f, -0.00957321f, 0.0210624f, 0.013331f, 0.0150954f, 0.02168f
    };

    LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                            inputToInputWeightsDimensions,         inputToInputWeightsValue,
                            inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                            inputToCellWeightsDimensions,          inputToCellWeightsValue,
                            inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                            recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                            recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                            recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                            recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                            cellToInputWeightsDimensions,          cellToInputWeightsValue,
                            cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                            cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                            inputGateBiasDimensions,               inputGateBiasValue,
                            forgetGateBiasDimensions,              forgetGateBiasValue,
                            cellBiasDimensions,                    cellBiasValue,
                            outputGateBiasDimensions,              outputGateBiasValue,
                            projectionWeightsDimensions,           projectionWeightsValue,
                            projectionBiasDimensions,              projectionBiasValue,
                            outputStateInDimensions,               outputStateInValue,
                            cellStateInDimensions,                 cellStateInValue,
                            activationFunctionDimensions,          activationFunctionValue,
                            cellClippingThresholdDimensions,       cellClippingThresholdValue,
                            projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                            inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                            forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                            cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                            outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                            scratchBufferDimensions,               scratchBufferValue,
                            outputStateOutDimensions,              outputStateOutValue,
                            cellStateOutDimensions,                cellStateOutValue,
                            outputDimensions,                      outputValue,
                            compute);
}

template <typename HalPolicy>
void LstmCifgPeepholeNoProjectionBatch2(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/lstm2.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/lstm2.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).
    // The batch size has been increased to 2 (it was 1 in the VTS test) with appropriate input and output values added.

    uint32_t batchSize = 2;
    uint32_t inputSize = 2;
    uint32_t numUnits = 4;
    uint32_t outputSize = numUnits;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{2.0f, 3.0f, 3.0f, 4.0f};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{0};
    std::vector<float> inputToInputWeightsValue;
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.55291498f, -0.42866567f,
                                                  0.13056988f, -0.36333650f,
                                                 -0.22755712f,  0.28253698f,
                                                  0.24407166f,  0.33826375f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.49770179f, -0.27711356f,
                                               -0.09624726f,  0.05100781f,
                                                0.04717243f,  0.48944736f,
                                               -0.38535351f, -0.17212132f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{ 0.10725588f, -0.02335852f,
                                                 -0.55932593f, -0.09426838f,
                                                 -0.44257352f,  0.54939759f,
                                                  0.01533556f,  0.42751634f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0}; // VTS was {4, 4} -> {0} ?
    std::vector<float> recurrentToInputWeightsValue;
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.13832897f, -0.05151010f, -0.23590070f, -0.16661474f,
                                                     -0.14340827f,  0.36986142f,  0.23414481f,  0.55899000f,
                                                      0.10798943f, -0.41174671f,  0.17751795f, -0.34484994f,
                                                     -0.35874045f, -0.11352962f,  0.27268326f,  0.54058349f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{ 0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f,
                                                    0.42957711f,  0.01841056f, -0.32764608f, -0.33027974f,
                                                   -0.10826075f,  0.20675004f,  0.19069612f, -0.03026325f,
                                                   -0.54532051f,  0.33003211f,  0.44901288f,  0.21193194f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{0.41613156f,  0.42610586f, -0.16495961f, -0.56638730f,
                                                     0.30579174f, -0.05115908f, -0.33941799f,  0.23364776f,
                                                     0.11178309f,  0.09481031f, -0.26424935f,  0.46261835f,
                                                     0.50248802f,  0.26114327f, -0.43736315f,  0.33149987f};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<float> cellToInputWeightsValue;
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue{0.47485286f, -0.51955009f, -0.24458408f, 0.31544167f};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue{-0.17135078f, 0.82760304f, 0.85573703f, -0.77109635f};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{0}; // VTS was {4} -> {0} ?
    std::vector<float> inputGateBiasValue;
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{1.0f, 1.0f, 1.0f, 1.0f};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue(numUnits, 0.0f);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue(numUnits, 0.0f);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{0};
    std::vector<float> projectionWeightsValue;
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<float> projectionBiasValue;

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 24:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 25:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 26:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 3};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 3, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue{-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f,
                                           -0.42734814f, -0.00478661f, 0.13455015f, -0.03560682f};
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue{-0.76044439f, -0.01804161f, 0.18226376f, -0.06493707f,
                                         -0.90477051f, -0.04355603f, 0.18475688f, -0.04158677f};
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue{-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f,
                                   -0.42734814f, -0.00478661f, 0.13455015f, -0.03560682f};

    LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                            inputToInputWeightsDimensions,         inputToInputWeightsValue,
                            inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                            inputToCellWeightsDimensions,          inputToCellWeightsValue,
                            inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                            recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                            recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                            recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                            recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                            cellToInputWeightsDimensions,          cellToInputWeightsValue,
                            cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                            cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                            inputGateBiasDimensions,               inputGateBiasValue,
                            forgetGateBiasDimensions,              forgetGateBiasValue,
                            cellBiasDimensions,                    cellBiasValue,
                            outputGateBiasDimensions,              outputGateBiasValue,
                            projectionWeightsDimensions,           projectionWeightsValue,
                            projectionBiasDimensions,              projectionBiasValue,
                            outputStateInDimensions,               outputStateInValue,
                            cellStateInDimensions,                 cellStateInValue,
                            activationFunctionDimensions,          activationFunctionValue,
                            cellClippingThresholdDimensions,       cellClippingThresholdValue,
                            projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                            inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                            forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                            cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                            outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                            scratchBufferDimensions,               scratchBufferValue,
                            outputStateOutDimensions,              outputStateOutValue,
                            cellStateOutDimensions,                cellStateOutValue,
                            outputDimensions,                      outputValue,
                            compute);
}

template <typename HalPolicy>
void LstmNoCifgPeepholeProjectionNoClippingLayerNorm(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/layer_norm_lstm.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/layer_norm_lstm.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 2;
    uint32_t inputSize = 5;
    uint32_t numUnits = 4;
    uint32_t outputSize = 3;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{ 0.7f,  0.8f,  0.1f,  0.2f,  0.3f,  // batch 0
                                   0.3f,  0.2f,  0.9f,  0.8f,  0.1f}; // batch 1

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{ 0.5,  0.6,  0.7, -0.8, -0.9,
                                                 0.1,  0.2,  0.3, -0.4,  0.5,
                                                -0.8,  0.7, -0.6,  0.5, -0.4,
                                                -0.5, -0.4, -0.3, -0.2, -0.1};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.6, -0.1,  0.3,  0.2,  0.9,
                                                 -0.5, -0.2, -0.4,  0.3, -0.8,
                                                 -0.4,  0.3, -0.5, -0.4, -0.6,
                                                  0.3, -0.4, -0.6, -0.5, -0.5};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.4, -0.3, -0.2, -0.1, -0.5,
                                                0.5, -0.2, -0.3, -0.2, -0.6,
                                                0.6, -0.1, -0.4, -0.3, -0.7,
                                                0.7, -0.9, -0.5,  0.8,  0.6};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{-0.8, -0.4, -0.2, -0.9, -0.1,
                                                 -0.7,  0.3, -0.3, -0.8, -0.2,
                                                  0.6, -0.2,  0.4, -0.7, -0.3,
                                                 -0.5,  0.1,  0.5, -0.6, -0.4};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{-0.2, -0.3,  0.4,
                                                     0.1, -0.5,  0.9,
                                                    -0.2, -0.3, -0.7,
                                                    0.05, -0.2, -0.6};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.5, -0.3, -0.5,
                                                     -0.2,  0.6,  0.4,
                                                      0.9,  0.3, -0.1,
                                                      0.2,  0.5,  0.2};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.3,  0.2,  0.1,
                                                   -0.3,  0.8,-0.08,
                                                   -0.2,  0.3,  0.8,
                                                   -0.6, -0.1,  0.2};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{ 0.3, -0.1,  0.1,
                                                     -0.2, -0.5, -0.7,
                                                     -0.2, -0.6, -0.1,
                                                     -0.4, -0.7, -0.2};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{numUnits};
    std::vector<float> cellToInputWeightsValue{0.05, 0.1, 0.25, 0.15};
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue{-0.02, -0.15, -0.25, -0.03};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue{0.1, -0.1, -0.5, 0.05};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<float> inputGateBiasValue{0.03, 0.15, 0.22, 0.38};
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{0.1, -0.3, -0.2, 0.1};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue{-0.05, 0.72, 0.25, 0.08};
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue{0.05, -0.01, 0.2, 0.1};
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{numUnits, outputSize};
    std::vector<float> projectionWeightsValue{-0.1,  0.2, 0.01,
                                              -0.2,  0.1,  0.5,
                                               0.3, 0.08, 0.07,
                                               0.2, -0.4,  0.2};
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<float> projectionBiasValue(outputSize, 0.0f);
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{numUnits};
    std::vector<float> inputLayerNormWeightsValue{0.1, 0.2, 0.3, 0.5};
    // 24: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<float> forgetLayerNormWeightsValue{0.2, 0.2, 0.4, 0.3};
    // 25: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<float> cellLayerNormWeightsValue{0.7, 0.2, 0.3, 0.8};
    // 26: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<float> outputLayerNormWeightsValue{0.6, 0.2, 0.2, 0.5};

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 4};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 4, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue { 0.02440767f,  0.12802738f, -0.00170918f,
                                            -0.00692428f,  0.08487406f,  0.06344498f};
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue {-0.45177122f,  0.37691566f,  0.22542511f,  0.23240635f,
                                          -0.25258583f,  0.33042118f,  0.01730525f,  0.36660123f};
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue{ 0.02440767f, 0.12802738f, -0.00170918f,
                                   -0.00692428f, 0.08487406f,  0.06344498f};

   LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                           inputToInputWeightsDimensions,         inputToInputWeightsValue,
                           inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                           inputToCellWeightsDimensions,          inputToCellWeightsValue,
                           inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                           recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                           recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                           recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                           recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                           cellToInputWeightsDimensions,          cellToInputWeightsValue,
                           cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                           cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                           inputGateBiasDimensions,               inputGateBiasValue,
                           forgetGateBiasDimensions,              forgetGateBiasValue,
                           cellBiasDimensions,                    cellBiasValue,
                           outputGateBiasDimensions,              outputGateBiasValue,
                           projectionWeightsDimensions,           projectionWeightsValue,
                           projectionBiasDimensions,              projectionBiasValue,
                           outputStateInDimensions,               outputStateInValue,
                           cellStateInDimensions,                 cellStateInValue,
                           activationFunctionDimensions,          activationFunctionValue,
                           cellClippingThresholdDimensions,       cellClippingThresholdValue,
                           projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                           inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                           forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                           cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                           outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                           scratchBufferDimensions,               scratchBufferValue,
                           outputStateOutDimensions,              outputStateOutValue,
                           cellStateOutDimensions,                cellStateOutValue,
                           outputDimensions,                      outputValue,
                           compute);
}

template <typename HalPolicy>
void LstmCifgPeepholeProjectionNoClippingLayerNorm(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/layer_norm_lstm.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/layer_norm_lstm.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 2;
    uint32_t inputSize = 5;
    uint32_t numUnits = 4;
    uint32_t outputSize = 3;

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<float> inputValue{ 0.7f, 0.8f, 0.1f, 0.2f, 0.3f,  // batch 0
                                   0.3f, 0.2f, 0.9f, 0.8f, 0.1f}; // batch 1

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{0};
    std::vector<float> inputToInputWeightsValue;
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.6, -0.1,  0.3,  0.2,  0.9,
                                                 -0.5, -0.2, -0.4,  0.3, -0.8,
                                                 -0.4,  0.3, -0.5, -0.4, -0.6,
                                                  0.3, -0.4, -0.6, -0.5, -0.5};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.4, -0.3, -0.2, -0.1, -0.5,
                                                0.5, -0.2, -0.3, -0.2, -0.6,
                                                0.6, -0.1, -0.4, -0.3, -0.7,
                                                0.7, -0.9, -0.5,  0.8,  0.6};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{-0.8, -0.4, -0.2, -0.9, -0.1,
                                                 -0.7,  0.3, -0.3, -0.8, -0.2,
                                                  0.6, -0.2,  0.4, -0.7, -0.3,
                                                 -0.5,  0.1,  0.5, -0.6, -0.4};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0};
    std::vector<float> recurrentToInputWeightsValue;
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.5, -0.3, -0.5,
                                                     -0.2,  0.6,  0.4,
                                                      0.9,  0.3, -0.1,
                                                      0.2,  0.5,  0.2};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.3,  0.2,  0.1,
                                                   -0.3,  0.8,-0.08,
                                                   -0.2,  0.3,  0.8,
                                                   -0.6, -0.1,  0.2};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{  0.3, -0.1,  0.1,
                                                      -0.2, -0.5, -0.7,
                                                      -0.2, -0.6, -0.1,
                                                      -0.4, -0.7, -0.2};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<float> cellToInputWeightsValue;
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue{-0.02, -0.15, -0.25, -0.03};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue{0.1, -0.1, -0.5, 0.05};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{0};
    std::vector<float> inputGateBiasValue;
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{0.1, -0.3, -0.2, 0.1};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue{-0.05, 0.72, 0.25, 0.08};
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue{0.05, -0.01, 0.2, 0.1};
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{numUnits, outputSize};
    std::vector<float> projectionWeightsValue{-0.1,  0.2, 0.01,
                                              -0.2,  0.1,  0.5,
                                               0.3, 0.08, 0.07,
                                               0.2, -0.4,  0.2};
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<float> projectionBiasValue(outputSize, 0.0f);
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.0f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.0f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t> activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> cellClippingThresholdDimensions{};
    std::vector<float> cellClippingThresholdValue{0.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t> projectionClippingThresholdDimensions{};
    std::vector<float> projectionClippingThresholdValue{0.0f};

    // Normalization:
    // 23: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{numUnits};
    std::vector<float> inputLayerNormWeightsValue{0.1, 0.2, 0.3, 0.5};
    // 24: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<float> forgetLayerNormWeightsValue{0.2, 0.2, 0.4, 0.3};
    // 25: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<float> cellLayerNormWeightsValue{0.7, 0.2, 0.3, 0.8};
    // 26: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<float> outputLayerNormWeightsValue{0.6, 0.2, 0.2, 0.5};

    // Outputs:
    //  0: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    // HOWEVER, by looking at the code, seems that it's the opposite: (cifg ? 3 : 4) * numUnits
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp:319
    //           android/frameworks/ml/nn/common/operations/LSTMTest.cpp:114
    //           tensorflow/tensorflow/contrib/lite/kernels/lstm.cc:332
    hidl_vec<uint32_t> scratchBufferDimensions{batchSize, numUnits * 3};
    std::vector<float> scratchBufferValue(batchSize * numUnits * 3, 0.0f);
    //  1: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<float> outputStateOutValue { 0.02129706f,  0.14081624f,  0.01127331f,
                                            -0.02263505f,  0.09169482f,  0.07691758f};
    //  2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue{-0.35102980f,  0.42610350f,  0.21463650f,  0.27716520f,
                                         -0.18855170f,  0.32522000f,  0.02036650f,  0.48967660f};
    //  3: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<float> outputValue{ 0.02129706f,  0.14081624f,  0.01127331f,
                                   -0.02263505f,  0.09169482f,  0.07691758f};

    LstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                            inputToInputWeightsDimensions,         inputToInputWeightsValue,
                            inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                            inputToCellWeightsDimensions,          inputToCellWeightsValue,
                            inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                            recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                            recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                            recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                            recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                            cellToInputWeightsDimensions,          cellToInputWeightsValue,
                            cellToForgetWeightsDimensions,         cellToForgetWeightsValue,
                            cellToOutputWeightsDimensions,         cellToOutputWeightsValue,
                            inputGateBiasDimensions,               inputGateBiasValue,
                            forgetGateBiasDimensions,              forgetGateBiasValue,
                            cellBiasDimensions,                    cellBiasValue,
                            outputGateBiasDimensions,              outputGateBiasValue,
                            projectionWeightsDimensions,           projectionWeightsValue,
                            projectionBiasDimensions,              projectionBiasValue,
                            outputStateInDimensions,               outputStateInValue,
                            cellStateInDimensions,                 cellStateInValue,
                            activationFunctionDimensions,          activationFunctionValue,
                            cellClippingThresholdDimensions,       cellClippingThresholdValue,
                            projectionClippingThresholdDimensions, projectionClippingThresholdValue,
                            inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                            forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                            cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                            outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                            scratchBufferDimensions,               scratchBufferValue,
                            outputStateOutDimensions,              outputStateOutValue,
                            cellStateOutDimensions,                cellStateOutValue,
                            outputDimensions,                      outputValue,
                            compute);
}

template <typename HalPolicy>
void QuantizedLstm(armnn::Compute compute)
{
    armnn::IgnoreUnused(compute);
    // This replicates android/frameworks/ml/nn/runtime/test/generated/vts_models/quantized_lstm.model.cpp
    // with values from android/frameworks/ml/nn/runtime/test/generated/examples/quantized_lstm.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of MODEL_INPUT tensors).

    uint32_t batchSize = 2;
    uint32_t inputSize = 2;
    uint32_t outputSize = 4;

    // Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<uint8_t> inputValue{166, 179, 50, 150};

    // 1: The input-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-input part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{outputSize, inputSize};
    std::vector<uint8_t> inputToInputWeightsValue{146, 250, 235, 171, 10, 218, 171, 108};
    // 2: The input-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-forget part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{outputSize, inputSize};
    std::vector<uint8_t> inputToForgetWeightsValue{24, 50, 132, 179, 158, 110, 3, 169};
    // 3: The input-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-cell part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> inputToCellWeightsDimensions{outputSize, inputSize};
    std::vector<uint8_t> inputToCellWeightsValue{133, 34, 29, 49, 206, 109, 54, 183};
    // 4: The input-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-output part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{outputSize, inputSize};
    std::vector<uint8_t> inputToOutputWeightsValue{195, 187, 11, 99, 109, 10, 218, 48};
    // 5: The recurrent-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-input part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{outputSize, outputSize};
    std::vector<uint8_t> recurrentToInputWeightsValue{254, 206, 77,  168, 71, 20,  215, 6,
                                                      223, 7,   118, 225, 59, 130, 174, 26};
    // 6: The recurrent-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-forget part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{outputSize, outputSize};
    std::vector<uint8_t> recurrentToForgetWeightsValue{137, 240, 103, 52, 68, 51, 237, 112,
                                                       0,   220, 89,  23, 69, 4,  207, 253};
    // 7: The recurrent-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-cell part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{outputSize, outputSize};
    std::vector<uint8_t> recurrentToCellWeightsValue{172, 60,  205, 65, 14,  0,  140, 168,
                                                     240, 223, 133, 56, 142, 64, 246, 216};
    // 8: The recurrent-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-output part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{outputSize, outputSize};
    std::vector<uint8_t> recurrentToOutputWeightsValue{106, 214, 67, 23,  59,  158, 45, 3,
                                                       119, 132, 49, 205, 129, 218, 11, 98};
    // 9: The input gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the
    //    bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    hidl_vec<uint32_t> inputGateBiasDimensions{outputSize};
    std::vector<int32_t> inputGateBiasValue{-7876, 13488, -726, 32839};
    // 10: The forget gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    hidl_vec<uint32_t> forgetGateBiasDimensions{outputSize};
    std::vector<int32_t> forgetGateBiasValue{9206, -46884, -11693, -38724};
    // 11:The cell bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the bias
    //    for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product of input
    //    and weights scales and zeroPoint equal to 0.
    hidl_vec<uint32_t> cellBiasDimensions{outputSize};
    std::vector<int32_t> cellBiasValue{39481, 48624, 48976, -21419};
    // 12:The output gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //    the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    hidl_vec<uint32_t> outputGateBiasDimensions{outputSize};
    std::vector<int32_t> outputGateBiasValue{-58999, -17050, -41852, -40538};

    //13: The previous cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape
    //    [numBatches, outputSize] specifying the cell state from the previous time step of the LSTM cell.
    //    It is quantized using a quantization range of -2^4, 2^4 * 32767/32768.
    hidl_vec<uint32_t> previousCellStateInDimensions{batchSize, outputSize};
    std::vector<int16_t> previousCellStateInValue{876, 1034, 955, -909, 761, 1029, 796, -1036};
    // 14: The previous output state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //     [numBathes, outputSize] specifying the output of the LSTM cell from previous time-step. Tensor
    //     is quantized with a fixed quantization range of -1, 127/128.
    hidl_vec<uint32_t> previousOutputInDimensions{batchSize, outputSize};
    std::vector<uint8_t> previousOutputInValue{136, 150, 140, 115, 135, 152, 138, 112};

    // 0: The cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape [numBatches, outputSize]
    //    which contains a cell state from the current time step. Tensor is quantized using a quantization range
    //    of -2^4, 2^4 * 32767/32768.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, outputSize};
    std::vector<int16_t> cellStateOutValue {1485, 1177, 1373, -1023, 1019, 1355, 1097, -1235};
    // 1: The output: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBathes, outputSize] which
    //      contains the output value. Tensor is quantized with a fixed quantization range of -1, 127/128.
    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<uint8_t> outputValue {140, 151, 146, 112, 136, 156, 142, 112};


    QuantizedLstmTestImpl<HalPolicy>(inputDimensions,                       inputValue,
                                     inputToInputWeightsDimensions,         inputToInputWeightsValue,
                                     inputToForgetWeightsDimensions,        inputToForgetWeightsValue,
                                     inputToCellWeightsDimensions,          inputToCellWeightsValue,
                                     inputToOutputWeightsDimensions,        inputToOutputWeightsValue,
                                     recurrentToInputWeightsDimensions,     recurrentToInputWeightsValue,
                                     recurrentToForgetWeightsDimensions,    recurrentToForgetWeightsValue,
                                     recurrentToCellWeightsDimensions,      recurrentToCellWeightsValue,
                                     recurrentToOutputWeightsDimensions,    recurrentToOutputWeightsValue,
                                     inputGateBiasDimensions,               inputGateBiasValue,
                                     forgetGateBiasDimensions,              forgetGateBiasValue,
                                     cellBiasDimensions,                    cellBiasValue,
                                     outputGateBiasDimensions,              outputGateBiasValue,
                                     previousOutputInDimensions,            previousOutputInValue,
                                     previousCellStateInDimensions,         previousCellStateInValue,
                                     cellStateOutDimensions,                cellStateOutValue,
                                     outputDimensions,                      outputValue);
}
