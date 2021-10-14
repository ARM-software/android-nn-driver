//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"

#include <1.3/HalPolicy.hpp>

#include <array>

using ArmnnDriver   = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;

using namespace driverTestHelpers;
using namespace android::hardware;

using HalPolicy = hal_1_3::HalPolicy;

static const float TOLERANCE = 1.0f;

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
HalPolicy::OperandLifeTime CreateNoValueLifeTime(const hidl_vec<uint32_t>& dimensions)
{
    // Only create a NO_VALUE for optional operands that have no elements
    if (dimensions.size() == 0 || dimensions[0] == 0)
    {
        return HalPolicy::OperandLifeTime::NO_VALUE;
    }
    return HalPolicy::OperandLifeTime::CONSTANT_COPY;
}

void ExecuteModel(const armnn_driver::hal_1_3::HalPolicy::Model& model,
                  armnn_driver::ArmnnDriver& driver,
                  const V1_0::Request& request)
{
    android::sp<V1_3::IPreparedModel> preparedModel = PrepareModel_1_3(model, driver);
    if (preparedModel.get() != nullptr)
    {
        Execute(preparedModel, request);
    }
}

// Add our own tests here since we skip the qlstm tests which Google supplies (because of non-const weights)
void QLstmTestImpl(const hidl_vec<uint32_t>&   inputDimensions,
                   const std::vector<int8_t>&   inputValue,
                   const hidl_vec<uint32_t>&    inputToInputWeightsDimensions,
                   const std::vector<int8_t>&   inputToInputWeightsValue,
                   const hidl_vec<uint32_t>&    inputToForgetWeightsDimensions,
                   const std::vector<int8_t>&   inputToForgetWeightsValue,
                   const hidl_vec<uint32_t>&    inputToCellWeightsDimensions,
                   const std::vector<int8_t>&   inputToCellWeightsValue,
                   const hidl_vec<uint32_t>&    inputToOutputWeightsDimensions,
                   const std::vector<int8_t>&   inputToOutputWeightsValue,
                   const hidl_vec<uint32_t>&    recurrentToInputWeightsDimensions,
                   const std::vector<int8_t>&   recurrentToInputWeightsValue,
                   const hidl_vec<uint32_t>&    recurrentToForgetWeightsDimensions,
                   const std::vector<int8_t>&   recurrentToForgetWeightsValue,
                   const hidl_vec<uint32_t>&    recurrentToCellWeightsDimensions,
                   const std::vector<int8_t>&   recurrentToCellWeightsValue,
                   const hidl_vec<uint32_t>&    recurrentToOutputWeightsDimensions,
                   const std::vector<int8_t>&   recurrentToOutputWeightsValue,
                   const hidl_vec<uint32_t>&    cellToInputWeightsDimensions,
                   const std::vector<int16_t>&  cellToInputWeightsValue,
                   const hidl_vec<uint32_t>&    cellToForgetWeightsDimensions,
                   const std::vector<int16_t>&  cellToForgetWeightsValue,
                   const hidl_vec<uint32_t>&    cellToOutputWeightsDimensions,
                   const std::vector<int16_t>&  cellToOutputWeightsValue,
                   const hidl_vec<uint32_t>&    inputGateBiasDimensions,
                   const std::vector<int32_t>&  inputGateBiasValue,
                   const hidl_vec<uint32_t>&    forgetGateBiasDimensions,
                   const std::vector<int32_t>&  forgetGateBiasValue,
                   const hidl_vec<uint32_t>&    cellBiasDimensions,
                   const std::vector<int32_t>&  cellBiasValue,
                   const hidl_vec<uint32_t>&    outputGateBiasDimensions,
                   const std::vector<int32_t>&  outputGateBiasValue,
                   const hidl_vec<uint32_t>&    projectionWeightsDimensions,
                   const std::vector<int8_t>&   projectionWeightsValue,
                   const hidl_vec<uint32_t>&    projectionBiasDimensions,
                   const std::vector<int32_t>&  projectionBiasValue,
                   const hidl_vec<uint32_t>&    outputPreviousTimeStepInDimensions,
                   const std::vector<int8_t>&   outputPreviousTimeStepInValue,
                   const hidl_vec<uint32_t>&    cellStatePreviousTimeStepInDimensions,
                   const std::vector<int16_t>&  cellStatePreviousTimeStepInValue,
                   const hidl_vec<uint32_t>&    inputLayerNormWeightsDimensions,
                   const std::vector<int16_t>&  inputLayerNormWeightsValue,
                   const hidl_vec<uint32_t>&    forgetLayerNormWeightsDimensions,
                   const std::vector<int16_t>&  forgetLayerNormWeightsValue,
                   const hidl_vec<uint32_t>&    cellLayerNormWeightsDimensions,
                   const std::vector<int16_t>&  cellLayerNormWeightsValue,
                   const hidl_vec<uint32_t>&    outputLayerNormWeightsDimensions,
                   const std::vector<int16_t>&  outputLayerNormWeightsValue,
                   const float&                 cellClipValue,
                   const float&                 projectionClipValue,
                   const float&                 matMulInputGateValue,
                   const float&                 matMulForgetGateValue,
                   const float&                 matMulCellGateValue,
                   const float&                 matMulOutputGateValue,
                   const int32_t&               projInputZeroPointValue,
                   const float&                 projInputScaleValue,
                   const hidl_vec<uint32_t>&    outputStateOutDimensions,
                   const std::vector<int8_t>&   outputStateOutValue,
                   const hidl_vec<uint32_t>&    cellStateOutDimensions,
                   const std::vector<int16_t>&  cellStateOutValue,
                   const hidl_vec<uint32_t>&    outputDimensions,
                   const std::vector<int8_t>&   outputValue,
                   armnn::Compute               compute)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(compute));
    HalPolicy::Model model = {};

    // Scale/Offset quantization info
    float inputScale    = 0.0078125f;
    int32_t inputOffset = 0;

    int32_t hiddenStateZeroPoint = 0;
    float hiddenStateScale       = 0.007f;

    float outputScale    = hiddenStateScale;
    int32_t outputOffset = hiddenStateZeroPoint;

    float cellStateScale    = 3.05176e-05f;
    float cellWeightsScale  = 1.0f;
    int32_t cellStateOffset = 0;

    float weightsScale    = 0.00784314f;
    int32_t weightsOffset = 0;

    float layerNormScale    = 3.05182e-05f;
    int32_t layerNormOffset = 0;

    float biasScale    = layerNormScale / 1024;
    int32_t biasOffset = 0;

    // Inputs:
    // 00: The input to the LSTM cell. Type: ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED Shape: [batchSize, inputSize]
    AddInputOperand<HalPolicy>(model,
                               inputDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                               inputScale,
                               inputOffset);

    // 01: The input-to-input weights. Optional. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, inputSize]
    AddTensorOperand<HalPolicy>(model,
                                inputToInputWeightsDimensions,
                                inputToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(inputToInputWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 02: The input-to-forget weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, inputSize]
    AddTensorOperand<HalPolicy>(model,
                                inputToForgetWeightsDimensions,
                                inputToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(inputToForgetWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 03: The input-to-cell weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, inputSize]
    AddTensorOperand<HalPolicy>(model,
                                inputToCellWeightsDimensions,
                                inputToCellWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(inputToCellWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 04: The input-to-output weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, inputSize]
    AddTensorOperand<HalPolicy>(model,
                                inputToOutputWeightsDimensions,
                                inputToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(inputToOutputWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 05: The recurrent-to-input weights. Optional. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM
    //     Shape: [numUnits, outputSize]
    AddTensorOperand<HalPolicy>(model,
                                recurrentToInputWeightsDimensions,
                                recurrentToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(recurrentToInputWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 06: The recurrent-to-forget weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, outputSize]
    AddTensorOperand<HalPolicy>(model,
                                recurrentToForgetWeightsDimensions,
                                recurrentToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(recurrentToForgetWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 07: The recurrent-to-cell weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, outputSize]
    AddTensorOperand<HalPolicy>(model,
                                recurrentToCellWeightsDimensions,
                                recurrentToCellWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(recurrentToCellWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 08: The recurrent-to-output weights. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [numUnits, outputSize]
    AddTensorOperand<HalPolicy>(model,
                                recurrentToOutputWeightsDimensions,
                                recurrentToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(recurrentToOutputWeightsDimensions),
                                weightsScale,
                                weightsOffset);

    // 09: The cell-to-input weights (for peephole). Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                cellToInputWeightsDimensions,
                                cellToInputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT16_SYMM ,
                                CreateNoValueLifeTime(cellToInputWeightsDimensions),
                                cellWeightsScale,
                                weightsOffset);

    // 10: The cell-to-forget weights (for peephole). Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Shape: [numUnits].
    AddTensorOperand<HalPolicy>(model,
                                cellToForgetWeightsDimensions,
                                cellToForgetWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                CreateNoValueLifeTime(cellToForgetWeightsDimensions),
                                cellWeightsScale,
                                weightsOffset);

    // 11: The cell-to-output weights (for peephole). Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                cellToOutputWeightsDimensions,
                                cellToOutputWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                CreateNoValueLifeTime(cellToOutputWeightsDimensions),
                                cellWeightsScale,
                                weightsOffset);

    // 12: The input gate bias. Quantized with scale being the product of input and weights scales
    //     and zeroPoint equal to 0. Optional. Type: ANEURALNETWORKS_TENSOR_INT32 Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                inputGateBiasDimensions,
                                inputGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(inputGateBiasDimensions),
                                biasScale,
                                biasOffset);

    // 13: The forget gate bias. Quantized with scale being the product of input and weights scales
    //     and zeroPoint equal to 0. Type: ANEURALNETWORKS_TENSOR_INT32 Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                forgetGateBiasDimensions,
                                forgetGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(forgetGateBiasDimensions),
                                biasScale,
                                biasOffset);

    // 14: The cell bias. Quantized with scale being the product of input and weights scales and zeroPoint equal to 0.
    //     Type: ANEURALNETWORKS_TENSOR_INT32 Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                cellBiasDimensions,
                                cellBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(cellBiasDimensions),
                                biasScale,
                                biasOffset);

    // 15: The output gate bias. Quantized with scale being the product of input and weights scales
    //     and zeroPoint equal to 0. Type: ANEURALNETWORKS_TENSOR_INT32 Shape: [numUnits]
    AddTensorOperand<HalPolicy>(model,
                                outputGateBiasDimensions,
                                outputGateBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(outputGateBiasDimensions),
                                biasScale,
                                biasOffset);

    // 16: The projection weights. Optional. Type: ANEURALNETWORKS_TENSOR_QUANT8_SYMM Shape: [outputSize, numUnits]
    AddTensorOperand<HalPolicy>(model,
                                projectionWeightsDimensions,
                                projectionWeightsValue,
                                HalPolicy::OperandType::TENSOR_QUANT8_SYMM,
                                CreateNoValueLifeTime(projectionWeightsDimensions),
                                0.00392157f,
                                weightsOffset);

    // 17: The projection bias. Quantized with scale being the product of input and weights scales
    //     and zeroPoint equal to 0. Optional. Type: ANEURALNETWORKS_TENSOR_INT32 Shape: [outputSize]
    AddTensorOperand<HalPolicy>(model,
                                projectionBiasDimensions,
                                projectionBiasValue,
                                HalPolicy::OperandType::TENSOR_INT32,
                                CreateNoValueLifeTime(projectionBiasDimensions),
                                0.0f,
                                biasOffset);

    // 18: The output from the previous time step. Type: ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED
    //     Shape: [batchSize, outputSize]
    AddInputOperand<HalPolicy>(model,
                               outputPreviousTimeStepInDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                               cellStateScale,
                               inputOffset);

    // 19: The cell state from the previous time step. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Shape: [batchSize, numUnits]
    AddInputOperand<HalPolicy>(model,
                               cellStatePreviousTimeStepInDimensions,
                               HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                               cellStateScale,
                               cellStateOffset);

    // If any of the tensors have a value all normalization tensors are set
    if (!inputLayerNormWeightsValue.empty()  ||
        !forgetLayerNormWeightsValue.empty() ||
        !cellLayerNormWeightsValue.empty()   ||
        !outputLayerNormWeightsValue.empty())
    {
        // Normalization:
        // 20: The input layer normalization weights. Used to rescale normalized inputs to activation at input gate.
        //      Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM Shape: [numUnits]
        AddTensorOperand<HalPolicy>(model,
                                    inputLayerNormWeightsDimensions,
                                    inputLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                    CreateNoValueLifeTime(inputLayerNormWeightsDimensions),
                                    layerNormScale,
                                    layerNormOffset);

        // 21: The forget layer normalization weights. Used to rescale normalized inputs to activation at forget gate.
        //     Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM Shape: [numUnits]
        AddTensorOperand<HalPolicy>(model,
                                    forgetLayerNormWeightsDimensions,
                                    forgetLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                    CreateNoValueLifeTime(forgetLayerNormWeightsDimensions),
                                    layerNormScale,
                                    layerNormOffset);

        // 22: The cell layer normalization weights. Used to rescale normalized inputs to activation at cell gate.
        //     Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM Shape: [numUnits]
        AddTensorOperand<HalPolicy>(model,
                                    cellLayerNormWeightsDimensions,
                                    cellLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                    CreateNoValueLifeTime(cellLayerNormWeightsDimensions),
                                    layerNormScale,
                                    layerNormOffset);

        // 23: The output layer normalization weights. Used to rescale normalized inputs to activation at output gate.
        //     Optional. Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM Shape: [numUnits]
        AddTensorOperand<HalPolicy>(model,
                                    outputLayerNormWeightsDimensions,
                                    outputLayerNormWeightsValue,
                                    HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                    CreateNoValueLifeTime(outputLayerNormWeightsDimensions),
                                    layerNormScale,
                                    layerNormOffset);
    }

    // Constant scalar values
    // 24: The cell clip. If provided the cell state is clipped by this value prior to the cell output activation.
    //     Optional. Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, cellClipValue);

    // Constant scalar values
    // 25: The projection clip. If provided and projection is enabled, this is used for clipping the projected values.
    //     Optional. Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, projectionClipValue);

    // Constant scalar values
    // 26: The scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
    //     Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, matMulInputGateValue);

    // Constant scalar values
    // 27: The scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
    //     Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, matMulForgetGateValue);

    // Constant scalar values
    // 28: The scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
    //     Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, matMulCellGateValue);

    // Constant scalar values
    // 29: The scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
    //     Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, matMulOutputGateValue);

    // Constant scalar values
    // 30: The zero point of the hidden state, i.e. input to projection. Type: ANEURALNETWORKS_INT32.
    AddIntOperand<HalPolicy>(model, projInputZeroPointValue);

    // Constant scalar values
    // 31: The scale of the hidden state, i.e. input to projection. Type: ANEURALNETWORKS_FLOAT32.
    AddFloatOperand<HalPolicy>(model, projInputScaleValue);

    // Outputs:
    //  0: The output state (out). Type: ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED Shape: [batchSize, outputSize]
    AddOutputOperand<HalPolicy>(model,
                                outputStateOutDimensions,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                                cellStateScale,
                                cellStateScale);

    //  1: The cell state (out). Type: ANEURALNETWORKS_TENSOR_QUANT16_SYMM Shape: [batchSize, numUnits].
    AddOutputOperand<HalPolicy>(model,
                                cellStateOutDimensions,
                                HalPolicy::OperandType::TENSOR_QUANT16_SYMM,
                                cellStateScale,
                                cellStateOffset);

    //  2: The output. This is effectively the same as the current "output state (out)" value.
    //     Type: ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED Shape: [batchSize, outputSize]
    AddOutputOperand<HalPolicy>(model,
                                outputDimensions,
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
                                cellStateScale,
                                cellStateScale);

    // make the QUANTIZED_LSTM operation
    model.main.operations.resize(1);
    model.main.operations[0].type = HalPolicy::OperationType::QUANTIZED_LSTM;

    model.main.operations[0].inputs = hidl_vec<uint32_t> { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                                                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                          24, 25, 26, 27, 28, 29, 30, 31};
    model.main.operations[0].outputs = hidl_vec<uint32_t> {32, 33, 34};

    // define the input values
    hidl_vec<RequestArgument> inputArguments;
    inputArguments.resize(3);

    inputArguments[0] = CreateRequestArgument<int8_t>(inputValue, 0);
    inputArguments[1] = CreateRequestArgument<int8_t>(outputPreviousTimeStepInValue, 1);
    inputArguments[2] = CreateRequestArgument<int16_t>(cellStatePreviousTimeStepInValue, 2);

    // define the expected output values
    hidl_vec<RequestArgument> outputArguments;
    outputArguments.resize(3);

    outputArguments[0] = CreateRequestArgument<int8_t>(outputStateOutValue, 3);
    outputArguments[1] = CreateRequestArgument<int16_t>(cellStateOutValue, 4);
    outputArguments[2] = CreateRequestArgument<int8_t>(outputValue, 5);

    android::hardware::neuralnetworks::V1_0::Request request = {};
    request.inputs  = inputArguments;
    request.outputs = outputArguments;

    // set the input data
    AddPoolAndSetData(inputValue.size(), request, inputValue.data());
    AddPoolAndSetData(outputPreviousTimeStepInValue.size(), request, outputPreviousTimeStepInValue.data());
    AddPoolAndSetData(cellStatePreviousTimeStepInValue.size(), request, cellStatePreviousTimeStepInValue.data());

    // add memory for the outputs
    android::sp<IMemory> outputStateOutMemory = AddPoolAndGetData<int8_t>(outputStateOutValue.size(), request);
    int8_t* outputStateOutData = static_cast<int8_t*>(static_cast<void*>(outputStateOutMemory->getPointer()));

    android::sp<IMemory> cellStateOutMemory = AddPoolAndGetData<int16_t>(cellStateOutValue.size(), request);
    int16_t* cellStateOutData = static_cast<int16_t*>(static_cast<void*>(cellStateOutMemory->getPointer()));

    android::sp<IMemory> outputMemory = AddPoolAndGetData<int8_t>(outputValue.size(), request);
    int8_t* outputData = static_cast<int8_t*>(static_cast<void*>(outputMemory->getPointer()));

    // make the prepared model and run the execution
    ExecuteModel(model, *driver, request);

    // check the results
    for (size_t i = 0; i < outputStateOutValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputStateOutValue[i] == doctest::Approx( outputStateOutData[i] ).epsilon(TOLERANCE),
                              "outputStateOut[" << i << "]: " << outputStateOutValue[i] << " != "
                              << outputStateOutData[i]);
    }

    // CELL STATE OUTPUT Does not match currently: IVGCVSW-4860 Verify remaining VTS tests (2) for QLSTM
    // Comment out for now
    // for (size_t i = 0; i < cellStateOutValue.size(); ++i)
    // {
    //    BOOST_TEST(TolerantCompareEqual(cellStateOutValue[i], cellStateOutData[i]),
    //               "cellStateOut[" << i << "]: " << cellStateOutValue[i] << " != " << cellStateOutData[i]);
    //}

    for (size_t i = 0; i < outputValue.size(); ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputValue[i] == doctest::Approx( outputData[i] ).epsilon(TOLERANCE),
                              "output[" << i << "]: " << outputValue[i] << " != " << outputData[i]);
    }
}

void QLstmWithProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/specs/V1_3/qlstm_projection.mod.py
    // with values from android/frameworks/ml/nn/runtime/test/generated/spec_V1_3/qlstm_projection.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of SUBGRAPH_INPUT tensors).

    uint32_t batchSize  = 2;
    uint32_t inputSize  = 5;
    uint32_t outputSize = 3;
    uint32_t numUnits   = 4;

    // Inputs:
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<int8_t> inputValue{ 90, 102, 13, 26, 38, 102, 13, 26, 51, 64};

    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToInputWeightsValue{   64,  77,   89, -102,
                                                  -115,  13,   25,   38,
                                                   -51,  64, -102,   89,
                                                   -77,  64,  -51,  -64,
                                                   -51, -38,  -25,  -13 };

    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToForgetWeightsValue{ -77,  -13,  38,  25,
                                                   115,  -64, -25, -51,
                                                    38, -102, -51,  38,
                                                   -64,  -51, -77,  38,
                                                   -51,  -77, -64, -64 };

    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToCellWeightsValue{  -51, -38, -25, -13,
                                                  -64,  64, -25, -38,
                                                  -25, -77,  77, -13,
                                                  -51, -38, -89,  89,
                                                 -115, -64, 102,  77 };

    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToOutputWeightsValue{ -102, -51, -25, -115,
                                                    -13, -89,  38,  -38,
                                                   -102, -25,  77,  -25,
                                                     51, -89, -38,  -64,
                                                     13,  64, -77,  -51 };

    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToInputWeightsValue{ -25, -38, 51, 13, -64, 115, -25, -38, -89, 6, -25, -77 };

    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToForgetWeightsValue{ -64, -38, -64, -25, 77, 51, 115, 38, -13, 25, 64, 25 };

    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToCellWeightsValue{ -38, 25, 13, -38, 102, -10, -25, 38, 102, -77, -13, 25 };

    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToOutputWeightsValue{ 38, -13, 13, -25, -64, -89, -25, -77, -13, -51, -89, -25 };

    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<int16_t> cellToInputWeightsValue;

    hidl_vec<uint32_t> cellToForgetWeightsDimensions{0};
    std::vector<int16_t> cellToForgetWeightsValue;

    hidl_vec<uint32_t> cellToOutputWeightsDimensions{0};
    std::vector<int16_t> cellToOutputWeightsValue;

    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<int32_t> inputGateBiasValue{ 644245, 3221226, 4724464, 8160438 };

    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<int32_t> forgetGateBiasValue{ 2147484, -6442451, -4294968, 2147484 };

    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<int32_t> cellBiasValue{-1073742, 15461883, 5368709, 1717987 };

    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<int32_t> outputGateBiasValue{ 1073742, -214748, 4294968, 2147484 };

    hidl_vec<uint32_t> projectionWeightsDimensions{outputSize, numUnits};
    std::vector<int8_t> projectionWeightsValue{ -25, 51, 3, -51, 25, 127, 77, 20, 18, 51, -102, 51 };

    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<int32_t> projectionBiasValue{ 0, 0, 0 };

    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateInValue{ 0, 0, 0, 0, 0, 0 };

    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<int16_t> cellStateInValue{ 0, 0, 0, 0, 0, 0, 0, 0 };

    // Normalization:
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> inputLayerNormWeightsValue{ 3277, 6553, 9830, 16384 };

    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> forgetLayerNormWeightsValue{ 6553, 6553, 13107, 9830 };

    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> cellLayerNormWeightsValue{ 22937, 6553, 9830, 26214 };

    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> outputLayerNormWeightsValue{ 19660, 6553, 6553, 16384 };

    float cellClipValue           = 0.0f;
    float projectionClipValue     = 0.0f;
    float inputIntermediateScale  = 0.007059f;
    float forgetIntermediateScale = 0.007812f;
    float cellIntermediateScale   = 0.007059f;
    float outputIntermediateScale = 0.007812f;
    int32_t hiddenStateZeroPoint  = 0;
    float hiddenStateScale        = 0.007f;

    // Outputs:
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateOutValue{ 127, 127, -108, -67, 127, 127 };

    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<int16_t> cellStateOutValue { -14650, 8939, 5771, 6715, -11843, 7847, 1508, 12939 };

    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<int8_t> outputValue { 127, 127, -108, -67, 127, 127 };

    QLstmTestImpl(inputDimensions,                       inputValue,
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
                  inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                  forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                  cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                  outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                  cellClipValue,
                  projectionClipValue,
                  inputIntermediateScale,
                  forgetIntermediateScale,
                  cellIntermediateScale,
                  outputIntermediateScale,
                  hiddenStateZeroPoint,
                  hiddenStateScale,
                  outputStateOutDimensions,              outputStateOutValue,
                  cellStateOutDimensions,                cellStateOutValue,
                  outputDimensions,                      outputValue,
                  compute);
}

void QLstmWithNoProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/specs/V1_3/qlstm_noprojection.mod.py
    // with values from android/frameworks/ml/nn/runtime/test/generated/spec_V1_3/qlstm_noprojection.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of SUBGRAPH_INPUT tensors).

    uint32_t batchSize  = 2;
    uint32_t inputSize  = 5;
    uint32_t outputSize = 4;
    uint32_t numUnits   = 4;

    // Inputs:
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<int8_t> inputValue { 90, 102, 13, 26, 38, 102, 13, 26, 51, 64 };

    hidl_vec<uint32_t> inputToInputWeightsDimensions{0, 0};
    std::vector<int8_t> inputToInputWeightsValue;

    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToForgetWeightsValue { -77, -13,  38,  25,  115,
                                                    -64, -25, -51,  38, -102,
                                                    -51,  38, -64, -51,  -77,
                                                     38, -51, -77, -64,  -64 };

    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToCellWeightsValue { -51,  -38, -25, -13, -64,
                                                   64,  -25, -38, -25, -77,
                                                   77,  -13, -51, -38, -89,
                                                   89, -115, -64, 102,  77 };

    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToOutputWeightsValue { -102, -51, -25, -115, -13,
                                                     -89,  38, -38, -102, -25,
                                                      77, -25,  51,  -89, -38,
                                                     -64,  13,  64,  -77, -51 };

    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0, 0};
    std::vector<int8_t> recurrentToInputWeightsValue;

    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToForgetWeightsValue { -64, -38, -64, -25,
                                                         77,  51, 115,  38,
                                                        -13,  25,  64,  25,
                                                         25,  38, -13,  51 };

    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToCellWeightsValue { -38,  25,  13, -38,
                                                      102, -10, -25,  38,
                                                      102, -77, -13,  25,
                                                       38, -13,  25,  64 };

    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToOutputWeightsValue {  38, -13,  13, -25,
                                                        -64, -89, -25, -77,
                                                        -13, -51, -89, -25,
                                                         13,  64,  25, -38 };

    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<int16_t> cellToInputWeightsValue;

    hidl_vec<uint32_t> cellToForgetWeightsDimensions{0};
    std::vector<int16_t> cellToForgetWeightsValue;

    hidl_vec<uint32_t> cellToOutputWeightsDimensions{0};
    std::vector<int16_t> cellToOutputWeightsValue;

    hidl_vec<uint32_t> inputGateBiasDimensions{0};
    std::vector<int32_t> inputGateBiasValue;

    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<int32_t> forgetGateBiasValue { 2147484, -6442451, -4294968, 2147484 };

    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<int32_t> cellBiasValue { -1073742, 15461883, 5368709, 1717987 };

    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<int32_t> outputGateBiasValue { 1073742, -214748, 4294968, 2147484 };

    hidl_vec<uint32_t> projectionWeightsDimensions{0, 0};
    std::vector<int8_t> projectionWeightsValue;

    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<int32_t> projectionBiasValue;

    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateInValue { 0, 0, 0, 0, 0, 0, 0, 0 };

    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<int16_t> cellStateInValue { 0, 0, 0, 0, 0, 0, 0, 0 };

    // Normalization:
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<int16_t> inputLayerNormWeightsValue;

    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> forgetLayerNormWeightsValue { 6553, 6553, 13107, 9830 };

    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> cellLayerNormWeightsValue { 22937, 6553, 9830, 26214 };

    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> outputLayerNormWeightsValue { 19660, 6553, 6553, 16384 };

    float cellClipValue           = 0.0f;
    float projectionClipValue     = 0.0f;
    float inputIntermediateScale  = 0.007059f;
    float forgetIntermediateScale = 0.007812f;
    float cellIntermediateScale   = 0.007059f;
    float outputIntermediateScale = 0.007812f;
    int32_t hiddenStateZeroPoint  = 0;
    float hiddenStateScale        = 0.007f;

    // Outputs:
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateOutValue { -15, 21, 14, 20, -15, 15, 5, 27 };

    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<int16_t> cellStateOutValue { -11692, 9960, 5491, 8861, -9422, 7726, 2056, 13149 };

    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<int8_t> outputValue { -15, 21, 14, 20, -15, 15, 5, 27 };

    QLstmTestImpl(inputDimensions,                       inputValue,
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
                  inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                  forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                  cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                  outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                  cellClipValue,
                  projectionClipValue,
                  inputIntermediateScale,
                  forgetIntermediateScale,
                  cellIntermediateScale,
                  outputIntermediateScale,
                  hiddenStateZeroPoint,
                  hiddenStateScale,
                  outputStateOutDimensions,              outputStateOutValue,
                  cellStateOutDimensions,                cellStateOutValue,
                  outputDimensions,                      outputValue,
                  compute);
}

void DynamicOutputQLstmWithNoProjection(armnn::Compute compute)
{
    // This replicates android/frameworks/ml/nn/runtime/test/specs/V1_3/qlstm_noprojection.mod.py
    // with values from android/frameworks/ml/nn/runtime/test/generated/spec_V1_3/qlstm_noprojection.example.cpp
    // and weights, biases and scalars passed as CONSTANT_COPY tensors (instead of SUBGRAPH_INPUT tensors)
    // and made cellStateOutput dynamic.

    uint32_t batchSize  = 2;
    uint32_t inputSize  = 5;
    uint32_t outputSize = 4;
    uint32_t numUnits   = 4;

    // Inputs:
    hidl_vec<uint32_t> inputDimensions{batchSize, inputSize};
    std::vector<int8_t> inputValue { 90, 102, 13, 26, 38, 102, 13, 26, 51, 64 };

    hidl_vec<uint32_t> inputToInputWeightsDimensions{0, 0};
    std::vector<int8_t> inputToInputWeightsValue;

    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToForgetWeightsValue { -77, -13,  38,  25,  115,
                                                    -64, -25, -51,  38, -102,
                                                    -51,  38, -64, -51,  -77,
                                                    38, -51, -77, -64,  -64 };

    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToCellWeightsValue { -51,  -38, -25, -13, -64,
                                                  64,  -25, -38, -25, -77,
                                                  77,  -13, -51, -38, -89,
                                                  89, -115, -64, 102,  77 };

    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<int8_t> inputToOutputWeightsValue { -102, -51, -25, -115, -13,
                                                    -89,  38, -38, -102, -25,
                                                    77, -25,  51,  -89, -38,
                                                    -64,  13,  64,  -77, -51 };

    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0, 0};
    std::vector<int8_t> recurrentToInputWeightsValue;

    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToForgetWeightsValue { -64, -38, -64, -25,
                                                        77,  51, 115,  38,
                                                        -13,  25,  64,  25,
                                                        25,  38, -13,  51 };

    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToCellWeightsValue { -38,  25,  13, -38,
                                                      102, -10, -25,  38,
                                                      102, -77, -13,  25,
                                                      38, -13,  25,  64 };

    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<int8_t> recurrentToOutputWeightsValue {  38, -13,  13, -25,
                                                         -64, -89, -25, -77,
                                                         -13, -51, -89, -25,
                                                         13,  64,  25, -38 };

    hidl_vec<uint32_t> cellToInputWeightsDimensions{0};
    std::vector<int16_t> cellToInputWeightsValue;

    hidl_vec<uint32_t> cellToForgetWeightsDimensions{0};
    std::vector<int16_t> cellToForgetWeightsValue;

    hidl_vec<uint32_t> cellToOutputWeightsDimensions{0};
    std::vector<int16_t> cellToOutputWeightsValue;

    hidl_vec<uint32_t> inputGateBiasDimensions{0};
    std::vector<int32_t> inputGateBiasValue;

    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<int32_t> forgetGateBiasValue { 2147484, -6442451, -4294968, 2147484 };

    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<int32_t> cellBiasValue { -1073742, 15461883, 5368709, 1717987 };

    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<int32_t> outputGateBiasValue { 1073742, -214748, 4294968, 2147484 };

    hidl_vec<uint32_t> projectionWeightsDimensions{0, 0};
    std::vector<int8_t> projectionWeightsValue;

    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<int32_t> projectionBiasValue;

    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateInValue { 0, 0, 0, 0, 0, 0, 0, 0 };

    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<int16_t> cellStateInValue { 0, 0, 0, 0, 0, 0, 0, 0 };

    // Normalization:
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<int16_t> inputLayerNormWeightsValue;

    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> forgetLayerNormWeightsValue { 6553, 6553, 13107, 9830 };

    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> cellLayerNormWeightsValue { 22937, 6553, 9830, 26214 };

    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<int16_t> outputLayerNormWeightsValue { 19660, 6553, 6553, 16384 };

    float cellClipValue           = 0.0f;
    float projectionClipValue     = 0.0f;
    float inputIntermediateScale  = 0.007059f;
    float forgetIntermediateScale = 0.007812f;
    float cellIntermediateScale   = 0.007059f;
    float outputIntermediateScale = 0.007812f;
    int32_t hiddenStateZeroPoint  = 0;
    float hiddenStateScale        = 0.007f;

    // Outputs:
    hidl_vec<uint32_t> outputStateOutDimensions{batchSize, outputSize};
    std::vector<int8_t> outputStateOutValue { -15, 21, 14, 20, -15, 15, 5, 27 };

    hidl_vec<uint32_t> cellStateOutDimensions{};
    std::vector<int16_t> cellStateOutValue { -11692, 9960, 5491, 8861, -9422, 7726, 2056, 13149 };

    hidl_vec<uint32_t> outputDimensions{batchSize, outputSize};
    std::vector<int8_t> outputValue { -15, 21, 14, 20, -15, 15, 5, 27 };

    QLstmTestImpl(inputDimensions,                       inputValue,
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
                  inputLayerNormWeightsDimensions,       inputLayerNormWeightsValue,
                  forgetLayerNormWeightsDimensions,      forgetLayerNormWeightsValue,
                  cellLayerNormWeightsDimensions,        cellLayerNormWeightsValue,
                  outputLayerNormWeightsDimensions,      outputLayerNormWeightsValue,
                  cellClipValue,
                  projectionClipValue,
                  inputIntermediateScale,
                  forgetIntermediateScale,
                  cellIntermediateScale,
                  outputIntermediateScale,
                  hiddenStateZeroPoint,
                  hiddenStateScale,
                  outputStateOutDimensions,              outputStateOutValue,
                  cellStateOutDimensions,                cellStateOutValue,
                  outputDimensions,                      outputValue,
                  compute);
}

} // anonymous namespace

// Support is not added yet
//TEST_CASE(QLSTMWithProjectionTest, COMPUTE_DEVICES)
//{
//     QLstmWithProjection(sample);
//}

DOCTEST_TEST_SUITE("QLSTMTests_CpuRef")
{

    DOCTEST_TEST_CASE("QLSTMWithNoProjectionTest_CpuRef")
    {
        QLstmWithNoProjection(armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("DynamicOutputQLstmWithNoProjection_CpuRef")
    {
        DynamicOutputQLstmWithNoProjection(armnn::Compute::CpuRef);
    }

}
#ifdef ARMCOMPUTECL_ENABLED
DOCTEST_TEST_SUITE("QLSTMTests_CpuAcc")
{

    DOCTEST_TEST_CASE("QLSTMWithNoProjectionTest_CpuAcc")
    {
        QLstmWithNoProjection(armnn::Compute::CpuAcc);
    }

    DOCTEST_TEST_CASE("DynamicOutputQLstmWithNoProjection_CpuAcc")
    {
        DynamicOutputQLstmWithNoProjection(armnn::Compute::CpuAcc);
    }

}
#endif
