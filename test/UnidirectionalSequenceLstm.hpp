//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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

// Add our own tests here since we fail the unidirectional sequence lstm
// tests which Google supplies (because of non-const weights)
template <typename HalPolicy>
void UnidirectionalSequenceLstmTestImpl(const hidl_vec<uint32_t>& inputDimensions,
                                        const std::vector<float>& inputValue,
                                        const hidl_vec<uint32_t>& inputToInputWeightsDimensions,
                                        const std::vector<float>& inputToInputWeightsValue,
                                        const hidl_vec<uint32_t>& inputToForgetWeightsDimensions,
                                        const std::vector<float>& inputToForgetWeightsValue,
                                        const hidl_vec<uint32_t>& inputToCellWeightsDimensions,
                                        const std::vector<float>& inputToCellWeightsValue,
                                        const hidl_vec<uint32_t>& inputToOutputWeightsDimensions,
                                        const std::vector<float>& inputToOutputWeightsValue,
                                        const hidl_vec<uint32_t>& recurrentToInputWeightsDimensions,
                                        const std::vector<float>& recurrentToInputWeightsValue,
                                        const hidl_vec<uint32_t>& recurrentToForgetWeightsDimensions,
                                        const std::vector<float>& recurrentToForgetWeightsValue,
                                        const hidl_vec<uint32_t>& recurrentToCellWeightsDimensions,
                                        const std::vector<float>& recurrentToCellWeightsValue,
                                        const hidl_vec<uint32_t>& recurrentToOutputWeightsDimensions,
                                        const std::vector<float>& recurrentToOutputWeightsValue,
                                        const hidl_vec<uint32_t>& cellToInputWeightsDimensions,
                                        const std::vector<float>& cellToInputWeightsValue,
                                        const hidl_vec<uint32_t>& cellToForgetWeightsDimensions,
                                        const std::vector<float>& cellToForgetWeightsValue,
                                        const hidl_vec<uint32_t>& cellToOutputWeightsDimensions,
                                        const std::vector<float>& cellToOutputWeightsValue,
                                        const hidl_vec<uint32_t>& inputGateBiasDimensions,
                                        const std::vector<float>& inputGateBiasValue,
                                        const hidl_vec<uint32_t>& forgetGateBiasDimensions,
                                        const std::vector<float>& forgetGateBiasValue,
                                        const hidl_vec<uint32_t>& cellBiasDimensions,
                                        const std::vector<float>& cellBiasValue,
                                        const hidl_vec<uint32_t>& outputGateBiasDimensions,
                                        const std::vector<float>& outputGateBiasValue,
                                        const hidl_vec<uint32_t>& projectionWeightsDimensions,
                                        const std::vector<float>& projectionWeightsValue,
                                        const hidl_vec<uint32_t>& projectionBiasDimensions,
                                        const std::vector<float>& projectionBiasValue,
                                        const hidl_vec<uint32_t>& outputStateInDimensions,
                                        const std::vector<float>& outputStateInValue,
                                        const hidl_vec<uint32_t>& cellStateInDimensions,
                                        const std::vector<float>& cellStateInValue,
                                        const hidl_vec<uint32_t>& activationFunctionDimensions,
                                        const std::vector<int32_t>& activationFunctionValue,
                                        const hidl_vec<uint32_t>& cellClippingThresholdDimensions,
                                        const std::vector<float>& cellClippingThresholdValue,
                                        const hidl_vec<uint32_t>& projectionClippingThresholdDimensions,
                                        const std::vector<float>& projectionClippingThresholdValue,
                                        const bool& timeMajorValue,
                                        const hidl_vec<uint32_t>& inputLayerNormWeightsDimensions,
                                        const std::vector<float>& inputLayerNormWeightsValue,
                                        const hidl_vec<uint32_t>& forgetLayerNormWeightsDimensions,
                                        const std::vector<float>& forgetLayerNormWeightsValue,
                                        const hidl_vec<uint32_t>& cellLayerNormWeightsDimensions,
                                        const std::vector<float>& cellLayerNormWeightsValue,
                                        const hidl_vec<uint32_t>& outputLayerNormWeightsDimensions,
                                        const std::vector<float>& outputLayerNormWeightsValue,
                                        const hidl_vec<uint32_t>& outputDimensions,
                                        const std::vector<float>& outputValue,
                                        const hidl_vec<uint32_t>&, // outputStateOutDimensions,
                                        const std::vector<float>&, // outputStateOutValue,
                                        const hidl_vec<uint32_t>&, // cellStateOutDimensions,
                                        const std::vector<float>&, // cellStateOutValue,
                                        armnn::Compute compute,
                                        float epsilonValue = 0)
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

    // 23: Time-major if true, batch-major if false.
    AddBoolOperand<HalPolicy>(model, timeMajorValue);

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    AddTensorOperand<HalPolicy>(model,
                                inputLayerNormWeightsDimensions,
                                inputLayerNormWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(inputLayerNormWeightsDimensions));
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    AddTensorOperand<HalPolicy>(model,
                                forgetLayerNormWeightsDimensions,
                                forgetLayerNormWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(forgetLayerNormWeightsDimensions));
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    AddTensorOperand<HalPolicy>(model,
                                cellLayerNormWeightsDimensions,
                                cellLayerNormWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(cellLayerNormWeightsDimensions));
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    AddTensorOperand<HalPolicy>(model,
                                outputLayerNormWeightsDimensions,
                                outputLayerNormWeightsValue,
                                HalPolicy::OperandType::TENSOR_FLOAT32,
                                CreateNoValueLifeTime(outputLayerNormWeightsDimensions));

    // Outputs:
    // 00: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    // [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    AddOutputOperand<HalPolicy>(model, outputDimensions);
    // 01: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    // [batch_size, output_size]. This output is optional and can be omitted. If this output
    // is present then output #2 must be present as well.
    //AddOutputOperand<HalPolicy>(model, hiddenStateOutDimensions);
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    // [batch_size, num_units]. This output is optional and can be omitted.
    //AddOutputOperand<HalPolicy>(model, cellStateOutDimensions);

    // make the lstm operation
    model.operations.resize(1);
    model.operations[0].type = HalPolicy::OperationType::UNIDIRECTIONAL_SEQUENCE_LSTM;

    model.operations[0].inputs = hidl_vec<uint32_t> {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                                                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
    model.operations[0].outputs = hidl_vec<uint32_t> {28};

    // define the input values
    hidl_vec<RequestArgument> inputArguments;
    inputArguments.resize(3);

    inputArguments[0] = CreateRequestArgument<float>(inputValue, 0);
    inputArguments[1] = CreateRequestArgument<float>(outputStateInValue, 1);
    inputArguments[2] = CreateRequestArgument<float>(cellStateInValue, 2);

    // define the expected output values
    hidl_vec<RequestArgument> outputArguments;
    outputArguments.resize(1);

    outputArguments[0] = CreateRequestArgument<float>(outputValue, 3);

    V1_0::Request request = {};
    request.inputs  = inputArguments;
    request.outputs = outputArguments;

    // set the input data
    AddPoolAndSetData(inputValue.size(), request, inputValue.data());
    AddPoolAndSetData(outputStateInValue.size(), request, outputStateInValue.data());
    AddPoolAndSetData(cellStateInValue.size(), request, cellStateInValue.data());

    // add memory for the outputs
    android::sp<IMemory> outputMemory = AddPoolAndGetData<float>(outputValue.size(), request);
    float* outputData = static_cast<float*>(static_cast<void*>(outputMemory->getPointer()));

    // make the prepared model and run the execution
    ExecuteModel(model, *driver, request);

    // check the results
    if (epsilonValue != 0)
    {
        for (size_t i = 0; i < outputValue.size(); ++i)
        {
            DOCTEST_CHECK_MESSAGE(outputValue[i] == doctest::Approx(outputData[i]).epsilon(epsilonValue),
                                  "outputValue[" << i << "]: " << outputValue[i] << " != " << outputData[i]);
        }
    }
    else
    {
        for (size_t i = 0; i < outputValue.size(); ++i)
        {
            DOCTEST_CHECK_MESSAGE(outputValue[i] == doctest::Approx(outputData[i]),
                                  "outputValue[" << i << "]: " << outputValue[i] << " != " << outputData[i]);
        }
    }
}

template<typename HalPolicy>
void UnidirectionalSequenceLstmLayerFloat32TestImpl(armnn::Compute compute)
{
    uint32_t batchSize  = 3;
    uint32_t timeSize   = 2;
    uint32_t inputSize  = 3;
    uint32_t outputSize = 4;
    uint32_t numUnits   = outputSize;

    // Inputs:
    // 00: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    //     [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length),
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, timeSize, inputSize};
    std::vector<float> inputValue{1., 2., 3., 4., 5., 4.,
                                  3., 2., 1., 2., 3., 4.,
                                  5., 4., 3., 2., 1., 2.};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{-0.49536117f, -0.0556083915f, -0.102400711f,
                                                -0.117484632f, 0.3298470976f, -0.1179017122f,
                                                0.214305695f, 0.42135173085f, 0.003878414626f,
                                                -0.348303917f, -0.1881275477f, 0.0343011027f};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                 -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                 -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                 0.13512269116f, -0.4932442977f, -0.10039821991f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.2504855627f, 0.184490025045f, -0.2480507493f,
                                               0.386399507f, -0.259465157985f, -0.16545993089f,
                                               -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                               -0.2277662414f, -0.55275535589f, 0.34184026718f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                 0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                 0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                 -0.51717478049f, 0.1889653282f, -0.367323637f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{-0.128009796112f, 0.1995525098f, -0.07745539397f, 0.1558421701f,
                                                    -0.265254765766f, -0.38837709614f, -0.05636804124f, 0.4259087456f,
                                                    0.17628988623f, 0.3877420127f, 0.53300309181f, -0.0959980934f,
                                                    0.00302857416f, 0.3266998827f, -0.142509296562f, -0.04433270756f};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                     -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                     -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                     -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                   -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                   0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                   0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{-0.32921677827f, 0.32624614238f, -0.1388191282f,
                                                     -0.17879831790f, -0.15185534954f, -0.16918526583f,
                                                     -0.10087361183f, -0.5436913968f, 0.016758225858f,
                                                     0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                     -0.12584099173f, -0.12319286912f, 0.2407919466f,
                                                     -0.08879069983f};
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
    hidl_vec<uint32_t>   activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   cellClippingThresholdDimensions{};
    std::vector<float>   cellClippingThresholdValue{10.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   projectionClippingThresholdDimensions{};
    std::vector<float>   projectionClippingThresholdValue{0.f};

    // 23: Time-major if true, batch-major if false.
    bool timeMajorValue = false;

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    // 0: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    //    [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    hidl_vec<uint32_t> outputDimensions{batchSize, timeSize, outputSize};
    std::vector<float> outputValue{-0.07149004f, -0.1621171f, -0.17516759f, -0.0232934225f,
                                   -0.16810727f, -0.41412935f, -0.5498753f, -0.00803578f,
                                   -0.06687349f, 0.204077631f, -0.4276504f, -0.03123213f,
                                   -0.12000261f, -0.0941918f, -0.45639035f, -0.02870186f,
                                   -0.03429216f, 0.20824050f, -0.6569892f, -0.004152651f,
                                   -0.10493034f, 0.14210969f, -0.58347696f, -0.03297536f};

    // 1: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, output_size]. This output is optional and can be omitted. If this output
    //    is present then output #2 must be present as well.
    hidl_vec<uint32_t> hiddenStateOutDimensions{batchSize, outputSize};
    std::vector<float> hiddenStateOutValue(batchSize * outputSize, 0.f);
    // 2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, num_units]. This output is optional and can be omitted.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue(batchSize * numUnits, 0.f);

    UnidirectionalSequenceLstmTestImpl<HalPolicy>(inputDimensions, inputValue,
                                                  inputToInputWeightsDimensions, inputToInputWeightsValue,
                                                  inputToForgetWeightsDimensions, inputToForgetWeightsValue,
                                                  inputToCellWeightsDimensions, inputToCellWeightsValue,
                                                  inputToOutputWeightsDimensions, inputToOutputWeightsValue,
                                                  recurrentToInputWeightsDimensions, recurrentToInputWeightsValue,
                                                  recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue,
                                                  recurrentToCellWeightsDimensions, recurrentToCellWeightsValue,
                                                  recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue,
                                                  cellToInputWeightsDimensions, cellToInputWeightsValue,
                                                  cellToForgetWeightsDimensions, cellToForgetWeightsValue,
                                                  cellToOutputWeightsDimensions, cellToOutputWeightsValue,
                                                  inputGateBiasDimensions, inputGateBiasValue,
                                                  forgetGateBiasDimensions, forgetGateBiasValue,
                                                  cellBiasDimensions, cellBiasValue,
                                                  outputGateBiasDimensions, outputGateBiasValue,
                                                  projectionWeightsDimensions, projectionWeightsValue,
                                                  projectionBiasDimensions, projectionBiasValue,
                                                  outputStateInDimensions, outputStateInValue,
                                                  cellStateInDimensions, cellStateInValue,
                                                  activationFunctionDimensions, activationFunctionValue,
                                                  cellClippingThresholdDimensions, cellClippingThresholdValue,
                                                  projectionClippingThresholdDimensions,
                                                  projectionClippingThresholdValue,
                                                  timeMajorValue,
                                                  inputLayerNormWeightsDimensions, inputLayerNormWeightsValue,
                                                  forgetLayerNormWeightsDimensions, forgetLayerNormWeightsValue,
                                                  cellLayerNormWeightsDimensions, cellLayerNormWeightsValue,
                                                  outputLayerNormWeightsDimensions, outputLayerNormWeightsValue,
                                                  outputDimensions, outputValue,
                                                  hiddenStateOutDimensions, hiddenStateOutValue,
                                                  cellStateOutDimensions, cellStateOutValue,
                                                  compute);
}

template<typename HalPolicy>
void UnidirectionalSequenceLstmLayerFloat32TimeMajorTestImpl(armnn::Compute compute)
{
    uint32_t batchSize  = 3;
    uint32_t timeSize   = 2;
    uint32_t inputSize  = 3;
    uint32_t outputSize = 4;
    uint32_t numUnits   = outputSize;

    // Inputs:
    // 00: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    //     [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length),
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{timeSize, batchSize, inputSize};
    std::vector<float> inputValue{1., 2., 3., 4., 5., 4.,
                                  3., 2., 1., 2., 3., 4.,
                                  5., 4., 3., 2., 1., 2.};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{0.27277296781539917f, 0.3813590407371521f, -0.394489049911499f,
                                                0.2782636880874634f, -0.3793870210647583f, -0.018918335437774658f,
                                                0.2724653482437134f, -0.19314253330230713f, -0.2947450876235962f,
                                                -0.30253493785858154f, 0.4241350293159485f, -0.22560018301010132f};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.2667974531650543f, -0.05505800247192383f, -0.20932340621948242f,
                                                 -0.14345619082450867f, 0.09666192531585693f, -0.2604355812072754f,
                                                 -0.2681812047958374f, -0.3314584493637085f, 0.4485899806022644f,
                                                 -0.23467743396759033f, 0.5072842240333557f, -0.4192768931388855f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.15782442688941956f, -0.027530014514923096f, 0.4789854884147644f,
                                               0.23227906227111816f, 0.28259342908859253f, -0.030095696449279785f,
                                               0.10071521997451782f, -0.08535495400428772f, 0.18563997745513916f,
                                               -0.3049069046974182f, -0.478048175573349f, 0.025234103202819824f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{-0.04584759473800659f, -0.2716066539287567f, 0.012970447540283203f,
                                                 -0.4729190170764923f, -0.37422770261764526f, 0.49352723360061646f,
                                                 0.3163864016532898f, -0.436781644821167f, -0.33074596524238586f,
                                                 -0.32885751128196716f, -0.40959352254867554f, -0.2124689817428589f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{0.23788475990f, -0.24948765337f, 0.50044941902f,
                                                    0.14431896805f, -0.115940228137f, -0.717082679f,
                                                    -0.17208620906f, 0.17850610617f, -0.16702319684f,
                                                    -0.11384502053f, -0.309785276245f, -0.3316611672f,
                                                    0.52380162477f, -0.06839632987f, -0.391478359627f,
                                                    -0.10756178963f};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{0.11383482068f, 0.1676601767f, -0.08550968004f, 0.03399394089f,
                                                     0.08042152225f, -0.2133381964f, 0.05182432704f, 0.38161808255f,
                                                     -0.5018365979f, -0.08043262364f, 0.07894329014f, -0.07547105155f,
                                                     0.12047368288f, 0.2986997961f, 0.0485043078f, -0.13372567296f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{0.0433832928545f, 0.07587072294f, -0.120520234107f, 0.604576051f,
                                                   -0.434353142986f, 0.009314475068f, 0.005085289478f, 0.08488202038f,
                                                   -0.00025437487886f, 0.15245915082f, -0.1936587542f, 0.004754020f,
                                                   -0.1582719236f, 0.3307867646f, 0.0236605107784f, 0.307716339826f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{-0.079031050201f, 0.041414566286f, -0.583727357285f,
                                                     0.1025384515f, -0.172372072937f, 0.09214124082f,
                                                     0.178184121827f, -0.2439443916f, 0.104485116899f,
                                                     0.2600405514f, 0.064414866268f, 0.24141204357f,
                                                     0.281875759363f, -0.14234502664f, 0.15126448862f,
                                                     -0.24421440064f};
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
    hidl_vec<uint32_t>   activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   cellClippingThresholdDimensions{};
    std::vector<float>   cellClippingThresholdValue{10.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   projectionClippingThresholdDimensions{};
    std::vector<float>   projectionClippingThresholdValue{0.f};

    // 23: Time-major if true, batch-major if false.
    bool timeMajorValue = true;

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    // 0: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    //    [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    hidl_vec<uint32_t> outputDimensions{timeSize, batchSize, outputSize};
    std::vector<float> outputValue{0.135657698f, 0.124672532f, 0.0212090332f, -0.0530203655f,
                                   0.106138252f, 0.0404792242f, 0.0151643595f, -0.00675163185f,
                                   -0.0128514022f, 0.0644884035f, 0.0709072053f, -0.0454045124f,
                                   0.16288602f,  0.16649379f,  0.02770456f, -0.03698075f,
                                   0.11171641f,  0.043119f  ,  0.0762981f , -0.01228541f,
                                   0.10439701f,  0.21439962f,  0.11919238f, -0.08390583f};

    // 1: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, output_size]. This output is optional and can be omitted. If this output
    //    is present then output #2 must be present as well.
    hidl_vec<uint32_t> hiddenStateOutDimensions{batchSize, outputSize};
    std::vector<float> hiddenStateOutValue(batchSize * outputSize, 0.f);
    // 2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, num_units]. This output is optional and can be omitted.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue(batchSize * numUnits, 0.f);

    UnidirectionalSequenceLstmTestImpl<HalPolicy>(inputDimensions, inputValue,
                                                  inputToInputWeightsDimensions, inputToInputWeightsValue,
                                                  inputToForgetWeightsDimensions, inputToForgetWeightsValue,
                                                  inputToCellWeightsDimensions, inputToCellWeightsValue,
                                                  inputToOutputWeightsDimensions, inputToOutputWeightsValue,
                                                  recurrentToInputWeightsDimensions, recurrentToInputWeightsValue,
                                                  recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue,
                                                  recurrentToCellWeightsDimensions, recurrentToCellWeightsValue,
                                                  recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue,
                                                  cellToInputWeightsDimensions, cellToInputWeightsValue,
                                                  cellToForgetWeightsDimensions, cellToForgetWeightsValue,
                                                  cellToOutputWeightsDimensions, cellToOutputWeightsValue,
                                                  inputGateBiasDimensions, inputGateBiasValue,
                                                  forgetGateBiasDimensions, forgetGateBiasValue,
                                                  cellBiasDimensions, cellBiasValue,
                                                  outputGateBiasDimensions, outputGateBiasValue,
                                                  projectionWeightsDimensions, projectionWeightsValue,
                                                  projectionBiasDimensions, projectionBiasValue,
                                                  outputStateInDimensions, outputStateInValue,
                                                  cellStateInDimensions, cellStateInValue,
                                                  activationFunctionDimensions, activationFunctionValue,
                                                  cellClippingThresholdDimensions, cellClippingThresholdValue,
                                                  projectionClippingThresholdDimensions,
                                                  projectionClippingThresholdValue,
                                                  timeMajorValue,
                                                  inputLayerNormWeightsDimensions, inputLayerNormWeightsValue,
                                                  forgetLayerNormWeightsDimensions, forgetLayerNormWeightsValue,
                                                  cellLayerNormWeightsDimensions, cellLayerNormWeightsValue,
                                                  outputLayerNormWeightsDimensions, outputLayerNormWeightsValue,
                                                  outputDimensions, outputValue,
                                                  hiddenStateOutDimensions, hiddenStateOutValue,
                                                  cellStateOutDimensions, cellStateOutValue,
                                                  compute);
}

template<typename HalPolicy>
void UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionTestImpl(armnn::Compute compute)
{
    uint32_t batchSize  = 2;
    uint32_t timeSize   = 3;
    uint32_t inputSize  = 4;
    uint32_t outputSize = 5;
    uint32_t numUnits   = 6;

    // Inputs:
    // 00: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    //     [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length),
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, timeSize, inputSize};
    std::vector<float> inputValue{1., 2., 3., 4., 5., 4.,
                                  3., 2., 1., 2., 3., 4.,
                                  5., 4., 3., 2., 1., 2.,
                                  1., 2., 3., 4., 5., 4.};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{0.021393683f, 0.06124551f, 0.046905167f, -0.014657677f,
                                                -0.03149463f, 0.09171803f, 0.14647801f, 0.10797193f,
                                                -0.0057968358f, 0.0019193048f, -0.2726754f, 0.10154029f,
                                                -0.018539885f, 0.080349885f, -0.10262385f, -0.022599787f,
                                                -0.09121155f, -0.008675967f, -0.045206103f, -0.0821282f,
                                                -0.008045952f, 0.015478081f, 0.055217247f, 0.038719587f};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{-0.0018401089f, -0.004852237f, 0.03698424f, 0.014181704f,
                                                 0.028273236f, -0.016726194f, -0.05249759f, -0.10204261f,
                                                 0.00861066f, -0.040979505f, -0.009899187f, 0.01923892f,
                                                 -0.028177269f, -0.08535103f, -0.14585495f, 0.10662567f,
                                                 -0.01909731f, -0.017883534f, -0.0047269356f, -0.045103323f,
                                                 0.0030784295f, 0.076784775f, 0.07463696f, 0.094531395f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.04580283f, -0.09549462f, -0.032418985f, -0.06454633f,
                                               -0.043528453f, 0.043018587f, -0.049152344f, -0.12418144f,
                                               -0.078985475f, -0.07596889f, 0.019484362f, -0.11434962f,
                                               -0.0074034138f, -0.06314844f, -0.092981495f, 0.0062155537f,
                                               -0.025034338f, -0.0028890965f, 0.048929527f, 0.06235075f,
                                               0.10665918f, -0.032036792f, -0.08505916f, -0.10843358f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{-0.0998932f, -0.07201956f, -0.052803773f, -0.15629593f,
                                                 -0.15001918f, -0.07650751f, 0.02359855f, -0.075155355f,
                                                 -0.08037709f, -0.15093534f, 0.029517552f, -0.04751393f,
                                                 0.010350531f, -0.02664851f, -0.016839722f, -0.023121163f,
                                                 0.0077019283f, 0.012851257f, -0.05040649f, -0.0129761f,
                                                 -0.021737747f, -0.038305793f, -0.06870586f, -0.01481247f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{-0.001374326f, -0.078856036f, 0.10672688f, 0.029162422f,
                                                    -0.11585556f, 0.02557986f, -0.13446963f, -0.035785314f,
                                                    -0.01244275f, 0.025961924f, -0.02337298f, -0.044228926f,
                                                    -0.055839065f, -0.046598054f, -0.010546039f, -0.06900766f,
                                                    0.027239809f, 0.022582639f, -0.013296484f, -0.05459212f,
                                                    0.08981f, -0.045407712f, 0.08682226f, -0.06867011f,
                                                    -0.14390695f, -0.02916037f, 0.000996957f, 0.091420636f,
                                                    0.14283475f, -0.07390571f};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.057784554f, -0.026057621f, -0.068447545f, -0.022581743f,
                                                     0.14811787f, 0.10826372f, 0.09471067f, 0.03987225f,
                                                     -0.0039523416f, 0.00030638507f, 0.053185795f, 0.10572994f,
                                                     0.08414449f, -0.022036452f, -0.00066928595f, -0.09203576f,
                                                     0.032950465f, -0.10985798f, -0.023809856f, 0.0021431844f,
                                                     -0.02196096f, -0.00326074f, 0.00058621005f, -0.074678116f,
                                                     -0.06193199f, 0.055729095f, 0.03736828f, 0.020123724f,
                                                     0.061878487f, -0.04729229f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.037322544f, 0.018592842f, 0.0056175636f, -0.06253426f,
                                                   0.055647098f, -0.05713207f, -0.05626563f, 0.005559383f,
                                                   0.03375411f, -0.025757805f, -0.088049285f, 0.06017052f,
                                                   -0.06570978f, 0.007384076f, 0.035123326f, -0.07920549f,
                                                   0.053676967f, 0.044480428f, -0.07663568f, 0.0071805613f,
                                                   0.08089997f, 0.05143358f, 0.038261272f, 0.03339287f,
                                                   -0.027673481f, 0.044746667f, 0.028349208f, 0.020090483f,
                                                   -0.019443132f, -0.030755889f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{0.025825322f, -0.05813119f, 0.09495884f,
                                                     -0.045984812f,-0.01255415f, -0.0026479573f,
                                                     -0.08196161f, -0.054914974f, -0.0046604523f,
                                                     -0.029587349f, -0.044576716f, -0.07480124f,
                                                     -0.082868785f, 0.023254942f, 0.027502948f,
                                                     -0.0039728214f, -0.08683098f, -0.08116779f,
                                                     -0.014675607f, -0.037924774f, -0.023314456f,
                                                     -0.007401714f, -0.09255757f, 0.029460307f,
                                                     -0.08829125f, -0.005139627f, -0.08989442f,
                                                     -0.0555066f, 0.13596267f, 0.025062224f};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{numUnits};
    std::vector<float> cellToInputWeightsValue{0.040369894f, 0.030746894f, 0.24704495f,
                                               0.018586371f, -0.037586458f, -0.15312155f};
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue{-0.01998659f, -0.15568835f, -0.24248174f,
                                                -0.012770197f, 0.041331276f, -0.072311886f};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue{0.08286371f, -0.08261836f, -0.51210177f,
                                                0.002913762f, 0.17764764f, -0.5495371f};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<float> inputGateBiasValue{0.02234832f, 0.14757581f, 0.18176508f,
                                          0.10380666f, 0.053110216f, -0.06928846f};
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{0.035185695f, -0.042891346f, -0.03032477f,
                                           0.23027696f, 0.11098921f, 0.08989442f};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue{-0.024379363f, 0.0055531194f, 0.23377132f,
                                     0.033463873f, -0.1483596f, 0.029460307f};
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue{0.046159424f, -0.0012809046f, 0.03563469f,
                                           0.12648113f, 0.027195795f, 0.35373217f};
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{numUnits, outputSize};
    std::vector<float> projectionWeightsValue{-0.009802181f, 0.09401916f, 0.0717386f, -0.13895074f, 0.09641832f,
                                              0.060420845f, 0.08539281f, 0.054285463f, 0.061395317f, 0.034448683f,
                                              -0.042991187f, 0.019801661f, -0.16840284f, -0.015726732f, -0.23041931f,
                                              -0.024478018f, -0.10959692f, -0.013875541f, 0.18600968f, -0.061274476f,
                                              0.0138165f, -0.08160894f, -0.07661644f, 0.032372914f, 0.16169067f,
                                              0.22465782f, -0.03993472f, -0.004017731f, 0.08633481f, -0.28869787f};
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<float> projectionBiasValue(outputSize, 0.f);

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t>   activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   cellClippingThresholdDimensions{};
    std::vector<float>   cellClippingThresholdValue{10.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   projectionClippingThresholdDimensions{};
    std::vector<float>   projectionClippingThresholdValue{0.f};

    // 23: Time-major if true, batch-major if false.
    bool timeMajorValue = false;

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    // 0: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    //    [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    hidl_vec<uint32_t> outputDimensions{batchSize, timeSize, outputSize};
    std::vector<float> outputValue{-0.0135612f, -0.0263441f, 0.0314008f, -0.00883455f, 0.00763052f,
                                   -0.00126877f, -0.0292959f, 0.0449957f, -0.00976195f, -0.00492338f,
                                   -0.0175702f, -0.0431753f, 0.0597117f, -0.0169154f, 0.0142087f,
                                   0.00472515f, -0.0196355f, 0.0342524f, -0.00407936f, -0.0253189f,
                                   -0.00512944f, -0.0293754f, 0.0512771f, -0.0151874f, -0.0246433f,
                                   -0.00744986f, -0.0345103f, 0.0450666f, -0.00944991f, 0.0127171f};

    // 1: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, output_size]. This output is optional and can be omitted. If this output
    //    is present then output #2 must be present as well.
    hidl_vec<uint32_t> hiddenStateOutDimensions{batchSize, outputSize};
    std::vector<float> hiddenStateOutValue(batchSize * outputSize, 0.f);
    // 2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, num_units]. This output is optional and can be omitted.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue(batchSize * numUnits, 0.f);

    UnidirectionalSequenceLstmTestImpl<HalPolicy>(inputDimensions, inputValue,
                                                  inputToInputWeightsDimensions, inputToInputWeightsValue,
                                                  inputToForgetWeightsDimensions, inputToForgetWeightsValue,
                                                  inputToCellWeightsDimensions, inputToCellWeightsValue,
                                                  inputToOutputWeightsDimensions, inputToOutputWeightsValue,
                                                  recurrentToInputWeightsDimensions, recurrentToInputWeightsValue,
                                                  recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue,
                                                  recurrentToCellWeightsDimensions, recurrentToCellWeightsValue,
                                                  recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue,
                                                  cellToInputWeightsDimensions, cellToInputWeightsValue,
                                                  cellToForgetWeightsDimensions, cellToForgetWeightsValue,
                                                  cellToOutputWeightsDimensions, cellToOutputWeightsValue,
                                                  inputGateBiasDimensions, inputGateBiasValue,
                                                  forgetGateBiasDimensions, forgetGateBiasValue,
                                                  cellBiasDimensions, cellBiasValue,
                                                  outputGateBiasDimensions, outputGateBiasValue,
                                                  projectionWeightsDimensions, projectionWeightsValue,
                                                  projectionBiasDimensions, projectionBiasValue,
                                                  outputStateInDimensions, outputStateInValue,
                                                  cellStateInDimensions, cellStateInValue,
                                                  activationFunctionDimensions, activationFunctionValue,
                                                  cellClippingThresholdDimensions, cellClippingThresholdValue,
                                                  projectionClippingThresholdDimensions,
                                                  projectionClippingThresholdValue,
                                                  timeMajorValue,
                                                  inputLayerNormWeightsDimensions, inputLayerNormWeightsValue,
                                                  forgetLayerNormWeightsDimensions, forgetLayerNormWeightsValue,
                                                  cellLayerNormWeightsDimensions, cellLayerNormWeightsValue,
                                                  outputLayerNormWeightsDimensions, outputLayerNormWeightsValue,
                                                  outputDimensions, outputValue,
                                                  hiddenStateOutDimensions, hiddenStateOutValue,
                                                  cellStateOutDimensions, cellStateOutValue,
                                                  compute, 0.0031454);
}

template<typename HalPolicy>
void UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTestImpl(armnn::Compute compute)
{
    uint32_t batchSize  = 3;
    uint32_t timeSize   = 2;
    uint32_t inputSize  = 3;
    uint32_t outputSize = 4;
    uint32_t numUnits   = 5;

    // Inputs:
    // 00: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    //     [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length),
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, timeSize, inputSize};
    std::vector<float> inputValue{1., 2., 3., 4., 5., 4.,
                                  3., 2., 1., 2., 3., 4.,
                                  5., 4., 3., 2., 1., 2.};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToInputWeightsValue{-0.49536117f, -0.0556083915f, -0.102400711f,
                                                -0.117484632f, 0.3298470976f, -0.1179017122f,
                                                0.214305695f, 0.42135173085f, 0.003878414626f,
                                                -0.348303917f, -0.1881275477f, 0.0343011027f,
                                                -0.38837709614f, -0.05636804124f, 0.4259087456f};
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                 -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                 -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                 0.13512269116f, -0.4932442977f, -0.10039821991f,
                                                 0.2726137042f, 0.09216640889f, -0.06551410215f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.2504855627f, 0.184490025045f, -0.2480507493f,
                                               0.386399507f, -0.259465157985f, -0.16545993089f,
                                               -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                               -0.2277662414f, -0.55275535589f, 0.34184026718f,
                                               0.3954237699f, -0.19407111404f, 0.30412107706f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                 0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                 0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                 -0.51717478049f, 0.1889653282f, -0.367323637f,
                                                 -0.12584099173f, -0.12319286912f, 0.2407919466f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToInputWeightsValue{-0.128009796112f, 0.1995525098f, -0.07745539397f, 0.1558421701f,
                                                    -0.265254765766f, -0.38837709614f, -0.05636804124f, 0.4259087456f,
                                                    0.17628988623f, 0.3877420127f, 0.53300309181f, -0.0959980934f,
                                                    0.00302857416f, 0.3266998827f, -0.142509296562f, -0.04433270756f,
                                                    0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f};
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                     -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                     -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                     -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f,
                                                     0.01841056f, -0.32764608f, -0.33027974f, -0.10826075f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                   -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                   0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                   0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f,
                                                   0.19069612f, -0.03026325f, -0.54532051f, 0.33003211f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{-0.32921677827f, 0.32624614238f, -0.1388191282f,
                                                     -0.17879831790f,-0.15185534954f, -0.16918526583f,
                                                     -0.10087361183f, -0.5436913968f, 0.016758225858f,
                                                     0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                     -0.12584099173f, -0.12319286912f, 0.2407919466f,
                                                     -0.08879069983f, 0.11178309f, 0.09481031f,
                                                     -0.26424935f, 0.46261835f};
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToInputWeightsDimensions{numUnits};
    std::vector<float> cellToInputWeightsValue{0.05f, 0.1f, 0.25f, 0.15f, -0.02f};
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToForgetWeightsDimensions{numUnits};
    std::vector<float> cellToForgetWeightsValue{-0.02f, -0.15f, -0.25f, -0.03f, 0.15f};
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellToOutputWeightsDimensions{numUnits};
    std::vector<float> cellToOutputWeightsValue{0.1f, -0.1f, -0.5f, 0.05f, 0.01f};
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> inputGateBiasDimensions{numUnits};
    std::vector<float> inputGateBiasValue{0.03f, 0.15f, 0.22f, 0.38f, 0.05f};
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{0.1f, -0.3f, -0.2f, 0.1f, 0.4f};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue{-0.05f, 0.72f, 0.25f, 0.08f, 0.1f};
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue{0.05f, -0.01f, 0.2f, 0.1f, -0.2f};
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{numUnits, outputSize};
    std::vector<float> projectionWeightsValue{-0.1f, 0.2f, 0.01f, -0.2f,
                                              0.1f, 0.5f,  0.3f, 0.08f,
                                              0.07f, 0.2f, -0.4f,  0.2f,
                                              0.5f, -0.4f, 0.3f, -0.2f,
                                              0.3f, 0.08f, -0.07f, 0.2f};
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{outputSize};
    std::vector<float> projectionBiasValue(outputSize, 0.f);

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t>   activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   cellClippingThresholdDimensions{};
    std::vector<float>   cellClippingThresholdValue{10.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   projectionClippingThresholdDimensions{};
    std::vector<float>   projectionClippingThresholdValue{0.f};

    // 23: Time-major if true, batch-major if false.
    bool timeMajorValue = false;

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{numUnits};
    std::vector<float> inputLayerNormWeightsValue{0.1f, 0.2f, 0.3f, 0.5f, 0.8f};
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{numUnits};
    std::vector<float> forgetLayerNormWeightsValue{0.1f, 0.2f, 0.3f, 0.5f, 0.2f};
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{numUnits};
    std::vector<float> cellLayerNormWeightsValue{0.7f, 0.2f, 0.3f, 0.8f, 0.5f};
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{numUnits};
    std::vector<float> outputLayerNormWeightsValue{0.6f, 0.2f, 0.2f, 0.5f, 0.1f};

    // Outputs:
    // 0: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    //    [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    hidl_vec<uint32_t> outputDimensions{batchSize, timeSize, outputSize};
    std::vector<float> outputValue{0.0642256f, 0.0343966f, 0.184122f, 0.114717f,
                                   0.11458f, 0.0407109f, 0.300327f, 0.174301f,
                                   0.0864761f, 0.0362912f, 0.178635f, 0.115689f,
                                   0.108008f, 0.0386623f, 0.273471f, 0.167115f,
                                   0.0859545f, 0.0331481f, 0.186051f, 0.11888f,
                                   0.106649f, 0.0276847f, 0.229863f, 0.166958f};

    // 1: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, output_size]. This output is optional and can be omitted. If this output
    //    is present then output #2 must be present as well.
    hidl_vec<uint32_t> hiddenStateOutDimensions{batchSize, outputSize};
    std::vector<float> hiddenStateOutValue(batchSize * outputSize, 0.f);
    // 2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, num_units]. This output is optional and can be omitted.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue(batchSize * numUnits, 0.f);

    UnidirectionalSequenceLstmTestImpl<HalPolicy>(inputDimensions, inputValue,
                                                  inputToInputWeightsDimensions, inputToInputWeightsValue,
                                                  inputToForgetWeightsDimensions, inputToForgetWeightsValue,
                                                  inputToCellWeightsDimensions, inputToCellWeightsValue,
                                                  inputToOutputWeightsDimensions, inputToOutputWeightsValue,
                                                  recurrentToInputWeightsDimensions, recurrentToInputWeightsValue,
                                                  recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue,
                                                  recurrentToCellWeightsDimensions, recurrentToCellWeightsValue,
                                                  recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue,
                                                  cellToInputWeightsDimensions, cellToInputWeightsValue,
                                                  cellToForgetWeightsDimensions, cellToForgetWeightsValue,
                                                  cellToOutputWeightsDimensions, cellToOutputWeightsValue,
                                                  inputGateBiasDimensions, inputGateBiasValue,
                                                  forgetGateBiasDimensions, forgetGateBiasValue,
                                                  cellBiasDimensions, cellBiasValue,
                                                  outputGateBiasDimensions, outputGateBiasValue,
                                                  projectionWeightsDimensions, projectionWeightsValue,
                                                  projectionBiasDimensions, projectionBiasValue,
                                                  outputStateInDimensions, outputStateInValue,
                                                  cellStateInDimensions, cellStateInValue,
                                                  activationFunctionDimensions, activationFunctionValue,
                                                  cellClippingThresholdDimensions, cellClippingThresholdValue,
                                                  projectionClippingThresholdDimensions,
                                                  projectionClippingThresholdValue,
                                                  timeMajorValue,
                                                  inputLayerNormWeightsDimensions, inputLayerNormWeightsValue,
                                                  forgetLayerNormWeightsDimensions, forgetLayerNormWeightsValue,
                                                  cellLayerNormWeightsDimensions, cellLayerNormWeightsValue,
                                                  outputLayerNormWeightsDimensions, outputLayerNormWeightsValue,
                                                  outputDimensions, outputValue,
                                                  hiddenStateOutDimensions, hiddenStateOutValue,
                                                  cellStateOutDimensions, cellStateOutValue,
                                                  compute);
}

template<typename HalPolicy>
void UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTestImpl(armnn::Compute compute)
{
    uint32_t batchSize  = 3;
    uint32_t timeSize   = 2;
    uint32_t inputSize  = 3;
    uint32_t outputSize = 4;
    uint32_t numUnits   = outputSize;

    // Inputs:
    // 00: The input: A 3-D tensor of shape: If time-major: [max_time, batch_size, input_size] If batch-major:
    //     [batch_size, max_time, input_size] where “max_time” is the number of timesteps (sequence length),
    //     “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    hidl_vec<uint32_t> inputDimensions{batchSize, timeSize, inputSize};
    std::vector<float> inputValue{1., 2., 3., 4., 5., 4.,
                                  3., 2., 1., 2., 3., 4.,
                                  5., 4., 3., 2., 1., 2.};

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    hidl_vec<uint32_t> inputToInputWeightsDimensions{0};
    std::vector<float> inputToInputWeightsValue;
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToForgetWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToForgetWeightsValue{0.2415594226f, 0.15400093799f, 0.4566498398f,
                                                 -0.3810434485f, 0.268383264f, -0.009807467424f,
                                                 -0.3522925403f, -0.24275735512f, -0.28344226125f,
                                                 0.13512269116f, -0.4932442977f, -0.10039821991f};
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    hidl_vec<uint32_t> inputToCellWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToCellWeightsValue{-0.2504855627f, 0.184490025045f, -0.2480507493f,
                                               0.386399507f, -0.259465157985f, -0.16545993089f,
                                               -0.4230232555f, 0.341664791103f, -0.18127849691f,
                                               -0.2277662414f, -0.55275535589f, 0.34184026718f};
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    hidl_vec<uint32_t> inputToOutputWeightsDimensions{numUnits, inputSize};
    std::vector<float> inputToOutputWeightsValue{0.2303854227f, 0.5218806862f, -0.4865379333f,
                                                 0.53969591851f, 0.23393625035f, -0.27140527306f,
                                                 0.50009280443f, 0.07511717046f, 0.3998299249f,
                                                 -0.51717478049f, 0.1889653282f, -0.367323637f};
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    hidl_vec<uint32_t> recurrentToInputWeightsDimensions{0};
    std::vector<float> recurrentToInputWeightsValue;
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToForgetWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToForgetWeightsValue{-0.09499983487f, -0.08814888417f, -0.04834804721f, 0.1516668247f,
                                                     -0.3967529535f, -0.06463699788f, 0.4952811002f, 0.003274492938f,
                                                     -0.0968840941f, 0.17928104102f, 0.0031281141592f, -0.3387276584f,
                                                     -0.3587934076f, 0.06705895066f, 0.22463923692f, 0.1961955726f};
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToCellWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToCellWeightsValue{-0.21938985582f, -0.3023648226f, -0.1170005202f, -0.3509177422f,
                                                   -0.4286288613f, 0.2726137042f, 0.09216640889f, -0.06551410215f,
                                                   0.20453298098f, 0.2393476665f, 0.11846517771f, 0.2630801796f,
                                                   0.3954237699f, -0.19407111404f, 0.30412107706f, -0.27342408554f};
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    hidl_vec<uint32_t> recurrentToOutputWeightsDimensions{numUnits, outputSize};
    std::vector<float> recurrentToOutputWeightsValue{-0.32921677827f, 0.32624614238f, -0.1388191282f,
                                                     -0.17879831790f, -0.15185534954f, -0.16918526583f,
                                                     -0.10087361183f, -0.5436913968f, 0.016758225858f,
                                                     0.30454617738f, -0.41493862867f, -0.005565764375f,
                                                     -0.12584099173f, -0.12319286912f, 0.2407919466f,
                                                     -0.08879069983f};
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
    hidl_vec<uint32_t> inputGateBiasDimensions{0};
    std::vector<float> inputGateBiasValue;
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> forgetGateBiasDimensions{numUnits};
    std::vector<float> forgetGateBiasValue{1., 1., 1., 1.};
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> cellBiasDimensions{numUnits};
    std::vector<float> cellBiasValue{0., 0., 0., 0.};
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    hidl_vec<uint32_t> outputGateBiasDimensions{numUnits};
    std::vector<float> outputGateBiasValue{0., 0., 0., 0.};
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    hidl_vec<uint32_t> projectionWeightsDimensions{0};
    std::vector<float> projectionWeightsValue;
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    hidl_vec<uint32_t> projectionBiasDimensions{0};
    std::vector<float> projectionBiasValue;

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    hidl_vec<uint32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<float> outputStateInValue(batchSize * outputSize, 0.f);
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    hidl_vec<uint32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<float> cellStateInValue(batchSize * numUnits, 0.f);

    // Constant scalar values (the VTS test adds these as tensors of dim {})
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    hidl_vec<uint32_t>   activationFunctionDimensions{};
    std::vector<int32_t> activationFunctionValue{4};
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   cellClippingThresholdDimensions{};
    std::vector<float>   cellClippingThresholdValue{10.0f};
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    hidl_vec<uint32_t>   projectionClippingThresholdDimensions{};
    std::vector<float>   projectionClippingThresholdValue{0.f};

    // 23: Time-major if true, batch-major if false.
    bool timeMajorValue = false;

    // Normalization:
    // 24:The input layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at input gate.
    hidl_vec<uint32_t> inputLayerNormWeightsDimensions{0};
    std::vector<float> inputLayerNormWeightsValue;
    // 25:The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at forget gate.
    hidl_vec<uint32_t> forgetLayerNormWeightsDimensions{0};
    std::vector<float> forgetLayerNormWeightsValue;
    // 26:The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at cell gate.
    hidl_vec<uint32_t> cellLayerNormWeightsDimensions{0};
    std::vector<float> cellLayerNormWeightsValue;
    // 27:The output layer normalization weights. A 1-D tensor of shape [num_units].
    //    Used to rescale normalized inputs to activation at output gate.
    hidl_vec<uint32_t> outputLayerNormWeightsDimensions{0};
    std::vector<float> outputLayerNormWeightsValue;

    // Outputs:
    // 0: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16. Shape:  if time-major:
    //    [max_time, batch_size, output_size] If batch-major: [batch_size, max_time, output_size]
    hidl_vec<uint32_t> outputDimensions{batchSize, timeSize, outputSize};
    std::vector<float> outputValue{-0.0129257f, -0.070531f, -0.153508f, -0.0392391f,
                                   -0.0300169f, -0.195717f, -0.528679f, -0.0818106f,
                                   -0.0332748f, 0.155429f, -0.353966f, -0.0801505f,
                                   -0.032312f, -0.0407911f, -0.435053f, -0.0932317f,
                                   -0.0108233f, 0.165584f, -0.640424f, -0.0447535f,
                                   -0.031675f, 0.125987f, -0.526695f, -0.110093f};

    // 1: The hidden state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, output_size]. This output is optional and can be omitted. If this output
    //    is present then output #2 must be present as well.
    hidl_vec<uint32_t> hiddenStateOutDimensions{batchSize, outputSize};
    std::vector<float> hiddenStateOutValue(batchSize * outputSize, 0.f);
    // 2: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32/16, of shape
    //    [batch_size, num_units]. This output is optional and can be omitted.
    hidl_vec<uint32_t> cellStateOutDimensions{batchSize, numUnits};
    std::vector<float> cellStateOutValue(batchSize * numUnits, 0.f);

    UnidirectionalSequenceLstmTestImpl<HalPolicy>(inputDimensions, inputValue,
                                                  inputToInputWeightsDimensions, inputToInputWeightsValue,
                                                  inputToForgetWeightsDimensions, inputToForgetWeightsValue,
                                                  inputToCellWeightsDimensions, inputToCellWeightsValue,
                                                  inputToOutputWeightsDimensions, inputToOutputWeightsValue,
                                                  recurrentToInputWeightsDimensions, recurrentToInputWeightsValue,
                                                  recurrentToForgetWeightsDimensions, recurrentToForgetWeightsValue,
                                                  recurrentToCellWeightsDimensions, recurrentToCellWeightsValue,
                                                  recurrentToOutputWeightsDimensions, recurrentToOutputWeightsValue,
                                                  cellToInputWeightsDimensions, cellToInputWeightsValue,
                                                  cellToForgetWeightsDimensions, cellToForgetWeightsValue,
                                                  cellToOutputWeightsDimensions, cellToOutputWeightsValue,
                                                  inputGateBiasDimensions, inputGateBiasValue,
                                                  forgetGateBiasDimensions, forgetGateBiasValue,
                                                  cellBiasDimensions, cellBiasValue,
                                                  outputGateBiasDimensions, outputGateBiasValue,
                                                  projectionWeightsDimensions, projectionWeightsValue,
                                                  projectionBiasDimensions, projectionBiasValue,
                                                  outputStateInDimensions, outputStateInValue,
                                                  cellStateInDimensions, cellStateInValue,
                                                  activationFunctionDimensions, activationFunctionValue,
                                                  cellClippingThresholdDimensions, cellClippingThresholdValue,
                                                  projectionClippingThresholdDimensions,
                                                  projectionClippingThresholdValue,
                                                  timeMajorValue,
                                                  inputLayerNormWeightsDimensions, inputLayerNormWeightsValue,
                                                  forgetLayerNormWeightsDimensions, forgetLayerNormWeightsValue,
                                                  cellLayerNormWeightsDimensions, cellLayerNormWeightsValue,
                                                  outputLayerNormWeightsDimensions, outputLayerNormWeightsValue,
                                                  outputDimensions, outputValue,
                                                  hiddenStateOutDimensions, hiddenStateOutValue,
                                                  cellStateOutDimensions, cellStateOutValue,
                                                  compute);
}