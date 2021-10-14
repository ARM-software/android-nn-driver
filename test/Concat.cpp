//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"
#include "TestTensor.hpp"

#include <array>
#include <log/log.h>

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_0::HalPolicy;
using RequestArgument = V1_0::RequestArgument;

namespace
{

void
ConcatTestImpl(const std::vector<const TestTensor*> & inputs,
                int32_t concatAxis,
                const TestTensor & expectedOutputTensor,
                armnn::Compute computeDevice,
                V1_0::ErrorStatus expectedPrepareStatus=V1_0::ErrorStatus::NONE,
                V1_0::ErrorStatus expectedExecStatus=V1_0::ErrorStatus::NONE)
{
    std::unique_ptr<ArmnnDriver> driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice));
    HalPolicy::Model model{};

    hidl_vec<uint32_t> modelInputIds;
    modelInputIds.resize(inputs.size()+1);
    for (uint32_t i = 0; i<inputs.size(); ++i)
    {
        modelInputIds[i] = i;
        AddInputOperand<HalPolicy>(model, inputs[i]->GetDimensions());
    }
    modelInputIds[inputs.size()] = inputs.size(); // add an id for the axis too
    AddIntOperand<HalPolicy>(model, concatAxis);
    AddOutputOperand<HalPolicy>(model, expectedOutputTensor.GetDimensions());

    // make the concat operation
    model.operations.resize(1);
    model.operations[0].type    = HalPolicy::OperationType::CONCATENATION;
    model.operations[0].inputs  = modelInputIds;
    model.operations[0].outputs = hidl_vec<uint32_t>{static_cast<uint32_t>(inputs.size()+1)};

    // make the prepared model
    V1_0::ErrorStatus prepareStatus = V1_0::ErrorStatus::NONE;
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModelWithStatus(model,
                                                                             *driver,
                                                                             prepareStatus,
                                                                             expectedPrepareStatus);
    DOCTEST_CHECK((int)prepareStatus == (int)expectedPrepareStatus);
    if (prepareStatus != V1_0::ErrorStatus::NONE)
    {
        // prepare failed, we cannot continue
        return;
    }

    DOCTEST_CHECK(preparedModel.get() != nullptr);
    if (preparedModel.get() == nullptr)
    {
        // don't spoil other tests if prepare failed
        return;
    }

    // construct the request
    hidl_vec<RequestArgument> inputArguments;
    hidl_vec<RequestArgument> outputArguments;
    inputArguments.resize(inputs.size());
    outputArguments.resize(1);

    // the request's memory pools will follow the same order as
    // the inputs
    for (uint32_t i = 0; i<inputs.size(); ++i)
    {
        V1_0::DataLocation inloc = {};
        inloc.poolIndex = i;
        inloc.offset = 0;
        inloc.length = inputs[i]->GetNumElements() * sizeof(float);
        RequestArgument input = {};
        input.location = inloc;
        input.dimensions = inputs[i]->GetDimensions();
        inputArguments[i] = input;
    }

    // and an additional memory pool is needed for the output
    {
        V1_0::DataLocation outloc = {};
        outloc.poolIndex = inputs.size();
        outloc.offset = 0;
        outloc.length = expectedOutputTensor.GetNumElements() * sizeof(float);
        RequestArgument output = {};
        output.location = outloc;
        output.dimensions = expectedOutputTensor.GetDimensions();
        outputArguments[0] = output;
    }

    // make the request based on the arguments
    V1_0::Request request = {};
    request.inputs  = inputArguments;
    request.outputs = outputArguments;

    // set the input data
    for (uint32_t i = 0; i<inputs.size(); ++i)
    {
        AddPoolAndSetData(inputs[i]->GetNumElements(),
                            request,
                            inputs[i]->GetData());
    }

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(expectedOutputTensor.GetNumElements(), request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    DOCTEST_CHECK(preparedModel.get() != nullptr);
    auto execStatus = Execute(preparedModel, request, expectedExecStatus);
    DOCTEST_CHECK((int)execStatus == (int)expectedExecStatus);

    if (execStatus == V1_0::ErrorStatus::NONE)
    {
        // check the result if there was no error
        const float * expectedOutput = expectedOutputTensor.GetData();
        for (unsigned int i=0; i<expectedOutputTensor.GetNumElements();++i)
        {
            DOCTEST_CHECK(outdata[i] == expectedOutput[i]);
        }
    }
}

/// Test cases...
void SimpleConcatAxis0(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{3, 1, 1, 1}, {0, 1, 2}};
    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void ConcatAxis0NoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{2, 1, 2, 1}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{3, 1, 2, 1}, {4, 5,
                                                    6, 7,
                                                    8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10, 11}};

    TestTensor expected{armnn::TensorShape{6, 1, 2, 1}, {0, 1,
                                                         2, 3,
                                                         4, 5,
                                                         6, 7,
                                                         8, 9,
                                                         10, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis1(armnn::Compute computeDevice)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 3, 1, 1}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void ConcatAxis1NoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1, 2, 2, 1}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 3, 2, 1}, {4, 5,
                                                    6, 7,
                                                    8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10, 11}};

    TestTensor expected{armnn::TensorShape{1, 6, 2, 1}, {0, 1,
                                                         2, 3,
                                                         4, 5,
                                                         6, 7,
                                                         8, 9,
                                                         10, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis1DoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{2, 2, 1, 1}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{2, 3, 1, 1}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{2, 1, 1, 1}, {10,
                                                    11}};

    TestTensor expected{armnn::TensorShape{2, 6, 1, 1}, {0, 1, 4, 5, 6, 10,
                                                         2, 3, 7, 8, 9, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis2(armnn::Compute computeDevice)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 1, 3, 1}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void ConcatAxis2NoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 3, 2}, {4, 5,
                                                    6, 7,
                                                    8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1, 2}, {10, 11}};

    TestTensor expected{armnn::TensorShape{1, 1, 6, 2}, {0, 1,
                                                         2, 3,
                                                         4, 5,
                                                         6, 7,
                                                         8, 9,
                                                         10, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis2DoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1, 2, 2, 1}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 2, 3, 1}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 2, 1, 1}, {10,
                                                    11}};

    TestTensor expected{armnn::TensorShape{1, 2, 6, 1}, {0, 1, 4, 5, 6, 10,
                                                         2, 3, 7, 8, 9, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis3(armnn::Compute computeDevice)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 1, 1, 3}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxis3DoInterleave(armnn::Compute computeDevice)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 2, 3}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10,
                                                    11}};

    TestTensor expected{armnn::TensorShape{1, 1, 2, 6}, {0, 1, 4, 5, 6, 10,
                                                         2, 3, 7, 8, 9, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void AxisTooBig(armnn::Compute computeDevice)
{
    int32_t axis = 4;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {0}};

    // The axis must be within the range of [-rank(values), rank(values))
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    TestTensor uncheckedOutput{armnn::TensorShape{1, 1, 1, 1}, {0}};
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn}, axis, uncheckedOutput, computeDevice, expectedParserStatus);
}

void AxisTooSmall(armnn::Compute computeDevice)
{
    int32_t axis = -5;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1, 1}, {0}};

    // The axis must be within the range of [-rank(values), rank(values))
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    TestTensor uncheckedOutput{armnn::TensorShape{1, 1, 1, 1}, {0}};
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn}, axis, uncheckedOutput, computeDevice, expectedParserStatus);
}

void TooFewInputs(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1, 1, 1, 1}, {0}};

    // We need at least two tensors to concatenate
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn}, axis, aIn, computeDevice, expectedParserStatus);
}

void MismatchedInputDimensions(armnn::Compute computeDevice)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 2, 3}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor mismatched{armnn::TensorShape{1, 1, 1, 1}, {10}};

    TestTensor expected{armnn::TensorShape{1, 1, 2, 6}, {0, 1, 4, 5, 6, 10,
                                                         2, 3, 7, 8, 9, 11}};

    // The input dimensions must be compatible
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn, &mismatched}, axis, expected, computeDevice, expectedParserStatus);
}

void MismatchedInputRanks(armnn::Compute computeDevice)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1, 1, 2}, {0, 1}};
    TestTensor bIn{armnn::TensorShape{1, 1}, {4}};
    TestTensor expected{armnn::TensorShape{1, 1, 3}, {0, 1, 4}};

    // The input dimensions must be compatible
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn}, axis, expected, computeDevice, expectedParserStatus);
}

void MismatchedOutputDimensions(armnn::Compute computeDevice)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 2, 3}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10,
                                                    11}};

    TestTensor mismatched{armnn::TensorShape{1, 1, 6, 2}, {0, 1, 4, 5, 6, 10,
                                                           2, 3, 7, 8, 9, 11}};

    // The input and output dimensions must be compatible
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, mismatched, computeDevice, expectedParserStatus);
}

void MismatchedOutputRank(armnn::Compute computeDevice)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 2, 3}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10,
                                                    11}};

    TestTensor mismatched{armnn::TensorShape{6, 2}, {0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    // The input and output ranks must match
    V1_0::ErrorStatus expectedParserStatus = V1_0::ErrorStatus::GENERAL_FAILURE;
    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, mismatched, computeDevice, expectedParserStatus);
}

void ValidNegativeAxis(armnn::Compute computeDevice)
{
    // this is the same as 3
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    int32_t axis = -1;
    TestTensor aIn{armnn::TensorShape{1, 1, 2, 2}, {0, 1,
                                                    2, 3}};
    TestTensor bIn{armnn::TensorShape{1, 1, 2, 3}, {4, 5, 6,
                                                    7, 8, 9}};
    TestTensor cIn{armnn::TensorShape{1, 1, 2, 1}, {10,
                                                    11}};

    TestTensor expected{armnn::TensorShape{1, 1, 2, 6}, {0, 1, 4, 5, 6, 10,
                                                         2, 3, 7, 8, 9, 11}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisZero3D(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{3, 1, 1}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisOne3D(armnn::Compute computeDevice)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 3, 1}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisTwo3D(armnn::Compute computeDevice)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1, 1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 1, 3}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisZero2D(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{3, 1}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisOne2D(armnn::Compute computeDevice)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1, 1}, {0}};
    TestTensor bIn{armnn::TensorShape{1, 1}, {1}};
    TestTensor cIn{armnn::TensorShape{1, 1}, {2}};

    TestTensor expected{armnn::TensorShape{1, 3}, {0, 1, 2}};

    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

void SimpleConcatAxisZero1D(armnn::Compute computeDevice)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1}, {0}};
    TestTensor bIn{armnn::TensorShape{1}, {1}};
    TestTensor cIn{armnn::TensorShape{1}, {2}};

    TestTensor expected{armnn::TensorShape{3}, {0, 1, 2}};
    ConcatTestImpl({&aIn, &bIn, &cIn}, axis, expected, computeDevice);
}

} // namespace <anonymous>

DOCTEST_TEST_SUITE("ConcatTests_CpuRef")
{

DOCTEST_TEST_CASE("SimpleConcatAxis0")
{
    SimpleConcatAxis0(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("ConcatAxis0NoInterleave")
{
    ConcatAxis0NoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis1")
{
    SimpleConcatAxis1(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("ConcatAxis1NoInterleave")
{
    ConcatAxis1NoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis1DoInterleave")
{
    SimpleConcatAxis1DoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis2")
{
    SimpleConcatAxis2(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("ConcatAxis2NoInterleave")
{
    ConcatAxis2NoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis2DoInterleave")
{
    SimpleConcatAxis2DoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis3")
{
    SimpleConcatAxis3(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxis3DoInterleave")
{
    SimpleConcatAxis3DoInterleave(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("AxisTooBig")
{
    AxisTooBig(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("AxisTooSmall")
{
    AxisTooSmall(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("TooFewInputs")
{
    TooFewInputs(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("MismatchedInputDimensions")
{
    MismatchedInputDimensions(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("MismatchedInputRanks")
{
    MismatchedInputRanks(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("MismatchedOutputDimensions")
{
    MismatchedOutputDimensions(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("MismatchedOutputRank")
{
    MismatchedOutputRank(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("ValidNegativeAxis")
{
    ValidNegativeAxis(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero3D")
{
    SimpleConcatAxisZero3D(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisOne3D")
{
    SimpleConcatAxisOne3D(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisTwo3D")
{
    SimpleConcatAxisTwo3D(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero2D")
{
    SimpleConcatAxisZero2D(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisOne2D")
{
    SimpleConcatAxisOne2D(armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero1D")
{
    SimpleConcatAxisZero1D(armnn::Compute::CpuRef);
}

}

#ifdef ARMCOMPUTECL_ENABLED
DOCTEST_TEST_SUITE("ConcatTests_GpuAcc")
{

DOCTEST_TEST_CASE("SimpleConcatAxis0")
{
    SimpleConcatAxis0(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("ConcatAxis0NoInterleave")
{
    ConcatAxis0NoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis1")
{
    SimpleConcatAxis1(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("ConcatAxis1NoInterleave")
{
    ConcatAxis1NoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis1DoInterleave")
{
    SimpleConcatAxis1DoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis2")
{
    SimpleConcatAxis2(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("ConcatAxis2NoInterleave")
{
    ConcatAxis2NoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis2DoInterleave")
{
    SimpleConcatAxis2DoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis3")
{
    SimpleConcatAxis3(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxis3DoInterleave")
{
    SimpleConcatAxis3DoInterleave(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("AxisTooBig")
{
    AxisTooBig(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("AxisTooSmall")
{
    AxisTooSmall(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("TooFewInputs")
{
    TooFewInputs(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("MismatchedInputDimensions")
{
    MismatchedInputDimensions(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("MismatchedInputRanks")
{
    MismatchedInputRanks(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("MismatchedOutputDimensions")
{
    MismatchedOutputDimensions(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("MismatchedOutputRank")
{
    MismatchedOutputRank(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("ValidNegativeAxis")
{
    ValidNegativeAxis(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero3D")
{
    SimpleConcatAxisZero3D(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisOne3D")
{
    SimpleConcatAxisOne3D(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisTwo3D")
{
    SimpleConcatAxisTwo3D(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero2D")
{
    SimpleConcatAxisZero2D(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisOne2D")
{
    SimpleConcatAxisOne2D(armnn::Compute::GpuAcc);
}

DOCTEST_TEST_CASE("SimpleConcatAxisZero1D")
{
    SimpleConcatAxisZero1D(armnn::Compute::GpuAcc);
}

}// End of GpuAcc Test Suite
#endif