//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "DriverTestHelpers.hpp"
#include "TestTensor.hpp"
#include <boost/test/unit_test.hpp>
#include <log/log.h>


BOOST_AUTO_TEST_SUITE(MergerTests)

using ArmnnDriver = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using namespace driverTestHelpers;

namespace
{

void
MergerTestImpl(const std::vector<const TestTensor*> & inputs,
                int32_t concatAxis,
                const TestTensor & expectedOutputTensor,
                ErrorStatus expectedPrepareStatus=ErrorStatus::NONE,
                ErrorStatus expectedExecStatus=ErrorStatus::NONE)
{
    std::unique_ptr<ArmnnDriver> driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    V1_0::Model model{};

    hidl_vec<uint32_t> modelInputIds;
    modelInputIds.resize(inputs.size()+1);
    for (uint32_t i = 0; i<inputs.size(); ++i)
    {
        modelInputIds[i] = i;
        AddInputOperand(model, inputs[i]->GetDimensions());
    }
    modelInputIds[inputs.size()] = inputs.size(); // add an id for the axis too
    AddIntOperand(model, concatAxis);
    AddOutputOperand(model, expectedOutputTensor.GetDimensions());

    // make the concat operation
    model.operations.resize(1);
    model.operations[0].type = V1_0::OperationType::CONCATENATION;
    model.operations[0].inputs  = modelInputIds;
    model.operations[0].outputs = hidl_vec<uint32_t>{static_cast<uint32_t>(inputs.size()+1)};

    // make the prepared model
    ErrorStatus prepareStatus=ErrorStatus::NONE;
    android::sp<IPreparedModel> preparedModel = PrepareModelWithStatus(model,
                                                                       *driver,
                                                                       prepareStatus,
                                                                       expectedPrepareStatus);
    BOOST_TEST(prepareStatus == expectedPrepareStatus);
    if (prepareStatus != ErrorStatus::NONE)
    {
        // prepare failed, we cannot continue
        return;
    }

    BOOST_TEST(preparedModel.get() != nullptr);
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
        DataLocation inloc = {};
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
        DataLocation outloc = {};
        outloc.poolIndex = inputs.size();
        outloc.offset = 0;
        outloc.length = expectedOutputTensor.GetNumElements() * sizeof(float);
        RequestArgument output = {};
        output.location = outloc;
        output.dimensions = expectedOutputTensor.GetDimensions();
        outputArguments[0] = output;
    }

    // make the request based on the arguments
    Request request = {};
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
    android::sp<IMemory> outMemory = AddPoolAndGetData(expectedOutputTensor.GetNumElements(), request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    auto execStatus = Execute(preparedModel, request, expectedExecStatus);
    BOOST_TEST(execStatus == expectedExecStatus);

    if (execStatus == ErrorStatus::NONE)
    {
        // check the result if there was no error
        const float * expectedOutput = expectedOutputTensor.GetData();
        for (unsigned int i=0; i<expectedOutputTensor.GetNumElements();++i)
        {
            BOOST_TEST(outdata[i] == expectedOutput[i]);
        }
    }
}

} // namespace <anonymous>

BOOST_AUTO_TEST_CASE(SimpleConcatAxis0)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{1}};
    TestTensor cIn{armnn::TensorShape{1,1,1,1},{2}};

    TestTensor expected{armnn::TensorShape{3,1,1,1},{0,1,2}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(ConcatAxis0_NoInterleave)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{2,1,2,1},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{3,1,2,1},{4,  5,
                                                6,  7,
                                                8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10, 11}};

    TestTensor expected{armnn::TensorShape{6,1,2,1},{0,  1,
                                                     2,  3,
                                                     4,  5,
                                                     6,  7,
                                                     8,  9,
                                                     10, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis1)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{1}};
    TestTensor cIn{armnn::TensorShape{1,1,1,1},{2}};

    TestTensor expected{armnn::TensorShape{1,3,1,1},{0,1,2}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(ConcatAxis1_NoInterleave)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{1,2,2,1},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,3,2,1},{4,  5,
                                                6,  7,
                                                8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10, 11}};

    TestTensor expected{armnn::TensorShape{1,6,2,1},{0,  1,
                                                     2,  3,
                                                     4,  5,
                                                     6,  7,
                                                     8,  9,
                                                     10, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis1_DoInterleave)
{
    int32_t axis = 1;
    TestTensor aIn{armnn::TensorShape{2,2,1,1},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{2,3,1,1},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{2,1,1,1},{10,
                                                11}};

    TestTensor expected{armnn::TensorShape{2,6,1,1},{0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis2)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{1}};
    TestTensor cIn{armnn::TensorShape{1,1,1,1},{2}};

    TestTensor expected{armnn::TensorShape{1,1,3,1},{0,1,2}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(ConcatAxis2_NoInterleave)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,3,2},{4,  5,
                                                6,  7,
                                                8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,1,2},{10, 11}};

    TestTensor expected{armnn::TensorShape{1,1,6,2},{0,  1,
                                                     2,  3,
                                                     4,  5,
                                                     6,  7,
                                                     8,  9,
                                                     10, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis2_DoInterleave)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1,2,2,1},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,2,3,1},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{1,2,1,1},{10,
                                                11}};

    TestTensor expected{armnn::TensorShape{1,2,6,1},{0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis3)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{1}};
    TestTensor cIn{armnn::TensorShape{1,1,1,1},{2}};

    TestTensor expected{armnn::TensorShape{1,1,1,3},{0,1,2}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(SimpleConcatAxis3_DoInterleave)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,2,3},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10,
                                                11}};

    TestTensor expected{armnn::TensorShape{1,1,2,6},{0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_CASE(AxisTooBig)
{
    int32_t axis = 4;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{0}};

    // The axis must be within the range of [-rank(values), rank(values))
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    TestTensor uncheckedOutput{armnn::TensorShape{1,1,1,1},{0}};
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn}, axis, uncheckedOutput, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(AxisTooSmall)
{
    int32_t axis = -5;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};
    TestTensor bIn{armnn::TensorShape{1,1,1,1},{0}};

    // The axis must be within the range of [-rank(values), rank(values))
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    TestTensor uncheckedOutput{armnn::TensorShape{1,1,1,1},{0}};
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn}, axis, uncheckedOutput, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(TooFewInputs)
{
    int32_t axis = 0;
    TestTensor aIn{armnn::TensorShape{1,1,1,1},{0}};

    // We need at least two tensors to concatenate
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn}, axis, aIn, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(MismatchedInputDimensions)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,2,3},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor mismatched{armnn::TensorShape{1,1,1,1},{10}};

    TestTensor expected{armnn::TensorShape{1,1,2,6},{0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    // The input dimensions must be compatible
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn, &mismatched}, axis, expected, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(MismatchedInputRanks)
{
    int32_t axis = 2;
    TestTensor aIn{armnn::TensorShape{1,1,2},{0,1}};
    TestTensor bIn{armnn::TensorShape{1,1},{4}};
    TestTensor expected{armnn::TensorShape{1,1,3},{0,1,4}};

    // The input dimensions must be compatible
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn}, axis, expected, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(MismatchedOutputDimensions)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,2,3},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10,
                                                11}};

    TestTensor mismatched{armnn::TensorShape{1,1,6,2},{0, 1, 4, 5, 6, 10,
                                                       2, 3, 7, 8, 9, 11}};

    // The input and output dimensions must be compatible
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn, &cIn}, axis, mismatched, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(MismatchedOutputRank)
{
    int32_t axis = 3;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,2,3},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10,
                                                11}};

    TestTensor mismatched{armnn::TensorShape{6,2},{0, 1, 4, 5, 6, 10,
                                                   2, 3, 7, 8, 9, 11}};

    // The input and output ranks must match
    ErrorStatus expectedParserStatus = ErrorStatus::GENERAL_FAILURE;
    MergerTestImpl({&aIn, &bIn, &cIn}, axis, mismatched, expectedParserStatus);
}

BOOST_AUTO_TEST_CASE(ValidNegativeAxis)
{
    // this is the same as 3
    // see: https://www.tensorflow.org/api_docs/python/tf/concat
    int32_t axis = -1;
    TestTensor aIn{armnn::TensorShape{1,1,2,2},{0,  1,
                                                2,  3}};
    TestTensor bIn{armnn::TensorShape{1,1,2,3},{4,  5,  6,
                                                7,  8,  9}};
    TestTensor cIn{armnn::TensorShape{1,1,2,1},{10,
                                                11}};

    TestTensor expected{armnn::TensorShape{1,1,2,6},{0, 1, 4, 5, 6, 10,
                                                     2, 3, 7, 8, 9, 11}};

    MergerTestImpl({&aIn, &bIn, &cIn}, axis, expected);
}

BOOST_AUTO_TEST_SUITE_END()
