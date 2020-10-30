//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../TestTensor.hpp"

#include "../1.1/HalPolicy.hpp"

#include <boost/test/data/test_case.hpp>

#include <array>

BOOST_AUTO_TEST_SUITE(MeanTests)

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_1::HalPolicy;

namespace
{

#ifndef ARMCOMPUTECL_ENABLED
    static const std::array<armnn::Compute, 1> COMPUTE_DEVICES = {{ armnn::Compute::CpuRef }};
#else
    static const std::array<armnn::Compute, 2> COMPUTE_DEVICES = {{ armnn::Compute::CpuRef, armnn::Compute::GpuAcc }};
#endif

void MeanTestImpl(const TestTensor& input,
                  const hidl_vec<uint32_t>& axisDimensions,
                  const int32_t* axisValues,
                  int32_t keepDims,
                  const TestTensor& expectedOutput,
                  bool fp16Enabled,
                  armnn::Compute computeDevice)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice, fp16Enabled));

    HalPolicy::Model model = {};

    AddInputOperand<HalPolicy>(model, input.GetDimensions());

    AddTensorOperand<HalPolicy>(model,
                                axisDimensions,
                                const_cast<int32_t*>(axisValues),
                                HalPolicy::OperandType::TENSOR_INT32);

    AddIntOperand<HalPolicy>(model, keepDims);

    AddOutputOperand<HalPolicy>(model, expectedOutput.GetDimensions());

    model.operations.resize(1);
    model.operations[0].type               = HalPolicy::OperationType::MEAN;
    model.operations[0].inputs             = hidl_vec<uint32_t>{ 0, 1, 2 };
    model.operations[0].outputs            = hidl_vec<uint32_t>{ 3 };
    model.relaxComputationFloat32toFloat16 = fp16Enabled;

    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // The request's memory pools will follow the same order as the inputs
    DataLocation inLoc    = {};
    inLoc.poolIndex       = 0;
    inLoc.offset          = 0;
    inLoc.length          = input.GetNumElements() * sizeof(float);
    RequestArgument inArg = {};
    inArg.location        = inLoc;
    inArg.dimensions      = input.GetDimensions();

    // An additional memory pool is needed for the output
    DataLocation outLoc    = {};
    outLoc.poolIndex       = 1;
    outLoc.offset          = 0;
    outLoc.length          = expectedOutput.GetNumElements() * sizeof(float);
    RequestArgument outArg = {};
    outArg.location        = outLoc;
    outArg.dimensions      = expectedOutput.GetDimensions();

    // Make the request based on the arguments
    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{ inArg };
    request.outputs = hidl_vec<RequestArgument>{ outArg };

    // Set the input data
    AddPoolAndSetData(input.GetNumElements(), request, input.GetData());

    // Add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(expectedOutput.GetNumElements(), request);
    const float* outputData = static_cast<const float*>(static_cast<void*>(outMemory->getPointer()));

    if (preparedModel.get() != nullptr)
    {
        V1_0::ErrorStatus execStatus = Execute(preparedModel, request);
        BOOST_TEST(execStatus == V1_0::ErrorStatus::NONE);
    }

    const float* expectedOutputData = expectedOutput.GetData();
    for (unsigned int i = 0; i < expectedOutput.GetNumElements(); i++)
    {
        BOOST_TEST(outputData[i] == expectedOutputData[i]);
    }
}

} // anonymous namespace

BOOST_DATA_TEST_CASE(MeanNoKeepDimsTest, COMPUTE_DEVICES)
{
    TestTensor input{ armnn::TensorShape{ 4, 3, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                                       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                                                       20.0f, 21.0f, 22.0f, 23.0f, 24.0f } };
    hidl_vec<uint32_t> axisDimensions = { 2 };
    int32_t axisValues[] = { 0, 1 };
    int32_t keepDims = 0;
    TestTensor expectedOutput{ armnn::TensorShape{ 2 }, { 12.0f, 13.0f } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, false, sample);
}

BOOST_DATA_TEST_CASE(MeanKeepDimsTest, COMPUTE_DEVICES)
{
    TestTensor input{ armnn::TensorShape{ 1, 1, 3, 2 }, { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f } };
    hidl_vec<uint32_t> axisDimensions = { 1 };
    int32_t axisValues[] = { 2 };
    int32_t keepDims = 1;
    TestTensor expectedOutput{ armnn::TensorShape{ 1, 1, 1, 2 }, {  2.0f, 2.0f } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, false, sample);
}

BOOST_DATA_TEST_CASE(MeanFp16NoKeepDimsTest, COMPUTE_DEVICES)
{
    TestTensor input{ armnn::TensorShape{ 4, 3, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                                       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                                                       20.0f, 21.0f, 22.0f, 23.0f, 24.0f } };
    hidl_vec<uint32_t> axisDimensions = { 2 };
    int32_t axisValues[] = { 0, 1 };
    int32_t keepDims = 0;
    TestTensor expectedOutput{ armnn::TensorShape{ 2 }, { 12.0f, 13.0f } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, sample);
}

BOOST_DATA_TEST_CASE(MeanFp16KeepDimsTest, COMPUTE_DEVICES)
{
    TestTensor input{ armnn::TensorShape{ 1, 1, 3, 2 }, { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f } };
    hidl_vec<uint32_t> axisDimensions = { 1 };
    int32_t axisValues[] = { 2 };
    int32_t keepDims = 1;
    TestTensor expectedOutput{ armnn::TensorShape{ 1, 1, 1, 2 }, {  2.0f, 2.0f } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, sample);
}

BOOST_AUTO_TEST_SUITE_END()
