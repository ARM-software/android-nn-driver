//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../TestHalfTensor.hpp"

#include <1.2/HalPolicy.hpp>

#include <array>

using Half = half_float::half;

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_2::HalPolicy;
using RequestArgument = V1_0::RequestArgument;

namespace
{

void MeanTestImpl(const TestHalfTensor& input,
                  const hidl_vec<uint32_t>& axisDimensions,
                  const int32_t* axisValues,
                  int32_t keepDims,
                  const TestHalfTensor& expectedOutput,
                  bool fp16Enabled,
                  armnn::Compute computeDevice)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice, fp16Enabled));

    HalPolicy::Model model = {};

    AddInputOperand<HalPolicy>(model, input.GetDimensions(), V1_2::OperandType::TENSOR_FLOAT16);

    AddTensorOperand<HalPolicy>(model,
                                axisDimensions,
                                const_cast<int32_t*>(axisValues),
                                HalPolicy::OperandType::TENSOR_INT32);

    AddIntOperand<HalPolicy>(model, keepDims);

    AddOutputOperand<HalPolicy>(model, expectedOutput.GetDimensions(), V1_2::OperandType::TENSOR_FLOAT16);

    model.operations.resize(1);
    model.operations[0].type               = HalPolicy::OperationType::MEAN;
    model.operations[0].inputs             = hidl_vec<uint32_t>{ 0, 1, 2 };
    model.operations[0].outputs            = hidl_vec<uint32_t>{ 3 };
    model.relaxComputationFloat32toFloat16 = fp16Enabled;

    //android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);
    android::sp<V1_2::IPreparedModel> preparedModel = PrepareModel_1_2(model, *driver);

    // The request's memory pools will follow the same order as the inputs
    V1_0::DataLocation inLoc = {};
    inLoc.poolIndex          = 0;
    inLoc.offset             = 0;
    inLoc.length             = input.GetNumElements() * sizeof(Half);
    RequestArgument inArg    = {};
    inArg.location           = inLoc;
    inArg.dimensions         = input.GetDimensions();

    // An additional memory pool is needed for the output
    V1_0::DataLocation outLoc = {};
    outLoc.poolIndex          = 1;
    outLoc.offset             = 0;
    outLoc.length             = expectedOutput.GetNumElements() * sizeof(Half);
    RequestArgument outArg    = {};
    outArg.location           = outLoc;
    outArg.dimensions         = expectedOutput.GetDimensions();

    // Make the request based on the arguments
    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{ inArg };
    request.outputs = hidl_vec<RequestArgument>{ outArg };

    // Set the input data
    AddPoolAndSetData(input.GetNumElements(), request, input.GetData());

    // Add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<Half>(expectedOutput.GetNumElements(), request);
    const Half* outputData = static_cast<const Half*>(static_cast<void*>(outMemory->getPointer()));

    if (preparedModel.get() != nullptr)
    {
        V1_0::ErrorStatus execStatus = Execute(preparedModel, request);
        DOCTEST_CHECK((int)execStatus == (int)V1_0::ErrorStatus::NONE);
    }

    const Half* expectedOutputData = expectedOutput.GetData();
    for (unsigned int i = 0; i < expectedOutput.GetNumElements(); i++)
    {
        DOCTEST_CHECK(outputData[i] == expectedOutputData[i]);
    }
}

} // anonymous namespace

DOCTEST_TEST_SUITE("MeanTests_1.2_CpuRef")
{

DOCTEST_TEST_CASE("MeanFp16NoKeepDimsTest_CpuRef")
{
    using namespace half_float::literal;

    TestHalfTensor input{ armnn::TensorShape{ 4, 3, 2 },
                      { 1.0_h, 2.0_h, 3.0_h, 4.0_h, 5.0_h, 6.0_h, 7.0_h, 8.0_h, 9.0_h, 10.0_h,
                        11.0_h, 12.0_h, 13.0_h, 14.0_h, 15.0_h, 16.0_h, 17.0_h, 18.0_h, 19.0_h,
                        20.0_h, 21.0_h, 22.0_h, 23.0_h, 24.0_h } };
    hidl_vec<uint32_t> axisDimensions = { 2 };
    int32_t axisValues[] = { 0, 1 };
    int32_t keepDims = 0;
    TestHalfTensor expectedOutput{ armnn::TensorShape{ 2 }, { 12.0_h, 13.0_h } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::CpuRef);
}

DOCTEST_TEST_CASE("MeanFp16KeepDimsTest_CpuRef")
{
    using namespace half_float::literal;

    TestHalfTensor input{ armnn::TensorShape{ 1, 1, 3, 2 }, { 1.0_h, 1.0_h, 2.0_h, 2.0_h, 3.0_h, 3.0_h } };
    hidl_vec<uint32_t> axisDimensions = { 1 };
    int32_t axisValues[] = { 2 };
    int32_t keepDims = 1;
    TestHalfTensor expectedOutput{ armnn::TensorShape{ 1, 1, 1, 2 }, {  2.0_h, 2.0_h } };

    MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::CpuRef);
}

}

#ifdef ARMCOMPUTECL_ENABLED
DOCTEST_TEST_SUITE("MeanTests_1.2_CpuAcc")
{
    DOCTEST_TEST_CASE("MeanFp16NoKeepDimsTest_CpuAcc")
    {
        using namespace half_float::literal;

        std::vector<Half> in = { 1.0_h, 2.0_h, 3.0_h, 4.0_h, 5.0_h, 6.0_h, 7.0_h, 8.0_h, 9.0_h, 10.0_h,
                            11.0_h, 12.0_h, 13.0_h, 14.0_h, 15.0_h, 16.0_h, 17.0_h, 18.0_h, 19.0_h,
                            20.0_h, 21.0_h, 22.0_h, 23.0_h, 24.0_h };
        TestHalfTensor input{ armnn::TensorShape{ 4, 3, 2 },
                           in};
        hidl_vec<uint32_t> axisDimensions = { 2 };
        int32_t axisValues[] = { 0, 1 };
        int32_t keepDims = 0;
        std::vector<Half> out = { 12.0_h, 13.0_h };
        TestHalfTensor expectedOutput{ armnn::TensorShape{ 2 }, out };

        MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::CpuAcc);
    }

    DOCTEST_TEST_CASE("MeanFp16KeepDimsTest_CpuAcc")
    {
        using namespace half_float::literal;

        std::vector<Half> in = { 1.0_h, 1.0_h, 2.0_h, 2.0_h, 3.0_h, 3.0_h };
        TestHalfTensor input{ armnn::TensorShape{ 1, 1, 3, 2 }, in };
        hidl_vec<uint32_t> axisDimensions = { 1 };
        int32_t axisValues[] = { 2 };
        int32_t keepDims = 1;
        std::vector<Half> out = {  2.0_h, 2.0_h };
        TestHalfTensor expectedOutput{ armnn::TensorShape{ 1, 1, 1, 2 }, out };

        MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::CpuAcc);
    }
}

DOCTEST_TEST_SUITE("MeanTests_1.2_GpuAcc")
{
    DOCTEST_TEST_CASE("MeanFp16NoKeepDimsTest_GpuAcc")
    {
        using namespace half_float::literal;

        TestHalfTensor input{ armnn::TensorShape{ 4, 3, 2 },
                          { 1.0_h, 2.0_h, 3.0_h, 4.0_h, 5.0_h, 6.0_h, 7.0_h, 8.0_h, 9.0_h, 10.0_h,
                            11.0_h, 12.0_h, 13.0_h, 14.0_h, 15.0_h, 16.0_h, 17.0_h, 18.0_h, 19.0_h,
                            20.0_h, 21.0_h, 22.0_h, 23.0_h, 24.0_h } };
        hidl_vec<uint32_t> axisDimensions = { 2 };
        int32_t axisValues[] = { 0, 1 };
        int32_t keepDims = 0;
        TestHalfTensor expectedOutput{ armnn::TensorShape{ 2 }, { 12.0_h, 13.0_h } };

        MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("MeanFp16KeepDimsTest_GpuAcc")
    {
        using namespace half_float::literal;

        TestHalfTensor input{ armnn::TensorShape{ 1, 1, 3, 2 }, { 1.0_h, 1.0_h, 2.0_h, 2.0_h, 3.0_h, 3.0_h } };
        hidl_vec<uint32_t> axisDimensions = { 1 };
        int32_t axisValues[] = { 2 };
        int32_t keepDims = 1;
        TestHalfTensor expectedOutput{ armnn::TensorShape{ 1, 1, 1, 2 }, {  2.0_h, 2.0_h } };

        MeanTestImpl(input, axisDimensions, axisValues, keepDims, expectedOutput, true, armnn::Compute::GpuAcc);
    }
}
#endif
