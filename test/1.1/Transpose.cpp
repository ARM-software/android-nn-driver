//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "../TestTensor.hpp"
#include <1.1/HalPolicy.hpp>

#include <log/log.h>
#include <OperationsUtils.h>

#include <array>
#include <cmath>

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_1::HalPolicy;
using RequestArgument = V1_0::RequestArgument;

namespace
{

void TransposeTestImpl(const TestTensor & inputs, int32_t perm[],
                       const TestTensor & expectedOutputTensor, armnn::Compute computeDevice)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice));
    HalPolicy::Model model = {};

    AddInputOperand<HalPolicy>(model,inputs.GetDimensions());

    AddTensorOperand<HalPolicy>(model,
                                hidl_vec<uint32_t>{4},
                                perm,
                                HalPolicy::OperandType::TENSOR_INT32);

    AddOutputOperand<HalPolicy>(model, expectedOutputTensor.GetDimensions());

    model.operations.resize(1);
    model.operations[0].type = HalPolicy::OperationType::TRANSPOSE;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1};
    model.operations[0].outputs = hidl_vec<uint32_t>{2};

    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // the request's memory pools will follow the same order as
    // the inputs
    V1_0::DataLocation inloc = {};
    inloc.poolIndex          = 0;
    inloc.offset             = 0;
    inloc.length             = inputs.GetNumElements() * sizeof(float);
    RequestArgument input    = {};
    input.location           = inloc;
    input.dimensions         = inputs.GetDimensions();

    // and an additional memory pool is needed for the output
    V1_0::DataLocation outloc = {};
    outloc.poolIndex          = 1;
    outloc.offset             = 0;
    outloc.length             = expectedOutputTensor.GetNumElements() * sizeof(float);
    RequestArgument output    = {};
    output.location           = outloc;
    output.dimensions         = expectedOutputTensor.GetDimensions();

    // make the request based on the arguments
    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    AddPoolAndSetData(inputs.GetNumElements(),
                      request,
                      inputs.GetData());

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(expectedOutputTensor.GetNumElements(), request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    if (preparedModel.get() != nullptr)
    {
        auto execStatus = Execute(preparedModel, request);
    }

    const float * expectedOutput = expectedOutputTensor.GetData();
    for (unsigned int i = 0; i < expectedOutputTensor.GetNumElements(); ++i)
    {
        DOCTEST_CHECK(outdata[i] == expectedOutput[i]);
    }
}

} // namespace

DOCTEST_TEST_SUITE("TransposeTests_CpuRef")
{
    DOCTEST_TEST_CASE("Transpose_CpuRef")
    {
        int32_t perm[] = {2, 3, 1, 0};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{2, 2, 2, 1},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuRef);
    }

    DOCTEST_TEST_CASE("TransposeNHWCToArmNN_CpuRef")
    {
        int32_t perm[] = {0, 3, 1, 2};
        TestTensor input{armnn::TensorShape{1, 2, 2, 3},{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}};
        TestTensor expected{armnn::TensorShape{1, 3, 2, 2},{1, 11, 21, 31, 2, 12, 22, 32, 3, 13, 23, 33}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuRef);
    }
    DOCTEST_TEST_CASE("TransposeArmNNToNHWC_CpuRef")
    {
        int32_t perm[] = {0, 2, 3, 1};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{1, 2, 2, 2},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuRef);
    }
}

#ifdef ARMCOMPUTECL_ENABLED
DOCTEST_TEST_SUITE("TransposeTests_CpuAcc")
{
    DOCTEST_TEST_CASE("Transpose_CpuAcc")
    {
        int32_t perm[] = {2, 3, 1, 0};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{2, 2, 2, 1},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuAcc);
    }

    DOCTEST_TEST_CASE("TransposeNHWCToArmNN_CpuAcc")
    {
        int32_t perm[] = {0, 3, 1, 2};
        TestTensor input{armnn::TensorShape{1, 2, 2, 3},{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}};
        TestTensor expected{armnn::TensorShape{1, 3, 2, 2},{1, 11, 21, 31, 2, 12, 22, 32, 3, 13, 23, 33}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuAcc);
    }

    DOCTEST_TEST_CASE("TransposeArmNNToNHWC_CpuAcc")
    {
        int32_t perm[] = {0, 2, 3, 1};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{1, 2, 2, 2},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::CpuAcc);
    }
}

DOCTEST_TEST_SUITE("TransposeTests_GpuAcc")
{
    DOCTEST_TEST_CASE("Transpose_GpuAcc")
    {
        int32_t perm[] = {2, 3, 1, 0};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{2, 2, 2, 1},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("TransposeNHWCToArmNN_GpuAcc")
    {
        int32_t perm[] = {0, 3, 1, 2};
        TestTensor input{armnn::TensorShape{1, 2, 2, 3},{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}};
        TestTensor expected{armnn::TensorShape{1, 3, 2, 2},{1, 11, 21, 31, 2, 12, 22, 32, 3, 13, 23, 33}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::GpuAcc);
    }

    DOCTEST_TEST_CASE("TransposeArmNNToNHWC_GpuAcc")
    {
        int32_t perm[] = {0, 2, 3, 1};
        TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
        TestTensor expected{armnn::TensorShape{1, 2, 2, 2},{1, 5, 2, 6, 3, 7, 4, 8}};

        TransposeTestImpl(input, perm, expected, armnn::Compute::GpuAcc);
    }
}
#endif

