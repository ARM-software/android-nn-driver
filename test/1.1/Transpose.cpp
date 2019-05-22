//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../DriverTestHelpers.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/array.hpp>
#include <log/log.h>
#include "../TestTensor.hpp"
#include "OperationsUtils.h"
#include <boost/test/data/test_case.hpp>

#include <cmath>

BOOST_AUTO_TEST_SUITE(TransposeTests)

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

namespace
{

static const boost::array<armnn::Compute, 2> COMPUTE_DEVICES = {{ armnn::Compute::CpuRef, armnn::Compute::GpuAcc }};

void TransposeTestImpl(const TestTensor & inputs, int32_t perm[],
                       const TestTensor & expectedOutputTensor, armnn::Compute computeDevice)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice));
    V1_1::Model model = {};

    AddInputOperand(model,inputs.GetDimensions());
    AddTensorOperand(model, hidl_vec<uint32_t>{4}, perm, V1_0::OperandType::TENSOR_INT32);
    AddOutputOperand(model, expectedOutputTensor.GetDimensions());

    model.operations.resize(1);
    model.operations[0].type = V1_1::OperationType::TRANSPOSE;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1};
    model.operations[0].outputs = hidl_vec<uint32_t>{2};

    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // the request's memory pools will follow the same order as
    // the inputs
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset = 0;
    inloc.length = inputs.GetNumElements() * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = inputs.GetDimensions();

    // and an additional memory pool is needed for the output
    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset = 0;
    outloc.length = expectedOutputTensor.GetNumElements() * sizeof(float);
    RequestArgument output = {};
    output.location = outloc;
    output.dimensions = expectedOutputTensor.GetDimensions();

    // make the request based on the arguments
    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    AddPoolAndSetData(inputs.GetNumElements(),
                      request,
                      inputs.GetData());

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(expectedOutputTensor.GetNumElements(), request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    auto execStatus = Execute(preparedModel, request);

    const float * expectedOutput = expectedOutputTensor.GetData();
    for (unsigned int i = 0; i < expectedOutputTensor.GetNumElements(); ++i)
    {
        BOOST_TEST(outdata[i] == expectedOutput[i]);
    }

}
} // namespace

BOOST_DATA_TEST_CASE(Transpose , COMPUTE_DEVICES)
{
    int32_t perm[] = {3, 2, 0, 1};
    TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
    TestTensor expected{armnn::TensorShape{2, 2, 2, 1},{1, 5, 2, 6, 3, 7, 4, 8}};

    TransposeTestImpl(input, perm, expected, sample);
}

BOOST_DATA_TEST_CASE(TransposeNHWCToArmNN , COMPUTE_DEVICES)
{
    int32_t perm[] = {0, 2, 3, 1};
    TestTensor input{armnn::TensorShape{1, 2, 2, 3},{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}};
    TestTensor expected{armnn::TensorShape{1, 3, 2, 2},{1, 11, 21, 31, 2, 12, 22, 32, 3, 13, 23, 33}};

    TransposeTestImpl(input, perm, expected, sample);
}

BOOST_DATA_TEST_CASE(TransposeArmNNToNHWC , COMPUTE_DEVICES)
{
    int32_t perm[] = {0, 3, 1, 2};
    TestTensor input{armnn::TensorShape{1, 2, 2, 2},{1, 2, 3, 4, 5, 6, 7, 8}};
    TestTensor expected{armnn::TensorShape{1, 2, 2, 2},{1, 5, 2, 6, 3, 7, 4, 8}};

    TransposeTestImpl(input, perm, expected, sample);
}

BOOST_AUTO_TEST_SUITE_END()

