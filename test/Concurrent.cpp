//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "DriverTestHelpers.hpp"
#include <boost/test/unit_test.hpp>
#include <log/log.h>

BOOST_AUTO_TEST_SUITE(ConcurrentDriverTests)

using ArmnnDriver = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using namespace android::nn;
using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

// Add our own test for concurrent execution
// The main point of this test is to check that multiple requests can be
// executed without waiting for the callback from previous execution.
// The operations performed are not significant.
BOOST_AUTO_TEST_CASE(ConcurrentExecute)
{
    ALOGI("ConcurrentExecute: entry");

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    neuralnetworks::V1_0::Model model = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // make the fully connected operation
    model.operations.resize(1);
    model.operations[0].type = neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared models
    const size_t maxRequests = 5;
    android::sp<IPreparedModel> preparedModels[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        preparedModels[i] = PrepareModel(model, *driver);
    }

    // construct the request data
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 3 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 1 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    // build the requests
    Request requests[maxRequests];
    android::sp<IMemory> outMemory[maxRequests];
    float* outdata[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        requests[i].inputs  = hidl_vec<RequestArgument>{input};
        requests[i].outputs = hidl_vec<RequestArgument>{output};
        // set the input data (matching source test)
        float indata[] = {2, 32, 16};
        AddPoolAndSetData(3, requests[i], indata);
        // add memory for the output
        outMemory[i] = AddPoolAndGetData(1, requests[i]);
        outdata[i] = static_cast<float*>(static_cast<void*>(outMemory[i]->getPointer()));
    }

    // invoke the execution of the requests
    ALOGI("ConcurrentExecute: executing requests");
    android::sp<ExecutionCallback> cb[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        cb[i] = ExecuteNoWait(preparedModels[i], requests[i]);
    }

    // wait for the requests to complete
    ALOGI("ConcurrentExecute: waiting for callbacks");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        cb[i]->wait();
    }

    // check the results
    ALOGI("ConcurrentExecute: validating results");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        BOOST_TEST(outdata[i][0] == 152);
    }
    ALOGI("ConcurrentExecute: exit");
}

BOOST_AUTO_TEST_SUITE_END()
