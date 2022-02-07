//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"

#include <log/log.h>

DOCTEST_TEST_SUITE("ConcurrentDriverTests")
{
using ArmnnDriver   = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using HalPolicy     = armnn_driver::hal_1_0::HalPolicy;
using RequestArgument = V1_0::RequestArgument;

using namespace android::nn;
using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

// Add our own test for concurrent execution
// The main point of this test is to check that multiple requests can be
// executed without waiting for the callback from previous execution.
// The operations performed are not significant.
DOCTEST_TEST_CASE("ConcurrentExecute")
{
    ALOGI("ConcurrentExecute: entry");

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    HalPolicy::Model model = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand<HalPolicy>(model, actValue);
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1});

    // make the fully connected operation
    model.operations.resize(1);
    model.operations[0].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared models
    const size_t maxRequests = 5;
    size_t preparedModelsSize = 0;
    android::sp<V1_0::IPreparedModel> preparedModels[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        auto preparedModel = PrepareModel(model, *driver);
        if (preparedModel.get() != nullptr)
        {
            preparedModels[i] = PrepareModel(model, *driver);
            preparedModelsSize++;
        }
    }

    DOCTEST_CHECK(maxRequests == preparedModelsSize);

    // construct the request data
    V1_0::DataLocation inloc = {};
    inloc.poolIndex          = 0;
    inloc.offset             = 0;
    inloc.length             = 3 * sizeof(float);
    RequestArgument input    = {};
    input.location           = inloc;
    input.dimensions         = hidl_vec<uint32_t>{};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex          = 1;
    outloc.offset             = 0;
    outloc.length             = 1 * sizeof(float);
    RequestArgument output    = {};
    output.location           = outloc;
    output.dimensions         = hidl_vec<uint32_t>{};

    // build the requests
    V1_0::Request requests[maxRequests];
    android::sp<IMemory> inMemory[maxRequests];
    android::sp<IMemory> outMemory[maxRequests];
    float indata[] = {2, 32, 16};
    float* outdata[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        requests[i].inputs  = hidl_vec<RequestArgument>{input};
        requests[i].outputs = hidl_vec<RequestArgument>{output};
        // set the input data (matching source test)
        inMemory[i] = AddPoolAndSetData<float>(3, requests[i], indata);
        // add memory for the output
        outMemory[i] = AddPoolAndGetData<float>(1, requests[i]);
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
        DOCTEST_CHECK(cb[i]);
        cb[i]->wait();
    }

    // check the results
    ALOGI("ConcurrentExecute: validating results");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        DOCTEST_CHECK(outdata[i][0] == 152);
    }
    ALOGI("ConcurrentExecute: exit");
}

}
