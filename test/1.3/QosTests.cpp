//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"

#include <1.3/HalPolicy.hpp>

DOCTEST_TEST_SUITE("QosTests")
{
using ArmnnDriver   = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;

using namespace android::nn;
using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_3::HalPolicy;

namespace
{

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

DOCTEST_TEST_CASE("ConcurrentExecuteWithQosPriority")
{
    ALOGI("ConcurrentExecuteWithQOSPriority: entry");

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    HalPolicy::Model model = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand<HalPolicy>(model,
                      hidl_vec<uint32_t>{1, 3},
                      weightValue,
                      HalPolicy::OperandType::TENSOR_FLOAT32,
                      V1_3::OperandLifeTime::CONSTANT_COPY);
    AddTensorOperand<HalPolicy>(model,
                      hidl_vec<uint32_t>{1},
                      biasValue,
                      HalPolicy::OperandType::TENSOR_FLOAT32,
                      V1_3::OperandLifeTime::CONSTANT_COPY);
    AddIntOperand<HalPolicy>(model, actValue);
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1});

    // make the fully connected operation
    model.main.operations.resize(1);
    model.main.operations[0].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model.main.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.main.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared models
    const size_t maxRequests = 45;
    size_t preparedModelsSize = 0;
    android::sp<V1_3::IPreparedModel> preparedModels[maxRequests];
    V1_3::ErrorStatus status(V1_3::ErrorStatus::NONE);
    size_t start = preparedModelsSize;
    for (size_t i = start; i < start+15; ++i)
    {
        preparedModels[i] = PrepareModelWithStatus_1_3(model, *driver, status, V1_3::Priority::LOW);
        preparedModelsSize++;
    }
    start = preparedModelsSize;
    for (size_t i = start; i < start+15; ++i)
    {
        preparedModels[i] = PrepareModelWithStatus_1_3(model, *driver, status, V1_3::Priority::MEDIUM);
        preparedModelsSize++;
    }
    start = preparedModelsSize;
    for (size_t i = start; i < start+15; ++i)
    {
        preparedModels[i] = PrepareModelWithStatus_1_3(model, *driver, status, V1_3::Priority::HIGH);
        preparedModelsSize++;
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
    android::sp<IMemory> outMemory[maxRequests];
    float* outdata[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        requests[i].inputs  = hidl_vec<RequestArgument>{input};
        requests[i].outputs = hidl_vec<RequestArgument>{output};
        // set the input data (matching source test)
        float inDataLow[] = {2, 32, 16};
        float inDataMedium[] = {1, 31, 11};
        float inDataHigh[] = {3, 33, 17};
        if (i < 15)
        {
            AddPoolAndSetData<float>(3, requests[i], inDataLow);
        }
        else if (i < 30)
        {
            AddPoolAndSetData<float>(3, requests[i], inDataMedium);
        }
        else
        {
            AddPoolAndSetData<float>(3, requests[i], inDataHigh);
        }
        // add memory for the output
        outMemory[i] = AddPoolAndGetData<float>(1, requests[i]);
        outdata[i] = static_cast<float*>(static_cast<void*>(outMemory[i]->getPointer()));
    }

    // invoke the execution of the requests
    ALOGI("ConcurrentExecuteWithQOSPriority: executing requests");
    android::sp<ExecutionCallback> cb[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        cb[i] = ExecuteNoWait(preparedModels[i], requests[i]);
    }

    // wait for the requests to complete
    ALOGI("ConcurrentExecuteWithQOSPriority: waiting for callbacks");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        DOCTEST_CHECK(cb[i]);
        cb[i]->wait();
    }

    // check the results
    ALOGI("ConcurrentExecuteWithQOSPriority: validating results");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        if (i < 15)
        {
            DOCTEST_CHECK(outdata[i][0] == 152);
        }
        else if (i < 30)
        {
            DOCTEST_CHECK(outdata[i][0] == 141);
        }
        else
        {
            DOCTEST_CHECK(outdata[i][0] == 159);
        }

    }
    ALOGI("ConcurrentExecuteWithQOSPriority: exit");
}

} // anonymous namespace

}