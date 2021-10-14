//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"

#include <log/log.h>

DOCTEST_TEST_SUITE("FullyConnectedTests")
{
using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_0::HalPolicy;

// Add our own test here since we fail the fc tests which Google supplies (because of non-const weights)
DOCTEST_TEST_CASE("FullyConnected")
{
    // this should ideally replicate fully_connected_float.model.cpp
    // but that uses slightly weird dimensions which I don't think we need to support for now

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
    model.operations[0].type = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    V1_0::DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 3 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 1 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data (matching source test)
    float indata[] = {2, 32, 16};
    AddPoolAndSetData<float>(3, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(1, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    if (preparedModel.get() != nullptr)
    {
        Execute(preparedModel, request);
    }

    // check the result
    DOCTEST_CHECK(outdata[0] == 152);
}

DOCTEST_TEST_CASE("TestFullyConnected4dInput")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](V1_0::ErrorStatus status, const std::vector<bool>& supported)
        {
            error = status;
            sup = supported;
        };

    HalPolicy::Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 1}; //identity
    float   biasValue[]   = {0, 0, 0, 0, 0, 0, 0, 0};

    // fully connected operation
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{8, 8}, weightValue);
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{8}, biasValue);
    AddIntOperand<HalPolicy>(model, actValue);
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 8});

    model.operations.resize(1);

    model.operations[0].type = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0,1,2,3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    V1_0::DataLocation inloc = {};
    inloc.poolIndex          = 0;
    inloc.offset             = 0;
    inloc.length             = 8 * sizeof(float);
    RequestArgument input    = {};
    input.location           = inloc;
    input.dimensions         = hidl_vec<uint32_t>{};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex          = 1;
    outloc.offset             = 0;
    outloc.length             = 8 * sizeof(float);
    RequestArgument output    = {};
    output.location           = outloc;
    output.dimensions         = hidl_vec<uint32_t>{};

    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    float indata[] = {1,2,3,4,5,6,7,8};
    AddPoolAndSetData(8, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(8, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    if (preparedModel != nullptr)
    {
        Execute(preparedModel, request);
    }

    // check the result
    DOCTEST_CHECK(outdata[0] == 1);
    DOCTEST_CHECK(outdata[1] == 2);
    DOCTEST_CHECK(outdata[2] == 3);
    DOCTEST_CHECK(outdata[3] == 4);
    DOCTEST_CHECK(outdata[4] == 5);
    DOCTEST_CHECK(outdata[5] == 6);
    DOCTEST_CHECK(outdata[6] == 7);
    DOCTEST_CHECK(outdata[7] == 8);
}

DOCTEST_TEST_CASE("TestFullyConnected4dInputReshape")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](V1_0::ErrorStatus status, const std::vector<bool>& supported)
        {
            error = status;
            sup = supported;
        };

    HalPolicy::Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 1}; //identity
    float   biasValue[]   = {0, 0, 0, 0, 0, 0, 0, 0};

    // fully connected operation
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 2, 2, 2});
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{8, 8}, weightValue);
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{8}, biasValue);
    AddIntOperand<HalPolicy>(model, actValue);
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 8});

    model.operations.resize(1);

    model.operations[0].type = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0,1,2,3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    V1_0::DataLocation inloc = {};
    inloc.poolIndex          = 0;
    inloc.offset             = 0;
    inloc.length             = 8 * sizeof(float);
    RequestArgument input    = {};
    input.location           = inloc;
    input.dimensions         = hidl_vec<uint32_t>{};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex          = 1;
    outloc.offset             = 0;
    outloc.length             = 8 * sizeof(float);
    RequestArgument output    = {};
    output.location           = outloc;
    output.dimensions         = hidl_vec<uint32_t>{};

    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    float indata[] = {1,2,3,4,5,6,7,8};
    AddPoolAndSetData(8, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(8, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    if (preparedModel != nullptr)
    {
        Execute(preparedModel, request);
    }

    // check the result
    DOCTEST_CHECK(outdata[0] == 1);
    DOCTEST_CHECK(outdata[1] == 2);
    DOCTEST_CHECK(outdata[2] == 3);
    DOCTEST_CHECK(outdata[3] == 4);
    DOCTEST_CHECK(outdata[4] == 5);
    DOCTEST_CHECK(outdata[5] == 6);
    DOCTEST_CHECK(outdata[6] == 7);
    DOCTEST_CHECK(outdata[7] == 8);
}

DOCTEST_TEST_CASE("TestFullyConnectedWeightsAsInput")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](V1_0::ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    HalPolicy::Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 1}; //identity
    float   biasValue[]   = {0, 0, 0, 0, 0, 0, 0, 0};

    // fully connected operation
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{8, 8});
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{8});
    AddIntOperand<HalPolicy>(model, actValue);
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 8});

    model.operations.resize(1);

    model.operations[0].type = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0,1,2,3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request for input
    V1_0::DataLocation inloc = {};
    inloc.poolIndex          = 0;
    inloc.offset             = 0;
    inloc.length             = 8 * sizeof(float);
    RequestArgument input    = {};
    input.location           = inloc;
    input.dimensions         = hidl_vec<uint32_t>{1, 1, 1, 8};

    // construct the request for weights as input
    V1_0::DataLocation wloc = {};
    wloc.poolIndex          = 1;
    wloc.offset             = 0;
    wloc.length             = 64 * sizeof(float);
    RequestArgument weights = {};
    weights.location        = wloc;
    weights.dimensions      = hidl_vec<uint32_t>{8, 8};

    // construct the request for bias as input
    V1_0::DataLocation bloc = {};
    bloc.poolIndex          = 2;
    bloc.offset             = 0;
    bloc.length             = 8 * sizeof(float);
    RequestArgument bias    = {};
    bias.location           = bloc;
    bias.dimensions         = hidl_vec<uint32_t>{8};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex          = 3;
    outloc.offset             = 0;
    outloc.length             = 8 * sizeof(float);
    RequestArgument output    = {};
    output.location           = outloc;
    output.dimensions         = hidl_vec<uint32_t>{1, 8};

    V1_0::Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input, weights, bias};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    float indata[] = {1,2,3,4,5,6,7,8};
    AddPoolAndSetData(8, request, indata);

    // set the weights data
    AddPoolAndSetData(64, request, weightValue);
    // set the bias data
    AddPoolAndSetData(8, request, biasValue);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(8, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    if (preparedModel != nullptr)
    {
        Execute(preparedModel, request);
    }

    // check the result
    DOCTEST_CHECK(outdata[0] == 1);
    DOCTEST_CHECK(outdata[1] == 2);
    DOCTEST_CHECK(outdata[2] == 3);
    DOCTEST_CHECK(outdata[3] == 4);
    DOCTEST_CHECK(outdata[4] == 5);
    DOCTEST_CHECK(outdata[5] == 6);
    DOCTEST_CHECK(outdata[6] == 7);
    DOCTEST_CHECK(outdata[7] == 8);
}

}
