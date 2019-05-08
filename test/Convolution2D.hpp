//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverTestHelpers.hpp"

#include <boost/test/unit_test.hpp>
#include <log/log.h>

#include <OperationsUtils.h>

BOOST_AUTO_TEST_SUITE(Convolution2DTests)

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

namespace driverTestHelpers
{

void SetModelFp16Flag(V1_0::Model& model, bool fp16Enabled);

#ifdef ARMNN_ANDROID_NN_V1_1
void SetModelFp16Flag(V1_1::Model& model, bool fp16Enabled);
#endif

template<typename HalPolicy>
void PaddingTestImpl(android::nn::PaddingScheme paddingScheme, bool fp16Enabled = false)
{
    using HalModel         = typename HalPolicy::Model;
    using HalOperationType = typename HalPolicy::OperationType;

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::GpuAcc, fp16Enabled));
    HalModel model = {};

    uint32_t outSize = paddingScheme == android::nn::kPaddingSame ? 2 : 1;

    // add operands
    float weightValue[] = {1.f, -1.f, 0.f, 1.f};
    float biasValue[]   = {0.f};

    AddInputOperand(model, hidl_vec<uint32_t>{1, 2, 3, 1});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 2, 2, 1}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, (int32_t)paddingScheme); // padding
    AddIntOperand(model, 2); // stride x
    AddIntOperand(model, 2); // stride y
    AddIntOperand(model, 0); // no activation
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, outSize, 1});

    // make the convolution operation
    model.operations.resize(1);
    model.operations[0].type = HalOperationType::CONV_2D;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3, 4, 5, 6};
    model.operations[0].outputs = hidl_vec<uint32_t>{7};

    // make the prepared model
    SetModelFp16Flag(model, fp16Enabled);
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    DataLocation inloc    = {};
    inloc.poolIndex       = 0;
    inloc.offset          = 0;
    inloc.length          = 6 * sizeof(float);
    RequestArgument input = {};
    input.location        = inloc;
    input.dimensions      = hidl_vec<uint32_t>{};

    DataLocation outloc    = {};
    outloc.poolIndex       = 1;
    outloc.offset          = 0;
    outloc.length          = outSize * sizeof(float);
    RequestArgument output = {};
    output.location        = outloc;
    output.dimensions      = hidl_vec<uint32_t>{};

    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data (matching source test)
    float indata[] = {1024.25f, 1.f, 0.f, 3.f, -1, -1024.25f};
    AddPoolAndSetData(6, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(outSize, request);
    float* outdata = reinterpret_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    Execute(preparedModel, request);

    // check the result
    switch (paddingScheme)
    {
    case android::nn::kPaddingValid:
        if (fp16Enabled)
        {
            BOOST_TEST(outdata[0] == 1022.f);
        }
        else
        {
            BOOST_TEST(outdata[0] == 1022.25f);
        }
        break;
    case android::nn::kPaddingSame:
        if (fp16Enabled)
        {
            BOOST_TEST(outdata[0] == 1022.f);
            BOOST_TEST(outdata[1] == 0.f);
        }
        else
        {
            BOOST_TEST(outdata[0] == 1022.25f);
            BOOST_TEST(outdata[1] == 0.f);
        }
        break;
    default:
        BOOST_TEST(false);
        break;
    }
}

} // namespace driverTestHelpers

BOOST_AUTO_TEST_SUITE_END()
