//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverTestHelpers.hpp"

#include <log/log.h>

#include <OperationsUtils.h>

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using RequestArgument = V1_0::RequestArgument;

namespace driverTestHelpers
{
#define ARMNN_ANDROID_FP16_TEST(result, fp16Expectation, fp32Expectation, fp16Enabled) \
   if (fp16Enabled) \
   { \
       DOCTEST_CHECK_MESSAGE((result == fp16Expectation || result == fp32Expectation), result << \
       " does not match either " << fp16Expectation << "[fp16] or " << fp32Expectation << "[fp32]"); \
   } else \
   { \
      DOCTEST_CHECK(result == fp32Expectation); \
   }

void SetModelFp16Flag(V1_0::Model& model, bool fp16Enabled);

void SetModelFp16Flag(V1_1::Model& model, bool fp16Enabled);

template<typename HalPolicy>
void PaddingTestImpl(android::nn::PaddingScheme paddingScheme, bool fp16Enabled = false)
{
    using HalModel         = typename HalPolicy::Model;
    using HalOperationType = typename HalPolicy::OperationType;

    armnn::Compute computeDevice = armnn::Compute::GpuAcc;

#ifndef ARMCOMPUTECL_ENABLED
    computeDevice = armnn::Compute::CpuRef;
#endif

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(computeDevice, fp16Enabled));
    HalModel model = {};

    uint32_t outSize = paddingScheme == android::nn::kPaddingSame ? 2 : 1;

    // add operands
    float weightValue[] = {1.f, -1.f, 0.f, 1.f};
    float biasValue[] = {0.f};

    AddInputOperand<HalPolicy>(model, hidl_vec < uint32_t > {1, 2, 3, 1});
    AddTensorOperand<HalPolicy>(model, hidl_vec < uint32_t > {1, 2, 2, 1}, weightValue);
    AddTensorOperand<HalPolicy>(model, hidl_vec < uint32_t > {1}, biasValue);
    AddIntOperand<HalPolicy>(model, (int32_t) paddingScheme); // padding
    AddIntOperand<HalPolicy>(model, 2); // stride x
    AddIntOperand<HalPolicy>(model, 2); // stride y
    AddIntOperand<HalPolicy>(model, 0); // no activation
    AddOutputOperand<HalPolicy>(model, hidl_vec < uint32_t > {1, 1, outSize, 1});

    // make the convolution operation
    model.operations.resize(1);
    model.operations[0].type = HalOperationType::CONV_2D;
    model.operations[0].inputs = hidl_vec < uint32_t > {0, 1, 2, 3, 4, 5, 6};
    model.operations[0].outputs = hidl_vec < uint32_t > {7};

    // make the prepared model
    SetModelFp16Flag(model, fp16Enabled);
    android::sp<V1_0::IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    V1_0::DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset = 0;
    inloc.length = 6 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec < uint32_t > {};

    V1_0::DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset = 0;
    outloc.length = outSize * sizeof(float);
    RequestArgument output = {};
    output.location = outloc;
    output.dimensions = hidl_vec < uint32_t > {};

    V1_0::Request request = {};
    request.inputs = hidl_vec < RequestArgument > {input};
    request.outputs = hidl_vec < RequestArgument > {output};

    // set the input data (matching source test)
    float indata[] = {1024.25f, 1.f, 0.f, 3.f, -1, -1024.25f};
    AddPoolAndSetData(6, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData<float>(outSize, request);
    float* outdata = reinterpret_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    if (preparedModel.get() != nullptr)
    {
        Execute(preparedModel, request);
    }

    // check the result
    switch (paddingScheme)
    {
        case android::nn::kPaddingValid:
            ARMNN_ANDROID_FP16_TEST(outdata[0], 1022.f, 1022.25f, fp16Enabled)
            break;
        case android::nn::kPaddingSame:
            ARMNN_ANDROID_FP16_TEST(outdata[0], 1022.f, 1022.25f, fp16Enabled)
            DOCTEST_CHECK(outdata[1] == 0.f);
            break;
        default:
            DOCTEST_CHECK(false);
            break;
    }
}

} // namespace driverTestHelpers
