//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"

#include <log/log.h>

DOCTEST_TEST_SUITE("GenericLayerTests")
{

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

using HalPolicy = hal_1_0::HalPolicy;

DOCTEST_TEST_CASE("GetSupportedOperations")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](V1_0::ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    HalPolicy::Model model0 = {};

    // Add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand<HalPolicy>(model0, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand<HalPolicy>(model0, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand<HalPolicy>(model0, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand<HalPolicy>(model0, actValue);
    AddOutputOperand<HalPolicy>(model0, hidl_vec<uint32_t>{1, 1});

    model0.operations.resize(1);

    // Make a correct fully connected operation
    model0.operations[0].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model0.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model0.operations[0].outputs = hidl_vec<uint32_t>{4};

    driver->getSupportedOperations(model0, cb);
    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(supported.size() == (size_t)1);
    DOCTEST_CHECK(supported[0] == true);

    V1_0::Model model1 = {};

    AddInputOperand<HalPolicy>(model1, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand<HalPolicy>(model1, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand<HalPolicy>(model1, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand<HalPolicy>(model1, actValue);
    AddOutputOperand<HalPolicy>(model1, hidl_vec<uint32_t>{1, 1});

    model1.operations.resize(2);

    // Make a correct fully connected operation
    model1.operations[0].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model1.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model1.operations[0].outputs = hidl_vec<uint32_t>{4};

    // Add an incorrect fully connected operation
    AddIntOperand<HalPolicy>(model1, actValue);
    AddOutputOperand<HalPolicy>(model1, hidl_vec<uint32_t>{1, 1});

    model1.operations[1].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model1.operations[1].inputs  = hidl_vec<uint32_t>{4}; // Only 1 input operand, expected 4
    model1.operations[1].outputs = hidl_vec<uint32_t>{5};

    driver->getSupportedOperations(model1, cb);

    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::INVALID_ARGUMENT);
    DOCTEST_CHECK(supported.empty());

    // Test Broadcast on add/mul operators
    HalPolicy::Model model2 = {};

    AddInputOperand<HalPolicy>(model2,
                               hidl_vec<uint32_t>{1, 1, 3, 4},
                               HalPolicy::OperandType::TENSOR_FLOAT32,
                               0.0f,
                               0,
                               2);
    AddInputOperand<HalPolicy>(model2,
                               hidl_vec<uint32_t>{4},
                               HalPolicy::OperandType::TENSOR_FLOAT32,
                               0.0f,
                               0,
                               2);
    AddIntOperand<HalPolicy>(model2, actValue, 2);
    AddOutputOperand<HalPolicy>(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand<HalPolicy>(model2, hidl_vec<uint32_t>{1, 1, 3, 4});

    model2.operations.resize(2);

    model2.operations[0].type    = HalPolicy::OperationType::ADD;
    model2.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model2.operations[0].outputs = hidl_vec<uint32_t>{3};

    model2.operations[1].type    = HalPolicy::OperationType::MUL;
    model2.operations[1].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model2.operations[1].outputs = hidl_vec<uint32_t>{4};

    driver->getSupportedOperations(model2, cb);
    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(supported.size() == (size_t)2);
    DOCTEST_CHECK(supported[0] == true);
    DOCTEST_CHECK(supported[1] == true);

    V1_0::Model model3 = {};

    AddInputOperand<HalPolicy>(model3,
                               hidl_vec<uint32_t>{1, 1, 3, 4},
                               HalPolicy::OperandType::TENSOR_INT32);
    AddInputOperand<HalPolicy>(model3,
                               hidl_vec<uint32_t>{4},
                               HalPolicy::OperandType::TENSOR_INT32);
    AddInputOperand<HalPolicy>(model3, hidl_vec<uint32_t>{1, 1, 3, 4});

    AddOutputOperand<HalPolicy>(model3, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand<HalPolicy>(model3,
                                hidl_vec<uint32_t>{1, 1, 3, 4},
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                1.f / 225.f);

    model3.operations.resize(1);

    // Add unsupported operation, should return no error but we don't support it
    model3.operations[0].type    = HalPolicy::OperationType::HASHTABLE_LOOKUP;
    model3.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model3.operations[0].outputs = hidl_vec<uint32_t>{3, 4};

    driver->getSupportedOperations(model3, cb);
    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(supported.size() == (size_t)1);
    DOCTEST_CHECK(supported[0] == false);

    HalPolicy::Model model4 = {};

    AddIntOperand<HalPolicy>(model4, 0);

    model4.operations.resize(1);

    // Add invalid operation
    model4.operations[0].type    = static_cast<HalPolicy::OperationType>(100);
    model4.operations[0].outputs = hidl_vec<uint32_t>{0};

    driver->getSupportedOperations(model4, cb);
    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::INVALID_ARGUMENT);
    DOCTEST_CHECK(supported.empty());
}

// The purpose of this test is to ensure that when encountering an unsupported operation
// it is skipped and getSupportedOperations() continues (rather than failing and stopping).
// As per IVGCVSW-710.
DOCTEST_TEST_CASE("UnsupportedLayerContinueOnFailure")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](V1_0::ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    HalPolicy::Model model = {};

    // Operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    // HASHTABLE_LOOKUP is unsupported at the time of writing this test, but any unsupported layer will do
    AddInputOperand<HalPolicy>(model,
                               hidl_vec<uint32_t>{1, 1, 3, 4},
                               HalPolicy::OperandType::TENSOR_INT32);
    AddInputOperand<HalPolicy>(model,
                               hidl_vec<uint32_t>{4},
                               HalPolicy::OperandType::TENSOR_INT32,
                               0.0f,
                               0,
                               2);
    AddInputOperand<HalPolicy>(model,
                               hidl_vec<uint32_t>{1, 1, 3, 4},
                               HalPolicy::OperandType::TENSOR_FLOAT32,
                               0.0f,
                               0,
                               2);

    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand<HalPolicy>(model,
                                hidl_vec<uint32_t>{1, 1, 3, 4},
                                HalPolicy::OperandType::TENSOR_QUANT8_ASYMM,
                                1.f / 225.f);

    // Fully connected is supported
    AddInputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 3});

    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand<HalPolicy>(model, hidl_vec<uint32_t>{1}, biasValue);

    AddIntOperand<HalPolicy>(model, actValue);

    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1});

    // EMBEDDING_LOOKUP is unsupported
    AddOutputOperand<HalPolicy>(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    model.operations.resize(3);

    // Unsupported
    model.operations[0].type    = HalPolicy::OperationType::HASHTABLE_LOOKUP;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model.operations[0].outputs = hidl_vec<uint32_t>{3, 4};

    // Supported
    model.operations[1].type    = HalPolicy::OperationType::FULLY_CONNECTED;
    model.operations[1].inputs  = hidl_vec<uint32_t>{5, 6, 7, 8};
    model.operations[1].outputs = hidl_vec<uint32_t>{9};

    // Unsupported
    model.operations[2].type    = HalPolicy::OperationType::EMBEDDING_LOOKUP;
    model.operations[2].inputs  = hidl_vec<uint32_t>{1, 2};
    model.operations[2].outputs = hidl_vec<uint32_t>{10};

    // We are testing that the unsupported layers return false and the test continues rather than failing and stopping
    driver->getSupportedOperations(model, cb);
    DOCTEST_CHECK((int)errorStatus == (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(supported.size() == (size_t)3);
    DOCTEST_CHECK(supported[0] == false);
    DOCTEST_CHECK(supported[1] == true);
    DOCTEST_CHECK(supported[2] == false);
}

// The purpose of this test is to ensure that when encountering an failure
// during mem pool mapping we properly report an error to the framework via a callback
DOCTEST_TEST_CASE("ModelToINetworkConverterMemPoolFail")
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    V1_0::ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](V1_0::ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    HalPolicy::Model model = {};

    model.pools = hidl_vec<hidl_memory>{hidl_memory("Unsuported hidl memory type", nullptr, 0)};

    // Memory pool mapping should fail, we should report an error
    driver->getSupportedOperations(model, cb);
    DOCTEST_CHECK((int)errorStatus != (int)V1_0::ErrorStatus::NONE);
    DOCTEST_CHECK(supported.empty());
}

}
