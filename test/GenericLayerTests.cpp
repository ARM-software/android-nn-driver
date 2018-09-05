//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DriverTestHelpers.hpp"
#include <boost/test/unit_test.hpp>
#include <log/log.h>

BOOST_AUTO_TEST_SUITE(GenericLayerTests)

using namespace android::hardware;
using namespace driverTestHelpers;
using namespace armnn_driver;

BOOST_AUTO_TEST_CASE(GetSupportedOperations)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    neuralnetworks::V1_0::Model model0 = {};

    // Add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand (model0, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model0, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model0, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand   (model0, actValue);
    AddOutputOperand(model0, hidl_vec<uint32_t>{1, 1});

    model0.operations.resize(1);

    // Make a correct fully connected operation
    model0.operations[0].type    = neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    model0.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model0.operations[0].outputs = hidl_vec<uint32_t>{4};

    driver->getSupportedOperations(model0, cb);
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::NONE);
    BOOST_TEST(supported.size() == (size_t)1);
    BOOST_TEST(supported[0] == true);

    neuralnetworks::V1_0::Model model1 = {};

    AddInputOperand (model1, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model1, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model1, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand   (model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});

    model1.operations.resize(2);

    // Make a correct fully connected operation
    model1.operations[0].type    = neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    model1.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model1.operations[0].outputs = hidl_vec<uint32_t>{4};

    // Add an incorrect fully connected operation
    AddIntOperand   (model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});
    model1.operations[1].type    = neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    model1.operations[1].inputs  = hidl_vec<uint32_t>{4}; // Only 1 input operand, expected 4
    model1.operations[1].outputs = hidl_vec<uint32_t>{5};

    driver->getSupportedOperations(model1, cb);

#if defined(ARMNN_ANDROID_P)
    // In Android P, android::nn::validateModel returns INVALID_ARGUMENT, because of the wrong number of inputs for the
    // fully connected layer (1 instead of 4)
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::INVALID_ARGUMENT);
    BOOST_TEST(supported.empty());
#else
    // In Android O, android::nn::validateModel indicates that the second (wrong) fully connected layer in unsupported
    // in the vector of flags returned by the callback
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::NONE);
    BOOST_TEST(supported.size() == (size_t)2);
    BOOST_TEST(supported[0] == true);
    BOOST_TEST(supported[1] == false);
#endif

    // Test Broadcast on add/mul operators
    neuralnetworks::V1_0::Model model2 = {};

    AddInputOperand (model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddInputOperand (model2, hidl_vec<uint32_t>{4});
    AddIntOperand   (model2, actValue);
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});

    model2.operations.resize(2);

    model2.operations[0].type    = neuralnetworks::V1_0::OperationType::ADD;
    model2.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model2.operations[0].outputs = hidl_vec<uint32_t>{3};

    model2.operations[1].type    = neuralnetworks::V1_0::OperationType::MUL;
    model2.operations[1].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model2.operations[1].outputs = hidl_vec<uint32_t>{4};

    driver->getSupportedOperations(model2, cb);
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::NONE);
    BOOST_TEST(supported.size() == (size_t)2);
    BOOST_TEST(supported[0] == true);
    BOOST_TEST(supported[1] == true);

    neuralnetworks::V1_0::Model model3 = {};

    AddInputOperand (model3, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddIntOperand   (model3, 2);
    AddOutputOperand(model3, hidl_vec<uint32_t>{1, 2, 2, 2});

    model3.operations.resize(1);

    // Add unsupported operation, should return no error but we don't support it
    model3.operations[0].type    = neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE;
    model3.operations[0].inputs  = hidl_vec<uint32_t>{0, 1};
    model3.operations[0].outputs = hidl_vec<uint32_t>{2};

    driver->getSupportedOperations(model3, cb);
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::NONE);
    BOOST_TEST(supported.size() == (size_t)1);
    BOOST_TEST(supported[0] == false);

    neuralnetworks::V1_0::Model model4 = {};

    AddIntOperand(model4, 0);

    model4.operations.resize(1);

    // Add invalid operation
    model4.operations[0].type    = static_cast<neuralnetworks::V1_0::OperationType>(100);
    model4.operations[0].outputs = hidl_vec<uint32_t>{0};

    driver->getSupportedOperations(model4, cb);
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::INVALID_ARGUMENT);
    BOOST_TEST(supported.empty());
}

// The purpose of this test is to ensure that when encountering an unsupported operation
// it is skipped and getSupportedOperations() continues (rather than failing and stopping).
// As per IVGCVSW-710.
BOOST_AUTO_TEST_CASE(UnsupportedLayerContinueOnFailure)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    neuralnetworks::V1_0::Model model = {};

    // Operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    // HASHTABLE_LOOKUP is unsupported at the time of writing this test, but any unsupported layer will do
    AddInputOperand (model, hidl_vec<uint32_t>{1, 1, 3, 4}, neuralnetworks::V1_0::OperandType::TENSOR_INT32);
    AddInputOperand (model, hidl_vec<uint32_t>{4},          neuralnetworks::V1_0::OperandType::TENSOR_INT32);
    AddInputOperand (model, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4}, neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM);

    // Fully connected is supported
    AddInputOperand (model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand   (model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // EMBEDDING_LOOKUP is unsupported
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    model.operations.resize(3);

    // Unsupported
    model.operations[0].type    = neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2};
    model.operations[0].outputs = hidl_vec<uint32_t>{3, 4};

    // Supported
    model.operations[1].type    = neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    model.operations[1].inputs  = hidl_vec<uint32_t>{5, 6, 7, 8};
    model.operations[1].outputs = hidl_vec<uint32_t>{9};

    // Unsupported
    model.operations[2].type    = neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP;
    model.operations[2].inputs  = hidl_vec<uint32_t>{1, 2};
    model.operations[2].outputs = hidl_vec<uint32_t>{10};

    // We are testing that the unsupported layers return false and the test continues rather than failing and stopping
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)errorStatus == (int)ErrorStatus::NONE);
    BOOST_TEST(supported.size() == (size_t)3);
    BOOST_TEST(supported[0] == false);
    BOOST_TEST(supported[1] == true);
    BOOST_TEST(supported[2] == false);
}

// The purpose of this test is to ensure that when encountering an failure
// during mem pool mapping we properly report an error to the framework via a callback
BOOST_AUTO_TEST_CASE(ModelToINetworkConverterMemPoolFail)
{
    auto driver = std::make_unique<ArmnnDriver>(armnn::Compute::CpuRef);

    ErrorStatus errorStatus;
    std::vector<bool> supported;

    auto cb = [&](ErrorStatus _errorStatus, const std::vector<bool>& _supported)
    {
        errorStatus = _errorStatus;
        supported = _supported;
    };

    neuralnetworks::V1_0::Model model = {};

    model.pools = hidl_vec<hidl_memory>{hidl_memory("Unsuported hidl memory type", nullptr, 0)};

    // Memory pool mapping should fail, we should report an error
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)errorStatus != (int)ErrorStatus::NONE);
    BOOST_TEST(supported.empty());
}

BOOST_AUTO_TEST_SUITE_END()
