//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "DriverTestHelpers.hpp"
#include <boost/test/unit_test.hpp>
#include <log/log.h>

BOOST_AUTO_TEST_SUITE(GenericLayerTests)

using ArmnnDriver = armnn_driver::ArmnnDriver;
using DriverOptions = armnn_driver::DriverOptions;
using namespace driverTestHelpers;

BOOST_AUTO_TEST_CASE(GetSupportedOperations)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model1 = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand(model1, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model1, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model1, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});

    // make a correct fully connected operation
    model1.operations.resize(2);
    model1.operations[0].type = OperationType::FULLY_CONNECTED;
    model1.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model1.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make an incorrect fully connected operation
    AddIntOperand(model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});
    model1.operations[1].type = OperationType::FULLY_CONNECTED;
    model1.operations[1].inputs = hidl_vec<uint32_t>{4};
    model1.operations[1].outputs = hidl_vec<uint32_t>{5};

    driver->getSupportedOperations(model1, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == true);
    BOOST_TEST(sup[1] == false);

    // Broadcast add/mul are not supported
    Model model2 = {};

    AddInputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddInputOperand(model2, hidl_vec<uint32_t>{4});
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});

    model2.operations.resize(2);

    model2.operations[0].type = OperationType::ADD;
    model2.operations[0].inputs = hidl_vec<uint32_t>{0,1};
    model2.operations[0].outputs = hidl_vec<uint32_t>{2};

    model2.operations[1].type = OperationType::MUL;
    model2.operations[1].inputs = hidl_vec<uint32_t>{0,1};
    model2.operations[1].outputs = hidl_vec<uint32_t>{3};

    driver->getSupportedOperations(model2, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);
    BOOST_TEST(sup[1] == false);

    Model model3 = {};

    // Add unsupported operation, should return no error but we don't support it
    AddInputOperand(model3, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddIntOperand(model3, 2);
    AddOutputOperand(model3, hidl_vec<uint32_t>{1, 2, 2, 2});
    model3.operations.resize(1);
    model3.operations[0].type = OperationType::DEPTH_TO_SPACE;
    model1.operations[0].inputs = hidl_vec<uint32_t>{0, 1};
    model3.operations[0].outputs = hidl_vec<uint32_t>{2};

    driver->getSupportedOperations(model3, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);

    // Add invalid operation
    Model model4 = {};
    AddIntOperand(model4, 0);
    model4.operations.resize(1);
    model4.operations[0].type = static_cast<OperationType>(100);
    model4.operations[0].outputs = hidl_vec<uint32_t>{0};

    driver->getSupportedOperations(model4, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::INVALID_ARGUMENT);
}

// The purpose of this test is to ensure that when encountering an unsupported operation
//      it is skipped and getSupportedOperations() continues (rather than failing and stopping).
//      As per IVGCVSW-710.
BOOST_AUTO_TEST_CASE(UnsupportedLayerContinueOnFailure)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    // broadcast add is unsupported at the time of writing this test, but any unsupported layer will do
    AddInputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddInputOperand(model, hidl_vec<uint32_t>{4});
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    // fully connected
    AddInputOperand(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // broadcast mul is unsupported
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    model.operations.resize(3);

    // unsupported
    model.operations[0].type = OperationType::ADD;
    model.operations[0].inputs = hidl_vec<uint32_t>{0,1};
    model.operations[0].outputs = hidl_vec<uint32_t>{2};

    // supported
    model.operations[1].type = OperationType::FULLY_CONNECTED;
    model.operations[1].inputs  = hidl_vec<uint32_t>{3, 4, 5, 6};
    model.operations[1].outputs = hidl_vec<uint32_t>{7};

    // unsupported
    model.operations[2].type = OperationType::MUL;
    model.operations[2].inputs = hidl_vec<uint32_t>{0,1};
    model.operations[2].outputs = hidl_vec<uint32_t>{8};

    // we are testing that the unsupported layers return false and the test continues
    //      rather than failing and stopping.
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);
    BOOST_TEST(sup[1] == true);
    BOOST_TEST(sup[2] == false);
}

// The purpose of this test is to ensure that when encountering an failure
//      during mem pool mapping we properly report an error to the framework via a callback
BOOST_AUTO_TEST_CASE(ModelToINetworkConverterMemPoolFail)
{
    auto driver = std::make_unique<ArmnnDriver>(armnn::Compute::CpuRef);

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model = {};

    model.pools = hidl_vec<hidl_memory>{hidl_memory("Unsuported hidl memory type", nullptr, 0)};

    //memory pool mapping should fail, we should report an error
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::GENERAL_FAILURE);
}

BOOST_AUTO_TEST_SUITE_END()
