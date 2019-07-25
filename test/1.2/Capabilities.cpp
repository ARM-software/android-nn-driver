//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../../1.2/ArmnnDriverImpl.hpp"

#include "Utils.h"

#include <boost/test/unit_test.hpp>
#include <boost/core/ignore_unused.hpp>

#include <sys/system_properties.h>

#include <cfloat>

using namespace std;

struct CapabilitiesFixture
{
    CapabilitiesFixture()
    {
        // CleanUp before the execution of each test
        CleanUp();
    }

    ~CapabilitiesFixture()
    {
        // CleanUp after the execution of each test
        CleanUp();
    }

    void CleanUp()
    {
        const char* nullStr = "";

        __system_property_set("Armnn.operandTypeTensorFloat32Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorFloat32Performance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeFloat32Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeFloat32Performance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeTensorFloat16Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorFloat16Performance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeFloat16Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeFloat16Performance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant8AsymmPerformance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant8AsymmPerformance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant16SymmPerformance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant16SymmPerformance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeTensorInt32Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorInt32Performance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeInt32Performance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeInt32Performance.powerUsage", nullStr);
    }
};

void CheckOperandType(const V1_2::Capabilities& capabilities, OperandType type, float execTime, float powerUsage)
{
    PerformanceInfo perfInfo = android::nn::lookup(capabilities.operandPerformance, type);
    BOOST_ASSERT(perfInfo.execTime == execTime);
    BOOST_ASSERT(perfInfo.powerUsage == powerUsage);
}

BOOST_FIXTURE_TEST_SUITE(CapabilitiesTests, CapabilitiesFixture)

BOOST_AUTO_TEST_CASE(PerformanceCapabilitiesWithRuntime)
{
    using namespace armnn_driver::hal_1_2;
    using namespace android::nn;

    auto getCapabilitiesFn = [&](ErrorStatus error, const V1_2::Capabilities& capabilities)
        {
            CheckOperandType(capabilities, OperandType::TENSOR_FLOAT32, 2.0f, 2.1f);
            CheckOperandType(capabilities, OperandType::FLOAT32, 2.2f, 2.3f);
            CheckOperandType(capabilities, OperandType::TENSOR_FLOAT16, 2.4f, 2.5f);
            CheckOperandType(capabilities, OperandType::FLOAT16, 2.6f, 2.7f);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_ASYMM, 2.8f, 2.9f);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT16_SYMM, 3.0f, 3.1f);
            CheckOperandType(capabilities, OperandType::TENSOR_INT32, 3.2f, 3.3f);
            CheckOperandType(capabilities, OperandType::INT32, 3.4f, 3.5f);

            // Unsupported operands take FLT_MAX value
            CheckOperandType(capabilities, OperandType::UINT32, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::BOOL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_SYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT16_ASYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_BOOL8, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::OEM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_OEM_BYTE, FLT_MAX, FLT_MAX);

            BOOST_ASSERT(error == ErrorStatus::NONE);
        };

    __system_property_set("Armnn.operandTypeTensorFloat32Performance.execTime", "2.0f");
    __system_property_set("Armnn.operandTypeTensorFloat32Performance.powerUsage", "2.1f");
    __system_property_set("Armnn.operandTypeFloat32Performance.execTime", "2.2f");
    __system_property_set("Armnn.operandTypeFloat32Performance.powerUsage", "2.3f");
    __system_property_set("Armnn.operandTypeTensorFloat16Performance.execTime", "2.4f");
    __system_property_set("Armnn.operandTypeTensorFloat16Performance.powerUsage", "2.5f");
    __system_property_set("Armnn.operandTypeFloat16Performance.execTime", "2.6f");
    __system_property_set("Armnn.operandTypeFloat16Performance.powerUsage", "2.7f");
    __system_property_set("Armnn.operandTypeTensorQuant8AsymmPerformance.execTime", "2.8f");
    __system_property_set("Armnn.operandTypeTensorQuant8AsymmPerformance.powerUsage", "2.9f");
    __system_property_set("Armnn.operandTypeTensorQuant16SymmPerformance.execTime", "3.0f");
    __system_property_set("Armnn.operandTypeTensorQuant16SymmPerformance.powerUsage", "3.1f");
    __system_property_set("Armnn.operandTypeTensorInt32Performance.execTime", "3.2f");
    __system_property_set("Armnn.operandTypeTensorInt32Performance.powerUsage", "3.3f");
    __system_property_set("Armnn.operandTypeInt32Performance.execTime", "3.4f");
    __system_property_set("Armnn.operandTypeInt32Performance.powerUsage", "3.5f");

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    ArmnnDriverImpl::getCapabilities_1_2(runtime, getCapabilitiesFn);
}

BOOST_AUTO_TEST_CASE(PerformanceCapabilitiesUndefined)
{
    using namespace armnn_driver::hal_1_2;
    using namespace android::nn;

    float defaultValue = .1f;

    auto getCapabilitiesFn = [&](ErrorStatus error, const V1_2::Capabilities& capabilities)
        {
            CheckOperandType(capabilities, OperandType::TENSOR_FLOAT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::FLOAT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::TENSOR_FLOAT16, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::FLOAT16, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_ASYMM, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT16_SYMM, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::TENSOR_INT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, OperandType::INT32, defaultValue, defaultValue);

            // Unsupported operands take FLT_MAX value
            CheckOperandType(capabilities, OperandType::UINT32, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::BOOL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_SYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT16_ASYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_BOOL8, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::OEM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, OperandType::TENSOR_OEM_BYTE, FLT_MAX, FLT_MAX);

            BOOST_ASSERT(error == ErrorStatus::NONE);
        };

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    ArmnnDriverImpl::getCapabilities_1_2(runtime, getCapabilitiesFn);
}

BOOST_AUTO_TEST_SUITE_END()