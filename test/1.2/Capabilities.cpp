//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../DriverTestHelpers.hpp"
#include "Utils.h"

#include <1.2/ArmnnDriverImpl.hpp>

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
        __system_property_set("Armnn.operandTypeTensorQuant8SymmPerformance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant8SymmPerformance.powerUsage", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.execTime", nullStr);
        __system_property_set("Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.powerUsage", nullStr);
    }
};

void CheckOperandType(const V1_2::Capabilities& capabilities, V1_2::OperandType type, float execTime, float powerUsage)
{
    using namespace armnn_driver::hal_1_2;
    V1_0::PerformanceInfo perfInfo = android::nn::lookup(capabilities.operandPerformance, type);
    DOCTEST_CHECK(perfInfo.execTime == execTime);
    DOCTEST_CHECK(perfInfo.powerUsage == powerUsage);
}

DOCTEST_TEST_SUITE("CapabilitiesTests")
{
DOCTEST_TEST_CASE_FIXTURE(CapabilitiesFixture, "PerformanceCapabilitiesWithRuntime")
{
    using namespace android::nn;

    auto getCapabilitiesFn = [&](V1_0::ErrorStatus error, const V1_2::Capabilities& capabilities)
        {
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_FLOAT32, 2.0f, 2.1f);
            CheckOperandType(capabilities, V1_2::OperandType::FLOAT32, 2.2f, 2.3f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_FLOAT16, 2.4f, 2.5f);
            CheckOperandType(capabilities, V1_2::OperandType::FLOAT16, 2.6f, 2.7f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT8_ASYMM, 2.8f, 2.9f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT16_SYMM, 3.0f, 3.1f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_INT32, 3.2f, 3.3f);
            CheckOperandType(capabilities, V1_2::OperandType::INT32, 3.4f, 3.5f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT8_SYMM, 2.8f, 2.9f);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL, 2.8f, 2.9f);

            // Unsupported operands take FLT_MAX value
            CheckOperandType(capabilities, V1_2::OperandType::UINT32, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::BOOL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT16_ASYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_BOOL8, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::OEM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_OEM_BYTE, FLT_MAX, FLT_MAX);

            bool result = (error == V1_0::ErrorStatus::NONE);
            DOCTEST_CHECK(result);
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
    __system_property_set("Armnn.operandTypeTensorQuant8SymmPerformance.execTime", "2.8f");
    __system_property_set("Armnn.operandTypeTensorQuant8SymmPerformance.powerUsage", "2.9f");
    __system_property_set("Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.execTime", "2.8f");
    __system_property_set("Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.powerUsage", "2.9f");

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    armnn_driver::hal_1_2::ArmnnDriverImpl::getCapabilities_1_2(runtime, getCapabilitiesFn);
}

DOCTEST_TEST_CASE_FIXTURE(CapabilitiesFixture, "PerformanceCapabilitiesUndefined")
{
    using namespace android::nn;

    float defaultValue = .1f;

    auto getCapabilitiesFn = [&](V1_0::ErrorStatus error, const V1_2::Capabilities& capabilities)
        {
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_FLOAT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::FLOAT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_FLOAT16, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::FLOAT16, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT8_ASYMM, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT16_SYMM, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_INT32, defaultValue, defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::INT32, defaultValue, defaultValue);
            CheckOperandType(capabilities,
                             V1_2::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
                             defaultValue,
                             defaultValue);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT8_SYMM, defaultValue, defaultValue);

            // Unsupported operands take FLT_MAX value
            CheckOperandType(capabilities, V1_2::OperandType::UINT32, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::BOOL, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_QUANT16_ASYMM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_BOOL8, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::OEM, FLT_MAX, FLT_MAX);
            CheckOperandType(capabilities, V1_2::OperandType::TENSOR_OEM_BYTE, FLT_MAX, FLT_MAX);

            bool result = (error == V1_0::ErrorStatus::NONE);
            DOCTEST_CHECK(result);
        };

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    armnn_driver::hal_1_2::ArmnnDriverImpl::getCapabilities_1_2(runtime, getCapabilitiesFn);
}

}