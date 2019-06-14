//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#ifndef LOG_TAG
#define LOG_TAG "ArmnnDriverTests"
#endif // LOG_TAG

#include "../ArmnnDriver.hpp"
#include <iosfwd>
#include <boost/test/unit_test.hpp>

namespace android
{
namespace hardware
{
namespace neuralnetworks
{
namespace V1_0
{

std::ostream& operator<<(std::ostream& os, ErrorStatus stat);

} // namespace android::hardware::neuralnetworks::V1_0
} // namespace android::hardware::neuralnetworks
} // namespace android::hardware
} // namespace android

namespace driverTestHelpers
{

std::ostream& operator<<(std::ostream& os, V1_0::ErrorStatus stat);

struct ExecutionCallback : public V1_0::IExecutionCallback
{
    ExecutionCallback() : mNotified(false) {}
    Return<void> notify(ErrorStatus status) override;
    /// wait until the callback has notified us that it is done
    Return<void> wait();

private:
    // use a mutex and a condition variable to wait for asynchronous callbacks
    std::mutex mMutex;
    std::condition_variable mCondition;
    // and a flag, in case we are notified before the wait call
    bool mNotified;
};

class PreparedModelCallback : public V1_0::IPreparedModelCallback
{
public:
    PreparedModelCallback()
        : m_ErrorStatus(ErrorStatus::NONE)
        , m_PreparedModel()
    { }
    ~PreparedModelCallback() override { }

    Return<void> notify(ErrorStatus status,
                        const android::sp<V1_0::IPreparedModel>& preparedModel) override;
    ErrorStatus GetErrorStatus() { return m_ErrorStatus; }
    android::sp<V1_0::IPreparedModel> GetPreparedModel() { return m_PreparedModel; }

private:
    ErrorStatus                  m_ErrorStatus;
    android::sp<V1_0::IPreparedModel>  m_PreparedModel;
};

hidl_memory allocateSharedMemory(int64_t size);

android::sp<IMemory> AddPoolAndGetData(uint32_t size, Request& request);

void AddPoolAndSetData(uint32_t size, Request& request, const float* data);

template<typename HalPolicy,
         typename HalModel   = typename HalPolicy::Model,
         typename HalOperand = typename HalPolicy::Operand>
void AddOperand(HalModel& model, const HalOperand& op)
{
    model.operands.resize(model.operands.size() + 1);
    model.operands[model.operands.size() - 1] = op;
}

template<typename HalPolicy, typename HalModel = typename HalPolicy::Model>
void AddIntOperand(HalModel& model, int32_t value)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(int32_t);

    HalOperand op    = {};
    op.type          = HalOperandType::INT32;
    op.dimensions    = hidl_vec<uint32_t>{};
    op.lifetime      = HalOperandLifeTime::CONSTANT_COPY;
    op.location      = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<int32_t*>(&model.operandValues[location.offset]) = value;

    AddOperand<HalPolicy>(model, op);
}

template<typename HalPolicy, typename HalModel = typename HalPolicy::Model>
void AddBoolOperand(HalModel& model, bool value)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(uint8_t);

    HalOperand op    = {};
    op.type          = HalOperandType::BOOL;
    op.dimensions    = hidl_vec<uint32_t>{};
    op.lifetime      = HalOperandLifeTime::CONSTANT_COPY;
    op.location      = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<uint8_t*>(&model.operandValues[location.offset]) = static_cast<uint8_t>(value);

    AddOperand<HalModel>(model, op);
}

template<typename T>
OperandType TypeToOperandType();

template<>
OperandType TypeToOperandType<float>();

template<>
OperandType TypeToOperandType<int32_t>();

template<typename HalPolicy,
         typename T,
         typename HalModel           = typename HalPolicy::Model,
         typename HalOperandType     = typename HalPolicy::OperandType,
         typename HalOperandLifeTime = typename HalPolicy::OperandLifeTime>
void AddTensorOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      const T* values,
                      HalOperandType operandType = HalOperandType::TENSOR_FLOAT32,
                      HalOperandLifeTime operandLifeTime = HalOperandLifeTime::CONSTANT_COPY)
{
    using HalOperand = typename HalPolicy::Operand;

    uint32_t totalElements = 1;
    for (uint32_t dim : dimensions)
    {
        totalElements *= dim;
    }

    DataLocation location = {};
    location.length = totalElements * sizeof(T);

    if(operandLifeTime == HalOperandLifeTime::CONSTANT_COPY)
    {
        location.offset = model.operandValues.size();
    }

    HalOperand op    = {};
    op.type          = operandType;
    op.dimensions    = dimensions;
    op.lifetime      = HalOperandLifeTime::CONSTANT_COPY;
    op.location      = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    for (uint32_t i = 0; i < totalElements; i++)
    {
        *(reinterpret_cast<T*>(&model.operandValues[location.offset]) + i) = values[i];
    }

    AddOperand<HalPolicy>(model, op);
}

template<typename HalPolicy,
         typename T,
         typename HalModel           = typename HalPolicy::Model,
         typename HalOperandType     = typename HalPolicy::OperandType,
         typename HalOperandLifeTime = typename HalPolicy::OperandLifeTime>
void AddTensorOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      const std::vector<T>& values,
                      HalOperandType operandType = HalPolicy::OperandType::TENSOR_FLOAT32,
                      HalOperandLifeTime operandLifeTime = HalOperandLifeTime::CONSTANT_COPY)
{
    AddTensorOperand<HalPolicy, T>(model, dimensions, values.data(), operandType, operandLifeTime);
}

template<typename HalPolicy,
         typename HalModel       = typename HalPolicy::Model,
         typename HalOperandType = typename HalPolicy::OperandType>
void AddInputOperand(HalModel& model,
                     const hidl_vec<uint32_t>& dimensions,
                     HalOperandType operandType = HalOperandType::TENSOR_FLOAT32)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    HalOperand op    = {};
    op.type          = operandType;
    op.scale         = operandType == HalOperandType::TENSOR_QUANT8_ASYMM ? 1.f / 255.f : 0.f;
    op.dimensions    = dimensions;
    op.lifetime      = HalOperandLifeTime::MODEL_INPUT;

    AddOperand<HalPolicy>(model, op);

    model.inputIndexes.resize(model.inputIndexes.size() + 1);
    model.inputIndexes[model.inputIndexes.size() - 1] = model.operands.size() - 1;
}

template<typename HalPolicy,
         typename HalModel       = typename HalPolicy::Model,
         typename HalOperandType = typename HalPolicy::OperandType>
void AddOutputOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      HalOperandType operandType = HalOperandType::TENSOR_FLOAT32)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    HalOperand op    = {};
    op.type          = operandType;
    op.scale         = operandType == HalOperandType::TENSOR_QUANT8_ASYMM ? 1.f / 255.f : 0.f;
    op.dimensions    = dimensions;
    op.lifetime      = HalOperandLifeTime::MODEL_OUTPUT;

    AddOperand<HalPolicy>(model, op);

    model.outputIndexes.resize(model.outputIndexes.size() + 1);
    model.outputIndexes[model.outputIndexes.size() - 1] = model.operands.size() - 1;
}

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_0::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         ErrorStatus& prepareStatus,
                                                         ErrorStatus expectedStatus = ErrorStatus::NONE);

#ifdef ARMNN_ANDROID_NN_V1_1

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_1::Model& model,
                                                   armnn_driver::ArmnnDriver& driver,
                                                   ErrorStatus& prepareStatus,
                                                   ErrorStatus expectedStatus = ErrorStatus::NONE);

#endif

template<typename HalModel>
android::sp<V1_0::IPreparedModel> PrepareModel(const HalModel& model,
                                               armnn_driver::ArmnnDriver& driver)
{
    ErrorStatus prepareStatus = ErrorStatus::NONE;
    return PrepareModelWithStatus(model, driver, prepareStatus);
}

ErrorStatus Execute(android::sp<V1_0::IPreparedModel> preparedModel,
                    const Request& request,
                    ErrorStatus expectedStatus = ErrorStatus::NONE);

android::sp<ExecutionCallback> ExecuteNoWait(android::sp<V1_0::IPreparedModel> preparedModel,
                                             const Request& request);

} // namespace driverTestHelpers
