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

template<typename HalModel>
void AddOperand(HalModel& model, const V1_0::Operand& op)
{
    model.operands.resize(model.operands.size() + 1);
    model.operands[model.operands.size() - 1] = op;
}

template<typename HalModel>
void AddIntOperand(HalModel& model, int32_t value)
{
    DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(int32_t);

    V1_0::Operand op    = {};
    op.type             = V1_0::OperandType::INT32;
    op.dimensions       = hidl_vec<uint32_t>{};
    op.lifetime         = V1_0::OperandLifeTime::CONSTANT_COPY;
    op.location         = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<int32_t*>(&model.operandValues[location.offset]) = value;

    AddOperand<HalModel>(model, op);
}

template<typename T>
OperandType TypeToOperandType();

template<>
OperandType TypeToOperandType<float>();

template<>
OperandType TypeToOperandType<int32_t>();

template<typename HalModel, typename T>
void AddTensorOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      const T* values,
                      V1_0::OperandType operandType = V1_0::OperandType::TENSOR_FLOAT32,
                      V1_0::OperandLifeTime operandLifeTime = V1_0::OperandLifeTime::CONSTANT_COPY)
{
    uint32_t totalElements = 1;
    for (uint32_t dim : dimensions)
    {
        totalElements *= dim;
    }

    DataLocation location = {};
    location.length = totalElements * sizeof(T);

    if(operandLifeTime == V1_0::OperandLifeTime::CONSTANT_COPY)
    {
        location.offset = model.operandValues.size();
    }

    V1_0::Operand op    = {};
    op.type             = operandType;
    op.dimensions       = dimensions;
    op.lifetime         = V1_0::OperandLifeTime::CONSTANT_COPY;
    op.location         = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    for (uint32_t i = 0; i < totalElements; i++)
    {
        *(reinterpret_cast<T*>(&model.operandValues[location.offset]) + i) = values[i];
    }

    AddOperand<HalModel>(model, op);
}

template<typename HalModel, typename T>
void AddTensorOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      const std::vector<T>& values,
                      V1_0::OperandType operandType = V1_0::OperandType::TENSOR_FLOAT32,
                      V1_0::OperandLifeTime operandLifeTime = V1_0::OperandLifeTime::CONSTANT_COPY)
{
    AddTensorOperand<HalModel, T>(model, dimensions, values.data(), operandType, operandLifeTime);
}

template<typename HalModel>
void AddInputOperand(HalModel& model,
                     const hidl_vec<uint32_t>& dimensions,
                     V1_0::OperandType operandType = V1_0::OperandType::TENSOR_FLOAT32)
{
    V1_0::Operand op    = {};
    op.type             = operandType;
    op.scale            = operandType == V1_0::OperandType::TENSOR_QUANT8_ASYMM ? 1.f / 255.f : 0.f;
    op.dimensions       = dimensions;
    op.lifetime         = V1_0::OperandLifeTime::MODEL_INPUT;

    AddOperand<HalModel>(model, op);

    model.inputIndexes.resize(model.inputIndexes.size() + 1);
    model.inputIndexes[model.inputIndexes.size() - 1] = model.operands.size() - 1;
}

template<typename HalModel>
void AddOutputOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      V1_0::OperandType operandType = V1_0::OperandType::TENSOR_FLOAT32)
{
    V1_0::Operand op    = {};
    op.type             = operandType;
    op.scale            = operandType == V1_0::OperandType::TENSOR_QUANT8_ASYMM ? 1.f / 255.f : 0.f;
    op.dimensions       = dimensions;
    op.lifetime         = V1_0::OperandLifeTime::MODEL_OUTPUT;

    AddOperand<HalModel>(model, op);

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
