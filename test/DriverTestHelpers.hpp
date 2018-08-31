//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#ifndef LOG_TAG
#define LOG_TAG "ArmnnDriverTests"
#endif // LOG_TAG

#include "../ArmnnDriver.hpp"
#include <iosfwd>

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

std::ostream& operator<<(std::ostream& os, android::hardware::neuralnetworks::V1_0::ErrorStatus stat);

struct ExecutionCallback : public IExecutionCallback
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

class PreparedModelCallback : public IPreparedModelCallback
{
public:
    PreparedModelCallback()
        : m_ErrorStatus(ErrorStatus::NONE)
        , m_PreparedModel()
    { }
    ~PreparedModelCallback() override { }

    Return<void> notify(ErrorStatus status,
                        const android::sp<IPreparedModel>& preparedModel) override;
    ErrorStatus GetErrorStatus() { return m_ErrorStatus; }
    android::sp<IPreparedModel> GetPreparedModel() { return m_PreparedModel; }

private:
    ErrorStatus                  m_ErrorStatus;
    android::sp<IPreparedModel>  m_PreparedModel;
};

hidl_memory allocateSharedMemory(int64_t size);

android::sp<IMemory> AddPoolAndGetData(uint32_t size, Request& request);

void AddPoolAndSetData(uint32_t size, Request& request, const float* data);

void AddOperand(::android::hardware::neuralnetworks::V1_0::Model& model, const Operand& op);

void AddIntOperand(::android::hardware::neuralnetworks::V1_0::Model& model, int32_t value);

template<typename T>
OperandType TypeToOperandType();

template<>
OperandType TypeToOperandType<float>();

template<>
OperandType TypeToOperandType<int32_t>();

template<typename T>
void AddTensorOperand(::android::hardware::neuralnetworks::V1_0::Model& model,
                      hidl_vec<uint32_t> dimensions,
                      T* values,
                      OperandType operandType = OperandType::TENSOR_FLOAT32)
{
    uint32_t totalElements = 1;
    for (uint32_t dim : dimensions)
    {
        totalElements *= dim;
    }

    DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = totalElements * sizeof(T);

    Operand op    = {};
    op.type       = operandType;
    op.dimensions = dimensions;
    op.lifetime   = OperandLifeTime::CONSTANT_COPY;
    op.location   = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    for (uint32_t i = 0; i < totalElements; i++)
    {
        *(reinterpret_cast<T*>(&model.operandValues[location.offset]) + i) = values[i];
    }

    AddOperand(model, op);
}

void AddInputOperand(::android::hardware::neuralnetworks::V1_0::Model& model,
                     hidl_vec<uint32_t> dimensions,
                     ::android::hardware::neuralnetworks::V1_0::OperandType operandType = OperandType::TENSOR_FLOAT32);

void AddOutputOperand(::android::hardware::neuralnetworks::V1_0::Model& model,
                      hidl_vec<uint32_t> dimensions,
                      ::android::hardware::neuralnetworks::V1_0::OperandType operandType = OperandType::TENSOR_FLOAT32);

android::sp<IPreparedModel> PrepareModel(const ::android::hardware::neuralnetworks::V1_0::Model& model,
                                         armnn_driver::ArmnnDriver& driver);

android::sp<IPreparedModel> PrepareModelWithStatus(const ::android::hardware::neuralnetworks::V1_0::Model& model,
                                                   armnn_driver::ArmnnDriver& driver,
                                                   ErrorStatus & prepareStatus,
                                                   ErrorStatus expectedStatus=ErrorStatus::NONE);

ErrorStatus Execute(android::sp<IPreparedModel> preparedModel,
                    const Request& request,
                    ErrorStatus expectedStatus=ErrorStatus::NONE);

android::sp<ExecutionCallback> ExecuteNoWait(android::sp<IPreparedModel> preparedModel,
                                             const Request& request);

} // namespace driverTestHelpers
