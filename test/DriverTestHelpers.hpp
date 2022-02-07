//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#ifndef LOG_TAG
#define LOG_TAG "ArmnnDriverTests"
#endif // LOG_TAG

#include "../ArmnnDriver.hpp"
#include <iosfwd>
#include <android/hidl/allocator/1.0/IAllocator.h>

// Some of the short name macros from 'third-party/doctest/doctest.h' clash with macros in
// 'system/core/base/include/android-base/logging.h' so we use the full DOCTEST macro names
#ifndef DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#endif // DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES

#include <doctest/doctest.h>

using RequestArgument = V1_0::RequestArgument;
using ::android::hidl::allocator::V1_0::IAllocator;

using ::android::hidl::memory::V1_0::IMemory;

namespace android
{
namespace hardware
{
namespace neuralnetworks
{
namespace V1_0
{

std::ostream& operator<<(std::ostream& os, V1_0::ErrorStatus stat);

} // namespace android::hardware::neuralnetworks::V1_0

#ifdef ARMNN_ANDROID_NN_V1_3
namespace V1_3
{

std::ostream& operator<<(std::ostream& os, V1_3::ErrorStatus stat);

} // namespace android::hardware::neuralnetworks::V1_3
#endif

} // namespace android::hardware::neuralnetworks
} // namespace android::hardware
} // namespace android

namespace driverTestHelpers
{

std::ostream& operator<<(std::ostream& os, V1_0::ErrorStatus stat);

#ifdef ARMNN_ANDROID_NN_V1_3
std::ostream& operator<<(std::ostream& os, V1_3::ErrorStatus stat);
#endif

struct ExecutionCallback : public V1_0::IExecutionCallback
{
    ExecutionCallback() : mNotified(false) {}
    Return<void> notify(V1_0::ErrorStatus status) override;
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
        : m_ErrorStatus(V1_0::ErrorStatus::NONE)
        , m_PreparedModel()
    { }
    ~PreparedModelCallback() override { }

    Return<void> notify(V1_0::ErrorStatus status,
                        const android::sp<V1_0::IPreparedModel>& preparedModel) override;
    V1_0::ErrorStatus GetErrorStatus() { return m_ErrorStatus; }
    android::sp<V1_0::IPreparedModel> GetPreparedModel() { return m_PreparedModel; }

private:
    V1_0::ErrorStatus                  m_ErrorStatus;
    android::sp<V1_0::IPreparedModel>  m_PreparedModel;
};

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

class PreparedModelCallback_1_2 : public V1_2::IPreparedModelCallback
{
public:
    PreparedModelCallback_1_2()
            : m_ErrorStatus(V1_0::ErrorStatus::NONE)
            , m_PreparedModel()
            , m_PreparedModel_1_2()
    { }
    ~PreparedModelCallback_1_2() override { }

    Return<void> notify(V1_0::ErrorStatus status, const android::sp<V1_0::IPreparedModel>& preparedModel) override;

    Return<void> notify_1_2(V1_0::ErrorStatus status, const android::sp<V1_2::IPreparedModel>& preparedModel) override;

    V1_0::ErrorStatus GetErrorStatus() { return m_ErrorStatus; }

    android::sp<V1_0::IPreparedModel> GetPreparedModel() { return m_PreparedModel; }

    android::sp<V1_2::IPreparedModel> GetPreparedModel_1_2() { return m_PreparedModel_1_2; }

private:
    V1_0::ErrorStatus                   m_ErrorStatus;
    android::sp<V1_0::IPreparedModel>  m_PreparedModel;
    android::sp<V1_2::IPreparedModel>  m_PreparedModel_1_2;
};

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

class PreparedModelCallback_1_3 : public V1_3::IPreparedModelCallback
{
public:
    PreparedModelCallback_1_3()
            : m_1_0_ErrorStatus(V1_0::ErrorStatus::NONE)
            , m_1_3_ErrorStatus(V1_3::ErrorStatus::NONE)
            , m_PreparedModel()
            , m_PreparedModel_1_2()
            , m_PreparedModel_1_3()
    { }
    ~PreparedModelCallback_1_3() override { }

    Return<void> notify(V1_0::ErrorStatus status, const android::sp<V1_0::IPreparedModel>& preparedModel) override;

    Return<void> notify_1_2(V1_0::ErrorStatus status, const android::sp<V1_2::IPreparedModel>& preparedModel) override;

    Return<void> notify_1_3(V1_3::ErrorStatus status, const android::sp<V1_3::IPreparedModel>& preparedModel) override;

    V1_0::ErrorStatus GetErrorStatus() { return m_1_0_ErrorStatus; }

    V1_3::ErrorStatus Get_1_3_ErrorStatus() { return m_1_3_ErrorStatus; }

    android::sp<V1_0::IPreparedModel> GetPreparedModel() { return m_PreparedModel; }

    android::sp<V1_2::IPreparedModel> GetPreparedModel_1_2() { return m_PreparedModel_1_2; }

    android::sp<V1_3::IPreparedModel> GetPreparedModel_1_3() { return m_PreparedModel_1_3; }

private:
    V1_0::ErrorStatus                   m_1_0_ErrorStatus;
    V1_3::ErrorStatus                   m_1_3_ErrorStatus;
    android::sp<V1_0::IPreparedModel>  m_PreparedModel;
    android::sp<V1_2::IPreparedModel>  m_PreparedModel_1_2;
    android::sp<V1_3::IPreparedModel>  m_PreparedModel_1_3;
};

#endif

hidl_memory allocateSharedMemory(int64_t size);

template<typename T>
android::sp<IMemory> AddPoolAndGetData(uint32_t size, V1_0::Request& request)
{
    hidl_memory pool;

    android::sp<IAllocator> allocator = IAllocator::getService("ashmem");
    allocator->allocate(sizeof(T) * size, [&](bool success, const hidl_memory& mem) {
        DOCTEST_CHECK(success);
        pool = mem;
    });

    request.pools.resize(request.pools.size() + 1);
    request.pools[request.pools.size() - 1] = pool;

    android::sp<IMemory> mapped = mapMemory(pool);
    mapped->update();
    return mapped;
}

template<typename T>
android::sp<IMemory> AddPoolAndSetData(uint32_t size, V1_0::Request& request, const T* data)
{
    android::sp<IMemory> memory = AddPoolAndGetData<T>(size, request);

    T* dst = static_cast<T*>(static_cast<void*>(memory->getPointer()));

    memcpy(dst, data, size * sizeof(T));

    return memory;
}

template<typename HalPolicy,
         typename HalModel   = typename HalPolicy::Model,
         typename HalOperand = typename HalPolicy::Operand>
void AddOperand(HalModel& model, const HalOperand& op)
{
    model.operands.resize(model.operands.size() + 1);
    model.operands[model.operands.size() - 1] = op;
}

template<typename HalPolicy, typename HalModel = typename HalPolicy::Model>
void AddBoolOperand(HalModel& model, bool value, uint32_t numberOfConsumers = 1)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    V1_0::DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(uint8_t);

    HalOperand op           = {};
    op.type                 = HalOperandType::BOOL;
    op.dimensions           = hidl_vec<uint32_t>{};
    op.lifetime             = HalOperandLifeTime::CONSTANT_COPY;
    op.location             = location;
    op.numberOfConsumers    = numberOfConsumers;

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
    typename HalModel       = typename HalPolicy::Model,
    typename HalOperandType = typename HalPolicy::OperandType>
void AddInputOperand(HalModel& model,
                     const hidl_vec<uint32_t>& dimensions,
                     HalOperandType operandType = HalOperandType::TENSOR_FLOAT32,
                     double scale = 0.f,
                     int offset = 0,
                     uint32_t numberOfConsumers = 1)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    HalOperand op           = {};
    op.type                 = operandType;
    op.scale                = scale;
    op.zeroPoint            = offset;
    op.dimensions           = dimensions;
    op.lifetime             = HalOperandLifeTime::MODEL_INPUT;
    op.numberOfConsumers    = numberOfConsumers;

    AddOperand<HalPolicy>(model, op);

    model.inputIndexes.resize(model.inputIndexes.size() + 1);
    model.inputIndexes[model.inputIndexes.size() - 1] = model.operands.size() - 1;
}

template<typename HalPolicy,
    typename HalModel       = typename HalPolicy::Model,
    typename HalOperandType = typename HalPolicy::OperandType>
void AddOutputOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      HalOperandType operandType = HalOperandType::TENSOR_FLOAT32,
                      double scale = 0.f,
                      int offset = 0,
                      uint32_t numberOfConsumers = 0)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    HalOperand op           = {};
    op.type                 = operandType;
    op.scale                = scale;
    op.zeroPoint            = offset;
    op.dimensions           = dimensions;
    op.lifetime             = HalOperandLifeTime::MODEL_OUTPUT;
    op.numberOfConsumers    = numberOfConsumers;

    AddOperand<HalPolicy>(model, op);

    model.outputIndexes.resize(model.outputIndexes.size() + 1);
    model.outputIndexes[model.outputIndexes.size() - 1] = model.operands.size() - 1;
}

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_0::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         V1_0::ErrorStatus& prepareStatus,
                                                         V1_0::ErrorStatus expectedStatus = V1_0::ErrorStatus::NONE);

#if defined(ARMNN_ANDROID_NN_V1_1) || defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_1::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         V1_0::ErrorStatus& prepareStatus,
                                                         V1_0::ErrorStatus expectedStatus = V1_0::ErrorStatus::NONE);

#endif

template<typename HalModel>
android::sp<V1_0::IPreparedModel> PrepareModel(const HalModel& model,
                                               armnn_driver::ArmnnDriver& driver)
{
    V1_0::ErrorStatus prepareStatus = V1_0::ErrorStatus::NONE;
    return PrepareModelWithStatus(model, driver, prepareStatus);
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

android::sp<V1_2::IPreparedModel> PrepareModelWithStatus_1_2(const armnn_driver::hal_1_2::HalPolicy::Model& model,
                                                            armnn_driver::ArmnnDriver& driver,
                                                            V1_0::ErrorStatus& prepareStatus,
                                                            V1_0::ErrorStatus expectedStatus = V1_0::ErrorStatus::NONE);

template<typename HalModel>
android::sp<V1_2::IPreparedModel> PrepareModel_1_2(const HalModel& model,
                                                   armnn_driver::ArmnnDriver& driver)
{
    V1_0::ErrorStatus prepareStatus = V1_0::ErrorStatus::NONE;
    return PrepareModelWithStatus_1_2(model, driver, prepareStatus);
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

template<typename HalPolicy>
void AddOperand(armnn_driver::hal_1_3::HalPolicy::Model& model,
                const armnn_driver::hal_1_3::HalPolicy::Operand& op)
{
    model.main.operands.resize(model.main.operands.size() + 1);
    model.main.operands[model.main.operands.size() - 1] = op;
}

template<typename HalPolicy>
void AddInputOperand(armnn_driver::hal_1_3::HalPolicy::Model& model,
                     const hidl_vec<uint32_t>& dimensions,
                     armnn_driver::hal_1_3::HalPolicy::OperandType operandType =
                     armnn_driver::hal_1_3::HalPolicy::OperandType::TENSOR_FLOAT32,
                     double scale = 0.f,
                     int offset = 0,
                     uint32_t numberOfConsumers = 1)
{
    using HalOperand         = typename armnn_driver::hal_1_3::HalPolicy::Operand;
    using HalOperandLifeTime = typename armnn_driver::hal_1_3::HalPolicy::OperandLifeTime;

    HalOperand op           = {};
    op.type                 = operandType;
    op.scale                = scale;
    op.zeroPoint            = offset;
    op.dimensions           = dimensions;
    op.lifetime             = HalOperandLifeTime::SUBGRAPH_INPUT;
    op.numberOfConsumers    = numberOfConsumers;

    AddOperand<HalPolicy>(model, op);

    model.main.inputIndexes.resize(model.main.inputIndexes.size() + 1);
    model.main.inputIndexes[model.main.inputIndexes.size() - 1] = model.main.operands.size() - 1;
}

template<typename HalPolicy>
void AddOutputOperand(armnn_driver::hal_1_3::HalPolicy::Model& model,
                      const hidl_vec<uint32_t>& dimensions,
                      armnn_driver::hal_1_3::HalPolicy::OperandType operandType =
                      armnn_driver::hal_1_3::HalPolicy::OperandType::TENSOR_FLOAT32,
                      double scale = 0.f,
                      int offset = 0,
                      uint32_t numberOfConsumers = 0)
{
    using HalOperand         = typename armnn_driver::hal_1_3::HalPolicy::Operand;
    using HalOperandLifeTime = typename armnn_driver::hal_1_3::HalPolicy::OperandLifeTime;

    HalOperand op           = {};
    op.type                 = operandType;
    op.scale                = scale;
    op.zeroPoint            = offset;
    op.dimensions           = dimensions;
    op.lifetime             = HalOperandLifeTime::SUBGRAPH_OUTPUT;
    op.numberOfConsumers    = numberOfConsumers;

    AddOperand<HalPolicy>(model, op);

    model.main.outputIndexes.resize(model.main.outputIndexes.size() + 1);
    model.main.outputIndexes[model.main.outputIndexes.size() - 1] = model.main.operands.size() - 1;
}

android::sp<V1_3::IPreparedModel> PrepareModelWithStatus_1_3(const armnn_driver::hal_1_3::HalPolicy::Model& model,
                                                            armnn_driver::ArmnnDriver& driver,
                                                            V1_3::ErrorStatus& prepareStatus,
                                                            V1_3::Priority priority = V1_3::Priority::LOW);

template<typename HalModel>
android::sp<V1_3::IPreparedModel> PrepareModel_1_3(const HalModel& model,
                                                   armnn_driver::ArmnnDriver& driver)
{
    V1_3::ErrorStatus prepareStatus = V1_3::ErrorStatus::NONE;
    return PrepareModelWithStatus_1_3(model, driver, prepareStatus);
}

#endif

template<typename HalPolicy,
    typename T,
    typename HalModel           = typename HalPolicy::Model,
    typename HalOperandType     = typename HalPolicy::OperandType,
    typename HalOperandLifeTime = typename HalPolicy::OperandLifeTime>
void AddTensorOperand(HalModel& model,
                      const hidl_vec<uint32_t>& dimensions,
                      const T* values,
                      HalOperandType operandType = HalOperandType::TENSOR_FLOAT32,
                      HalOperandLifeTime operandLifeTime = V1_0::OperandLifeTime::CONSTANT_COPY,
                      double scale = 0.f,
                      int offset = 0,
                      uint32_t numberOfConsumers = 1)
{
    using HalOperand = typename HalPolicy::Operand;

    uint32_t totalElements = 1;
    for (uint32_t dim : dimensions)
    {
        totalElements *= dim;
    }

    V1_0::DataLocation location = {};
    location.length = totalElements * sizeof(T);

    if(operandLifeTime == HalOperandLifeTime::CONSTANT_COPY)
    {
        location.offset = model.operandValues.size();
    }

    HalOperand op           = {};
    op.type                 = operandType;
    op.dimensions           = dimensions;
    op.scale                = scale;
    op.zeroPoint            = offset;
    op.lifetime             = HalOperandLifeTime::CONSTANT_COPY;
    op.location             = location;
    op.numberOfConsumers    = numberOfConsumers;

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
                      HalOperandLifeTime operandLifeTime = V1_0::OperandLifeTime::CONSTANT_COPY,
                      double scale = 0.f,
                      int offset = 0,
                      uint32_t numberOfConsumers = 1)
{
    AddTensorOperand<HalPolicy, T>(model,
                                   dimensions,
                                   values.data(),
                                   operandType,
                                   operandLifeTime,
                                   scale,
                                   offset,
                                   numberOfConsumers);
}

template<typename HalPolicy, typename HalModel = typename HalPolicy::Model>
void AddIntOperand(HalModel& model, int32_t value, uint32_t numberOfConsumers = 1)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    V1_0::DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(int32_t);

    HalOperand op           = {};
    op.type                 = HalOperandType::INT32;
    op.dimensions           = hidl_vec<uint32_t>{};
    op.lifetime             = HalOperandLifeTime::CONSTANT_COPY;
    op.location             = location;
    op.numberOfConsumers    = numberOfConsumers;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<int32_t*>(&model.operandValues[location.offset]) = value;

    AddOperand<HalPolicy>(model, op);
}

template<typename HalPolicy, typename HalModel = typename HalPolicy::Model>
void AddFloatOperand(HalModel& model,
                     float value,
                     uint32_t numberOfConsumers = 1)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    V1_0::DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(float);

    HalOperand op           = {};
    op.type                 = HalOperandType::FLOAT32;
    op.dimensions           = hidl_vec<uint32_t>{};
    op.lifetime             = HalOperandLifeTime::CONSTANT_COPY;
    op.location             = location;
    op.numberOfConsumers    = numberOfConsumers;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<float*>(&model.operandValues[location.offset]) = value;

    AddOperand<HalPolicy>(model, op);
}

V1_0::ErrorStatus Execute(android::sp<V1_0::IPreparedModel> preparedModel,
                          const V1_0::Request& request,
                          V1_0::ErrorStatus expectedStatus = V1_0::ErrorStatus::NONE);

android::sp<ExecutionCallback> ExecuteNoWait(android::sp<V1_0::IPreparedModel> preparedModel,
                                             const V1_0::Request& request);

} // namespace driverTestHelpers
