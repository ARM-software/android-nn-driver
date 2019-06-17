//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DriverTestHelpers.hpp"
#include <log/log.h>
#include <boost/test/unit_test.hpp>

namespace android
{
namespace hardware
{
namespace neuralnetworks
{
namespace V1_0
{

std::ostream& operator<<(std::ostream& os, ErrorStatus stat)
{
   return os << static_cast<int>(stat);
}

} // namespace android::hardware::neuralnetworks::V1_0
} // namespace android::hardware::neuralnetworks
} // namespace android::hardware
} // namespace android

namespace driverTestHelpers
{

using namespace android::hardware;
using namespace armnn_driver;

Return<void> ExecutionCallback::notify(ErrorStatus status)
{
    (void)status;
    ALOGI("ExecutionCallback::notify invoked");
    std::lock_guard<std::mutex> executionLock(mMutex);
    mNotified = true;
    mCondition.notify_one();
    return Void();
}

Return<void> ExecutionCallback::wait()
{
    ALOGI("ExecutionCallback::wait invoked");
    std::unique_lock<std::mutex> executionLock(mMutex);
    while (!mNotified)
    {
        mCondition.wait(executionLock);
    }
    mNotified = false;
    return Void();
}

Return<void> PreparedModelCallback::notify(ErrorStatus status,
                                           const android::sp<V1_0::IPreparedModel>& preparedModel)
{
    m_ErrorStatus = status;
    m_PreparedModel = preparedModel;
    return Void();
}

// lifted from common/Utils.cpp
hidl_memory allocateSharedMemory(int64_t size)
{
    hidl_memory memory;

    const std::string& type      = "ashmem";
    android::sp<IAllocator>     allocator = IAllocator::getService(type);
    allocator->allocate(size, [&](bool success, const hidl_memory& mem) {
        if (!success)
        {
            ALOGE("unable to allocate %li bytes of %s", size, type.c_str());
        }
        else
        {
            memory = mem;
        }
    });

    return memory;
}

android::sp<IMemory> AddPoolAndGetData(uint32_t size, Request& request)
{
    hidl_memory pool;

    android::sp<IAllocator> allocator = IAllocator::getService("ashmem");
    allocator->allocate(sizeof(float) * size, [&](bool success, const hidl_memory& mem) {
        BOOST_TEST(success);
        pool = mem;
    });

    request.pools.resize(request.pools.size() + 1);
    request.pools[request.pools.size() - 1] = pool;

    android::sp<IMemory> mapped = mapMemory(pool);
    mapped->update();
    return mapped;
}

void AddPoolAndSetData(uint32_t size, Request& request, const float* data)
{
    android::sp<IMemory> memory = AddPoolAndGetData(size, request);

    float* dst = static_cast<float*>(static_cast<void*>(memory->getPointer()));

    memcpy(dst, data, size * sizeof(float));
}

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_0::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         ErrorStatus& prepareStatus,
                                                         ErrorStatus expectedStatus)
{
    android::sp<PreparedModelCallback> cb(new PreparedModelCallback());
    driver.prepareModel(model, cb);

    prepareStatus = cb->GetErrorStatus();
    BOOST_TEST(prepareStatus == expectedStatus);
    if (expectedStatus == ErrorStatus::NONE)
    {
        BOOST_TEST((cb->GetPreparedModel() != nullptr));
    }
    return cb->GetPreparedModel();
}

#if defined(ARMNN_ANDROID_NN_V1_1) || defined(ARMNN_ANDROID_NN_V1_2)

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_1::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         ErrorStatus& prepareStatus,
                                                         ErrorStatus expectedStatus)
{
    android::sp<PreparedModelCallback> cb(new PreparedModelCallback());
    driver.prepareModel_1_1(model, V1_1::ExecutionPreference::LOW_POWER, cb);

    prepareStatus = cb->GetErrorStatus();
    BOOST_TEST(prepareStatus == expectedStatus);
    if (expectedStatus == ErrorStatus::NONE)
    {
        BOOST_TEST((cb->GetPreparedModel() != nullptr));
    }
    return cb->GetPreparedModel();
}

#endif

ErrorStatus Execute(android::sp<V1_0::IPreparedModel> preparedModel,
                    const Request& request,
                    ErrorStatus expectedStatus)
{
    BOOST_TEST(preparedModel.get() != nullptr);
    android::sp<ExecutionCallback> cb(new ExecutionCallback());
    ErrorStatus execStatus = preparedModel->execute(request, cb);
    BOOST_TEST(execStatus == expectedStatus);
    ALOGI("Execute: waiting for callback to be invoked");
    cb->wait();
    return execStatus;
}

android::sp<ExecutionCallback> ExecuteNoWait(android::sp<V1_0::IPreparedModel> preparedModel, const Request& request)
{
    android::sp<ExecutionCallback> cb(new ExecutionCallback());
    BOOST_TEST(preparedModel->execute(request, cb) == ErrorStatus::NONE);
    ALOGI("ExecuteNoWait: returning callback object");
    return cb;
}

template<>
OperandType TypeToOperandType<float>()
{
    return OperandType::TENSOR_FLOAT32;
}

template<>
OperandType TypeToOperandType<int32_t>()
{
    return OperandType::TENSOR_INT32;
}

} // namespace driverTestHelpers
