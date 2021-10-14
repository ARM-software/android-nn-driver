//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"
#include <log/log.h>

namespace android
{
namespace hardware
{
namespace neuralnetworks
{
namespace V1_0
{

std::ostream& operator<<(std::ostream& os, V1_0::ErrorStatus stat)
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

Return<void> ExecutionCallback::notify(V1_0::ErrorStatus status)
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

Return<void> PreparedModelCallback::notify(V1_0::ErrorStatus status,
                                           const android::sp<V1_0::IPreparedModel>& preparedModel)
{
    m_ErrorStatus = status;
    m_PreparedModel = preparedModel;
    return Void();
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

Return<void> PreparedModelCallback_1_2::notify(V1_0::ErrorStatus status,
                                               const android::sp<V1_0::IPreparedModel>& preparedModel)
{
    m_ErrorStatus = status;
    m_PreparedModel = preparedModel;
    return Void();
}

Return<void> PreparedModelCallback_1_2::notify_1_2(V1_0::ErrorStatus status,
                                                   const android::sp<V1_2::IPreparedModel>& preparedModel)
{
    m_ErrorStatus = status;
    m_PreparedModel_1_2 = preparedModel;
    return Void();
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

Return<void> PreparedModelCallback_1_3::notify(V1_0::ErrorStatus status,
                                               const android::sp<V1_0::IPreparedModel>& preparedModel)
{
    m_1_0_ErrorStatus = status;
    m_PreparedModel = preparedModel;
    return Void();
}

Return<void> PreparedModelCallback_1_3::notify_1_2(V1_0::ErrorStatus status,
                                                   const android::sp<V1_2::IPreparedModel>& preparedModel)
{
    m_1_0_ErrorStatus = status;
    m_PreparedModel_1_2 = preparedModel;
    return Void();
}

Return<void> PreparedModelCallback_1_3::notify_1_3(V1_3::ErrorStatus status,
                                                   const android::sp<V1_3::IPreparedModel>& preparedModel)
{
    m_1_3_ErrorStatus = status;
    m_PreparedModel_1_3 = preparedModel;
    return Void();
}

#endif

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

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_0::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         V1_0::ErrorStatus& prepareStatus,
                                                         V1_0::ErrorStatus expectedStatus)
{
    android::sp<PreparedModelCallback> cb(new PreparedModelCallback());
    driver.prepareModel(model, cb);

    prepareStatus = cb->GetErrorStatus();
    DOCTEST_CHECK((int)prepareStatus == (int)expectedStatus);
    if (expectedStatus == V1_0::ErrorStatus::NONE)
    {
        DOCTEST_CHECK((cb->GetPreparedModel() != nullptr));
    }
    return cb->GetPreparedModel();
}

#if defined(ARMNN_ANDROID_NN_V1_1) || defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

android::sp<V1_0::IPreparedModel> PrepareModelWithStatus(const V1_1::Model& model,
                                                         armnn_driver::ArmnnDriver& driver,
                                                         V1_0::ErrorStatus& prepareStatus,
                                                         V1_0::ErrorStatus expectedStatus)
{
    android::sp<PreparedModelCallback> cb(new PreparedModelCallback());
    driver.prepareModel_1_1(model, V1_1::ExecutionPreference::LOW_POWER, cb);

    prepareStatus = cb->GetErrorStatus();
    DOCTEST_CHECK((int)prepareStatus == (int)expectedStatus);
    if (expectedStatus == V1_0::ErrorStatus::NONE)
    {
        DOCTEST_CHECK((cb->GetPreparedModel() != nullptr));
    }
    return cb->GetPreparedModel();
}

#endif

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

android::sp<V1_2::IPreparedModel> PrepareModelWithStatus_1_2(const armnn_driver::hal_1_2::HalPolicy::Model& model,
                                                             armnn_driver::ArmnnDriver& driver,
                                                             V1_0::ErrorStatus& prepareStatus,
                                                             V1_0::ErrorStatus expectedStatus)
{
    android::sp<PreparedModelCallback_1_2> cb(new PreparedModelCallback_1_2());

    android::hardware::hidl_vec<android::hardware::hidl_handle> emptyHandle1;
    android::hardware::hidl_vec<android::hardware::hidl_handle> emptyHandle2;
    armnn_driver::ArmnnDriver::HidlToken emptyToken;

    driver.prepareModel_1_2(model, V1_1::ExecutionPreference::LOW_POWER, emptyHandle1, emptyHandle2, emptyToken, cb);

    prepareStatus = cb->GetErrorStatus();
    DOCTEST_CHECK((int)prepareStatus == (int)expectedStatus);
    if (expectedStatus == V1_0::ErrorStatus::NONE)
    {
        DOCTEST_CHECK((cb->GetPreparedModel_1_2() != nullptr));
    }
    return cb->GetPreparedModel_1_2();
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

android::sp<V1_3::IPreparedModel> PrepareModelWithStatus_1_3(const armnn_driver::hal_1_3::HalPolicy::Model& model,
                                                             armnn_driver::ArmnnDriver& driver,
                                                             V1_3::ErrorStatus& prepareStatus,
                                                             V1_3::Priority priority)
{
    android::sp<PreparedModelCallback_1_3> cb(new PreparedModelCallback_1_3());

    android::hardware::hidl_vec<android::hardware::hidl_handle> emptyHandle1;
    android::hardware::hidl_vec<android::hardware::hidl_handle> emptyHandle2;
    armnn_driver::ArmnnDriver::HidlToken emptyToken;

    driver.prepareModel_1_3(model,
                            V1_1::ExecutionPreference::LOW_POWER,
                            priority,
                            {},
                            emptyHandle1,
                            emptyHandle2,
                            emptyToken,
                            cb);

    prepareStatus = cb->Get_1_3_ErrorStatus();
    if (prepareStatus == V1_3::ErrorStatus::NONE)
    {
        DOCTEST_CHECK((cb->GetPreparedModel_1_3() != nullptr));
    }
    return cb->GetPreparedModel_1_3();
}

#endif

V1_0::ErrorStatus Execute(android::sp<V1_0::IPreparedModel> preparedModel,
                          const V1_0::Request& request,
                          V1_0::ErrorStatus expectedStatus)
{
    DOCTEST_CHECK(preparedModel.get() != nullptr);
    android::sp<ExecutionCallback> cb(new ExecutionCallback());
    V1_0::ErrorStatus execStatus = preparedModel->execute(request, cb);
    DOCTEST_CHECK((int)execStatus == (int)expectedStatus);
    ALOGI("Execute: waiting for callback to be invoked");
    cb->wait();
    return execStatus;
}

android::sp<ExecutionCallback> ExecuteNoWait(android::sp<V1_0::IPreparedModel> preparedModel,
                                             const V1_0::Request& request)
{
    DOCTEST_CHECK(preparedModel.get() != nullptr);
    android::sp<ExecutionCallback> cb(new ExecutionCallback());
    V1_0::ErrorStatus execStatus = preparedModel->execute(request, cb);
    DOCTEST_CHECK((int)execStatus == (int)V1_0::ErrorStatus::NONE);
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
