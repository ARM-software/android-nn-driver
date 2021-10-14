//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "RequestThread.hpp"
#include "ArmnnPreparedModel.hpp"

#ifdef ARMNN_ANDROID_NN_V1_2
#include "ArmnnPreparedModel_1_2.hpp"
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
#include "ArmnnPreparedModel_1_2.hpp"
#include "ArmnnPreparedModel_1_3.hpp"
#endif

#include <log/log.h>

using namespace android;

namespace armnn_driver
{

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
RequestThread<PreparedModel, HalVersion, CallbackContext>::RequestThread()
{
    ALOGV("RequestThread::RequestThread()");
    m_Thread = std::make_unique<std::thread>(&RequestThread::Process, this);
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
RequestThread<PreparedModel, HalVersion, CallbackContext>::~RequestThread()
{
    ALOGV("RequestThread::~RequestThread()");

    try
    {
        // Coverity fix: The following code may throw an exception of type std::length_error.

        // This code is meant to to terminate the inner thread gracefully by posting an EXIT message
        // to the thread's message queue. However, according to Coverity, this code could throw an exception and fail.
        // Since only one static instance of RequestThread is used in the driver (in ArmnnPreparedModel),
        // this destructor is called only when the application has been closed, which means that
        // the inner thread will be terminated anyway, although abruptly, in the event that the destructor code throws.
        // Wrapping the destructor's code with a try-catch block simply fixes the Coverity bug.

        // Post an EXIT message to the thread
        std::shared_ptr<AsyncExecuteData> nulldata(nullptr);
        auto pMsg = std::make_shared<ThreadMsg>(ThreadMsgType::EXIT, nulldata);
        PostMsg(pMsg);
        // Wait for the thread to terminate, it is deleted automatically
        m_Thread->join();
    }
    catch (const std::exception&) { } // Swallow any exception.
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
void RequestThread<PreparedModel, HalVersion, CallbackContext>::PostMsg(PreparedModel<HalVersion>* model,
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& memPools,
        std::shared_ptr<armnn::InputTensors>& inputTensors,
        std::shared_ptr<armnn::OutputTensors>& outputTensors,
        CallbackContext callbackContext)
{
    ALOGV("RequestThread::PostMsg(...)");
    auto data = std::make_shared<AsyncExecuteData>(model,
                                                   memPools,
                                                   inputTensors,
                                                   outputTensors,
                                                   callbackContext);
    auto pMsg = std::make_shared<ThreadMsg>(ThreadMsgType::REQUEST, data);
    PostMsg(pMsg);
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
void RequestThread<PreparedModel, HalVersion, CallbackContext>::PostMsg(std::shared_ptr<ThreadMsg>& pMsg)
{
    ALOGV("RequestThread::PostMsg(pMsg)");
    // Add a message to the queue and notify the request thread
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_Queue.push(pMsg);
    m_Cv.notify_one();
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
void RequestThread<PreparedModel, HalVersion, CallbackContext>::Process()
{
    ALOGV("RequestThread::Process()");
    while (true)
    {
        std::shared_ptr<ThreadMsg> pMsg(nullptr);
        {
            // Wait for a message to be added to the queue
            // This is in a separate scope to minimise the lifetime of the lock
            std::unique_lock<std::mutex> lock(m_Mutex);
            while (m_Queue.empty())
            {
                m_Cv.wait(lock);
            }
            // get the message to process from the front of the queue
            pMsg = m_Queue.front();
            m_Queue.pop();
        }

        switch (pMsg->type)
        {
            case ThreadMsgType::REQUEST:
            {
                ALOGV("RequestThread::Process() - request");
                // invoke the asynchronous execution method
                PreparedModel<HalVersion>* model = pMsg->data->m_Model;
                model->ExecuteGraph(pMsg->data->m_MemPools,
                                    *(pMsg->data->m_InputTensors),
                                    *(pMsg->data->m_OutputTensors),
                                    pMsg->data->m_CallbackContext);
                break;
            }

            case ThreadMsgType::EXIT:
            {
                ALOGV("RequestThread::Process() - exit");
                // delete all remaining messages (there should not be any)
                std::unique_lock<std::mutex> lock(m_Mutex);
                while (!m_Queue.empty())
                {
                    m_Queue.pop();
                }
                return;
            }

            default:
                // this should be unreachable
                throw armnn::RuntimeException("ArmNN: RequestThread: invalid message type");
        }
    }
}

///
/// Class template specializations
///

template class RequestThread<ArmnnPreparedModel, hal_1_0::HalPolicy, CallbackContext_1_0>;

#ifdef ARMNN_ANDROID_NN_V1_1
template class RequestThread<armnn_driver::ArmnnPreparedModel, hal_1_1::HalPolicy, CallbackContext_1_0>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_2
template class RequestThread<ArmnnPreparedModel, hal_1_1::HalPolicy, CallbackContext_1_0>;
template class RequestThread<ArmnnPreparedModel, hal_1_2::HalPolicy, CallbackContext_1_0>;
template class RequestThread<ArmnnPreparedModel_1_2, hal_1_2::HalPolicy, CallbackContext_1_2>;
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
template class RequestThread<ArmnnPreparedModel, hal_1_1::HalPolicy, CallbackContext_1_0>;
template class RequestThread<ArmnnPreparedModel, hal_1_2::HalPolicy, CallbackContext_1_0>;
template class RequestThread<ArmnnPreparedModel, hal_1_3::HalPolicy, CallbackContext_1_0>;
template class RequestThread<ArmnnPreparedModel_1_2, hal_1_2::HalPolicy, CallbackContext_1_2>;
#endif

} // namespace armnn_driver
