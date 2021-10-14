//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "ArmnnPreparedModel_1_3.hpp"
#include "RequestThread_1_3.hpp"

#include <log/log.h>

using namespace android;

namespace armnn_driver
{

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
RequestThread_1_3<PreparedModel, HalVersion, CallbackContext>::RequestThread_1_3()
{
    ALOGV("RequestThread_1_3::RequestThread_1_3()");
    m_Thread = std::make_unique<std::thread>(&RequestThread_1_3::Process, this);
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
RequestThread_1_3<PreparedModel, HalVersion, CallbackContext>::~RequestThread_1_3()
{
    ALOGV("RequestThread_1_3::~RequestThread_1_3()");

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
void RequestThread_1_3<PreparedModel, HalVersion, CallbackContext>::PostMsg(PreparedModel<HalVersion>* model,
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& memPools,
        std::shared_ptr<armnn::InputTensors>& inputTensors,
        std::shared_ptr<armnn::OutputTensors>& outputTensors,
        CallbackContext callbackContext)
{
    ALOGV("RequestThread_1_3::PostMsg(...)");
    auto data = std::make_shared<AsyncExecuteData>(model,
                                                   memPools,
                                                   inputTensors,
                                                   outputTensors,
                                                   callbackContext);
    auto pMsg = std::make_shared<ThreadMsg>(ThreadMsgType::REQUEST, data);
    PostMsg(pMsg, model->GetModelPriority());
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
void RequestThread_1_3<PreparedModel, HalVersion, CallbackContext>::PostMsg(std::shared_ptr<ThreadMsg>& pMsg,
                                                                        V1_3::Priority priority)
{
    ALOGV("RequestThread_1_3::PostMsg(pMsg)");
    // Add a message to the queue and notify the request thread
    std::unique_lock<std::mutex> lock(m_Mutex);
    switch (priority) {
        case V1_3::Priority::HIGH:
            m_HighPriorityQueue.push(pMsg);
            break;
        case V1_3::Priority::LOW:
            m_LowPriorityQueue.push(pMsg);
            break;
        case V1_3::Priority::MEDIUM:
        default:
            m_MediumPriorityQueue.push(pMsg);
    }
    m_Cv.notify_one();
}

template <template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
void RequestThread_1_3<PreparedModel, HalVersion, CallbackContext>::Process()
{
    ALOGV("RequestThread_1_3::Process()");
    int retireRate = RETIRE_RATE;
    int highPriorityCount = 0;
    int mediumPriorityCount = 0;
    while (true)
    {
        std::shared_ptr<ThreadMsg> pMsg(nullptr);
        {
            // Wait for a message to be added to the queue
            // This is in a separate scope to minimise the lifetime of the lock
            std::unique_lock<std::mutex> lock(m_Mutex);
            while (m_HighPriorityQueue.empty() && m_MediumPriorityQueue.empty() && m_LowPriorityQueue.empty())
            {
                m_Cv.wait(lock);
            }
            // Get the message to process from the front of each queue based on priority from high to low
            // Get high priority first if it does not exceed the retire rate
            if (!m_HighPriorityQueue.empty() && highPriorityCount < retireRate)
            {
                pMsg = m_HighPriorityQueue.front();
                m_HighPriorityQueue.pop();
                highPriorityCount += 1;
            }
            // If high priority queue is empty or the count exceeds the retire rate, get medium priority message
            else if (!m_MediumPriorityQueue.empty() && mediumPriorityCount < retireRate)
            {
                pMsg = m_MediumPriorityQueue.front();
                m_MediumPriorityQueue.pop();
                mediumPriorityCount += 1;
                // Reset high priority count
                highPriorityCount = 0;
            }
            // If medium priority queue is empty or the count exceeds the retire rate, get low priority message
            else if (!m_LowPriorityQueue.empty())
            {
                pMsg = m_LowPriorityQueue.front();
                m_LowPriorityQueue.pop();
                // Reset high and medium priority count
                highPriorityCount = 0;
                mediumPriorityCount = 0;
            }
            else
            {
                // Reset high and medium priority count
                highPriorityCount = 0;
                mediumPriorityCount = 0;
                continue;
            }
        }

        switch (pMsg->type)
        {
            case ThreadMsgType::REQUEST:
            {
                ALOGV("RequestThread_1_3::Process() - request");
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
                ALOGV("RequestThread_1_3::Process() - exit");
                // delete all remaining messages (there should not be any)
                std::unique_lock<std::mutex> lock(m_Mutex);
                while (!m_HighPriorityQueue.empty())
                {
                    m_HighPriorityQueue.pop();
                }
                while (!m_MediumPriorityQueue.empty())
                {
                    m_MediumPriorityQueue.pop();
                }
                while (!m_LowPriorityQueue.empty())
                {
                    m_LowPriorityQueue.pop();
                }
                return;
            }

            default:
                // this should be unreachable
                throw armnn::RuntimeException("ArmNN: RequestThread_1_3: invalid message type");
        }
    }
}

///
/// Class template specializations
///

template class RequestThread_1_3<ArmnnPreparedModel_1_3, hal_1_3::HalPolicy, CallbackContext_1_3>;

} // namespace armnn_driver
