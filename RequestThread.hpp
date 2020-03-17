//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"

#include <CpuExecutor.h>
#include <armnn/ArmNN.hpp>

namespace armnn_driver
{
using TimePoint = std::chrono::steady_clock::time_point;
static const TimePoint g_Min = std::chrono::steady_clock::time_point::min();

template<template <typename HalVersion> class PreparedModel, typename HalVersion, typename CallbackContext>
class RequestThread
{
public:
    /// Constructor creates the thread
    RequestThread();

    /// Destructor terminates the thread
    ~RequestThread();

    /// Add a message to the thread queue.
    /// @param[in] model pointer to the prepared model handling the request
    /// @param[in] memPools pointer to the memory pools vector for the tensors
    /// @param[in] inputTensors pointer to the input tensors for the request
    /// @param[in] outputTensors pointer to the output tensors for the request
    /// @param[in] callback the android notification callback
    void PostMsg(PreparedModel<HalVersion>* model,
                 std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& memPools,
                 std::shared_ptr<armnn::InputTensors>& inputTensors,
                 std::shared_ptr<armnn::OutputTensors>& outputTensors,
                 CallbackContext callbackContext);

private:
    RequestThread(const RequestThread&) = delete;
    RequestThread& operator=(const RequestThread&) = delete;

    /// storage for a prepared model and args for the asyncExecute call
    struct AsyncExecuteData
    {
        AsyncExecuteData(PreparedModel<HalVersion>* model,
                         std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>>& memPools,
                         std::shared_ptr<armnn::InputTensors>& inputTensors,
                         std::shared_ptr<armnn::OutputTensors>& outputTensors,
                         CallbackContext callbackContext)
            : m_Model(model)
            , m_MemPools(memPools)
            , m_InputTensors(inputTensors)
            , m_OutputTensors(outputTensors)
            , m_CallbackContext(callbackContext)
        {
        }

        PreparedModel<HalVersion>* m_Model;
        std::shared_ptr<std::vector<::android::nn::RunTimePoolInfo>> m_MemPools;
        std::shared_ptr<armnn::InputTensors> m_InputTensors;
        std::shared_ptr<armnn::OutputTensors> m_OutputTensors;
        CallbackContext m_CallbackContext;
    };
    enum class ThreadMsgType
    {
        EXIT,                   // exit the thread
        REQUEST                 // user request to process
    };

    /// storage for the thread message type and data
    struct ThreadMsg
    {
        ThreadMsg(ThreadMsgType msgType,
                  std::shared_ptr<AsyncExecuteData>& msgData)
            : type(msgType)
            , data(msgData)
        {
        }

        ThreadMsgType type;
        std::shared_ptr<AsyncExecuteData> data;
    };

    /// Add a prepared thread message to the thread queue.
    /// @param[in] threadMsg the message to add to the queue
    void PostMsg(std::shared_ptr<ThreadMsg>& pThreadMsg);

    /// Entry point for the request thread
    void Process();

    std::unique_ptr<std::thread> m_Thread;
    std::queue<std::shared_ptr<ThreadMsg>> m_Queue;
    std::mutex m_Mutex;
    std::condition_variable m_Cv;
};

} // namespace armnn_driver
