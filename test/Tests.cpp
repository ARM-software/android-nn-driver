//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#define LOG_TAG "ArmnnDriverTests"
#define BOOST_TEST_MODULE armnn_driver_tests
#include <boost/test/unit_test.hpp>
#include <log/log.h>

#include "../ArmnnDriver.hpp"
#include "../SystemPropertiesUtils.hpp"

#include "OperationsUtils.h"

#include <condition_variable>

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

}
}
}
}

BOOST_AUTO_TEST_SUITE(DriverTests)

using namespace armnn_driver;
using namespace android::nn;
using namespace android;

BOOST_AUTO_TEST_CASE(Init)
{
    // Making the driver object on the stack causes a weird libc error, so make it on the heap instead
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    DeviceStatus status = driver->getStatus();
    // Note double-parentheses to avoid compile error from Boost trying to printf the DeviceStatus
    BOOST_TEST((status == DeviceStatus::AVAILABLE));
}

BOOST_AUTO_TEST_CASE(TestCapabilities)
{
    // Making the driver object on the stack causes a weird libc error, so make it on the heap instead
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    Capabilities cap;

    ArmnnDriver::getCapabilities_cb cb = [&](ErrorStatus status, const Capabilities& capabilities)
    {
        error = status;
        cap = capabilities;
    };

    driver->getCapabilities(cb);

    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(cap.float32Performance.execTime > 0.f);
    BOOST_TEST(cap.float32Performance.powerUsage > 0.f);
    BOOST_TEST(cap.quantized8Performance.execTime > 0.f);
    BOOST_TEST(cap.quantized8Performance.powerUsage > 0.f);
}

BOOST_AUTO_TEST_CASE(SystemProperties)
{
    // Test default value
    {
        auto p = __system_property_find("thisDoesNotExist");
        BOOST_TEST((p == nullptr));

        int defaultValue = ParseSystemProperty("thisDoesNotExist", -4);
        BOOST_TEST((defaultValue == -4));
    }

    //  Test default value from bad data type
    {
        __system_property_set("thisIsNotFloat", "notfloat");
        float defaultValue = ParseSystemProperty("thisIsNotFloat", 0.1f);
        BOOST_TEST((defaultValue == 0.1f));
    }

    // Test fetching bool values
    {
        __system_property_set("myTestBool", "1");
        bool b = ParseSystemProperty("myTestBool", false);
        BOOST_TEST((b == true));
    }
    {
        __system_property_set("myTestBool", "0");
        bool b = ParseSystemProperty("myTestBool", true);
        BOOST_TEST((b == false));
    }

    // Test fetching int
    {
        __system_property_set("myTestInt", "567");
        int i = ParseSystemProperty("myTestInt", 890);
        BOOST_TEST((i==567));
    }

    // Test fetching float
    {
        __system_property_set("myTestFloat", "1.2f");
        float f = ParseSystemProperty("myTestFloat", 3.4f);
        BOOST_TEST((f==1.2f));
    }
}

// The following are helpers for writing unit tests for the driver
namespace
{

struct ExecutionCallback : public IExecutionCallback
{
    ExecutionCallback()
        : mNotified(false)
    {
    }

    Return<void> notify(ErrorStatus status) override
    {
        (void)status;
        ALOGI("ExecutionCallback::notify invoked");
        std::lock_guard<std::mutex> executionLock(mMutex);
        mNotified = true;
        mCondition.notify_one();
        return Void();
    }

    /// wait until the callback has notified us that it is done
    Return<void> wait()
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
    {
    }

    ~PreparedModelCallback() override
    {
    }

    Return<void> notify(ErrorStatus status, const sp<IPreparedModel>& preparedModel) override
    {
        m_ErrorStatus = status;
        m_PreparedModel = preparedModel;
        return Void();
    }

    ErrorStatus GetErrorStatus()
    {
        return m_ErrorStatus;
    }

    sp<IPreparedModel> GetPreparedModel()
    {
        return m_PreparedModel;
    }


private:
    ErrorStatus        m_ErrorStatus;
    sp<IPreparedModel> m_PreparedModel;
};



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

void AddPoolAndSetData(uint32_t size, Request& request, float* data)
{
    android::sp<IMemory> memory = AddPoolAndGetData(size, request);

    float* dst = static_cast<float*>(static_cast<void*>(memory->getPointer()));

    memcpy(dst, data, size * sizeof(float));
}

void AddOperand(Model& model, const Operand& op)
{
    model.operands.resize(model.operands.size() + 1);
    model.operands[model.operands.size() - 1] = op;
}

void AddIntOperand(Model& model, int32_t value)
{
    DataLocation location = {};
    location.offset = model.operandValues.size();
    location.length = sizeof(int32_t);

    Operand op    = {};
    op.type = OperandType::INT32;
    op.dimensions = hidl_vec<uint32_t>{};
    op.lifetime   = OperandLifeTime::CONSTANT_COPY;
    op.location   = location;

    model.operandValues.resize(model.operandValues.size() + location.length);
    *reinterpret_cast<int32_t*>(&model.operandValues[location.offset]) = value;

    AddOperand(model, op);
}

template<typename T>
OperandType TypeToOperandType();

template<>
OperandType TypeToOperandType<float>()
{
    return OperandType::TENSOR_FLOAT32;
};

template<>
OperandType TypeToOperandType<int32_t>()
{
    return OperandType::TENSOR_INT32;
};



template<typename T>
void AddTensorOperand(Model& model, hidl_vec<uint32_t> dimensions, T* values)
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
    op.type       = TypeToOperandType<T>();
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

void AddInputOperand(Model& model, hidl_vec<uint32_t> dimensions)
{
    Operand op    = {};
    op.type       = OperandType::TENSOR_FLOAT32;
    op.dimensions = dimensions;
    op.lifetime   = OperandLifeTime::MODEL_INPUT;

    AddOperand(model, op);

    model.inputIndexes.resize(model.inputIndexes.size() + 1);
    model.inputIndexes[model.inputIndexes.size() - 1] = model.operands.size() - 1;
}

void AddOutputOperand(Model& model, hidl_vec<uint32_t> dimensions)
{
    Operand op = {};
    op.type       = OperandType::TENSOR_FLOAT32;
    op.dimensions = dimensions;
    op.lifetime   = OperandLifeTime::MODEL_OUTPUT;

    AddOperand(model, op);

    model.outputIndexes.resize(model.outputIndexes.size() + 1);
    model.outputIndexes[model.outputIndexes.size() - 1] = model.operands.size() - 1;
}

android::sp<IPreparedModel> PrepareModel(const Model& model, ArmnnDriver& driver)
{

    sp<PreparedModelCallback> cb(new PreparedModelCallback());
    driver.prepareModel(model, cb);

    BOOST_TEST((cb->GetErrorStatus() == ErrorStatus::NONE));
    BOOST_TEST((cb->GetPreparedModel() != nullptr));

    return cb->GetPreparedModel();
}

void Execute(android::sp<IPreparedModel> preparedModel, const Request& request)
{
    sp<ExecutionCallback> cb(new ExecutionCallback());
    BOOST_TEST(preparedModel->execute(request, cb) == ErrorStatus::NONE);
    ALOGI("Execute: waiting for callback to be invoked");
    cb->wait();
}

sp<ExecutionCallback> ExecuteNoWait(android::sp<IPreparedModel> preparedModel, const Request& request)
{
    sp<ExecutionCallback> cb(new ExecutionCallback());
    BOOST_TEST(preparedModel->execute(request, cb) == ErrorStatus::NONE);
    ALOGI("ExecuteNoWait: returning callback object");
    return cb;
}
}

// Add our own test here since we fail the fc tests which Google supplies (because of non-const weights)
BOOST_AUTO_TEST_CASE(FullyConnected)
{
    // this should ideally replicate fully_connected_float.model.cpp
    // but that uses slightly weird dimensions which I don't think we need to support for now

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    Model model = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // make the fully connected operation
    model.operations.resize(1);
    model.operations[0].type = OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 3 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 1 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data (matching source test)
    float indata[] = {2, 32, 16};
    AddPoolAndSetData(3, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(1, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    Execute(preparedModel, request);

    // check the result
    BOOST_TEST(outdata[0] == 152);
}

// Add our own test for concurrent execution
// The main point of this test is to check that multiple requests can be
// executed without waiting for the callback from previous execution.
// The operations performed are not significant.
BOOST_AUTO_TEST_CASE(ConcurrentExecute)
{
    ALOGI("ConcurrentExecute: entry");

    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    Model model = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // make the fully connected operation
    model.operations.resize(1);
    model.operations[0].type = OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared models
    const size_t maxRequests = 5;
    android::sp<IPreparedModel> preparedModels[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        preparedModels[i] = PrepareModel(model, *driver);
    }

    // construct the request data
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 3 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 1 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    // build the requests
    Request requests[maxRequests];
    android::sp<IMemory> outMemory[maxRequests];
    float* outdata[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        requests[i].inputs  = hidl_vec<RequestArgument>{input};
        requests[i].outputs = hidl_vec<RequestArgument>{output};
        // set the input data (matching source test)
        float indata[] = {2, 32, 16};
        AddPoolAndSetData(3, requests[i], indata);
        // add memory for the output
        outMemory[i] = AddPoolAndGetData(1, requests[i]);
        outdata[i] = static_cast<float*>(static_cast<void*>(outMemory[i]->getPointer()));
    }

    // invoke the execution of the requests
    ALOGI("ConcurrentExecute: executing requests");
    sp<ExecutionCallback> cb[maxRequests];
    for (size_t i = 0; i < maxRequests; ++i)
    {
        cb[i] = ExecuteNoWait(preparedModels[i], requests[i]);
    }

    // wait for the requests to complete
    ALOGI("ConcurrentExecute: waiting for callbacks");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        cb[i]->wait();
    }

    // check the results
    ALOGI("ConcurrentExecute: validating results");
    for (size_t i = 0; i < maxRequests; ++i)
    {
        BOOST_TEST(outdata[i][0] == 152);
    }
    ALOGI("ConcurrentExecute: exit");
}

BOOST_AUTO_TEST_CASE(GetSupportedOperations)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model1 = {};

    // add operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    AddInputOperand(model1, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model1, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model1, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});

    // make a correct fully connected operation
    model1.operations.resize(2);
    model1.operations[0].type = OperationType::FULLY_CONNECTED;
    model1.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3};
    model1.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make an incorrect fully connected operation
    AddIntOperand(model1, actValue);
    AddOutputOperand(model1, hidl_vec<uint32_t>{1, 1});
    model1.operations[1].type = OperationType::FULLY_CONNECTED;
    model1.operations[1].inputs = hidl_vec<uint32_t>{4};
    model1.operations[1].outputs = hidl_vec<uint32_t>{5};

    driver->getSupportedOperations(model1, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == true);
    BOOST_TEST(sup[1] == false);

    // Broadcast add/mul are not supported
    Model model2 = {};

    AddInputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddInputOperand(model2, hidl_vec<uint32_t>{4});
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddOutputOperand(model2, hidl_vec<uint32_t>{1, 1, 3, 4});

    model2.operations.resize(2);

    model2.operations[0].type = OperationType::ADD;
    model2.operations[0].inputs = hidl_vec<uint32_t>{0,1};
    model2.operations[0].outputs = hidl_vec<uint32_t>{2};

    model2.operations[1].type = OperationType::MUL;
    model2.operations[1].inputs = hidl_vec<uint32_t>{0,1};
    model2.operations[1].outputs = hidl_vec<uint32_t>{3};

    driver->getSupportedOperations(model2, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);
    BOOST_TEST(sup[1] == false);

    Model model3 = {};

    // Add unsupported operation, should return no error but we don't support it
    AddInputOperand(model3, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddIntOperand(model3, 2);
    AddOutputOperand(model3, hidl_vec<uint32_t>{1, 2, 2, 2});
    model3.operations.resize(1);
    model3.operations[0].type = OperationType::DEPTH_TO_SPACE;
    model1.operations[0].inputs = hidl_vec<uint32_t>{0, 1};
    model3.operations[0].outputs = hidl_vec<uint32_t>{2};

    driver->getSupportedOperations(model3, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);

    // Add invalid operation
    Model model4 = {};
    AddIntOperand(model4, 0);
    model4.operations.resize(1);
    model4.operations[0].type = static_cast<OperationType>(100);
    model4.operations[0].outputs = hidl_vec<uint32_t>{0};

    driver->getSupportedOperations(model4, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::INVALID_ARGUMENT);
}

// The purpose of this test is to ensure that when encountering an unsupported operation
//      it is skipped and getSupportedOperations() continues (rather than failing and stopping).
//      As per IVGCVSW-710.
BOOST_AUTO_TEST_CASE(UnsupportedLayerContinueOnFailure)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {2, 4, 1};
    float   biasValue[]   = {4};

    // broadcast add is unsupported at the time of writing this test, but any unsupported layer will do
    AddInputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});
    AddInputOperand(model, hidl_vec<uint32_t>{4});
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    // fully connected
    AddInputOperand(model, hidl_vec<uint32_t>{1, 3});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 3}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1});

    // broadcast mul is unsupported
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, 3, 4});

    model.operations.resize(3);

    // unsupported
    model.operations[0].type = OperationType::ADD;
    model.operations[0].inputs = hidl_vec<uint32_t>{0,1};
    model.operations[0].outputs = hidl_vec<uint32_t>{2};

    // supported
    model.operations[1].type = OperationType::FULLY_CONNECTED;
    model.operations[1].inputs  = hidl_vec<uint32_t>{3, 4, 5, 6};
    model.operations[1].outputs = hidl_vec<uint32_t>{7};

    // unsupported
    model.operations[2].type = OperationType::MUL;
    model.operations[2].inputs = hidl_vec<uint32_t>{0,1};
    model.operations[2].outputs = hidl_vec<uint32_t>{8};

    // we are testing that the unsupported layers return false and the test continues
    //      rather than failing and stopping.
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::NONE);
    BOOST_TEST(sup[0] == false);
    BOOST_TEST(sup[1] == true);
    BOOST_TEST(sup[2] == false);
}

// The purpose of this test is to ensure that when encountering an failure
//      during mem pool mapping we properly report an error to the framework via a callback
BOOST_AUTO_TEST_CASE(ModelToINetworkConverterMemPoolFail)
{
    auto driver = std::make_unique<ArmnnDriver>(armnn::Compute::CpuRef);

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
    {
        error = status;
        sup = supported;
    };

    Model model = {};

    model.pools = hidl_vec<hidl_memory>{hidl_memory("Unsuported hidl memory type", nullptr, 0)};

    //memory pool mapping should fail, we should report an error
    driver->getSupportedOperations(model, cb);
    BOOST_TEST((int)error == (int)ErrorStatus::GENERAL_FAILURE);
}

namespace
{

void PaddingTestImpl(android::nn::PaddingScheme paddingScheme)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));
    Model model  = {};

    uint32_t outSize = paddingScheme == kPaddingSame ? 2 : 1;

    // add operands
    float weightValue[] = {1, -1, 0, 1};
    float biasValue[]   = {0};

    AddInputOperand(model, hidl_vec<uint32_t>{1, 2, 3, 1});
    AddTensorOperand(model, hidl_vec<uint32_t>{1, 2, 2, 1}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{1}, biasValue);
    AddIntOperand(model, (int32_t)paddingScheme); // padding
    AddIntOperand(model, 2); // stride x
    AddIntOperand(model, 2); // stride y
    AddIntOperand(model, 0); // no activation
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 1, outSize, 1});

    // make the convolution operation
    model.operations.resize(1);
    model.operations[0].type = OperationType::CONV_2D;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0, 1, 2, 3, 4, 5, 6};
    model.operations[0].outputs = hidl_vec<uint32_t>{7};

    // make the prepared model
    android::sp<IPreparedModel> preparedModel = PrepareModel(model, *driver);

    // construct the request
    DataLocation inloc    = {};
    inloc.poolIndex       = 0;
    inloc.offset          = 0;
    inloc.length          = 6 * sizeof(float);
    RequestArgument input = {};
    input.location        = inloc;
    input.dimensions      = hidl_vec<uint32_t>{};

    DataLocation outloc    = {};
    outloc.poolIndex       = 1;
    outloc.offset          = 0;
    outloc.length          = outSize * sizeof(float);
    RequestArgument output = {};
    output.location        = outloc;
    output.dimensions      = hidl_vec<uint32_t>{};

    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};


    // set the input data (matching source test)
    float indata[] = {4, 1, 0, 3, -1, 2};
    AddPoolAndSetData(6, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(outSize, request);
    float*               outdata   = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    Execute(preparedModel, request);

    // check the result
    if (paddingScheme == kPaddingValid)
    {
        BOOST_TEST(outdata[0] == 2);
    }
    else if (paddingScheme == kPaddingSame)
    {
        BOOST_TEST(outdata[0] == 2);
        BOOST_TEST(outdata[1] == 0);
    }
    else
    {
        BOOST_TEST(false);
    }
}

}

BOOST_AUTO_TEST_CASE(ConvValidPadding)
{
    PaddingTestImpl(kPaddingValid);
}

BOOST_AUTO_TEST_CASE(ConvSamePadding)
{
    PaddingTestImpl(kPaddingSame);
}

BOOST_AUTO_TEST_CASE(TestFullyConnected4dInput)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
        {
            error = status;
            sup = supported;
        };

    Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 1}; //identity
    float   biasValue[]   = {0, 0, 0, 0, 0, 0, 0, 0};

    // fully connected operation
    AddInputOperand(model, hidl_vec<uint32_t>{1, 1, 1, 8});
    AddTensorOperand(model, hidl_vec<uint32_t>{8, 8}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{8}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 8});

    model.operations.resize(1);

    model.operations[0].type = OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0,1,2,3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<IPreparedModel> preparedModel = PrepareModel(model, *driver);


    // construct the request
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 8 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 8 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    float indata[] = {1,2,3,4,5,6,7,8};
    AddPoolAndSetData(8, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(8, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    Execute(preparedModel, request);

    // check the result
    BOOST_TEST(outdata[0] == 1);
    BOOST_TEST(outdata[1] == 2);
    BOOST_TEST(outdata[2] == 3);
    BOOST_TEST(outdata[3] == 4);
    BOOST_TEST(outdata[4] == 5);
    BOOST_TEST(outdata[5] == 6);
    BOOST_TEST(outdata[6] == 7);
    BOOST_TEST(outdata[7] == 8);
}

BOOST_AUTO_TEST_CASE(TestFullyConnected4dInputReshape)
{
    auto driver = std::make_unique<ArmnnDriver>(DriverOptions(armnn::Compute::CpuRef));

    ErrorStatus error;
    std::vector<bool> sup;

    ArmnnDriver::getSupportedOperations_cb cb = [&](ErrorStatus status, const std::vector<bool>& supported)
        {
            error = status;
            sup = supported;
        };

    Model model = {};

    // operands
    int32_t actValue      = 0;
    float   weightValue[] = {1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 1}; //identity
    float   biasValue[]   = {0, 0, 0, 0, 0, 0, 0, 0};

    // fully connected operation
    AddInputOperand(model, hidl_vec<uint32_t>{1, 2, 2, 2});
    AddTensorOperand(model, hidl_vec<uint32_t>{8, 8}, weightValue);
    AddTensorOperand(model, hidl_vec<uint32_t>{8}, biasValue);
    AddIntOperand(model, actValue);
    AddOutputOperand(model, hidl_vec<uint32_t>{1, 8});

    model.operations.resize(1);

    model.operations[0].type = OperationType::FULLY_CONNECTED;
    model.operations[0].inputs  = hidl_vec<uint32_t>{0,1,2,3};
    model.operations[0].outputs = hidl_vec<uint32_t>{4};

    // make the prepared model
    android::sp<IPreparedModel> preparedModel = PrepareModel(model, *driver);


    // construct the request
    DataLocation inloc = {};
    inloc.poolIndex = 0;
    inloc.offset    = 0;
    inloc.length    = 8 * sizeof(float);
    RequestArgument input = {};
    input.location = inloc;
    input.dimensions = hidl_vec<uint32_t>{};

    DataLocation outloc = {};
    outloc.poolIndex = 1;
    outloc.offset    = 0;
    outloc.length    = 8 * sizeof(float);
    RequestArgument output = {};
    output.location  = outloc;
    output.dimensions = hidl_vec<uint32_t>{};

    Request request = {};
    request.inputs  = hidl_vec<RequestArgument>{input};
    request.outputs = hidl_vec<RequestArgument>{output};

    // set the input data
    float indata[] = {1,2,3,4,5,6,7,8};
    AddPoolAndSetData(8, request, indata);

    // add memory for the output
    android::sp<IMemory> outMemory = AddPoolAndGetData(8, request);
    float* outdata = static_cast<float*>(static_cast<void*>(outMemory->getPointer()));

    // run the execution
    Execute(preparedModel, request);

    // check the result
    BOOST_TEST(outdata[0] == 1);
    BOOST_TEST(outdata[1] == 2);
    BOOST_TEST(outdata[2] == 3);
    BOOST_TEST(outdata[3] == 4);
    BOOST_TEST(outdata[4] == 5);
    BOOST_TEST(outdata[5] == 6);
    BOOST_TEST(outdata[6] == 7);
    BOOST_TEST(outdata[7] == 8);
}

BOOST_AUTO_TEST_SUITE_END()
