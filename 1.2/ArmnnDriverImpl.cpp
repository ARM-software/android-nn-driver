//
// Copyright © 2017, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmnnDriverImpl.hpp"
#include "../ArmnnPreparedModel_1_2.hpp"
#include "../ModelToINetworkConverter.hpp"
#include "../SystemPropertiesUtils.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <log/log.h>
#include <sys/stat.h>
#include <chrono>

namespace
{

const char *g_RelaxedFloat32toFloat16PerformanceExecTime    = "ArmNN.relaxedFloat32toFloat16Performance.execTime";
const char *g_RelaxedFloat32toFloat16PerformancePowerUsage  = "ArmNN.relaxedFloat32toFloat16Performance.powerUsage";

const char *g_OperandTypeTensorFloat32PerformanceExecTime   = "Armnn.operandTypeTensorFloat32Performance.execTime";
const char *g_OperandTypeTensorFloat32PerformancePowerUsage = "Armnn.operandTypeTensorFloat32Performance.powerUsage";

const char *g_OperandTypeFloat32PerformanceExecTime         = "Armnn.operandTypeFloat32Performance.execTime";
const char *g_OperandTypeFloat32PerformancePowerUsage       = "Armnn.operandTypeFloat32Performance.powerUsage";

const char *g_OperandTypeTensorFloat16PerformanceExecTime   = "Armnn.operandTypeTensorFloat16Performance.execTime";
const char *g_OperandTypeTensorFloat16PerformancePowerUsage = "Armnn.operandTypeTensorFloat16Performance.powerUsage";

const char *g_OperandTypeFloat16PerformanceExecTime         = "Armnn.operandTypeFloat16Performance.execTime";
const char *g_OperandTypeFloat16PerformancePowerUsage       = "Armnn.operandTypeFloat16Performance.powerUsage";

const char *g_OperandTypeTensorQuant8AsymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant8AsymmPerformance.execTime";
const char *g_OperandTypeTensorQuant8AsymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant8AsymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant16SymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant16SymmPerformance.execTime";
const char *g_OperandTypeTensorQuant16SymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant16SymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant8SymmPerformanceExecTime =
        "Armnn.operandTypeTensorQuant8SymmPerformance.execTime";
const char *g_OperandTypeTensorQuant8SymmPerformancePowerUsage =
        "Armnn.operandTypeTensorQuant8SymmPerformance.powerUsage";

const char *g_OperandTypeTensorQuant8SymmPerChannelPerformanceExecTime =
    "Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.execTime";
const char *g_OperandTypeTensorQuant8SymmPerChannelPerformancePowerUsage =
    "Armnn.operandTypeTensorQuant8SymmPerChannelPerformance.powerUsage";


const char *g_OperandTypeTensorInt32PerformanceExecTime     = "Armnn.operandTypeTensorInt32Performance.execTime";
const char *g_OperandTypeTensorInt32PerformancePowerUsage   = "Armnn.operandTypeTensorInt32Performance.powerUsage";

const char *g_OperandTypeInt32PerformanceExecTime           = "Armnn.operandTypeInt32Performance.execTime";
const char *g_OperandTypeInt32PerformancePowerUsage         = "Armnn.operandTypeInt32Performance.powerUsage";


void NotifyCallbackAndCheck(const android::sp<V1_2::IPreparedModelCallback>& callback,
                            V1_0::ErrorStatus errorStatus,
                            const android::sp<V1_2::IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify_1_2(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("ArmnnDriverImpl::prepareModel: hidl callback failed to return properly: %s ",
              returned.description().c_str());
    }
}

Return<V1_0::ErrorStatus> FailPrepareModel(V1_0::ErrorStatus error,
                                           const std::string& message,
                                           const android::sp<V1_2::IPreparedModelCallback>& callback)
{
    ALOGW("ArmnnDriverImpl::prepareModel: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

} // anonymous namespace

namespace armnn_driver
{
namespace hal_1_2
{

Return<V1_0::ErrorStatus> ArmnnDriverImpl::prepareArmnnModel_1_2(
       const armnn::IRuntimePtr& runtime,
       const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
       const DriverOptions& options,
       const V1_2::Model& model,
       const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
       const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
       const HidlToken& token,
       const android::sp<V1_2::IPreparedModelCallback>& cb,
       bool float32ToFloat16)
{
    ALOGV("ArmnnDriverImpl::prepareArmnnModel_1_2()");

    std::chrono::time_point<std::chrono::system_clock> prepareModelTimepoint = std::chrono::system_clock::now();

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriverImpl::prepareModel: Invalid callback passed to prepareModel");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!runtime)
    {
        return FailPrepareModel(V1_0::ErrorStatus::DEVICE_UNAVAILABLE, "Device unavailable", cb);
    }

    if (!android::nn::validateModel(model))
    {
        return FailPrepareModel(V1_0::ErrorStatus::INVALID_ARGUMENT, "Invalid model passed as input", cb);
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    std::set<unsigned int> unsupportedOperations;
    ModelToINetworkConverter<HalPolicy> modelConverter(options.GetBackends(),
                                                       model,
                                                       unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "ModelToINetworkConverter failed", cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Serialize the network graph to a .armnn file if an output directory
    // has been specified in the drivers' arguments.
    std::vector<uint8_t> dataCacheData;
    bool serializeToFile = dataCacheHandle.size() < 1 ? false : true;
    auto serializedNetworkFileName =
        SerializeNetwork(*modelConverter.GetINetwork(),
                         options.GetRequestInputsAndOutputsDumpDir(),
                         dataCacheData,
                         serializeToFile);

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptionsOpaque OptOptions;
    OptOptions.SetReduceFp32ToFp16(float32ToFloat16);
    OptOptions.SetProfilingEnabled(options.IsGpuProfilingEnabled());

    int cachedFd = -1;
    bool saveCachedNetwork = options.SaveCachedNetwork();

    unsigned int numberOfCachedModelFiles = 0;
    if (modelCacheHandle.size() > 0)
    {
        unsigned int index = 0;
        for (auto& backend : options.GetBackends())
        {
            // modelCacheHandle size should be equal to numberOfCachedModelFiles
            // modelCacheHandle vector should be in same order as backends
            auto numberOfCacheFiles = GetNumberOfCacheFiles(backend);
            if (numberOfCacheFiles > 0)
            {
                numberOfCachedModelFiles += numberOfCacheFiles;
                if (modelCacheHandle[index]->numFds == 1)
                {
                    if (backend == armnn::Compute::GpuAcc)
                    {
                        cachedFd = modelCacheHandle[index]->data[0];
                        saveCachedNetwork = true;
                    }
                }
                index += numberOfCachedModelFiles;
            }
        }
    }

    armnn::BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "SaveCachedNetwork", saveCachedNetwork },
        { "CachedNetworkFilePath", options.GetCachedNetworkFilePath() },
        { "MLGOTuningFilePath", options.GetClMLGOTunedParametersFile() },
        { "CachedFileDescriptor", cachedFd }
    });

    armnn::BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "NumberOfThreads", options.GetNumberOfThreads() }
    });
    OptOptions.AddModelOption(gpuAcc);
    OptOptions.AddModelOption(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception &e)
    {
        std::stringstream message;
        message << "Exception (" << e.what() << ") caught from optimize.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        std::stringstream message;
        message << "Invalid optimized network";
        for (const std::string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet,
                                                               options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    std::string msg;
    armnn::INetworkProperties networkProperties(options.isAsyncModelExecutionEnabled(),
                                                MemorySource::Undefined,
                                                MemorySource::Undefined,
                                                options.IsGpuProfilingEnabled());

    auto numInputs  = getMainModel(model).inputIndexes.size();
    auto numOutputs = getMainModel(model).outputIndexes.size();
    try
    {
        if (runtime->LoadNetwork(netId, move(optNet), msg, networkProperties) != armnn::Status::Success)
        {
            return FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, msg, cb);
        }
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what()<< ") caught from LoadNetwork.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Now that we have a networkId for the graph rename the exported files to use it
    // so that we can associate the graph file and the input/output tensor exported files
    RenameExportedFiles(serializedNetworkFileName,
                        dotGraphFileName,
                        options.GetRequestInputsAndOutputsDumpDir(),
                        netId);

    std::unique_ptr<ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>> preparedModel(
            new ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>(
                    netId,
                    runtime.get(),
                    model,
                    options.GetRequestInputsAndOutputsDumpDir(),
                    options.IsGpuProfilingEnabled(),
                    options.isAsyncModelExecutionEnabled(),
                    options.getNoOfArmnnThreads(),
                    options.isImportEnabled(),
                    options.isExportEnabled()));

    // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
    // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
    // Only run this if the GpuAcc backend has been added to options
    if (std::find(options.GetBackends().begin(),
                  options.GetBackends().end(),
                  armnn::Compute::GpuAcc) != options.GetBackends().end())
    {
        if (!preparedModel->ExecuteWithDummyInputs(numInputs, numOutputs))
        {
            return FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Network could not be executed", cb);
        }

        if (clTunedParameters &&
            options.GetClTunedParametersMode() == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters)
        {
            // Now that we've done one inference the CL kernel parameters will have been tuned,
            // so save the updated file.
            try
            {
                clTunedParameters->Save(options.GetClTunedParametersFile().c_str());
            }
            catch (std::exception& error)
            {
                ALOGE("ArmnnDriverImpl::prepareModel: Failed to save CL tuned parameters file '%s': %s",
                      options.GetClTunedParametersFile().c_str(), error.what());
            }
        }
    }

    size_t hashValue = 0;
    // Cache the model
    if (dataCacheHandle.size() > 0)
    {
        // Cache the Arm NN model, should be only 1
        if (dataCacheHandle.size() != 1)
        {
            NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());
            return V1_0::ErrorStatus::NONE;
        }

        if (dataCacheHandle[0]->numFds != 1)
        {
            ALOGW("ArmnnDriverImpl::prepareArmnnModel_1_3: Cannot cache the data, numFds != 1.");
            NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());
            return V1_0::ErrorStatus::NONE;
        }

        if (dataCacheHandle[0]->data[0] < 0)
        {
            ALOGW("ArmnnDriverImpl::prepareArmnnModel_1_3: Cannot cache the data, fd < 0");
            NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());
            return V1_0::ErrorStatus::NONE;
        }

        int dataCacheFileAccessMode = fcntl(dataCacheHandle[0]->data[0], F_GETFL) & O_ACCMODE;
        if (dataCacheFileAccessMode != O_RDWR)
        {
            ALOGW("ArmnnDriverImpl::prepareModelFromCache_1_2(): Invalid Access Mode.");
            NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());
            return V1_0::ErrorStatus::NONE;
        }

        write(dataCacheHandle[0]->data[0], dataCacheData.data(), dataCacheData.size());
        hashValue = CacheDataHandlerInstance().Hash(dataCacheData);
    }

    if (modelCacheHandle.size() > 0)
    {
        if (modelCacheHandle.size() != numberOfCachedModelFiles)
        {
            NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());
            return V1_0::ErrorStatus::NONE;
        }
        for (uint32_t i = 0; i < modelCacheHandle.size(); ++i)
        {
            if (modelCacheHandle[i]->numFds == 1)
            {
                int modelCacheFileAccessMode = fcntl(modelCacheHandle[i]->data[0], F_GETFL) & O_ACCMODE;
                if (modelCacheFileAccessMode != O_RDONLY)
                {
                    struct stat statBuffer;
                    if (fstat(modelCacheHandle[i]->data[0], &statBuffer) == 0)
                    {
                        long modelDataSize = statBuffer.st_size;
                        if (modelDataSize > 0)
                        {
                            std::vector <uint8_t> modelData(modelDataSize);
                            pread(modelCacheHandle[i]->data[0], modelData.data(), modelData.size(), 0);
                            hashValue ^= CacheDataHandlerInstance().Hash(modelData);
                        }
                    }
                }
            }
        }
    }
    if (hashValue != 0)
    {
        CacheDataHandlerInstance().Register(token, hashValue, dataCacheData.size());
    }

    NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());

    ALOGV("ArmnnDriverImpl::prepareModel cache timing = %lld µs", std::chrono::duration_cast<std::chrono::microseconds>
         (std::chrono::system_clock::now() - prepareModelTimepoint).count());

    return V1_0::ErrorStatus::NONE;
}

Return<V1_0::ErrorStatus> ArmnnDriverImpl::prepareModelFromCache(
    const armnn::IRuntimePtr& runtime,
    const DriverOptions& options,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>& modelCacheHandle,
    const android::hardware::hidl_vec<android::hardware::hidl_handle>& dataCacheHandle,
    const HidlToken& token,
    const android::sp<V1_2::IPreparedModelCallback>& cb,
    bool float32ToFloat16)
{
    ALOGV("ArmnnDriverImpl::prepareModelFromCache()");
    std::chrono::time_point<std::chrono::system_clock> modelFromCacheTimepoint = std::chrono::system_clock::now();

    if (cb.get() == nullptr)
    {
        ALOGW("ArmnnDriverImpl::prepareModelFromCache: Invalid callback passed to prepareModel");
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!runtime)
    {
        return FailPrepareModel(V1_0::ErrorStatus::DEVICE_UNAVAILABLE, "Device unavailable", cb);
    }

    if (token.size() != ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN)
    {
        FailPrepareModel(V1_0::ErrorStatus::INVALID_ARGUMENT, "Invalid token passed!", cb);
        return V1_0::ErrorStatus::INVALID_ARGUMENT;
    }

    // DataCacheHandle size should always be 1
    // Arm NN model
    if (dataCacheHandle.size() != 1)
    {
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "No data cache!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    // Check if model files cached they match the expected value
    unsigned int numberOfCachedModelFiles = 0;
    for (auto& backend : options.GetBackends())
    {
        numberOfCachedModelFiles += GetNumberOfCacheFiles(backend);
    }
    if (modelCacheHandle.size() != numberOfCachedModelFiles)
    {
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Invalid model cache!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    if (dataCacheHandle[0]->numFds != 1)
    {
        ALOGW("ArmnnDriverImpl::prepareModelFromCache: Cannot read from the cache data, numFds != 1.");
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "No data cache!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    if (dataCacheHandle[0]->data[0] < 0)
    {
        ALOGW("ArmnnDriverImpl::prepareModelFromCache: Cannot read from the cache data, fd < 0");
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "No data cache!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    int dataCacheFileAccessMode = fcntl(dataCacheHandle[0]->data[0], F_GETFL) & O_ACCMODE;
    if (dataCacheFileAccessMode != O_RDWR)
    {
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Invalid Access Mode!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    auto dataSize = CacheDataHandlerInstance().GetCacheSize(token);
    if (dataSize == 0)
    {
        ALOGW("ArmnnDriverImpl::prepareModelFromCache: Invalid data to deserialize!");
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Invalid data to deserialize!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    int offset = 0;
    {
        struct stat statBuffer;
        if (fstat(dataCacheHandle[0]->data[0], &statBuffer) == 0)
        {
            unsigned long bufferSize = statBuffer.st_size;
            if (bufferSize != dataSize)
            {
                ALOGW("ArmnnDriverImpl::prepareModelFromCache: Invalid data to deserialize!");
                FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Invalid data to deserialize!", cb);
                return V1_0::ErrorStatus::GENERAL_FAILURE;
            }
        }
    }
    std::vector<uint8_t> dataCacheData(dataSize);
    pread(dataCacheHandle[0]->data[0], dataCacheData.data(), dataCacheData.size(), offset);
    auto hashValue = CacheDataHandlerInstance().Hash(dataCacheData);

    int gpuAccCachedFd = -1;
    bool saveCachedNetwork = false;
    if (modelCacheHandle.size() > 0)
    {
        unsigned int index = 0;
        for (auto& backend : options.GetBackends())
        {
            // modelCacheHandle size should be equal to numberOfCachedModelFiles
            // modelCacheHandle vector should be in same order as backends
            auto numberOfCacheFiles = GetNumberOfCacheFiles(backend);
            if (numberOfCacheFiles > 0)
            {
                if (modelCacheHandle[index]->numFds != 1)
                {
                    ALOGW("ArmnnDriverImpl::prepareModelFromCache: Cannot read from the model cache, numFds != 1.");
                    FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE,
                                     "Cannot read from the model cache, numFds != 1.", cb);
                    return V1_0::ErrorStatus::GENERAL_FAILURE;
                }
                auto cachedFd = modelCacheHandle[index]->data[0];

                int modelCacheFileAccessMode = fcntl(cachedFd, F_GETFL) & O_ACCMODE;
                if (modelCacheFileAccessMode != O_RDWR)
                {
                    FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Invalid Access Mode!", cb);
                    return V1_0::ErrorStatus::GENERAL_FAILURE;
                }

                struct stat statBuffer;
                if (cachedFd != -1 && fstat(cachedFd, &statBuffer) == 0)
                {
                    long modelDataSize = statBuffer.st_size;
                    if (modelDataSize <= 0)
                    {
                        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "Wrong cached model size!", cb);
                        return V1_0::ErrorStatus::NONE;
                    }
                    std::vector<uint8_t> modelData(modelDataSize);
                    pread(cachedFd, modelData.data(), modelData.size(), 0);
                    hashValue ^= CacheDataHandlerInstance().Hash(modelData);

                    // For GpuAcc numberOfCachedFiles is 1
                    if (backend == armnn::Compute::GpuAcc)
                    {
                        gpuAccCachedFd = cachedFd;
                    }
                }
                index += numberOfCacheFiles;
            }
        }
    }

    if (!CacheDataHandlerInstance().Validate(token, hashValue, dataCacheData.size()))
    {
        ALOGW("ArmnnDriverImpl::prepareModelFromCache: ValidateHash() failed!");
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, "ValidateHash Failed!", cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    // Deserialize the network..
    armnn::INetworkPtr network = armnn::INetworkPtr(nullptr, [](armnn::INetwork*){});
    try
    {
        network = armnnDeserializer::IDeserializer::Create()->CreateNetworkFromBinary(dataCacheData);
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what() << ") caught from Deserializer.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptionsOpaque OptOptions;
    OptOptions.SetReduceFp32ToFp16(float32ToFloat16);
    OptOptions.SetProfilingEnabled(options.IsGpuProfilingEnabled());

    armnn::BackendOptions gpuAcc("GpuAcc",
                                 {
                                         {"FastMathEnabled",       options.IsFastMathEnabled()},
                                         {"SaveCachedNetwork",     saveCachedNetwork},
                                         {"CachedNetworkFilePath", options.GetCachedNetworkFilePath()},
                                         {"MLGOTuningFilePath",    options.GetClMLGOTunedParametersFile()},
                                         {"CachedFileDescriptor",  gpuAccCachedFd}
                                 });

    armnn::BackendOptions cpuAcc("CpuAcc",
                                 {
                                         {"FastMathEnabled", options.IsFastMathEnabled()},
                                         {"NumberOfThreads", options.GetNumberOfThreads()}
                                 });
    OptOptions.AddModelOption(gpuAcc);
    OptOptions.AddModelOption(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*network.get(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what() << ") caught from optimize.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        std::stringstream message;
        message << "Invalid optimized network";
        for (const std::string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet,
                                                               options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    std::string msg;
    armnn::INetworkProperties networkProperties(options.isAsyncModelExecutionEnabled(),
                                                MemorySource::Undefined,
                                                MemorySource::Undefined,
                                                options.IsGpuProfilingEnabled());

    try
    {
        if (runtime->LoadNetwork(netId, move(optNet), msg, networkProperties) != armnn::Status::Success)
        {
            return FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, msg, cb);
        }
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what() << ") caught from LoadNetwork.";
        FailPrepareModel(V1_0::ErrorStatus::GENERAL_FAILURE, message.str(), cb);
        return V1_0::ErrorStatus::NONE;
    }

    std::unique_ptr<ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>> preparedModel(
            new ArmnnPreparedModel_1_2<hal_1_2::HalPolicy>(
                    netId,
                    runtime.get(),
                    options.GetRequestInputsAndOutputsDumpDir(),
                    options.IsGpuProfilingEnabled(),
                    options.isAsyncModelExecutionEnabled(),
                    options.getNoOfArmnnThreads(),
                    options.isImportEnabled(),
                    options.isExportEnabled(),
                    true));

    NotifyCallbackAndCheck(cb, V1_0::ErrorStatus::NONE, preparedModel.release());

    ALOGV("ArmnnDriverImpl::prepareModelFromCache cache timing = %lld µs",
          std::chrono::duration_cast<std::chrono::microseconds>
          (std::chrono::system_clock::now() - modelFromCacheTimepoint).count());

    return V1_0::ErrorStatus::NONE;
}

Return<void> ArmnnDriverImpl::getCapabilities_1_2(const armnn::IRuntimePtr& runtime,
                                                  V1_2::IDevice::getCapabilities_1_2_cb cb)
{
    ALOGV("hal_1_2::ArmnnDriverImpl::getCapabilities()");

    V1_2::Capabilities capabilities;

    float defaultValue = .1f;

    if (runtime)
    {
        capabilities.relaxedFloat32toFloat16PerformanceScalar.execTime =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformanceExecTime, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceScalar.powerUsage =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformancePowerUsage, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceTensor.execTime =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformanceExecTime, defaultValue);

        capabilities.relaxedFloat32toFloat16PerformanceTensor.powerUsage =
                ParseSystemProperty(g_RelaxedFloat32toFloat16PerformancePowerUsage, defaultValue);

        // Set the base value for all operand types
        #if defined(ARMNN_ANDROID_R) || defined(ARMNN_ANDROID_S)
        capabilities.operandPerformance = nonExtensionOperandPerformance<HalVersion::V1_2>({FLT_MAX, FLT_MAX});
        #else
        capabilities.operandPerformance = nonExtensionOperandPerformance({FLT_MAX, FLT_MAX});
        #endif

        // Load supported operand types
        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_FLOAT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorFloat32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorFloat32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::FLOAT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeFloat32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeFloat32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_FLOAT16,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorFloat16PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorFloat16PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::FLOAT16,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeFloat16PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeFloat16PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_QUANT8_ASYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant8AsymmPerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_QUANT8_SYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_QUANT16_SYMM,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorQuant16SymmPerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorQuant16SymmPerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
               {
                   .execTime =
                   ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerChannelPerformanceExecTime, defaultValue),
                   .powerUsage =
                   ParseSystemProperty(g_OperandTypeTensorQuant8SymmPerChannelPerformancePowerUsage, defaultValue)
               });

        update(&capabilities.operandPerformance, V1_2::OperandType::TENSOR_INT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeTensorInt32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeTensorInt32PerformancePowerUsage, defaultValue)
                });

        update(&capabilities.operandPerformance, V1_2::OperandType::INT32,
                {
                    .execTime = ParseSystemProperty(g_OperandTypeInt32PerformanceExecTime, defaultValue),
                    .powerUsage = ParseSystemProperty(g_OperandTypeInt32PerformancePowerUsage, defaultValue)
                });

        cb(V1_0::ErrorStatus::NONE, capabilities);
    }
    else
    {
        capabilities.relaxedFloat32toFloat16PerformanceScalar.execTime   = 0;
        capabilities.relaxedFloat32toFloat16PerformanceScalar.powerUsage = 0;
        capabilities.relaxedFloat32toFloat16PerformanceTensor.execTime   = 0;
        capabilities.relaxedFloat32toFloat16PerformanceTensor.powerUsage = 0;

        // Set the base value for all operand types
        #if defined(ARMNN_ANDROID_R) || defined(ARMNN_ANDROID_S)
        capabilities.operandPerformance = nonExtensionOperandPerformance<HalVersion::V1_2>({0.f, 0.0f});
        #else
        capabilities.operandPerformance = nonExtensionOperandPerformance({0.f, 0.0f});
        #endif

        cb(V1_0::ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
    }

    return Void();
}

} // namespace hal_1_2
} // namespace armnn_driver