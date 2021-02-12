//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "Utils.hpp"
#include "Half.hpp"

#include <armnnSerializer/ISerializer.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnn/Utils.hpp>
#include <armnn/utility/Assert.hpp>
#include <Filesystem.hpp>
#include <log/log.h>

#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <sstream>
#include <cstdio>
#include <time.h>

using namespace android;
using namespace android::hardware;
using namespace android::hidl::memory::V1_0;

namespace armnn_driver
{
const armnn::PermutationVector g_DontPermute{};

namespace
{

void SwizzleAndroidNn4dTensorToArmNn(const armnn::TensorShape& inTensorShape, const void* input,
                                     void* output, size_t dataTypeSize, const armnn::PermutationVector& mappings)
{
    assert(inTensorShape.GetNumDimensions() == 4U);

    armnnUtils::Permute(armnnUtils::Permuted(inTensorShape, mappings), mappings, input, output, dataTypeSize);
}

} // anonymous namespace

void SwizzleAndroidNn4dTensorToArmNn(const armnn::TensorInfo& tensor, const void* input, void* output,
                                     const armnn::PermutationVector& mappings)
{
    assert(tensor.GetNumDimensions() == 4U);

    armnn::DataType dataType = tensor.GetDataType();
    switch (dataType)
    {
    case armnn::DataType::Float16:
    case armnn::DataType::Float32:
    case armnn::DataType::QAsymmU8:
    case armnn::DataType::QSymmS8:
    case armnn::DataType::QAsymmS8:
        SwizzleAndroidNn4dTensorToArmNn(tensor.GetShape(), input, output, armnn::GetDataTypeSize(dataType), mappings);
        break;
    default:
        ALOGW("Unknown armnn::DataType for swizzling");
        assert(0);
    }
}

void* GetMemoryFromPool(V1_0::DataLocation location, const std::vector<android::nn::RunTimePoolInfo>& memPools)
{
    // find the location within the pool
    assert(location.poolIndex < memPools.size());

    const android::nn::RunTimePoolInfo& memPool = memPools[location.poolIndex];

    uint8_t* memPoolBuffer = memPool.getBuffer();

    uint8_t* memory = memPoolBuffer + location.offset;

    return memory;
}

armnn::TensorInfo GetTensorInfoForOperand(const V1_0::Operand& operand)
{
    using namespace armnn;
    DataType type;

    switch (operand.type)
    {
        case V1_0::OperandType::TENSOR_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case V1_0::OperandType::TENSOR_QUANT8_ASYMM:
            type = armnn::DataType::QAsymmU8;
            break;
        case V1_0::OperandType::TENSOR_INT32:
            type = armnn::DataType::Signed32;
            break;
        default:
            throw UnsupportedOperand<V1_0::OperandType>(operand.type);
    }

    TensorInfo ret;
    if (operand.dimensions.size() == 0)
    {
        TensorShape tensorShape(Dimensionality::NotSpecified);
        ret = TensorInfo(tensorShape, type);
    }
    else
    {
        bool dimensionsSpecificity[5] = { true, true, true, true, true };
        int count = 0;
        std::for_each(operand.dimensions.data(),
                      operand.dimensions.data() +  operand.dimensions.size(),
                      [&](const unsigned int val)
                      {
                          if (val == 0)
                          {
                              dimensionsSpecificity[count] = false;
                          }
                          count++;
                      });

        TensorShape tensorShape(operand.dimensions.size(), operand.dimensions.data(), dimensionsSpecificity);
        ret = TensorInfo(tensorShape, type);
    }

    ret.SetQuantizationScale(operand.scale);
    ret.SetQuantizationOffset(operand.zeroPoint);

    return ret;
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)// Using ::android::hardware::neuralnetworks::V1_2

armnn::TensorInfo GetTensorInfoForOperand(const V1_2::Operand& operand)
{
    using namespace armnn;
    bool perChannel = false;

    DataType type;
    switch (operand.type)
    {
        case V1_2::OperandType::TENSOR_BOOL8:
            type = armnn::DataType::Boolean;
            break;
        case V1_2::OperandType::TENSOR_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case V1_2::OperandType::TENSOR_FLOAT16:
            type = armnn::DataType::Float16;
            break;
        case V1_2::OperandType::TENSOR_QUANT8_ASYMM:
            type = armnn::DataType::QAsymmU8;
            break;
        case V1_2::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            perChannel=true;
            ARMNN_FALLTHROUGH;
        case V1_2::OperandType::TENSOR_QUANT8_SYMM:
            type = armnn::DataType::QSymmS8;
            break;
        case V1_2::OperandType::TENSOR_QUANT16_SYMM:
            type = armnn::DataType::QSymmS16;
            break;
        case V1_2::OperandType::TENSOR_INT32:
            type = armnn::DataType::Signed32;
            break;
        default:
            throw UnsupportedOperand<V1_2::OperandType>(operand.type);
    }

    TensorInfo ret;
    if (operand.dimensions.size() == 0)
    {
        TensorShape tensorShape(Dimensionality::NotSpecified);
        ret = TensorInfo(tensorShape, type);
    }
    else
    {
        bool dimensionsSpecificity[5] = { true, true, true, true, true };
        int count = 0;
        std::for_each(operand.dimensions.data(),
                      operand.dimensions.data() +  operand.dimensions.size(),
                      [&](const unsigned int val)
                      {
                          if (val == 0)
                          {
                              dimensionsSpecificity[count] = false;
                          }
                          count++;
                      });

        TensorShape tensorShape(operand.dimensions.size(), operand.dimensions.data(), dimensionsSpecificity);
        ret = TensorInfo(tensorShape, type);
    }

    if (perChannel)
    {
        // ExtraParams is expected to be of type channelQuant
        ARMNN_ASSERT(operand.extraParams.getDiscriminator() ==
                     V1_2::Operand::ExtraParams::hidl_discriminator::channelQuant);

        auto perAxisQuantParams = operand.extraParams.channelQuant();

        ret.SetQuantizationScales(perAxisQuantParams.scales);
        ret.SetQuantizationDim(MakeOptional<unsigned int>(perAxisQuantParams.channelDim));
    }
    else
    {
        ret.SetQuantizationScale(operand.scale);
        ret.SetQuantizationOffset(operand.zeroPoint);
    }

    return ret;
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_3

armnn::TensorInfo GetTensorInfoForOperand(const V1_3::Operand& operand)
{
    using namespace armnn;
    bool perChannel = false;
    bool isScalar   = false;

    DataType type;
    switch (operand.type)
    {
        case V1_3::OperandType::TENSOR_BOOL8:
            type = armnn::DataType::Boolean;
            break;
        case V1_3::OperandType::TENSOR_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case V1_3::OperandType::TENSOR_FLOAT16:
            type = armnn::DataType::Float16;
            break;
        case V1_3::OperandType::TENSOR_QUANT8_ASYMM:
            type = armnn::DataType::QAsymmU8;
            break;
        case V1_3::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            perChannel=true;
            ARMNN_FALLTHROUGH;
        case V1_3::OperandType::TENSOR_QUANT8_SYMM:
            type = armnn::DataType::QSymmS8;
            break;
        case V1_3::OperandType::TENSOR_QUANT16_SYMM:
            type = armnn::DataType::QSymmS16;
            break;
        case V1_3::OperandType::TENSOR_INT32:
            type = armnn::DataType::Signed32;
            break;
        case V1_3::OperandType::INT32:
            type = armnn::DataType::Signed32;
            isScalar = true;
            break;
        case V1_3::OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            type = armnn::DataType::QAsymmS8;
            break;
        default:
            throw UnsupportedOperand<V1_3::OperandType>(operand.type);
    }

    TensorInfo ret;
    if (isScalar)
    {
        ret = TensorInfo(TensorShape(armnn::Dimensionality::Scalar), type);
    }
    else
    {
        if (operand.dimensions.size() == 0)
        {
            TensorShape tensorShape(Dimensionality::NotSpecified);
            ret = TensorInfo(tensorShape, type);
        }
        else
        {
            bool dimensionsSpecificity[5] = { true, true, true, true, true };
            int count = 0;
            std::for_each(operand.dimensions.data(),
                          operand.dimensions.data() +  operand.dimensions.size(),
                          [&](const unsigned int val)
                          {
                              if (val == 0)
                              {
                                  dimensionsSpecificity[count] = false;
                              }
                              count++;
                          });

            TensorShape tensorShape(operand.dimensions.size(), operand.dimensions.data(), dimensionsSpecificity);
            ret = TensorInfo(tensorShape, type);
        }
    }

    if (perChannel)
    {
        // ExtraParams is expected to be of type channelQuant
        ARMNN_ASSERT(operand.extraParams.getDiscriminator() ==
                     V1_2::Operand::ExtraParams::hidl_discriminator::channelQuant);

        auto perAxisQuantParams = operand.extraParams.channelQuant();

        ret.SetQuantizationScales(perAxisQuantParams.scales);
        ret.SetQuantizationDim(MakeOptional<unsigned int>(perAxisQuantParams.channelDim));
    }
    else
    {
        ret.SetQuantizationScale(operand.scale);
        ret.SetQuantizationOffset(operand.zeroPoint);
    }
    return ret;
}

#endif

std::string GetOperandSummary(const V1_0::Operand& operand)
{
    return android::hardware::details::arrayToString(operand.dimensions, operand.dimensions.size()) + " " +
        toString(operand.type);
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3) // Using ::android::hardware::neuralnetworks::V1_2

std::string GetOperandSummary(const V1_2::Operand& operand)
{
    return android::hardware::details::arrayToString(operand.dimensions, operand.dimensions.size()) + " " +
           toString(operand.type);
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_3

std::string GetOperandSummary(const V1_3::Operand& operand)
{
    return android::hardware::details::arrayToString(operand.dimensions, operand.dimensions.size()) + " " +
           toString(operand.type);
}

#endif

using DumpElementFunction = void (*)(const armnn::ConstTensor& tensor,
    unsigned int elementIndex,
    std::ofstream& fileStream);

namespace
{
template <typename ElementType, typename PrintableType = ElementType>
void DumpTensorElement(const armnn::ConstTensor& tensor, unsigned int elementIndex, std::ofstream& fileStream)
{
    const ElementType* elements = reinterpret_cast<const ElementType*>(tensor.GetMemoryArea());
    fileStream << static_cast<PrintableType>(elements[elementIndex]) << ",";
}

constexpr const char* MemoryLayoutString(const armnn::ConstTensor& tensor)
{
    const char* str = "";

    switch (tensor.GetNumDimensions())
    {
        case 4:  { str = "(BHWC) "; break; }
        case 3:  { str = "(HWC) "; break; }
        case 2:  { str = "(HW) "; break; }
        default: { str = ""; break; }
    }

    return str;
}
} // namespace

void DumpTensor(const std::string& dumpDir,
    const std::string& requestName,
    const std::string& tensorName,
    const armnn::ConstTensor& tensor)
{
    // The dump directory must exist in advance.
    fs::path dumpPath = dumpDir;
    const fs::path fileName = dumpPath / (requestName + "_" + tensorName + ".dump");

    std::ofstream fileStream;
    fileStream.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        ALOGW("Could not open file %s for writing", fileName.c_str());
        return;
    }

    DumpElementFunction dumpElementFunction = nullptr;

    switch (tensor.GetDataType())
    {
        case armnn::DataType::Float32:
        {
            dumpElementFunction = &DumpTensorElement<float>;
            break;
        }
        case armnn::DataType::QAsymmU8:
        {
            dumpElementFunction = &DumpTensorElement<uint8_t, uint32_t>;
            break;
        }
        case armnn::DataType::Signed32:
        {
            dumpElementFunction = &DumpTensorElement<int32_t>;
            break;
        }
        case armnn::DataType::Float16:
        {
            dumpElementFunction = &DumpTensorElement<armnn::Half>;
            break;
        }
        case armnn::DataType::QAsymmS8:
        {
            dumpElementFunction = &DumpTensorElement<int8_t, int32_t>;
            break;
        }
        case armnn::DataType::Boolean:
        {
            dumpElementFunction = &DumpTensorElement<bool>;
            break;
        }
        default:
        {
            dumpElementFunction = nullptr;
        }
    }

    if (dumpElementFunction != nullptr)
    {
        const unsigned int numDimensions = tensor.GetNumDimensions();

        const unsigned int batch = (numDimensions == 4) ? tensor.GetShape()[numDimensions - 4] : 1;

        const unsigned int height = (numDimensions >= 3)
                                    ? tensor.GetShape()[numDimensions - 3]
                                    : (numDimensions >= 2) ? tensor.GetShape()[numDimensions - 2] : 1;

        const unsigned int width = (numDimensions >= 3)
                                   ? tensor.GetShape()[numDimensions - 2]
                                   : (numDimensions >= 1) ? tensor.GetShape()[numDimensions - 1] : 0;

        const unsigned int channels = (numDimensions >= 3) ? tensor.GetShape()[numDimensions - 1] : 1;

        fileStream << "# Number of elements " << tensor.GetNumElements() << std::endl;
        fileStream << "# Dimensions " << MemoryLayoutString(tensor);
        fileStream << "[" << tensor.GetShape()[0];
        for (unsigned int d = 1; d < numDimensions; d++)
        {
            fileStream << "," << tensor.GetShape()[d];
        }
        fileStream << "]" << std::endl;

        for (unsigned int e = 0, b = 0; b < batch; ++b)
        {
            if (numDimensions >= 4)
            {
                fileStream << "# Batch " << b << std::endl;
            }
            for (unsigned int c = 0; c < channels; c++)
            {
                if (numDimensions >= 3)
                {
                    fileStream << "# Channel " << c << std::endl;
                }
                for (unsigned int h = 0; h < height; h++)
                {
                    for (unsigned int w = 0; w < width; w++, e += channels)
                    {
                        (*dumpElementFunction)(tensor, e, fileStream);
                    }
                    fileStream << std::endl;
                }
                e -= channels - 1;
                if (c < channels)
                {
                    e -= ((height * width) - 1) * channels;
                }
            }
            fileStream << std::endl;
        }
        fileStream << std::endl;
    }
    else
    {
        fileStream << "Cannot dump tensor elements: Unsupported data type "
            << static_cast<unsigned int>(tensor.GetDataType()) << std::endl;
    }

    if (!fileStream.good())
    {
        ALOGW("An error occurred when writing to file %s", fileName.c_str());
    }
}

void DumpJsonProfilingIfRequired(bool gpuProfilingEnabled,
                                 const std::string& dumpDir,
                                 armnn::NetworkId networkId,
                                 const armnn::IProfiler* profiler)
{
    // Check if profiling is required.
    if (!gpuProfilingEnabled)
    {
        return;
    }

    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return;
    }

    ARMNN_ASSERT(profiler);

    // Set the name of the output profiling file.
    fs::path dumpPath = dumpDir;
    const fs::path fileName = dumpPath / (std::to_string(networkId) + "_profiling.json");

    // Open the ouput file for writing.
    std::ofstream fileStream;
    fileStream.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        ALOGW("Could not open file %s for writing", fileName.c_str());
        return;
    }

    // Write the profiling info to a JSON file.
    profiler->Print(fileStream);
}

std::string ExportNetworkGraphToDotFile(const armnn::IOptimizedNetwork& optimizedNetwork,
                                        const std::string& dumpDir)
{
    std::string fileName;
    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return fileName;
    }

    std::string timestamp = GetFileTimestamp();
    if (timestamp.empty())
    {
        return fileName;
    }

    // Set the name of the output .dot file.
    fs::path dumpPath = dumpDir;
    fs::path tempFilePath = dumpPath / (timestamp + "_networkgraph.dot");
    fileName = tempFilePath.string();

    ALOGV("Exporting the optimized network graph to file: %s", fileName.c_str());

    // Write the network graph to a dot file.
    std::ofstream fileStream;
    fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        ALOGW("Could not open file %s for writing", fileName.c_str());
        return fileName;
    }

    if (optimizedNetwork.SerializeToDot(fileStream) != armnn::Status::Success)
    {
        ALOGW("An error occurred when writing to file %s", fileName.c_str());
    }
    return fileName;
}

std::string SerializeNetwork(const armnn::INetwork& network, const std::string& dumpDir)
{
    std::string fileName;
    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return fileName;
    }

    std::string timestamp = GetFileTimestamp();
    if (timestamp.empty())
    {
        return fileName;
    }

    auto serializer(armnnSerializer::ISerializer::Create());

    // Serialize the Network
    serializer->Serialize(network);

    // Set the name of the output .armnn file.
    fs::path dumpPath = dumpDir;
    fs::path tempFilePath = dumpPath / (timestamp + "_network.armnn");
    fileName = tempFilePath.string();

    // Save serialized network to a file
    std::ofstream serializedFile(fileName, std::ios::out | std::ios::binary);
    bool serialized = serializer->SaveSerializedToStream(serializedFile);
    if (!serialized)
    {
        ALOGW("An error occurred when serializing to file %s", fileName.c_str());
    }
    return fileName;
}

bool IsDynamicTensor(const armnn::TensorInfo& tensorInfo)
{
    if (tensorInfo.GetShape().GetDimensionality() == armnn::Dimensionality::NotSpecified)
    {
        return true;
    }
    // Account for the usage of the TensorShape empty constructor
    if (tensorInfo.GetNumDimensions() == 0)
    {
        return true;
    }
    return !tensorInfo.GetShape().AreAllDimensionsSpecified();
}

bool AreDynamicTensorsSupported()
{
#if defined(ARMNN_ANDROID_NN_V1_3)
    return true;
#else
    return false;
#endif
}

std::string GetFileTimestamp()
{
    // used to get a timestamp to name diagnostic files (the ArmNN serialized graph
    // and getSupportedOperations.txt files)
    timespec ts;
    int iRet = clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    std::stringstream ss;
    if (iRet == 0)
    {
        ss << std::to_string(ts.tv_sec) << "_" << std::to_string(ts.tv_nsec);
    }
    else
    {
        ALOGW("clock_gettime failed with errno %s : %s", std::to_string(errno).c_str(), std::strerror(errno));
    }
    return ss.str();
}

void RenameExportedFiles(const std::string& existingSerializedFileName,
                         const std::string& existingDotFileName,
                         const std::string& dumpDir,
                         const armnn::NetworkId networkId)
{
    if (dumpDir.empty())
    {
        return;
    }
    RenameFile(existingSerializedFileName, std::string("_network.armnn"), dumpDir, networkId);
    RenameFile(existingDotFileName, std::string("_networkgraph.dot"), dumpDir, networkId);
}

void RenameFile(const std::string& existingName,
                const std::string& extension,
                const std::string& dumpDir,
                const armnn::NetworkId networkId)
{
    if (existingName.empty() || dumpDir.empty())
    {
        return;
    }

    fs::path dumpPath = dumpDir;
    const fs::path newFileName = dumpPath / (std::to_string(networkId) + extension);
    int iRet = rename(existingName.c_str(), newFileName.c_str());
    if (iRet != 0)
    {
        std::stringstream ss;
        ss << "rename of [" << existingName << "] to [" << newFileName << "] failed with errno "
           << std::to_string(errno) << " : " << std::strerror(errno);
        ALOGW(ss.str().c_str());
    }
}

void CommitPools(std::vector<::android::nn::RunTimePoolInfo>& memPools)
{
    if (memPools.empty())
    {
        return;
    }
    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (auto& pool : memPools)
    {
        // Type android::nn::RunTimePoolInfo has changed between Android P & Q and Android R, where
        // update() has been removed and flush() added.
#if defined(ARMNN_ANDROID_R) || defined(ARMNN_ANDROID_S) // Use the new Android implementation.
        pool.flush();
#else
        pool.update();
#endif
    }
}
} // namespace armnn_driver
