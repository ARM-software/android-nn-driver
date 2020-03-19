//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include "Utils.hpp"
#include "Half.hpp"

#include <armnnUtils/Permute.hpp>

#include <armnn/Utils.hpp>

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
        SwizzleAndroidNn4dTensorToArmNn(tensor.GetShape(), input, output, armnn::GetDataTypeSize(dataType), mappings);
        break;
    default:
        ALOGW("Unknown armnn::DataType for swizzling");
        assert(0);
    }
}

void* GetMemoryFromPool(DataLocation location, const std::vector<android::nn::RunTimePoolInfo>& memPools)
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
    armnn::DataType type;

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

    armnn::TensorInfo ret(operand.dimensions.size(), operand.dimensions.data(), type);

    ret.SetQuantizationScale(operand.scale);
    ret.SetQuantizationOffset(operand.zeroPoint);

    return ret;
}

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2

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

    TensorInfo ret(operand.dimensions.size(), operand.dimensions.data(), type);
    if (perChannel)
    {
        // ExtraParams is expected to be of type channelQuant
        BOOST_ASSERT(operand.extraParams.getDiscriminator() ==
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

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2

std::string GetOperandSummary(const V1_2::Operand& operand)
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
    const std::string fileName = boost::str(boost::format("%1%/%2%_%3%.dump") % dumpDir % requestName % tensorName);

    std::ofstream fileStream;
    fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);

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

    BOOST_ASSERT(profiler);

    // Set the name of the output profiling file.
    const std::string fileName = boost::str(boost::format("%1%/%2%_%3%.json")
                                            % dumpDir
                                            % std::to_string(networkId)
                                            % "profiling");

    // Open the ouput file for writing.
    std::ofstream fileStream;
    fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);

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
    fileName = boost::str(boost::format("%1%/%2%_networkgraph.dot")
                          % dumpDir
                          % timestamp);

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

bool IsDynamicTensor(const armnn::TensorInfo& outputInfo)
{
    // Dynamic tensors have at least one 0-sized dimension
    return outputInfo.GetNumElements() == 0u;
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

void RenameGraphDotFile(const std::string& oldName, const std::string& dumpDir, const armnn::NetworkId networkId)
{
    if (dumpDir.empty())
    {
        return;
    }
    if (oldName.empty())
    {
        return;
    }
    const std::string newFileName = boost::str(boost::format("%1%/%2%_networkgraph.dot")
                                               % dumpDir
                                               % std::to_string(networkId));
    int iRet = rename(oldName.c_str(), newFileName.c_str());
    if (iRet != 0)
    {
        std::stringstream ss;
        ss << "rename of [" << oldName << "] to [" << newFileName << "] failed with errno " << std::to_string(errno)
           << " : " << std::strerror(errno);
        ALOGW(ss.str().c_str());
    }
}



} // namespace armnn_driver
