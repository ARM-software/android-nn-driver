//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <armnn/ArmNN.hpp>

#include <CpuExecutor.h>
#include <HalInterfaces.h>
#include <NeuralNetworks.h>

#include <boost/format.hpp>
#include <log/log.h>

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;
#endif

namespace armnn_driver
{

extern const armnn::PermutationVector g_DontPermute;

template <typename OperandType>
class UnsupportedOperand: public std::runtime_error
{
public:
    UnsupportedOperand(const OperandType type)
        : std::runtime_error("Operand type is unsupported")
        , m_type(type)
    {}

    OperandType m_type;
};

/// Swizzles tensor data in @a input according to the dimension mappings.
void SwizzleAndroidNn4dTensorToArmNn(const armnn::TensorInfo& tensor, const void* input, void* output,
                                     const armnn::PermutationVector& mappings);

/// Returns a pointer to a specific location in a pool
void* GetMemoryFromPool(DataLocation location,
                        const std::vector<android::nn::RunTimePoolInfo>& memPools);

/// Can throw UnsupportedOperand
armnn::TensorInfo GetTensorInfoForOperand(const V1_0::Operand& operand);

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2
armnn::TensorInfo GetTensorInfoForOperand(const V1_2::Operand& operand);
#endif

std::string GetOperandSummary(const V1_0::Operand& operand);

#ifdef ARMNN_ANDROID_NN_V1_2 // Using ::android::hardware::neuralnetworks::V1_2
std::string GetOperandSummary(const V1_2::Operand& operand);
#endif

template <typename HalModel>
std::string GetModelSummary(const HalModel& model)
{
    std::stringstream result;

    result << model.inputIndexes.size() << " input(s), " << model.operations.size() << " operation(s), " <<
        model.outputIndexes.size() << " output(s), " << model.operands.size() << " operand(s)" << std::endl;

    result << "Inputs: ";
    for (uint32_t i = 0; i < model.inputIndexes.size(); i++)
    {
        result << GetOperandSummary(model.operands[model.inputIndexes[i]]) << ", ";
    }
    result << std::endl;

    result << "Operations: ";
    for (uint32_t i = 0; i < model.operations.size(); i++)
    {
        result << toString(model.operations[i].type).c_str() << ", ";
    }
    result << std::endl;

    result << "Outputs: ";
    for (uint32_t i = 0; i < model.outputIndexes.size(); i++)
    {
        result << GetOperandSummary(model.operands[model.outputIndexes[i]]) << ", ";
    }
    result << std::endl;

    return result.str();
}

void DumpTensor(const std::string& dumpDir,
                const std::string& requestName,
                const std::string& tensorName,
                const armnn::ConstTensor& tensor);

void DumpJsonProfilingIfRequired(bool gpuProfilingEnabled,
                                 const std::string& dumpDir,
                                 armnn::NetworkId networkId,
                                 const armnn::IProfiler* profiler);

template <typename HalModel>
void ExportNetworkGraphToDotFile(const armnn::IOptimizedNetwork& optimizedNetwork,
                                 const std::string& dumpDir,
                                 const HalModel& model)
{
    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return;
    }

    // Get the memory address of the model and convert it to a hex string (of at least a '0' character).
    size_t modelAddress = uintptr_t(&model);
    std::stringstream ss;
    ss << std::uppercase << std::hex << std::setfill('0') << std::setw(1) << modelAddress;
    std::string modelAddressHexString = ss.str();

    // Set the name of the output .dot file.
    const std::string fileName = boost::str(boost::format("%1%/networkgraph_%2%.dot")
                                            % dumpDir
                                            % modelAddressHexString);

    ALOGV("Exporting the optimized network graph to file: %s", fileName.c_str());

    // Write the network graph to a dot file.
    std::ofstream fileStream;
    fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        ALOGW("Could not open file %s for writing", fileName.c_str());
        return;
    }

    if (optimizedNetwork.SerializeToDot(fileStream) != armnn::Status::Success)
    {
        ALOGW("An error occurred when writing to file %s", fileName.c_str());
    }
}

/// Checks if a tensor info represents a dynamic tensor
bool IsDynamicTensor(const armnn::TensorInfo& outputInfo);

} // namespace armnn_driver
