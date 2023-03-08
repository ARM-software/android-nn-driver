//
// Copyright © 2017-2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <armnn/ArmNN.hpp>

#include <CpuExecutor.h>
#include <HalInterfaces.h>
#include <NeuralNetworks.h>
#include <Utils.h>

#include <fmt/format.h>

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;
namespace V1_1 = ::android::hardware::neuralnetworks::V1_1;

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
namespace V1_3 = ::android::hardware::neuralnetworks::V1_3;
#endif

namespace armnn_driver
{

#ifdef ARMNN_ANDROID_R
using DataLocation = ::android::nn::hal::DataLocation;
#endif

inline const V1_0::Model&    getMainModel(const V1_0::Model& model) { return model; }
inline const V1_1::Model&    getMainModel(const V1_1::Model& model) { return model; }

#if defined (ARMNN_ANDROID_NN_V1_2) || defined (ARMNN_ANDROID_NN_V1_3)
inline const V1_2::Model&    getMainModel(const V1_2::Model& model) { return model; }
#endif

#ifdef ARMNN_ANDROID_NN_V1_3
inline const V1_3::Subgraph& getMainModel(const V1_3::Model& model) { return model.main; }
#endif

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
void SwizzleAndroidNn4dTensorToArmNn(armnn::TensorInfo& tensor, const void* input, void* output,
                                     const armnn::PermutationVector& mappings);

/// Returns a pointer to a specific location in a pool
void* GetMemoryFromPool(V1_0::DataLocation location,
                        const std::vector<android::nn::RunTimePoolInfo>& memPools);

/// Can throw UnsupportedOperand
armnn::TensorInfo GetTensorInfoForOperand(const V1_0::Operand& operand);

std::string GetOperandSummary(const V1_0::Operand& operand);

// Returns true for any quantized data type, false for the rest.
bool isQuantizedOperand(const V1_0::OperandType& operandType);

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3) // Using ::android::hardware::neuralnetworks::V1_2
armnn::TensorInfo GetTensorInfoForOperand(const V1_2::Operand& operand);

std::string GetOperandSummary(const V1_2::Operand& operand);

bool isQuantizedOperand(const V1_2::OperandType& operandType);
#endif

#ifdef ARMNN_ANDROID_NN_V1_3 // Using ::android::hardware::neuralnetworks::V1_3
armnn::TensorInfo GetTensorInfoForOperand(const V1_3::Operand& operand);

std::string GetOperandSummary(const V1_3::Operand& operand);

bool isQuantizedOperand(const V1_3::OperandType& operandType);
#endif

template <typename HalModel>
std::string GetModelSummary(const HalModel& model)
{
    std::stringstream result;

    result << getMainModel(model).inputIndexes.size() << " input(s), "
           << getMainModel(model).operations.size() << " operation(s), "
           << getMainModel(model).outputIndexes.size() << " output(s), "
           << getMainModel(model).operands.size() << " operand(s) "
           << std::endl;

    result << "Inputs: ";
    for (uint32_t i = 0; i < getMainModel(model).inputIndexes.size(); i++)
    {
        result << GetOperandSummary(getMainModel(model).operands[getMainModel(model).inputIndexes[i]]) << ", ";
    }
    result << std::endl;

    result << "Operations: ";
    for (uint32_t i = 0; i < getMainModel(model).operations.size(); i++)
    {
        result << toString(getMainModel(model).operations[i].type).c_str() << ", ";
    }
    result << std::endl;

    result << "Outputs: ";
    for (uint32_t i = 0; i < getMainModel(model).outputIndexes.size(); i++)
    {
        result << GetOperandSummary(getMainModel(model).operands[getMainModel(model).outputIndexes[i]]) << ", ";
    }
    result << std::endl;

    return result.str();
}

template <typename TensorType>
void DumpTensor(const std::string& dumpDir,
                const std::string& requestName,
                const std::string& tensorName,
                const TensorType& tensor);

void DumpJsonProfilingIfRequired(bool gpuProfilingEnabled,
                                 const std::string& dumpDir,
                                 armnn::NetworkId networkId,
                                 const armnn::IProfiler* profiler);

std::string ExportNetworkGraphToDotFile(const armnn::IOptimizedNetwork& optimizedNetwork,
                                        const std::string& dumpDir);

std::string SerializeNetwork(const armnn::INetwork& network,
                             const std::string& dumpDir,
                             std::vector<uint8_t>& dataCacheData,
                             bool dataCachingActive = true);

void RenameExportedFiles(const std::string& existingSerializedFileName,
                         const std::string& existingDotFileName,
                         const std::string& dumpDir,
                         const armnn::NetworkId networkId);

void RenameFile(const std::string& existingName,
                const std::string& extension,
                const std::string& dumpDir,
                const armnn::NetworkId networkId);

/// Checks if a tensor info represents a dynamic tensor
bool IsDynamicTensor(const armnn::TensorInfo& outputInfo);

/// Checks for ArmNN support of dynamic tensors.
bool AreDynamicTensorsSupported(void);

std::string GetFileTimestamp();

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)
inline V1_2::OutputShape ComputeShape(const armnn::TensorInfo& info)
{
    V1_2::OutputShape shape;

    armnn::TensorShape tensorShape = info.GetShape();
    // Android will expect scalars as a zero dimensional tensor
    if(tensorShape.GetDimensionality() == armnn::Dimensionality::Scalar)
    {
         shape.dimensions = android::hardware::hidl_vec<uint32_t>{};
    }
    else
    {
        android::hardware::hidl_vec<uint32_t> dimensions;
        const unsigned int numDims = tensorShape.GetNumDimensions();
        dimensions.resize(numDims);
        for (unsigned int outputIdx = 0u; outputIdx < numDims; ++outputIdx)
        {
            dimensions[outputIdx] = tensorShape[outputIdx];
        }
        shape.dimensions = dimensions;
    }

    shape.isSufficient = true;

    return shape;
}
#endif

void CommitPools(std::vector<::android::nn::RunTimePoolInfo>& memPools);

template <typename ErrorStatus, typename Request>
ErrorStatus ValidateRequestArgument(const Request& request,
                                    const armnn::TensorInfo& tensorInfo,
                                    const V1_0::RequestArgument& requestArgument,
                                    std::string descString);
} // namespace armnn_driver
