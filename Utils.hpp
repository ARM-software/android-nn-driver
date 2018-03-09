//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include <armnn/ArmNN.hpp>
#include <CpuExecutor.h>

#include <vector>
#include <string>

namespace armnn_driver
{

extern const armnn::PermutationVector g_DontPermute;

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
armnn::TensorInfo GetTensorInfoForOperand(const Operand& operand);

std::string GetOperandSummary(const Operand& operand);
std::string GetModelSummary(const Model& model);

void DumpTensor(const std::string& dumpDir,
    const std::string& requestName,
    const std::string& tensorName,
    const armnn::ConstTensor& tensor);

}
