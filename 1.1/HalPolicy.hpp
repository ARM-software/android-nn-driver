//
// Copyright © 2017-2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../ConversionUtils.hpp"

#include <HalInterfaces.h>

namespace V1_1 = ::android::hardware::neuralnetworks::V1_1;

namespace armnn_driver
{
namespace hal_1_1
{

class HalPolicy
{
public:
    using Model                     = V1_1::Model;
    using Operand                   = V1_0::Operand;
    using OperandLifeTime           = V1_0::OperandLifeTime;
    using OperandType               = V1_0::OperandType;
    using Operation                 = V1_1::Operation;
    using OperationType             = V1_1::OperationType;
    using getSupportedOperations_cb = V1_1::IDevice::getSupportedOperations_1_1_cb;
    using ErrorStatus               = V1_0::ErrorStatus;

    static bool ConvertOperation(const Operation& operation, const Model& model, ConversionData& data);

private:
    static bool ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation);

    static bool ConvertMean(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertPad(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data);
    static bool ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data);
};

} // namespace hal_1_1
} // namespace armnn_driver
