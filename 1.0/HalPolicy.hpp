//
// Copyright © 2017-2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../ConversionUtils.hpp"

#include <HalInterfaces.h>

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;

namespace armnn_driver
{
namespace hal_1_0
{

class HalPolicy
{
public:
    using Model                     = V1_0::Model;
    using Operand                   = V1_0::Operand;
    using OperandLifeTime           = V1_0::OperandLifeTime;
    using OperandType               = V1_0::OperandType;
    using Operation                 = V1_0::Operation;
    using OperationType             = V1_0::OperationType;
    using getSupportedOperations_cb = V1_0::IDevice::getSupportedOperations_cb;
    using ErrorStatus               = V1_0::ErrorStatus;

    static bool ConvertOperation(const Operation& operation, const Model& model, ConversionData& data);

private:
    static bool ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation);

    static bool ConvertFloor(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data);

    static bool ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReshape(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertResizeBilinear(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTanH(const Operation& operation, const Model& model, ConversionData& data);

    static bool ValidateConv2dParameters(const Operation& operation);

    static bool ValidateDepthwiseConv2dParameters(const Operation& operation);
};

} // namespace hal_1_0
} // namespace armnn_driver
