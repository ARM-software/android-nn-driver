//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../ConversionUtils.hpp"

#include <HalInterfaces.h>

#include <armnn/Types.hpp>

namespace armnn_driver
{
namespace hal_1_2
{

class HalPolicy
{
public:
    using Model                     = V1_2::Model;
    using Operand                   = V1_2::Operand;
    using OperandLifeTime           = V1_0::OperandLifeTime;
    using OperandType               = V1_2::OperandType;
    using Operation                 = V1_2::Operation;
    using OperationType             = V1_2::OperationType;
    using ExecutionCallback         = V1_2::IExecutionCallback;
    using getSupportedOperations_cb = V1_2::IDevice::getSupportedOperations_1_2_cb;

    static bool ConvertOperation(const Operation& operation, const Model& model, ConversionData& data);

private:
    static bool ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaximum(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMinimum(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPad(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              armnn::ResizeMethod resizeMethod);

    static bool ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTanH(const Operation& operation, const Model& model, ConversionData& data);
};

} // namespace hal_1_2
} // namespace armnn_driver
