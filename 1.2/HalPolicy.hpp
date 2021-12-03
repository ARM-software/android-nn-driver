//
// Copyright © 2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../ConversionUtils.hpp"
#include "../ConversionUtils_1_2.hpp"

#include <HalInterfaces.h>

#include <armnn/Types.hpp>

namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;

namespace armnn_driver
{
class DriverOptions;
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
    using ErrorStatus               = V1_0::ErrorStatus;
    using DeviceType                = V1_2::DeviceType;

    static DeviceType GetDeviceTypeFromOptions(const DriverOptions& options);

    static bool ConvertOperation(const Operation& operation, const Model& model, ConversionData& data);

private:
    static bool ConvertArgMinMax(const Operation& operation,
                                 const Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction);

    static bool ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertCast(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertChannelShuffle(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  armnn::ComparisonOperation comparisonOperation);

    static bool ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation);

    static bool ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        armnn::UnaryOperation unaryOperation);

    static bool ConvertFloor(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGather(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data);

    static bool ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMean(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPad(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReduce(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ReduceOperation reduce_operation);

    static bool ConvertReLu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReshape(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              armnn::ResizeMethod resizeMethod);

    static bool ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTanH(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertUnidirectionalSequenceLstm(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data);
};

} // namespace hal_1_2
} // namespace armnn_driver
