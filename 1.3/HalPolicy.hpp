//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../ConversionUtils.hpp"
#include "../ConversionUtils_1_2.hpp"
#include "../ConversionUtils_1_3.hpp"

#include <HalInterfaces.h>

#include <armnn/Types.hpp>

namespace V1_3 = ::android::hardware::neuralnetworks::V1_3;

namespace armnn_driver
{
namespace hal_1_3
{

class HalPolicy
{
public:
    using Model                     = V1_3::Model;
    using Operand                   = V1_3::Operand;
    using OperandLifeTime           = V1_3::OperandLifeTime;
    using OperandType               = V1_3::OperandType;
    using Operation                 = V1_3::Operation;
    using OperationType             = V1_3::OperationType;
    using ExecutionCallback         = V1_3::IExecutionCallback;
    using getSupportedOperations_cb = V1_3::IDevice::getSupportedOperations_1_3_cb;
    using ErrorStatus               = V1_3::ErrorStatus;

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

    static bool ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation);

    static bool ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        armnn::UnaryOperation unaryOperation);

    static bool ConvertElu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFill(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFloor(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGather(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertHardSwish(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data);

    static bool ConvertLogicalBinary(const Operation& operation,
                                     const Model& model,
                                     ConversionData& data,
                                     armnn::LogicalBinaryOperation logicalOperation);

    static bool ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMean(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPad(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantizedLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertRank(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReduce(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ReduceOperation reduceOperation);

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

} // namespace hal_1_3
} // namespace armnn_driver
