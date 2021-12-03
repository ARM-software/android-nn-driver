//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

namespace armnn_driver
{
namespace hal_1_3
{

using namespace armnn;

namespace
{

} // anonymouse namespace

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case V1_3::OperationType::ABS:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Abs);
        case V1_3::OperationType::ADD:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Add);
        case V1_3::OperationType::ARGMAX:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Max);
        case V1_3::OperationType::ARGMIN:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Min);
        case V1_3::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_3::OperationType::BATCH_TO_SPACE_ND:
            return ConvertBatchToSpaceNd(operation, model, data);
        case V1_3::OperationType::CAST:
            return ConvertCast(operation, model, data);
        case V1_3::OperationType::CHANNEL_SHUFFLE:
            return ConvertChannelShuffle(operation, model, data);
        case V1_3::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_3::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_3::OperationType::DEPTH_TO_SPACE:
            return ConvertDepthToSpace(operation, model, data);
        case V1_3::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case V1_3::OperationType::DEQUANTIZE:
            return ConvertDequantize(operation, model, data);
        case V1_3::OperationType::DIV:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Div);
        case V1_3::OperationType::ELU:
            return ConvertElu(operation, model, data);
        case V1_3::OperationType::EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::Equal);
        case V1_3::OperationType::EXP:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Exp);
        case V1_3::OperationType::EXPAND_DIMS:
            return ConvertExpandDims(operation, model, data);
        case V1_3::OperationType::FILL:
            return ConvertFill(operation, model, data);
        case V1_3::OperationType::FLOOR:
            return ConvertFloor(operation, model, data);
        case V1_3::OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation, model, data);
        case V1_3::OperationType::GATHER:
            return ConvertGather(operation, model, data);
        case V1_3::OperationType::GREATER:
            return ConvertComparison(operation, model, data, ComparisonOperation::Greater);
        case V1_3::OperationType::GREATER_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::GreaterOrEqual);
        case V1_3::OperationType::GROUPED_CONV_2D:
            return ConvertGroupedConv2d(operation, model, data);
        case V1_3::OperationType::HARD_SWISH:
            return ConvertHardSwish(operation, model, data);
        case V1_3::OperationType::INSTANCE_NORMALIZATION:
            return ConvertInstanceNormalization(operation, model, data);
        case V1_3::OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation, model, data);
        case V1_3::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case V1_3::OperationType::LESS:
            return ConvertComparison(operation, model, data, ComparisonOperation::Less);
        case V1_3::OperationType::LESS_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::LessOrEqual);
        case V1_3::OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation, model, data);
        case V1_3::OperationType::LOG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Log);
        case V1_3::OperationType::LOGICAL_AND:
            return ConvertLogicalBinary(operation, model, data, LogicalBinaryOperation::LogicalAnd);
        case V1_3::OperationType::LOGICAL_NOT:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::LogicalNot);
        case V1_3::OperationType::LOGICAL_OR:
            return ConvertLogicalBinary(operation, model, data, LogicalBinaryOperation::LogicalOr);
        case V1_3::OperationType::LOGISTIC:
            return ConvertLogistic(operation, model, data);
        case V1_3::OperationType::LOG_SOFTMAX:
            return ConvertLogSoftmax(operation, model, data);
        case V1_3::OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        case V1_3::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case V1_3::OperationType::MAXIMUM:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Maximum);
        case V1_3::OperationType::MEAN:
            return ConvertMean(operation, model, data);
        case V1_3::OperationType::MINIMUM:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Minimum);
        case V1_3::OperationType::MUL:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Mul);
        case V1_3::OperationType::NEG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Neg);
        case V1_3::OperationType::NOT_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::NotEqual);
        case V1_3::OperationType::PAD:
            return ConvertPad(operation, model, data);
        case V1_3::OperationType::PAD_V2:
            return ConvertPadV2(operation, model, data);
        case V1_3::OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case V1_3::OperationType::QUANTIZE:
            return ConvertQuantize(operation, model, data);
        case V1_3::OperationType::QUANTIZED_LSTM:
            return ConvertQuantizedLstm(operation, model, data);
        case V1_3::OperationType::QUANTIZED_16BIT_LSTM:
            return ConvertQuantized16BitLstm(operation, model, data);
        case V1_3::OperationType::RANK:
            return ConvertRank(operation, model, data);
        case V1_3::OperationType::REDUCE_MAX:
            return ConvertReduce(operation, model, data, ReduceOperation::Max);
        case V1_3::OperationType::REDUCE_MIN:
            return ConvertReduce(operation, model, data, ReduceOperation::Min);
        case V1_3::OperationType::REDUCE_PROD:
            return ConvertReduce(operation, model, data, ReduceOperation::Prod);
        case V1_3::OperationType::REDUCE_SUM:
            return ConvertReduce(operation, model, data, ReduceOperation::Sum);
        case V1_3::OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case V1_3::OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case V1_3::OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case V1_3::OperationType::RESHAPE:
            return ConvertReshape(operation, model, data);
        case V1_3::OperationType::RESIZE_BILINEAR:
            return ConvertResize(operation, model, data, ResizeMethod::Bilinear);
        case V1_3::OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResize(operation, model, data, ResizeMethod::NearestNeighbor);
        case V1_3::OperationType::RSQRT:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Rsqrt);
        case V1_3::OperationType::SIN:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Sin);
        case V1_3::OperationType::SOFTMAX:
            return ConvertSoftmax(operation, model, data);
        case V1_3::OperationType::SPACE_TO_BATCH_ND  :
            return ConvertSpaceToBatchNd(operation, model, data);
        case V1_3::OperationType::SPACE_TO_DEPTH:
            return ConvertSpaceToDepth(operation, model, data);
        case V1_3::OperationType::SQRT:
            return ConvertSqrt(operation, model, data);
        case V1_3::OperationType::SQUEEZE:
            return ConvertSqueeze(operation, model, data);
        case V1_3::OperationType::STRIDED_SLICE:
            return ConvertStridedSlice(operation, model, data);
        case V1_3::OperationType::SUB:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Sub);
        case V1_3::OperationType::TRANSPOSE:
            return ConvertTranspose(operation, model, data);
        case V1_3::OperationType::TRANSPOSE_CONV_2D:
            return ConvertTransposeConv2d(operation, model, data);
        case V1_3::OperationType::TANH:
            return ConvertTanH(operation, model, data);
        case V1_3::OperationType::UNIDIRECTIONAL_SEQUENCE_LSTM:
            return ConvertUnidirectionalSequenceLstm(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ConvertArgMinMax(const V1_3::Operation& operation,
                                 const V1_3::Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction)
{
    ALOGV("hal_1_3::HalPolicy::ConvertArgMinMax()");
    return ::ConvertArgMinMax<hal_1_3::HalPolicy>(operation, model, data, argMinMaxFunction);
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertAveragePool2d()");
    return ConvertPooling2d<hal_1_3::HalPolicy>(operation, __func__, PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertBatchToSpaceNd()");
    return ::ConvertBatchToSpaceNd<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertCast(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertCast()");
    return ::ConvertCast<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertChannelShuffle(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertChannelShuffle()");
    return ::ConvertChannelShuffle<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  ComparisonOperation comparisonOperation)
{
    ALOGV("hal_1_3::HalPolicy::ConvertComparison()");
    return ::ConvertComparison_1_2<hal_1_3::HalPolicy>(operation, model, data, comparisonOperation);
}

bool HalPolicy::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertConcatenation()");
    return ::ConvertConcatenation<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertConv2d()");
    return ::ConvertConv2d_1_2<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertDepthToSpace()");
    return ::ConvertDepthToSpace<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertDepthwiseConv2d()");
    return ::ConvertDepthwiseConv2d_1_2<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertDequantize()");
    return ::ConvertDequantize_1_2<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         BinaryOperation binaryOperation)
{
    ALOGV("hal_1_3::HalPolicy::ConvertElementwiseBinary()");
    return ::ConvertElementwiseBinary<hal_1_3::HalPolicy>(operation, model, data, binaryOperation);
}

bool HalPolicy::ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        UnaryOperation unaryOperation)
{
    ALOGV("hal_1_3::HalPolicy::ConvertElementwiseUnary()");
    return ::ConvertElementwiseUnary<hal_1_3::HalPolicy>(operation, model, data, unaryOperation);
}

bool HalPolicy::ConvertElu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertElu()");
    return ::ConvertElu<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertExpandDims()");
    return ::ConvertExpandDims<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFill(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertFill()");
    return ::ConvertFill<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertFloor()");
    return ::ConvertFloor<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertFullyConnected()");
    return ::ConvertFullyConnected<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertGather(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertGather()");
    return ::ConvertGather<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertGroupedConv2d()");
    return ::ConvertGroupedConv2d<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertHardSwish(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertHardSwish()");
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::HardSwish;

    return ::ConvertToActivation<hal_1_3::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertInstanceNormalization()");
    return ::ConvertInstanceNormalization<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertL2Normalization()");
    return ::ConvertL2Normalization<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertL2Pool2d()");
    return ConvertPooling2d<hal_1_3::HalPolicy>(operation, __func__, PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertLocalResponseNormalization()");
    return ::ConvertLocalResponseNormalization<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogicalBinary(const Operation& operation,
                                     const Model& model,
                                     ConversionData& data,
                                     armnn::LogicalBinaryOperation logicalOperation)
{
    ALOGV("hal_1_3::HalPolicy::ConvertLogicalBinary()");
    return ::ConvertLogicalBinary<hal_1_3::HalPolicy>(operation, model, data, logicalOperation);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertLogistic()");
    return ::ConvertLogistic<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertLogSoftmax()");
    return ::ConvertLogSoftmax<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertLstm()");
    return ::ConvertLstm<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertMaxPool2d()");
    return ConvertPooling2d<hal_1_3::HalPolicy>(operation, __func__, PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertMean()");
    return ::ConvertMean<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPad(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertPad()");
    return ::ConvertPad<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertPadV2()");
    return ::ConvertPadV2<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertPrelu()");
    return ::ConvertPrelu<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertQuantize()");
    return ::ConvertQuantize<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertQuantizedLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertQuantizedLstm()");
    return ::ConvertQuantizedLstm<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertQuantized16BitLstm()");
    return ::ConvertQuantized16BitLstm<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertRank(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertRank()");
    return ::ConvertRank<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReduce(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ReduceOperation reduceOperation)
{
    ALOGV("hal_1_3::HalPolicy::ConvertReduce()");
    return ::ConvertReduce<hal_1_3::HalPolicy>(operation, model, data, reduceOperation);
}

bool HalPolicy::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertReLu()");
    return ::ConvertReLu<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertReLu1()");
    return ::ConvertReLu1<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertReLu6()");
    return ::ConvertReLu6<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertReshape()");
    return ::ConvertReshape<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ResizeMethod resizeMethod)
{
    ALOGV("hal_1_3::HalPolicy::ConvertResize()");
    return ::ConvertResize<hal_1_3::HalPolicy>(operation, model, data, resizeMethod);
}

bool HalPolicy::ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertSpaceToBatchNd()");
    return ::ConvertSpaceToBatchNd<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertSpaceToDepth()");
    return ::ConvertSpaceToDepth<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertSoftmax()");
    return ::ConvertSoftmax<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertTanH()");
    return ::ConvertTanH<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertTransposeConv2d()");
    return ::ConvertTransposeConv2d<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertSqrt()");
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Sqrt;

    return ::ConvertToActivation<hal_1_3::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertSqueeze()");
    return ::ConvertSqueeze<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertStridedSlice()");
    return ::ConvertStridedSlice<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertTranspose()");
    return ::ConvertTranspose<hal_1_3::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertUnidirectionalSequenceLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_3::HalPolicy::ConvertUnidirectionalSequenceLstm()");
    return ::ConvertUnidirectionalSequenceLstm<hal_1_3::HalPolicy>(operation, model, data);
}

} // namespace hal_1_3
} // namespace armnn_driver
