//
// Copyright © 2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"
#include "DriverOptions.hpp"

namespace armnn_driver
{
namespace hal_1_2
{

using namespace armnn;

namespace
{

} // anonymous namespace

HalPolicy::DeviceType HalPolicy::GetDeviceTypeFromOptions(const DriverOptions& options)
{
        // Query backends list from the options
        auto backends = options.GetBackends();
        // Return first backend
        if(backends.size()>0)
        {
            const auto &first_backend = backends[0];
            if(first_backend.IsCpuAcc()||first_backend.IsCpuRef())
            {
                return V1_2::DeviceType::CPU;
            }
            else if(first_backend.IsGpuAcc())
            {
                return V1_2::DeviceType::GPU;
            }
            else
            {
                return V1_2::DeviceType::ACCELERATOR;
            }
        }
        else
        {
            return V1_2::DeviceType::CPU;
        }
}

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case V1_2::OperationType::ABS:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Abs);
        case V1_2::OperationType::ADD:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Add);
        case V1_2::OperationType::ARGMAX:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Max);
        case V1_2::OperationType::ARGMIN:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Min);
        case V1_2::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case V1_2::OperationType::BATCH_TO_SPACE_ND:
            return ConvertBatchToSpaceNd(operation, model, data);
        case V1_2::OperationType::CAST:
            return ConvertCast(operation, model, data);
        case V1_2::OperationType::CHANNEL_SHUFFLE:
            return ConvertChannelShuffle(operation, model, data);
        case V1_2::OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case V1_2::OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case V1_2::OperationType::DEPTH_TO_SPACE:
            return ConvertDepthToSpace(operation, model, data);
        case V1_2::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case V1_2::OperationType::DEQUANTIZE:
            return ConvertDequantize(operation, model, data);
        case V1_2::OperationType::DIV:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Div);
        case V1_2::OperationType::EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::Equal);
        case V1_2::OperationType::EXP:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Exp);
        case V1_2::OperationType::EXPAND_DIMS:
            return ConvertExpandDims(operation, model, data);
        case V1_2::OperationType::FLOOR:
            return ConvertFloor(operation, model, data);
        case V1_2::OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation, model, data);
        case V1_2::OperationType::GATHER:
            return ConvertGather(operation, model, data);
        case V1_2::OperationType::GREATER:
            return ConvertComparison(operation, model, data, ComparisonOperation::Greater);
        case V1_2::OperationType::GREATER_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::GreaterOrEqual);
        case V1_2::OperationType::GROUPED_CONV_2D:
            return ConvertGroupedConv2d(operation, model, data);
        case V1_2::OperationType::INSTANCE_NORMALIZATION:
            return ConvertInstanceNormalization(operation, model, data);
        case V1_2::OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation, model, data);
        case V1_2::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case V1_2::OperationType::LESS:
            return ConvertComparison(operation, model, data, ComparisonOperation::Less);
        case V1_2::OperationType::LESS_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::LessOrEqual);
        case V1_2::OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation, model, data);
        case V1_2::OperationType::LOG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Log);
        case V1_2::OperationType::LOGISTIC:
            return ConvertLogistic(operation, model, data);
        case V1_2::OperationType::LOG_SOFTMAX:
            return ConvertLogSoftmax(operation, model, data);
        case V1_2::OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        case V1_2::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case V1_2::OperationType::MAXIMUM:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Maximum);
        case V1_2::OperationType::MEAN:
            return ConvertMean(operation, model, data);
        case V1_2::OperationType::MINIMUM:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Minimum);
        case V1_2::OperationType::MUL:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Mul);
        case V1_2::OperationType::NEG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Neg);
        case V1_2::OperationType::NOT_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::NotEqual);
        case V1_2::OperationType::PAD:
            return ConvertPad(operation, model, data);
        case V1_2::OperationType::PAD_V2:
            return ConvertPadV2(operation, model, data);
        case V1_2::OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case V1_2::OperationType::QUANTIZE:
            return ConvertQuantize(operation, model, data);
        case V1_2::OperationType::QUANTIZED_16BIT_LSTM:
            return ConvertQuantized16BitLstm(operation, model, data);
        case V1_2::OperationType::REDUCE_MAX:
            return ConvertReduce(operation, model, data, ReduceOperation::Max);
        case V1_2::OperationType::REDUCE_MIN:
            return ConvertReduce(operation, model, data, ReduceOperation::Min);
        case V1_2::OperationType::REDUCE_PROD:
            return ConvertReduce(operation, model, data, ReduceOperation::Prod);
        case V1_2::OperationType::REDUCE_SUM:
            return ConvertReduce(operation, model, data, ReduceOperation::Sum);
        case V1_2::OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case V1_2::OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case V1_2::OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case V1_2::OperationType::RESHAPE:
            return ConvertReshape(operation, model, data);
        case V1_2::OperationType::RESIZE_BILINEAR:
            return ConvertResize(operation, model, data, ResizeMethod::Bilinear);
        case V1_2::OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResize(operation, model, data, ResizeMethod::NearestNeighbor);
        case V1_2::OperationType::RSQRT:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Rsqrt);
        case V1_2::OperationType::SIN:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Sin);
        case V1_2::OperationType::SOFTMAX:
            return ConvertSoftmax(operation, model, data);
        case V1_2::OperationType::SPACE_TO_BATCH_ND  :
            return ConvertSpaceToBatchNd(operation, model, data);
        case V1_2::OperationType::SPACE_TO_DEPTH:
            return ConvertSpaceToDepth(operation, model, data);
        case V1_2::OperationType::SQRT:
            return ConvertSqrt(operation, model, data);
        case V1_2::OperationType::SQUEEZE:
            return ConvertSqueeze(operation, model, data);
        case V1_2::OperationType::STRIDED_SLICE:
            return ConvertStridedSlice(operation, model, data);
        case V1_2::OperationType::SUB:
            return ConvertElementwiseBinary(operation, model, data, BinaryOperation::Sub);
        case V1_2::OperationType::TRANSPOSE:
            return ConvertTranspose(operation, model, data);
        case V1_2::OperationType::TRANSPOSE_CONV_2D:
            return ConvertTransposeConv2d(operation, model, data);
        case V1_2::OperationType::TANH:
            return ConvertTanH(operation, model, data);
        case V1_2::OperationType::UNIDIRECTIONAL_SEQUENCE_LSTM:
            return ConvertUnidirectionalSequenceLstm(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

bool HalPolicy::ConvertArgMinMax(const V1_2::Operation& operation,
                                 const V1_2::Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction)
{
    ALOGV("hal_1_2::HalPolicy::ConvertArgMinMax()");
    return ::ConvertArgMinMax<hal_1_2::HalPolicy>(operation, model, data, argMinMaxFunction);
}

bool HalPolicy::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertAveragePool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::Average, model, data);
}

bool HalPolicy::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertBatchToSpaceNd()");
    return ::ConvertBatchToSpaceNd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertCast(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertCast()");
    return ::ConvertCast<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertChannelShuffle(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertChannelShuffle()");
    return ::ConvertChannelShuffle<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  ComparisonOperation comparisonOperation)
{
    ALOGV("hal_1_2::HalPolicy::ConvertComparison()");
    return ::ConvertComparison_1_2<hal_1_2::HalPolicy>(operation, model, data, comparisonOperation);
}

bool HalPolicy::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertConcatenation()");
    return ::ConvertConcatenation<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertConv2d()");
    return ::ConvertConv2d_1_2<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDepthToSpace()");
    return ::ConvertDepthToSpace<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDepthwiseConv2d()");
    return ::ConvertDepthwiseConv2d_1_2<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertDequantize()");
    return ::ConvertDequantize_1_2<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         BinaryOperation binaryOperation)
{
    ALOGV("hal_1_2::HalPolicy::ConvertElementwiseBinary()");
    return ::ConvertElementwiseBinary<hal_1_2::HalPolicy>(operation, model, data, binaryOperation);
}

bool HalPolicy::ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        UnaryOperation unaryOperation)
{
    ALOGV("hal_1_2::HalPolicy::ConvertElementwiseUnary()");
    return ::ConvertElementwiseUnary<hal_1_2::HalPolicy>(operation, model, data, unaryOperation);
}

bool HalPolicy::ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertExpandDims()");
    return ::ConvertExpandDims<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertFloor()");
    return ::ConvertFloor<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertFullyConnected()");
    return ::ConvertFullyConnected<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertGather (const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertGather()");
    return ::ConvertGather<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertGroupedConv2d()");
    return ::ConvertGroupedConv2d<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertInstanceNormalization()");
    return ::ConvertInstanceNormalization<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertL2Normalization()");
    return ::ConvertL2Normalization<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertL2Pool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::L2, model, data);
}

bool HalPolicy::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLocalResponseNormalization()");
    return ::ConvertLocalResponseNormalization<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLogistic()");
    return ::ConvertLogistic<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLogSoftmax()");
    return ::ConvertLogSoftmax<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMaxPool2d()");
    return ConvertPooling2d<hal_1_2::HalPolicy>(operation, __func__, PoolingAlgorithm::Max, model, data);
}

bool HalPolicy::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertMean()");
    return ::ConvertMean<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPad(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPad()");
    return ::ConvertPad<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPadV2()");
    return ::ConvertPadV2<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertPrelu()");
    return ::ConvertPrelu<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertQuantize()");
    return ::ConvertQuantize<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertQuantized16BitLstm()");
    return ::ConvertQuantized16BitLstm<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReduce(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ReduceOperation reduceOperation)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReduce()");
    return ::ConvertReduce<hal_1_2::HalPolicy>(operation, model, data, reduceOperation);
}

bool HalPolicy::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu()");
    return ::ConvertReLu<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu1()");
    return ::ConvertReLu1<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReLu6()");
    return ::ConvertReLu6<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertReshape()");
    return ::ConvertReshape<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ResizeMethod resizeMethod)
{
    ALOGV("hal_1_2::HalPolicy::ConvertResize()");
    return ::ConvertResize<hal_1_2::HalPolicy>(operation, model, data, resizeMethod);
}

bool HalPolicy::ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSpaceToBatchNd()");
    return ::ConvertSpaceToBatchNd<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSpaceToDepth()");
    return ::ConvertSpaceToDepth<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSoftmax()");
    return ::ConvertSoftmax<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertTanH()");
    return ::ConvertTanH<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertLstm()");
    return ::ConvertLstm<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSqrt()");
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Sqrt;

    return ::ConvertToActivation<hal_1_2::HalPolicy>(operation, __func__, desc, model, data);
}

bool HalPolicy::ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertSqueeze()");
    return ::ConvertSqueeze<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertStridedSlice()");
    return ::ConvertStridedSlice<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertTranspose()");
    return ::ConvertTranspose<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertTransposeConv2d()");
    return ::ConvertTransposeConv2d<hal_1_2::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertUnidirectionalSequenceLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_2::HalPolicy::ConvertUnidirectionalSequenceLstm()");
    return ::ConvertUnidirectionalSequenceLstm<hal_1_2::HalPolicy>(operation, model, data);
}

} // namespace hal_1_2
} // namespace armnn_driver
