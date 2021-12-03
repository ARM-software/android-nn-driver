//
// Copyright © 2017-2019,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "Utils.hpp"

#include "../1.0/HalPolicy.hpp"

namespace
{
static std::vector<V1_0::OperationType> opsEquivalentInV10({
    V1_0::OperationType::ADD,
    V1_0::OperationType::AVERAGE_POOL_2D,
    V1_0::OperationType::CONCATENATION,
    V1_0::OperationType::CONV_2D,
    V1_0::OperationType::DEPTH_TO_SPACE,
    V1_0::OperationType::DEPTHWISE_CONV_2D,
    V1_0::OperationType::DEQUANTIZE,
    V1_0::OperationType::FLOOR,
    V1_0::OperationType::FULLY_CONNECTED,
    V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION,
    V1_0::OperationType::LOGISTIC,
    V1_0::OperationType::LSTM,
    V1_0::OperationType::L2_NORMALIZATION,
    V1_0::OperationType::L2_POOL_2D,
    V1_0::OperationType::MAX_POOL_2D,
    V1_0::OperationType::MUL,
    V1_0::OperationType::RELU,
    V1_0::OperationType::RELU1,
    V1_0::OperationType::RELU6,
    V1_0::OperationType::SOFTMAX,
    V1_0::OperationType::SPACE_TO_DEPTH,
    V1_0::OperationType::TANH,
    V1_0::OperationType::RESHAPE,
    V1_0::OperationType::RESIZE_BILINEAR,
});

bool CompliantWithVersion10(const V1_1::Operation & operation)
{
    std::vector<V1_0::OperationType>::iterator it;
    it = std::find(opsEquivalentInV10.begin(), opsEquivalentInV10.end(),
                   static_cast<V1_0::OperationType>(operation.type));

    if(it != opsEquivalentInV10.end())
    {
        return true;
    }
    return false;
}

V1_0::Operation ConvertOperationToVersion10(const V1_1::Operation & operation)
{
    V1_0::Operation v10Operation;
    v10Operation.type = static_cast<V1_0::OperationType>(operation.type);
    v10Operation.inputs = operation.inputs;
    v10Operation.outputs = operation.outputs;
    return v10Operation;
}
}

namespace armnn_driver
{
namespace hal_1_1
{

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    if (CompliantWithVersion10(operation))
    {
        hal_1_0::HalPolicy::Operation v10Operation = ConvertOperationToVersion10(operation);
        hal_1_0::HalPolicy::Model v10Model = convertToV1_0(model);

        return hal_1_0::HalPolicy::ConvertOperation(v10Operation, v10Model, data);
    }
    else
    {
        switch (operation.type)
        {
            case V1_1::OperationType::DIV:
                return ConvertElementwiseBinary(operation, model, data, armnn::BinaryOperation::Div);
            case V1_1::OperationType::SUB:
                return ConvertElementwiseBinary(operation, model, data, armnn::BinaryOperation::Sub);
            case V1_1::OperationType::MEAN:
                return ConvertMean(operation, model, data);
            case V1_1::OperationType::PAD:
                return ConvertPad(operation, model, data);
            case V1_1::OperationType::SPACE_TO_BATCH_ND:
                return ConvertSpaceToBatchNd(operation, model, data);
            case V1_1::OperationType::SQUEEZE:
                return ConvertSqueeze(operation, model, data);
            case V1_1::OperationType::STRIDED_SLICE:
                return ConvertStridedSlice(operation, model, data);
            case V1_1::OperationType::TRANSPOSE:
                return ConvertTranspose(operation, model, data);
            case V1_1::OperationType::BATCH_TO_SPACE_ND:
                return ConvertBatchToSpaceNd(operation, model, data);
            default:
                return Fail("%s: Operation type %s not supported in ArmnnDriver",
                            __func__, toString(operation.type).c_str());
        }
    }
}

bool HalPolicy::ConvertElementwiseBinary(const Operation& operation,
                                         const Model& model,
                                         ConversionData& data,
                                         armnn::BinaryOperation binaryOperation)
{
    ALOGV("hal_1_1::HalPolicy::ConvertElementwiseBinary()");
    return ::ConvertElementwiseBinary<hal_1_1::HalPolicy>(operation, model, data, binaryOperation);
}

bool HalPolicy::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertMean()");
    return ::ConvertMean<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertPad(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertPad()");
    return ::ConvertPad<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertSpaceToBatchNd()");
    return ::ConvertSpaceToBatchNd<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertSqueeze()");
    return ::ConvertSqueeze<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertStridedSlice()");
    return ::ConvertStridedSlice<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertTranspose()");
    return ::ConvertTranspose<hal_1_1::HalPolicy>(operation, model, data);
}

bool HalPolicy::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    ALOGV("hal_1_1::HalPolicy::ConvertBatchToSpaceNd()");
    return ::ConvertBatchToSpaceNd<hal_1_1::HalPolicy>(operation, model, data);
}

} // namespace hal_1_1
} // namespace armnn_driver
