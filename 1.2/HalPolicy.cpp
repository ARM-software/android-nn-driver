//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "../1.0/HalPolicy.hpp"
#include "../1.1/HalPolicy.hpp"

namespace armnn_driver
{
namespace hal_1_2
{

bool HandledByV1_0(V1_2::OperationType operationType)
{
    switch (static_cast<V1_0::OperationType>(operationType))
    {
        case V1_0::OperationType::ADD:
        case V1_0::OperationType::AVERAGE_POOL_2D:
        case V1_0::OperationType::CONCATENATION:
        case V1_0::OperationType::DEPTH_TO_SPACE:
        case V1_0::OperationType::DEQUANTIZE:
        case V1_0::OperationType::EMBEDDING_LOOKUP:
        case V1_0::OperationType::FLOOR:
        case V1_0::OperationType::FULLY_CONNECTED:
        case V1_0::OperationType::HASHTABLE_LOOKUP:
        case V1_0::OperationType::L2_NORMALIZATION:
        case V1_0::OperationType::L2_POOL_2D:
        case V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case V1_0::OperationType::LOGISTIC:
        case V1_0::OperationType::LSH_PROJECTION:
        case V1_0::OperationType::LSTM:
        case V1_0::OperationType::MAX_POOL_2D:
        case V1_0::OperationType::MUL:
        case V1_0::OperationType::RELU:
        case V1_0::OperationType::RELU1:
        case V1_0::OperationType::RELU6:
        case V1_0::OperationType::RESHAPE:
        case V1_0::OperationType::RESIZE_BILINEAR:
        case V1_0::OperationType::RNN:
        case V1_0::OperationType::SOFTMAX:
        case V1_0::OperationType::SPACE_TO_DEPTH:
        case V1_0::OperationType::SVDF:
        case V1_0::OperationType::TANH:
        case V1_0::OperationType::OEM_OPERATION:
            return true;
        default:
            return false;
    }
}

bool HandledByV1_1(V1_2::OperationType operationType)
{
    if (HandledByV1_0(operationType))
    {
        return true;
    }
    switch (static_cast<V1_1::OperationType>(operationType))
    {
        case V1_1::OperationType::BATCH_TO_SPACE_ND:
        case V1_1::OperationType::DIV:
        case V1_1::OperationType::MEAN:
        case V1_1::OperationType::PAD:
        case V1_1::OperationType::SPACE_TO_BATCH_ND:
        case V1_1::OperationType::SQUEEZE:
        case V1_1::OperationType::STRIDED_SLICE:
        case V1_1::OperationType::SUB:
        case V1_1::OperationType::TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool HandledByV1_0(const V1_2::Operation& operation)
{
    return HandledByV1_0(operation.type);
}

bool HandledByV1_1(const V1_2::Operation& operation)
{
    return HandledByV1_1(operation.type);
}

V1_0::OperationType CastToV1_0(V1_2::OperationType type)
{
    return static_cast<V1_0::OperationType>(type);
}

V1_1::OperationType CastToV1_1(V1_2::OperationType type)
{
    return static_cast<V1_1::OperationType>(type);
}

V1_0::Operation ConvertToV1_0(const V1_2::Operation& operation)
{
    V1_0::Operation op;
    op.type = CastToV1_0(operation.type);
    op.inputs = operation.inputs;
    op.outputs = operation.outputs;
    return op;
}

V1_1::Operation ConvertToV1_1(const V1_2::Operation& operation)
{
    V1_1::Operation op;
    op.type = CastToV1_1(operation.type);
    op.inputs = operation.inputs;
    op.outputs = operation.outputs;
    return op;
}

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    if (HandledByV1_0(operation) && compliantWithV1_0(model))
    {
        hal_1_0::HalPolicy::Operation v10Operation = ConvertToV1_0(operation);
        hal_1_0::HalPolicy::Model v10Model = convertToV1_0(model);

        return hal_1_0::HalPolicy::ConvertOperation(v10Operation, v10Model, data);
    }
    else if (HandledByV1_1(operation) && compliantWithV1_1(model))
    {
        hal_1_1::HalPolicy::Operation v11Operation = ConvertToV1_1(operation);
        hal_1_1::HalPolicy::Model v11Model = convertToV1_1(model);

        return hal_1_1::HalPolicy::ConvertOperation(v11Operation, v11Model, data);
    }
    switch (operation.type)
    {
        case V1_2::OperationType::CONV_2D:
            return ConvertConv2d<Operand, OperandType, Operation, Model>(operation, model, data);
        case V1_2::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d<Operand, OperandType, Operation, Model>(operation, model, data);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

} // namespace hal_1_2
} // namespace armnn_driver