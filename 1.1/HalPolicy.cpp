//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HalPolicy.hpp"

#include "../1.0/HalPolicy.hpp"

namespace armnn_driver
{
namespace hal_1_1
{

bool HalPolicy::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    if (compliantWithV1_0(operation))
    {
        hal_1_0::HalPolicy::Operation v10Operation = convertToV1_0(operation);
        hal_1_0::HalPolicy::Model v10Model = convertToV1_0(model);

        return hal_1_0::HalPolicy::ConvertOperation(v10Operation, v10Model, data);
    }
    else
    {
        switch (operation.type)
        {
            case V1_1::OperationType::DIV:
                return ConvertDiv(operation, model, data);
            case V1_1::OperationType::SUB:
                return ConvertSub(operation, model, data);
            case V1_1::OperationType::MEAN:
                return ConvertMean(operation, model, data);
            default:
                return Fail("%s: Operation type %s not supported in ArmnnDriver",
                            __func__, toString(operation.type).c_str());
        }
    }
}

bool HalPolicy::ConvertDiv(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsDivisionSupported,
                          data.m_Compute,
                          input0.GetTensorInfo(),
                          input1.GetTensorInfo(),
                          outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddDivisionLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer, data);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer)
    {
        BroadcastTensor(input0, input1, startLayer, *data.m_Network);
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer, model, data);
    }

    return Fail("%s: ProcessActivation failed", __func__);
}

bool HalPolicy::ConvertSub(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsSubtractionSupported,
                          data.m_Compute,
                          input0.GetTensorInfo(),
                          input1.GetTensorInfo(),
                          outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddSubtractionLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer, data);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer)
    {
        BroadcastTensor(input0, input1, startLayer, *data.m_Network);
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer, model, data);
    }

    return Fail("%s: ProcessActivation failed", __func__);
}

bool HalPolicy::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();

    armnn::MeanDescriptor descriptor;

    const Operand* axisOperand = GetInputOperand(operation, 1, model);
    if (axisOperand)
    {
        std::vector<int32_t> axis;
        GetTensorInt32Values(*axisOperand, axis, model, data);
        unsigned int rank = inputInfo.GetNumDimensions();
        // convert the axis to unsigned int.
        for (auto& i : axis)
        {
            unsigned int unsignedAxis = (i + rank) % rank;
            if (std::find(descriptor.m_Axis.begin(), descriptor.m_Axis.end(), unsignedAxis) == descriptor.m_Axis.end())
            {
                descriptor.m_Axis.push_back(unsignedAxis);
            }
        }
    }

    int32_t keepDims;
    GetInputInt32(operation, 2, keepDims, model, data);
    if (keepDims > 0)
    {
        descriptor.m_KeepDims = true;
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (!IsLayerSupported(__func__,
                          armnn::IsMeanSupported,
                          data.m_Compute,
                          inputInfo,
                          outputInfo,
                          descriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddMeanLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

} // namespace hal_1_1
} // namespace armnn_driver