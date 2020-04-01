//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ConversionUtils_1_2.hpp"

using Half = half_float::half;

namespace armnn_driver
{

using namespace armnn;
using namespace android::nn;

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertElu(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input0.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Determine data type of input tensor
    HalOperandType inputType;
    if (!GetOperandType<HalPolicy>(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Elu;

    // Read alpha
    if (inputType == HalOperandType::TENSOR_FLOAT16)
    {
        Half alpha;

        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT16, alpha, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_A = static_cast<float>(alpha);
    }
    else if (inputType == HalOperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::FLOAT32, desc.m_A, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    return ::ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

} // armnn_driver namespace