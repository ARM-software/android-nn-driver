//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

#include "../ConversionUtils.hpp"

namespace armnn_driver
{

inline armnn::TensorShape FlattenFullyConnectedInput(const armnn::TensorShape& inputShape,
                                                     const armnn::TensorShape& weightsShape)
{
    if (inputShape.GetNumDimensions() > 2U)
    {
        unsigned int totalInputElements = inputShape.GetNumElements();
        unsigned int inputSize = weightsShape[1];

        unsigned int batchSize = totalInputElements / inputSize;

        if(totalInputElements % batchSize != 0)
        {
            throw std::runtime_error("Failed to deduce tensor shape");
        }

        return armnn::TensorShape({batchSize, inputSize});
    }
    else
    {
        return inputShape;
    }
}

inline bool VerifyFullyConnectedShapes(const armnn::TensorShape& inputShape,
                                       const armnn::TensorShape& weightsShape,
                                       const armnn::TensorShape& outputShape,
                                       bool  transposeWeightMatrix)
{
    unsigned int dimIdx = transposeWeightMatrix ? 0 : 1;
    return (inputShape[0] == outputShape[0] && weightsShape[dimIdx] == outputShape[1]);
}

}