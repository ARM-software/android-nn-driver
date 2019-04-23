//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

#include "../ConversionUtils.hpp"

namespace armnn_driver
{

inline armnn::TensorShape FlattenFullyConnectedInput(const armnn::TensorShape &inputShape,
                                                     const armnn::TensorShape &weightsShape)
{
    if (inputShape.GetNumDimensions() > 2U)
    {
        unsigned int dim0 = inputShape[0];
        unsigned int dim1 = inputShape[1];

        for (unsigned int i = 2U; i < inputShape.GetNumDimensions(); ++i)
        {
            dim1 *= inputShape[i];
        }

        unsigned int divisor = weightsShape[1] / dim1;
        if(dim0 % divisor != 0)
        {
            throw std::runtime_error("Failed to deduce tensor shape");
        }

        return armnn::TensorShape({dim0 / divisor, dim1 * divisor});
    }
    else
    {
        return inputShape;
    }
}

}