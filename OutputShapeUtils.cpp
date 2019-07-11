//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputShapeUtils.hpp"

#include <algorithm>
#include <vector>

namespace armnn_driver
{

using namespace armnn;

bool IsDynamicOutput(const TensorInfo& outputInfo)
{
    return outputInfo.GetNumElements() == 0u;
}

TensorShape InferPadOutputShape(const TensorShape& inputShape,
                                const std::vector<std::pair<unsigned int, unsigned int>>& padList)
{
    const unsigned int numDims = inputShape.GetNumDimensions();

    std::vector<unsigned int> outputDims;
    TensorShape outputShape = TensorShape(numDims);
    for (unsigned int dim = 0; dim < numDims; ++dim)
    {
        unsigned int dimSize = inputShape[dim];
        const std::pair<unsigned int, unsigned int>& dimPadding = padList[dim];
        dimSize += dimPadding.first;
        dimSize += dimPadding.second;
        outputShape[dim] = dimSize;
    }
    return outputShape;
}

TensorShape InferPreluOutputShape(const TensorShape& inputShape, const TensorShape& alphaShape)
{
    // NOTE: The inferred PReLU output size will be the maximum size along each dimension
    // of input and alpha, starting with the trailing dimensions, and working its way forward.
    //
    // Example: inputShape={4, 1, 2}, alphaShape={5, 4, 3, 1} => outputShape={5, 4, 3, 2}

    const unsigned int numInputDims = inputShape.GetNumDimensions();
    const unsigned int numAlphaDims = alphaShape.GetNumDimensions();

    const unsigned int maxNumDims = std::max(numInputDims, numAlphaDims);

    TensorShape outputShape = TensorShape(maxNumDims);
    for (unsigned int reverseIdx = 1u; reverseIdx <= maxNumDims; ++reverseIdx)
    {
        const int inputIdx = numInputDims - reverseIdx;
        const int alphaIdx = numAlphaDims - reverseIdx;

        const unsigned int inputDimSize = inputIdx >= 0 ? inputShape[inputIdx] : 0u;
        const unsigned int alphaDimSize = alphaIdx >= 0 ? alphaShape[alphaIdx] : 0u;

        const unsigned int outputIdx = maxNumDims - reverseIdx;
        outputShape[outputIdx] = std::max(inputDimSize, alphaDimSize);
    }

    return outputShape;
}

} // namespace armnn_driver