//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputShapeUtils.hpp"

#include <algorithm>
#include <vector>

namespace
{

using namespace armnn;

TensorShape CalculateMaxShape(const TensorShape& inShape0, const TensorShape& inShape1)
{
    // NOTE: The inferred output size will be the maximum size along each dimension
    // of inShape0 and inShape1, starting with the trailing dimensions, and working its way forward.
    //
    // Example: inShape0={4, 1, 2}, inShape1={5, 4, 3, 1} => outputShape={5, 4, 3, 2}

    const unsigned int numInput0Dims = inShape0.GetNumDimensions();
    const unsigned int numInput1Dims = inShape1.GetNumDimensions();

    const unsigned int maxNumDims = std::max(numInput0Dims, numInput1Dims);

    TensorShape outputShape = TensorShape(maxNumDims);
    for (unsigned int reverseIdx = 1u; reverseIdx <= maxNumDims; ++reverseIdx)
    {
        const int input0Idx = numInput0Dims - reverseIdx;
        const int input1Idx = numInput1Dims - reverseIdx;

        const unsigned int input0DimSize = input0Idx >= 0 ? inShape0[input0Idx] : 0u;
        const unsigned int input1DimSize = input1Idx >= 0 ? inShape1[input1Idx] : 0u;

        const unsigned int outputIdx = maxNumDims - reverseIdx;
        outputShape[outputIdx] = std::max(input0DimSize, input1DimSize);
    }

    return outputShape;
}

} // namespace annonymous


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
    return CalculateMaxShape(inputShape, alphaShape);
}

TensorShape InferSubOutputShape(const TensorShape& input0Shape, const TensorShape& input1Shape)
{
    return CalculateMaxShape(input0Shape, input1Shape);
}

} // namespace armnn_driver