//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputShapeUtils.hpp"

#include <DataLayoutIndexed.hpp>

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

template<typename ConvolutionDescriptor>
TensorShape InferConvolution2dOutputShapeImpl(const TensorShape& inputShape,
                                              const TensorShape& kernelShape,
                                              const ConvolutionDescriptor& descriptor,
                                              bool isDepthwiseConvolution)
{
    if (inputShape.GetNumDimensions() != 4)
    {
        throw InvalidArgumentException("Input shape must be 4D");
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndex(descriptor.m_DataLayout);

    const unsigned int cIndex = dataLayoutIndex.GetChannelsIndex();
    const unsigned int wIndex = dataLayoutIndex.GetWidthIndex();
    const unsigned int hIndex = dataLayoutIndex.GetHeightIndex();

    const unsigned int wInput = inputShape[wIndex];
    const unsigned int hInput = inputShape[hIndex];

    const unsigned int wKernel  = isDepthwiseConvolution ? kernelShape[2] : kernelShape[wIndex];
    const unsigned int wDilated = wKernel + (descriptor.m_DilationX - 1) * (wKernel - 1);

    const unsigned int wRead   = (wInput + descriptor.m_PadLeft + descriptor.m_PadRight) - wDilated;
    const unsigned int wOutput = 1 + (wRead / descriptor.m_StrideX);

    const unsigned int hKernel  = isDepthwiseConvolution ? kernelShape[3] : kernelShape[hIndex];
    const unsigned int hDilated = hKernel + (descriptor.m_DilationY - 1) * (hKernel - 1);

    const unsigned int hRead   = (hInput + descriptor.m_PadTop + descriptor.m_PadBottom) - hDilated;
    const unsigned int hOutput = 1 + (hRead / descriptor.m_StrideY);

    TensorShape outputShape(4);
    outputShape[0]      = inputShape[0];
    outputShape[cIndex] = kernelShape[0];
    outputShape[wIndex] = wOutput;
    outputShape[hIndex] = hOutput;

    if (isDepthwiseConvolution)
    {
        outputShape[cIndex] *= inputShape[cIndex];
    }

    return outputShape;
}

} // anonymous namespace

namespace armnn_driver
{

using namespace armnn;

bool IsDynamicOutput(const TensorInfo& outputInfo)
{
    return outputInfo.GetNumElements() == 0u;
}

TensorShape InferConvolution2dOutputShape(const TensorShape& inputShape,
                                          const TensorShape& kernelShape,
                                          const Convolution2dDescriptor& descriptor)
{
    return InferConvolution2dOutputShapeImpl(inputShape, kernelShape, descriptor, false);
}

TensorShape InferDepthwiseConvolution2dOutputShape(const TensorShape& inputShape,
                                                   const TensorShape& kernelShape,
                                                   const DepthwiseConvolution2dDescriptor& descriptor)
{
    return InferConvolution2dOutputShapeImpl(inputShape, kernelShape, descriptor, true);
}

TensorShape InferMaximumOutputShape(const armnn::TensorShape& input0Shape,
                                    const armnn::TensorShape& input1Shape)
{
    return CalculateMaxShape(input0Shape, input1Shape);
}

TensorShape InferMinimumOutputShape(const armnn::TensorShape& input0Shape,
                                    const armnn::TensorShape& input1Shape)
{
    return CalculateMaxShape(input0Shape, input1Shape);
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

TensorShape InferResizeOutputShape(const TensorShape& inputShape, const ResizeDescriptor& descriptor)
{
    if (inputShape.GetNumDimensions() != 4)
    {
        throw InvalidArgumentException("Input shape for Resize must be 4D");
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);

    const unsigned int cIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int wIndex = dataLayoutIndexed.GetWidthIndex();
    const unsigned int hIndex = dataLayoutIndexed.GetHeightIndex();

    TensorShape outputShape(4);
    outputShape[0]      = inputShape[0];
    outputShape[cIndex] = inputShape[cIndex];
    outputShape[wIndex] = descriptor.m_TargetWidth;
    outputShape[hIndex] = descriptor.m_TargetHeight;

    return outputShape;
}

TensorShape InferSubOutputShape(const TensorShape& input0Shape, const TensorShape& input1Shape)
{
    return CalculateMaxShape(input0Shape, input1Shape);
}

} // namespace armnn_driver