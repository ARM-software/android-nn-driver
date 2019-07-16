//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

namespace armnn_driver
{

bool IsDynamicOutput(const armnn::TensorInfo& outputInfo);

armnn::TensorShape InferConvolution2dOutputShape(const armnn::TensorShape& inputShape,
                                                 const armnn::TensorShape& kernelShape,
                                                 const armnn::Convolution2dDescriptor& descriptor);

armnn::TensorShape InferDepthwiseConvolution2dOutputShape(const armnn::TensorShape& inputShape,
                                                          const armnn::TensorShape& kernelShape,
                                                          const armnn::DepthwiseConvolution2dDescriptor& descriptor);

armnn::TensorShape InferMaximumOutputShape(const armnn::TensorShape& input0Shape,
                                           const armnn::TensorShape& input1Shape);

armnn::TensorShape InferPadOutputShape(const armnn::TensorShape& inputShape,
                                       const std::vector<std::pair<unsigned int, unsigned int>>& padList);

armnn::TensorShape InferPreluOutputShape(const armnn::TensorShape& inputShape, const armnn::TensorShape& alphaShape);

armnn::TensorShape InferResizeOutputShape(const armnn::TensorShape& inputShape,
                                          const armnn::ResizeDescriptor& descriptor);

armnn::TensorShape InferSubOutputShape(const armnn::TensorShape& input0Shape, const armnn::TensorShape& input1Shape);

} // namespace armnn_driver


