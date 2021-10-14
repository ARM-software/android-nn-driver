//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestTensor.hpp"

namespace driverTestHelpers
{

hidl_vec<uint32_t> TestTensor::GetDimensions() const
{
    hidl_vec<uint32_t> dimensions;
    dimensions.resize(m_Shape.GetNumDimensions());
    for (uint32_t i=0; i<m_Shape.GetNumDimensions(); ++i)
    {
        dimensions[i] = m_Shape[i];
    }
    return dimensions;
}

unsigned int TestTensor::GetNumElements() const
{
    return m_Shape.GetNumElements();
}

const float * TestTensor::GetData() const
{
    DOCTEST_CHECK(m_Data.empty() == false);
    return &m_Data[0];
}

} // namespace driverTestHelpers
