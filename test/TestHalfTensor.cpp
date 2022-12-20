//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestHalfTensor.hpp"

namespace driverTestHelpers
{

hidl_vec<uint32_t> TestHalfTensor::GetDimensions() const
{
    hidl_vec<uint32_t> dimensions;
    dimensions.resize(m_Shape.GetNumDimensions());
    for (uint32_t i=0; i<m_Shape.GetNumDimensions(); ++i)
    {
        dimensions[i] = m_Shape[i];
    }
    return dimensions;
}

unsigned int TestHalfTensor::GetNumElements() const
{
    return m_Shape.GetNumElements();
}

const Half * TestHalfTensor::GetData() const
{
    DOCTEST_CHECK(m_Data.empty() == false);
    return &m_Data[0];
}

} // namespace driverTestHelpers
