//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ArmnnDriver.hpp>
#include "DriverTestHelpers.hpp"

#include <half/half.hpp>

using Half = half_float::half;

namespace driverTestHelpers
{

class TestHalfTensor
{
public:
    TestHalfTensor(const armnn::TensorShape & shape,
               const std::vector<Half> & data)
        : m_Shape{shape}
        , m_Data{data}
    {
        DOCTEST_CHECK(m_Shape.GetNumElements() == m_Data.size());
    }

    hidl_vec<uint32_t> GetDimensions() const;
    unsigned int GetNumElements() const;
    const Half * GetData() const;

private:
    armnn::TensorShape   m_Shape;
    std::vector<Half>   m_Data;
};

} // driverTestHelpers
