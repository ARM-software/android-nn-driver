//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ArmnnDriver.hpp>
#include "DriverTestHelpers.hpp"

namespace driverTestHelpers
{

class TestTensor
{
public:
    TestTensor(const armnn::TensorShape & shape,
               const std::vector<float> & data)
    : m_Shape{shape}
    , m_Data{data}
    {
        DOCTEST_CHECK(m_Shape.GetNumElements() == m_Data.size());
    }

    hidl_vec<uint32_t> GetDimensions() const;
    unsigned int GetNumElements() const;
    const float * GetData() const;

private:
    armnn::TensorShape   m_Shape;
    std::vector<float>   m_Data;
};

} // driverTestHelpers
