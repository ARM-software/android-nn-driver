//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "../ArmnnDriver.hpp"

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
        BOOST_ASSERT(m_Shape.GetNumElements() == m_Data.size());
    }

    hidl_vec<uint32_t> GetDimensions() const;
    unsigned int GetNumElements() const;
    const float * GetData() const;

private:
    armnn::TensorShape   m_Shape;
    std::vector<float>   m_Data;
};

} // driverTestHelpers
