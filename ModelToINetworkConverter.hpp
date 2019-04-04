//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ConversionUtils.hpp"

#include <armnn/ArmNN.hpp>

#include <set>
#include <vector>

namespace armnn_driver
{

enum class ConversionResult
{
    Success,
    ErrorMappingPools,
    UnsupportedFeature
};

// A helper template class performing the conversion from an AndroidNN driver Model representation,
// to an armnn::INetwork object
template<typename HalPolicy>
class ModelToINetworkConverter
{
public:
    using HalModel = typename HalPolicy::Model;

    ModelToINetworkConverter(const std::vector<armnn::BackendId>& backends,
                             const HalModel& model,
                             const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Data.m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

    // Shared aggregate input/output/internal data
    ConversionData m_Data;

    // Input data
    const HalModel&               m_Model;
    const std::set<unsigned int>& m_ForcedUnsupportedOperations;

    // Output data
    ConversionResult         m_ConversionResult;
    std::map<uint32_t, bool> m_OperationSupported;
};

} // armnn_driver
