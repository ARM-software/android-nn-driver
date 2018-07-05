//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include <armnn/ArmNN.hpp>

#include <memory>
#include <set>
#include <string>

// For Android O, explicitly declare the V1_0 HAL namespace to shorten type declarations,
// as the namespace is not defined in HalInterfaces.h.
namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;

namespace armnn_driver
{

class DriverOptions
{
public:
    DriverOptions(armnn::Compute computeDevice);
    DriverOptions(int argc, char** argv);
    DriverOptions(DriverOptions&& other) = default;

    armnn::Compute GetComputeDevice() const { return m_ComputeDevice; }
    bool IsVerboseLoggingEnabled() const { return m_VerboseLogging; }
    const std::string& GetRequestInputsAndOutputsDumpDir() const { return m_RequestInputsAndOutputsDumpDir; }
    bool UseAndroidNnCpuExecutor() const { return m_UseAndroidNnCpuExecutor; }
    const std::set<unsigned int>& GetForcedUnsupportedOperations() const { return m_ForcedUnsupportedOperations; }
    const std::string& GetClTunedParametersFile() const { return m_ClTunedParametersFile; }
    armnn::IClTunedParameters::Mode GetClTunedParametersMode() const { return m_ClTunedParametersMode; }

private:
    armnn::Compute m_ComputeDevice;
    bool m_VerboseLogging;
    bool m_UseAndroidNnCpuExecutor;
    std::string m_RequestInputsAndOutputsDumpDir;
    std::set<unsigned int> m_ForcedUnsupportedOperations;
    std::string m_ClTunedParametersFile;
    armnn::IClTunedParameters::Mode m_ClTunedParametersMode;
};

class ArmnnDriver : public V1_0::IDevice {
public:
    ArmnnDriver(DriverOptions options);
    virtual ~ArmnnDriver() {}
    virtual Return<void> getCapabilities(V1_0::IDevice::getCapabilities_cb _hidl_cb) override;
    virtual Return<void> getSupportedOperations(const V1_0::Model &model,
                                                V1_0::IDevice::getSupportedOperations_cb _hidl_cb) override;
    virtual Return<ErrorStatus> prepareModel(const V1_0::Model &model,
                                      const android::sp<IPreparedModelCallback>& callback);
    virtual Return<DeviceStatus> getStatus() override;

private:
    armnn::IRuntimePtr m_Runtime;
    armnn::IClTunedParametersPtr m_ClTunedParameters;
    DriverOptions m_Options;
};

}
