//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ArmnnDriverImpl.hpp"
#include "../1.0/ArmnnDriverImpl.hpp"

#include <OperationsUtils.h>

#include <log/log.h>
#include <boost/assert.hpp>

#include <ValidateHal.h>

using namespace std;
using namespace android;
using namespace android::nn;
using namespace android::hardware;

namespace
{

void NotifyCallbackAndCheck(const sp<IPreparedModelCallback>& callback,
                            ErrorStatus errorStatus,
                            const sp<IPreparedModel>& preparedModelPtr)
{
    Return<void> returned = callback->notify(errorStatus, preparedModelPtr);
    // This check is required, if the callback fails and it isn't checked it will bring down the service
    if (!returned.isOk())
    {
        ALOGE("V1_1::ArmnnDriverImpl::prepareModel_1_1: hidl callback failed to return properly: %s ",
              returned.description().c_str());
    }
}

Return<ErrorStatus> FailPrepareModel(ErrorStatus error,
                                     const string& message,
                                     const sp<IPreparedModelCallback>& callback)
{
    ALOGW("V1_1::ArmnnDriverImpl::prepareModel_1_1: %s", message.c_str());
    NotifyCallbackAndCheck(callback, error, nullptr);
    return error;
}

} // namespace

namespace armnn_driver
{
namespace V1_1
{

Return<void> ArmnnDriverImpl::getCapabilities_1_1(
        const armnn::IRuntimePtr& runtime,
        neuralnetworks::V1_1::IDevice::getCapabilities_1_1_cb cb)
{
    ALOGV("V1_1::ArmnnDriverImpl::getCapabilities_1_1()");

    neuralnetworks::V1_0::IDevice::getCapabilities_cb cb_1_0 =
            [&](ErrorStatus status, const neuralnetworks::V1_0::Capabilities& capabilities)
    {
        BOOST_ASSERT_MSG(compliantWithV1_1(capabilities),
                         "V1_1::ArmnnDriverImpl: V1_0::Capabilities not compliant with V1_1::Capabilities");

        cb(status, convertToV1_1(capabilities));
    };

    V1_0::ArmnnDriverImpl::getCapabilities(runtime, cb_1_0);

    return Void();
}

Return<void> ArmnnDriverImpl::getSupportedOperations_1_1(
        const armnn::IRuntimePtr& runtime,
        const DriverOptions& options,
        const neuralnetworks::V1_1::Model& model,
        neuralnetworks::V1_1::IDevice::getSupportedOperations_1_1_cb cb)
{
    ALOGV("V1_1::ArmnnDriverImpl::getSupportedOperations_1_1()");

    if(compliantWithV1_0(model))
    {
        V1_0::ArmnnDriverImpl::getSupportedOperations(runtime, options, convertToV1_0(model), cb);
    }
    else
    {
        std::vector<bool> result;

        if (!runtime)
        {
            ALOGW("V1_1::ArmnnDriverImpl::getSupportedOperations_1_1: Device unavailable");
            cb(ErrorStatus::DEVICE_UNAVAILABLE, result);
            return Void();
        }

        if (!android::nn::validateModel(model))
        {
            ALOGW("V1_1::ArmnnDriverImpl::getSupportedOperations_1_1: Invalid model passed as input");
            cb(ErrorStatus::INVALID_ARGUMENT, result);
            return Void();
        }

        result.assign(model.operations.size(), false);
        cb(ErrorStatus::NONE, result);
    }

    return Void();
}

Return<ErrorStatus> ArmnnDriverImpl::prepareModel_1_1(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const neuralnetworks::V1_1::Model& model,
        const sp<IPreparedModelCallback>& cb)
{
    ALOGV("V1_1::ArmnnDriverImpl::prepareModel_1_1()");

    if(compliantWithV1_0(model))
    {
        return V1_0::ArmnnDriverImpl::prepareModel(runtime, clTunedParameters, options, convertToV1_0(model), cb,
                                                   model.relaxComputationFloat32toFloat16 && options.GetFp16Enabled());
    }
    else
    {
        if (cb.get() == nullptr)
        {
            ALOGW("V1_1::ArmnnDriverImpl::prepareModel_1_1: Invalid callback passed to prepareModel");
            return ErrorStatus::INVALID_ARGUMENT;
        }

        if (!runtime)
        {
            return FailPrepareModel(ErrorStatus::DEVICE_UNAVAILABLE,
                                    "V1_1::ArmnnDriverImpl::prepareModel_1_1: Device unavailable", cb);
        }

        if (!android::nn::validateModel(model))
        {
            return FailPrepareModel(ErrorStatus::INVALID_ARGUMENT,
                                    "V1_1::ArmnnDriverImpl::prepareModel_1_1: Invalid model passed as input", cb);
        }

        FailPrepareModel(ErrorStatus::GENERAL_FAILURE,
                         "V1_1::ArmnnDriverImpl::prepareModel_1_1: Unsupported model", cb);
        return ErrorStatus::NONE;
    }
}

} // armnn_driver::namespace V1_1
} // namespace armnn_driver
