# ArmNN Android Neural Networks driver

This directory contains the ArmNN driver for the Android Neural Networks API, implementing the android.hardware.neuralnetworks@1.0 HAL.

## Integration guide

### Prerequisites

1. Android source tree for Android O MR1 or later, in the directory `<ANDROID_ROOT>`
2. Mali OpenCL driver integrated into the Android source tree

### Procedure

1. Place this source directory at `<ANDROID_ROOT>/vendor/arm/android-nn-driver`
2. Run setup.sh
3. Update the Android build environment to add the ArmNN driver. This ensures that the driver service
is built and copied to the `system/vendor/bin/hw` directory in the Android image.
To update the build environment, add to the contents of the variable `PRODUCT_PACKAGES`
within the device-specific makefile that is located in the `<ANDROID_ROOT>/device/<manufacturer>/<product>`
directory. This file is normally called `device.mk`:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.0-service-armnn
</pre>
4. Build Android as normal, i.e. run `make` in `<ANDROID_ROOT>`
5. To confirm that the ArmNN driver has been built, check for driver service executable at
`<ANDROID_ROOT>/out/target/product/<product>/system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn`

### Testing

1. Run the ArmNN driver service executable in the background
<pre>adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn &</pre>
2. Run some code that exercises the Android Neural Networks API, for example Android's
`NeuralNetworksTest` unit tests (note this is an optional component that must be built).
<pre>adb shell /data/nativetest/NeuralNetworksTest/NeuralNetworksTest > NeuralNetworkTest.log</pre>
3. To confirm that the ArmNN driver is being used to service the Android Neural Networks API requests,
check for messages in logcat with the `ArmnnDriver` tag.
