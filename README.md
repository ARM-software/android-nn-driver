# ArmNN Android Neural Networks driver

This directory contains the ArmNN driver for the Android Neural Networks API, implementing the android.hardware.neuralnetworks@1.0 HAL.

For more information about supported operations and configurations, see NnapiSupport.txt

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
<pre>
<ANDROID_ROOT>/out/target/product/<product>/system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn
</pre>

### Testing

1. Run the ArmNN driver service executable in the background
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn &
</pre>
2. Run some code that exercises the Android Neural Networks API, for example Android's
`NeuralNetworksTest` unit tests (note this is an optional component that must be built).
<pre>
adb shell /data/nativetest/NeuralNetworksTest/NeuralNetworksTest > NeuralNetworkTest.log
</pre>
3. To confirm that the ArmNN driver is being used to service the Android Neural Networks API requests,
check for messages in logcat with the `ArmnnDriver` tag.

### Using ClTuner

ClTuner is a feature of the Compute Library that finds optimum values for OpenCL tuning parameters. The recommended way of using it with ArmNN is to generate the tuning data during development of the Android image for a device, and use it in read-only mode during normal operation:

1. Run the ArmNN driver service executable in tuning mode. The path to the tuning data must be writable by the service:
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn --cl-tuned-parameters-file &lt;PATH_TO_TUNING_DATA&gt; --cl-tuned-parameters-mode UpdateTunedParameters &
</pre>
2. Run a representative set of Android NNAPI testing loads. In this mode of operation, each NNAPI workload will be slow the first time it is executed, as the tuning parameters are being selected. Subsequent executions will use the tuning data which has been generated.
3. Stop the service.
4. Deploy the tuned parameters file to a location readable by the ArmNN driver service (for example, to a location within /vendor/etc).
5. During normal operation, pass the location of the tuning data to the driver service (this would normally be done by passing arguments via Android init in the service .rc definition):
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn --cl-tuned-parameters-file &lt;PATH_TO_TUNING_DATA&gt; &
</pre>
