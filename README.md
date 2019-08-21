# ArmNN Android Neural Networks driver

This directory contains the ArmNN driver for the Android Neural Networks API, implementing the android.hardware.neuralnetworks@1.0, android.hardware.neuralnetworks@1.1 and  android.hardware.neuralnetworks@1.2 HALs.

For more information about supported operations and configurations, see NnapiSupport.txt

## Integration guide

### Prerequisites

1. Android source tree for Android P FSK-R3 or later, in the directory `<ANDROID_ROOT>`
1. Android source tree for Android Q FSK-2 or later, in the directory `<ANDROID_ROOT>`
2. Mali OpenCL driver integrated into the Android source tree

### Procedure

1. Place this source directory at `<ANDROID_ROOT>/vendor/arm/android-nn-driver`
2. Run setup.sh
3. Update the Android build environment to add the ArmNN driver. This ensures that the driver service
is built and copied to the `system/vendor/bin/hw` directory in the Android image.
To update the build environment, add to the contents of the variable `PRODUCT_PACKAGES`
within the device-specific makefile that is located in the `<ANDROID_ROOT>/device/<manufacturer>/<product>`
directory. This file is normally called `device.mk`:

For Android P or Q, using NN API version (1.0), the following should be added to `device.mk`:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.0-service-armnn
</pre>

For Android P or Q, a new version of the NN API is available (1.1),
thus the following should be added to `device.mk` instead:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.1-service-armnn
</pre> `Android.mk` contains the module definition of both versions of the ArmNN driver.

For Android Q, a new version of the NN API is available (1.2),
thus the following should be added to `device.mk` instead:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.2-service-armnn
</pre> `Android.mk` contains the module definition of both versions of the ArmNN driver.

Similarly, the Neon or CL backend can be enabled/disabled by setting ARMNN_COMPUTE_CL_ENABLE or
ARMNN_COMPUTE_NEON_ENABLE in `device.mk`:
<pre>
ARMNN_COMPUTE_CL_ENABLE := 1
</pre>

For Android P and Android Q the vendor manifest.xml requires the Neural Network HAL information.
For Android P use HAL version 1.1 as below. For Android Q substitute 1.2 where necessary.
```xml
<hal format="hidl">
    <name>android.hardware.neuralnetworks</name>
    <transport>hwbinder</transport>
    <version>1.1</version>
    <interface>
        <name>IDevice</name>
        <instance>armnn</instance>
    </interface>
    <fqname>@1.1::IDevice/armnn</fqname>
</hal>
```

4. Build Android as normal, i.e. run `make` in `<ANDROID_ROOT>`
5. To confirm that the ArmNN driver has been built, check for driver service executable at

Android P
<pre>
<ANDROID_ROOT>/out/target/product/<product>/system/vendor/bin/hw
</pre>
For example, if the ArmNN driver has been built with the NN API 1.0, check for the following file:
<pre>
<ANDROID_ROOT>/out/target/product/<product>/system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn
</pre>

Android Q has a different path:
<pre>
<ANDROID_ROOT>/out/target/product/<product>/vendor/bin/hw
</pre>

### Testing

1. Run the ArmNN driver service executable in the background.
The following examples assume that the 1.0 version of the driver is being used:
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn &
</pre>
2. Run some code that exercises the Android Neural Networks API, for example Android's
`NeuralNetworksTest` unit tests (note this is an optional component that must be built).
<pre>
adb shell /data/nativetest/NeuralNetworksTest_static/NeuralNetworksTest_static > NeuralNetworkTest.log
</pre>
3. To confirm that the ArmNN driver is being used to service the Android Neural Networks API requests,
check for messages in logcat with the `ArmnnDriver` tag.

### Using the GPU tuner

The GPU tuner is a feature of the Compute Library that finds optimum values for GPU acceleration tuning parameters.
There are three levels of tuning: exhaustive, normal and rapid.
Exhaustive means that all lws values are tested.
Normal means that a reduced number of lws values are tested, but that generally is sufficient to have a performance close enough to the exhaustive approach.
Rapid means that only 3 lws values should be tested for each kernel.
The recommended way of using it with ArmNN is to generate the tuning data during development of the Android image for a device, and use it in read-only mode during normal operation:

1. Run the ArmNN driver service executable in tuning mode. The path to the tuning data must be writable by the service.
The following examples assume that the 1.0 version of the driver is being used:
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn --cl-tuned-parameters-file &lt;PATH_TO_TUNING_DATA&gt; --cl-tuned-parameters-mode UpdateTunedParameters --cl-tuning-level exhaustive &
</pre>
2. Run a representative set of Android NNAPI testing loads. In this mode of operation, each NNAPI workload will be slow the first time it is executed, as the tuning parameters are being selected. Subsequent executions will use the tuning data which has been generated.
3. Stop the service.
4. Deploy the tuned parameters file to a location readable by the ArmNN driver service (for example, to a location within /vendor/etc).
5. During normal operation, pass the location of the tuning data to the driver service (this would normally be done by passing arguments via Android init in the service .rc definition):
<pre>
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.0-service-armnn --cl-tuned-parameters-file &lt;PATH_TO_TUNING_DATA&gt; &
</pre>

### License

The android-nn-driver is provided under the [MIT](https://spdx.org/licenses/MIT.html) license.
See [LICENSE](LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

    SPDX-License-Identifier: MIT

This enables machine processing of license information based on the SPDX License Identifiers that are available here: http://spdx.org/licenses/
