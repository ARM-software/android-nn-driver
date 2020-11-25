Integration guide
=================

This document describes how to integrate the Arm NN Android NNAPI driver into an Android source tree.

### Prerequisites

1. Android source tree for Android P (we have tested against Android P version 9.0.0_r3) , in the directory `<ANDROID_ROOT>`
2. Android source tree for Android Q (we have tested against Android Q version 10.0.0_r39), in the directory `<ANDROID_ROOT>`
2. Android source tree for Android R (we have tested against Android R version 11.0.0_r3), in the directory `<ANDROID_ROOT>`
3. Mali OpenCL driver integrated into the Android source tree

### Procedure

1. Place this source directory at `<ANDROID_ROOT>/vendor/arm/android-nn-driver`
2. Run setup.sh
3. Update the Android build environment to add the ArmNN driver. This ensures that the driver service
is built and copied to the `system/vendor/bin/hw` directory in the Android image.
To update the build environment, add to the contents of the variable `PRODUCT_PACKAGES`
within the device-specific makefile that is located in the `<ANDROID_ROOT>/device/<manufacturer>/<product>`
directory. This file is normally called `device.mk`:

For Android P, Q or R, using NN API version (1.0), the following should be added to `device.mk`:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.0-service-armnn
</pre>

For Android P, Q or R, a new version of the NN API is available (1.1),
thus the following should be added to `device.mk` instead:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.1-service-armnn
</pre> `Android.mk` contains the module definition of both versions of the ArmNN driver.

For Android Q, a new version of the NN API is available (1.2),
thus the following should be added to `device.mk` instead:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.2-service-armnn
</pre> `Android.mk` contains the module definition of all three versions (1.0, 1.1 and 1.2) of the ArmNN driver.

For Android R, new version of the NN API is available (1.3),
thus the following should be added to `device.mk` instead:
<pre>
PRODUCT_PACKAGES += android.hardware.neuralnetworks@1.3-service-armnn
</pre>
`Android.mk` contains the module definition of all versions (1.0, 1.1, 1.2 and 1.3) of the ArmNN driver.

Similarly, the Neon, CL or reference backend can be enabled/disabled by setting ARMNN_COMPUTE_CL_ENABLE,
ARMNN_COMPUTE_NEON_ENABLE or ARMNN_REF_ENABLE in `device.mk`:
<pre>
ARMNN_COMPUTE_CL_ENABLE := 1
</pre>

For Android P, Q and R the vendor manifest.xml requires the Neural Network HAL information.
For Android P use HAL version 1.1 as below. For Android Q substitute 1.2 where necessary. For Android R substitute 1.3 where necessary.
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

Android Q and later has a different path:
<pre>
<ANDROID_ROOT>/out/target/product/<product>/vendor/bin/hw
</pre>

### Testing

1. Run the ArmNN driver service executable in the background.
Use the corresponding version of the driver for the Android version you are running.
i.e
android.hardware.neuralnetworks@1.1-service-armnn for Android P,
android.hardware.neuralnetworks@1.2-service-armnn for Android Q and
android.hardware.neuralnetworks@1.3-service-armnn for Android R
<pre>
It is also possible to use a specific backend by using the -c option.
The following is an example of using the CpuAcc backend for Android Q:
adb shell /system/vendor/bin/hw/android.hardware.neuralnetworks@1.2-service-armnn -c CpuAcc &
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
