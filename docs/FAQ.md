Frequently asked questions
==========================

These are issues that have been seen when using the Arm NN Android NNAPI driver. The guidance here could be of interest to Android system integrators including the driver in an Android platform.

Problems seen when running VTS tests through vts-tradefed
---------------------------------------------------------

An issue has been seen in some systems when running the NNAPI VTS tests using vts-tradefed, in a system configured to use the Arm NN GPU backend.
When the total time taken to run all tests in a suite exceeds the timeout defined for the plan, an error is supposed to be reported through vts-tradefed. However,
some users have seen a situation where the device side of the tests gets killed, and the vts-tradefed console continues to run and reports that all further
tests "Fail". In this case the "Failed" tests are actually not being executed.

This has been seen when running large test suites using the Arm NN GPU backend (Mali OpenCL) as the time needed to compile hundreds or thousands of OpenCL kernels
was not taken into account when setting an appropriate test timeout for Android 8.1 and Android 9.0. This issue was fixed in the AOSP master branch in change
https://android.googlesource.com/platform/test/vts-testcase/hal/+/f74899c6c09b52703e6db0323dffb4ae52539db4 so should not be seen in Android 10 or later.

An acceptable workaround is to increase the timeout defined in AndroidTest.xml, in a similar way to https://android.googlesource.com/platform/test/vts-testcase/hal/+/f74899c6c09b52703e6db0323dffb4ae52539db4.

Instance Normalization test failures 
------------------------------------

There is a known issue in the Android NNAPI implementation of Instance Normalization that has been verified as fixed from Android 10 r39 onwards. Using the Arm NN Android NNAPI driver with versions of the Android 10 VTS and CTS tests that do not have that fix will generate multiple Instance Normalization test failures. 

VTS and CTS test failures
-------------------------

With Android 10 R2 CTS some errors and crashes were discovered in the 19.08 and 19.11 releases of armnn, the android-nn-driver and ComputeLibrary. 19.08.01 and 19.11.01 releases of armnn, the android-nn-driver and ComputeLibrary were prepared that fix all these issues on CpuAcc and GpuAcc. If using 19.08 or 19.11 we recommend that you upgrade to the latest releases.

These fixes also required patches to be made to the Android Q test framework. You may encounter CTS and VTS test failures when attempting to build copies of the android-nn-driver against older versions of Android Q.

These test failures include:

* ComputeMode/GeneratedTests.avg_pool_v1_2 Float16 tests 
* ComputeMode/GeneratedTests.instance_normalization tests
* TestRandomGraph/SingleOperationTest.INSTANCE_NORMALIZATION_V1_2 tests
* TestRandomGraph/SingleOperationTest.PRELU_V1_2 tests
* Some TestRandomGraph/RandomGraphTest tests which include avg_pool or instance_normalization operators.
* Some TestRandomGraph/RandomGraphTest tests which use Float16 input.

In order to fix these failures you will have to update to a version of Android Q that includes the following patches: https://android-review.googlesource.com/q/project:platform%252Fframeworks%252Fml+branch:android10-tests-dev+status:merged

The Android 10 R3 CTS that can be downloaded from https://source.android.com/compatibility/cts/downloads contains all these patches. 

There is a known issue that even with these patches CTS tests "TestRandomGraph/RandomGraphTest#LargeGraph_TENSOR_FLOAT16_Rank3/41" and "TestRandomGraph/RandomGraphTest#LargeGraph_TENSOR_FLOAT16_Rank2/20 " will still fail on CpuRef. These failures are caused by a LogSoftmax layer followed by a Floor layer which blows up the slight difference between fp16 to fp32. This issue only affects CpuRef with Android Q. These tests are not failing for Android R.
