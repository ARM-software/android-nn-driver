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

