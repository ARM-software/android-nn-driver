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

Problems seen when trying to build the android-nn-driver obtained from GitHub
-----------------------------------------------------------------------------

Some users have encountered difficulties when attempting to build copies of the android-nn-driver obtained from GitHub. The build reports missing module source paths from boost_1_64_0, clframework or armnn. These errors can look like this:

'error: vendor/arm/android-nn-driver/Android.bp:892:1: module "libboost_program_options" variant "android_arm_variant": module source path "vendor/arm/android-nn-driver/boost_1_64_0" does not exist'
'error: vendor/arm/android-nn-driver/Android.bp:45:1: variant "android_arm64_armv7": module "armnn-arm_compute" "module source path "vendor/arm/android-nn-driver/clframework/build/android-arm64v8a/src/core/CL" does not exist'

These errors are due to missing dependencies or incompatiblities between the android-nn-driver and armnn or clframework versions. The android-nn-driver requires boost_1_40_0 to build. The versions of android-nn-driver, armnn and clframework will have to match for them to work together. So the 19.08 version of android-nn-driver, clframework and armnn will work together but none of them will work with earlier or later versions of the others. 

In order to ensure that the correct versions of boost, armnn and the clframework are obtained you can do the following:

1. Delete or move any boost, armnn or clframework directories from the android-nn-driver directory.
2. Run the setup.sh script in the android-nn-driver directory. 

This will download the correct versions of boost, armnn and the clframework and the android-nn-driver should build correctly. Alternatively you can go to the GitHub pages for android-nn-driver, armnn and computelibrary (clframework) and download versions with the same release tag. 

For 19.08 these would be:

https://github.com/ARM-software/android-nn-driver/tree/v19.08
https://github.com/ARM-software/armnn/tree/v19.08
https://github.com/ARM-software/computelibrary/tree/v19.08

The correct version of boost (1_64_0) can be downloaded from:

https://www.boost.org/

Instance Normalization test failures 
------------------------------------

There is a known issue in the Android NNAPI implementation of Instance Normalization that will be fixed in an upcoming revision of Android 10. Using the Arm NN Android NNAPI driver with versions of the Android 10 VTS and CTS tests that do not have that fix will generate multiple Instance Normalization failures. 

