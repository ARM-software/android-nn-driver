#
# Copyright © 2017 ARM Ltd. All rights reserved.
# See LICENSE file in the project root for full license information.
#

LOCAL_PATH := $(call my-dir)

# Configure these paths if you move the source or Khronos headers
#
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../../mali/product/khronos/original
NN_HEADER_PATH := $(LOCAL_PATH)/../../../../frameworks/ml/nn/runtime/include
ARMNN_HEADER_PATH := $(LOCAL_PATH)/../armnn/include
ARMNN_DRIVER_HEADER_PATH := $(LOCAL_PATH)/..

##########################
# armnn-driver-tests@1.0 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.0
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG
ifeq ($(PLATFORM_VERSION),9)
# Required to build with the changes made to the Android ML framework starting from Android P,
# regardless of the HAL version used for the build.
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_P
endif

LOCAL_SRC_FILES := \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        Convolution2D.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Lstm.cpp \
        Merger.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libarmnn-driver@1.0 \
        libneuralnetworks_common \
        libarmnn \
        libboost_log \
        libboost_system \
        libboost_unit_test_framework \
        libboost_thread \
        armnn-arm_compute

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libtextclassifier_hash \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        libOpenCL
ifeq ($(PLATFORM_VERSION),9)
# Required to build the 1.0 version of the NN Driver on Android P and later versions,
# as the 1.0 version of the NN API needs the 1.1 HAL headers to be included regardless.
LOCAL_SHARED_LIBRARIES+= \
        android.hardware.neuralnetworks@1.1
endif

include $(BUILD_EXECUTABLE)

##########################
# armnn-driver-tests@1.1 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.1
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG \
        -DARMNN_ANDROID_P \
        -DARMNN_ANDROID_NN_V1_1

LOCAL_SRC_FILES := \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        Convolution2D.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Lstm.cpp \
        Merger.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libarmnn-driver@1.1 \
        libneuralnetworks_common \
        libarmnn \
        libboost_log \
        libboost_system \
        libboost_unit_test_framework \
        libboost_thread \
        armnn-arm_compute

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libtextclassifier_hash \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        libOpenCL

include $(BUILD_EXECUTABLE)

