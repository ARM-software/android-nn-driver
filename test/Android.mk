#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

LOCAL_PATH := $(call my-dir)

# Configure these paths if you move the source or Khronos headers
#
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../../mali/product/khronos/original
ARMNN_HEADER_PATH := $(LOCAL_PATH)/../armnn/include
ARMNN_THIRD_PARTY_PATH   := $(LOCAL_PATH)/../armnn/third-party
ARMNN_UTILS_HEADER_PATH := $(LOCAL_PATH)/../armnn/src/armnnUtils
ARMNN_DRIVER_HEADER_PATH := $(LOCAL_PATH)/..

NN_HEADER_PATH := $(LOCAL_PATH)/../../../../frameworks/ml/nn/runtime/include
ifeq ($(S_OR_LATER),1)
NN_HEADER_PATH := $(LOCAL_PATH)/../../../../packages/modules/NeuralNetworks/runtime/include
endif

##########################
# armnn-driver-tests@1.0 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.0
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif # R or later

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif # S or later

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

LOCAL_SRC_FILES := \
        1.0/Convolution2D.cpp \
        1.0/FullyConnectedReshape.cpp \
        1.0/Lstm.cpp \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Concat.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_unit_test_framework \
        libflatbuffers-framework \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.0

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
        android.hidl.memory@1.0

ifeq ($(P_OR_LATER),1)
# Required to build the 1.0 version of the NN Driver on Android P and later versions,
# as the 1.0 version of the NN API needs the 1.1 HAL headers to be included regardless.
LOCAL_SHARED_LIBRARIES+= \
        android.hardware.neuralnetworks@1.1
endif # PLATFORM_VERSION == 9

ifeq ($(Q_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hardware.neuralnetworks@1.2
endif # PLATFORM_VERSION == Q

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

ifeq ($(P_OR_LATER),1)
# The following target is available starting from Android P

##########################
# armnn-driver-tests@1.1 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.1
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
#PRODUCT_PACKAGES_ENG := libarmnn
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG \
        -DARMNN_ANDROID_NN_V1_1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif # R or later

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif # S or later

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

LOCAL_SRC_FILES := \
        1.0/Convolution2D.cpp \
        1.1/Convolution2D.cpp \
        1.1/Mean.cpp \
        1.1/Transpose.cpp \
        1.0/Lstm.cpp \
        1.1/Lstm.cpp \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Concat.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_unit_test_framework \
        libflatbuffers-framework \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.1

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
        android.hidl.memory@1.0

ifeq ($(Q_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hardware.neuralnetworks@1.2
endif # PLATFORM_VERSION == Q

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == 9

ifeq ($(Q_OR_LATER),1)
# The following target is available starting from Android Q

##########################
# armnn-driver-tests@1.2 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.2
LOCAL_MODULE_TAGS := optional

LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG \
        -DBOOST_NO_AUTO_PTR \
        -DARMNN_ANDROID_NN_V1_2

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif # R or later

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif # S or later

LOCAL_SRC_FILES := \
        1.0/Convolution2D.cpp \
        1.1/Convolution2D.cpp \
        1.1/Mean.cpp \
        1.1/Transpose.cpp \
        1.2/Dilation.cpp \
        1.2/Capabilities.cpp \
        1.0/Lstm.cpp \
        1.1/Lstm.cpp \
        1.2/Lstm.cpp \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Concat.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_unit_test_framework \
        libflatbuffers-framework \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.2

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libcutils \
        libfmq \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libnativewindow \
        libtextclassifier_hash \
        libui \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == Q

ifeq ($(R_OR_LATER),1)
# The following target is available starting from Android R

##########################
# armnn-driver-tests@1.3 #
##########################
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-driver-tests@1.3
LOCAL_MODULE_TAGS := optional

LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -O0 \
        -UNDEBUG \
        -DBOOST_NO_AUTO_PTR \
        -DARMNN_ANDROID_NN_V1_3

ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

LOCAL_SRC_FILES := \
        1.0/Convolution2D.cpp \
        1.1/Convolution2D.cpp \
        1.1/Mean.cpp \
        1.1/Transpose.cpp \
        1.2/Dilation.cpp \
        1.2/Capabilities.cpp \
        1.0/Lstm.cpp \
        1.1/Lstm.cpp \
        1.2/Lstm.cpp \
        1.3/QLstm.cpp \
        1.3/QosTests.cpp \
        Tests.cpp \
        UtilsTests.cpp \
        Concurrent.cpp \
        FullyConnected.cpp \
        GenericLayerTests.cpp \
        DriverTestHelpers.cpp \
        SystemProperties.cpp \
        Concat.cpp \
        TestTensor.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_unit_test_framework \
        libflatbuffers-framework \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.3

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libcutils \
        libfmq \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libnativewindow \
        libtextclassifier_hash \
        libui \
        libutils \
        libsync \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2 \
        android.hardware.neuralnetworks@1.3 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == R