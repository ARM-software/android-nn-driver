#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

ANDROID_NN_DRIVER_LOCAL_PATH := $(call my-dir)
LOCAL_PATH := $(ANDROID_NN_DRIVER_LOCAL_PATH)

# Configure these paths if you move the source or Khronos headers
ARMNN_HEADER_PATH := $(LOCAL_PATH)/armnn/include
ARMNN_UTILS_HEADER_PATH := $(LOCAL_PATH)/armnn/src/armnnUtils
OPENCL_HEADER_PATH := $(LOCAL_PATH)/clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../frameworks/ml/nn/runtime/include

#######################
# libarmnn-driver@1.0 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.0
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -Werror \
        -Wno-format-security
ifeq ($(PLATFORM_VERSION),9)
# Required to build with the changes made to the Android ML framework starting from Android P,
# regardless of the HAL version used for the build.
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_P
endif # PLATFORM_VERSION == 9
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

LOCAL_SRC_FILES := \
        1.0/ArmnnDriverImpl.cpp \
        1.0/HalPolicy.cpp \
        ArmnnDriverImpl.cpp \
        DriverOptions.cpp \
        ArmnnDevice.cpp \
        ArmnnPreparedModel.cpp \
        ModelToINetworkConverter.cpp \
        RequestThread.cpp \
        Utils.cpp \
        ConversionUtils.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := libarmnn

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
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
endif # PLATFORM_VERSION == 9

include $(BUILD_STATIC_LIBRARY)

ifeq ($(PLATFORM_VERSION),9)
# The following target is available starting from Android P

#######################
# libarmnn-driver@1.1 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.1
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_P \
        -DARMNN_ANDROID_NN_V1_1
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

LOCAL_SRC_FILES := \
        1.0/ArmnnDriverImpl.cpp \
        1.0/HalPolicy.cpp \
        1.1/ArmnnDriverImpl.cpp \
        1.1/HalPolicy.cpp \
        ArmnnDriverImpl.cpp \
        DriverOptions.cpp \
        ArmnnDevice.cpp \
        ArmnnPreparedModel.cpp \
        ModelToINetworkConverter.cpp \
        RequestThread.cpp \
        Utils.cpp \
        ConversionUtils.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := libarmnn

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        libOpenCL

include $(BUILD_STATIC_LIBRARY)

endif # PLATFORM_VERSION == 9

#####################################################
# android.hardware.neuralnetworks@1.0-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.0-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-service-armnn.rc
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.0

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        libdl \
        libhardware \
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
endif # PLATFORM_VERSION == 9

include $(BUILD_EXECUTABLE)

ifeq ($(PLATFORM_VERSION),9)
# The following target is available starting from Android P

#####################################################
# android.hardware.neuralnetworks@1.1-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.1-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.1-service-armnn.rc
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_1
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.1

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        libdl \
        libhardware \
        liblog \
        libtextclassifier_hash \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        libOpenCL

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == 9

##########################
# armnn module and tests #
##########################
# Note we use ANDROID_NN_DRIVER_LOCAL_PATH rather than LOCAL_PATH because LOCAL_PATH will be overwritten
# when including other .mk files that set it.
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/armnn/Android.mk
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/test/Android.mk
