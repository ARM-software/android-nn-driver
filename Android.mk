#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

ANDROID_NN_DRIVER_LOCAL_PATH := $(call my-dir)
LOCAL_PATH := $(ANDROID_NN_DRIVER_LOCAL_PATH)

P_OR_LATER := 0
Q_OR_LATER := 0

ifeq ($(PLATFORM_VERSION),9)
P_OR_LATER := 1
endif # PLATFORM_VERSION == 9
ifeq ($(PLATFORM_VERSION),P)
P_OR_LATER := 1
endif # PLATFORM_VERSION == P

ifeq ($(PLATFORM_VERSION),10)
P_OR_LATER := 1
Q_OR_LATER := 1
endif # PLATFORM_VERSION == 10
ifeq ($(PLATFORM_VERSION),Q)
P_OR_LATER := 1
Q_OR_LATER := 1
endif # PLATFORM_VERSION == Q

CPP_VERSION := c++14

ifeq ($(Q_OR_LATER),1)
CPP_VERSION := c++17
endif

# Configure these paths if you move the source or Khronos headers
ARMNN_HEADER_PATH := $(LOCAL_PATH)/armnn/include
ARMNN_UTILS_HEADER_PATH := $(LOCAL_PATH)/armnn/src/armnnUtils
OPENCL_HEADER_PATH := $(LOCAL_PATH)/clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../frameworks/ml/nn/runtime/include

# Variables to control CL/NEON/reference backend support
#
# They can be optionally passed from the command line to build the backends programmatically
# For example, to disable CL support, do from the top of the Android source tree:
# ARMNN_COMPUTE_CL_ENABLED=0 make
# Or export it as an environment variable, export ARMNN_COMPUTE_CL_ENABLED=0, and then run the make command
#
# Set the following default values to '0' to disable support for a specific backend
ifndef ARMNN_COMPUTE_CL_ENABLED
# ARMNN_COMPUTE_CL_ENABLED is undefined, use the following default value
ARMNN_COMPUTE_CL_ENABLED := 1
endif
ifndef ARMNN_COMPUTE_NEON_ENABLED
# ARMNN_COMPUTE_NEON_ENABLED is undefined, use the following default value
ARMNN_COMPUTE_NEON_ENABLED := 1
endif
ifndef ARMNN_REF_ENABLED
# ARMNN_REF_ENABLED is undefined, use the following default value
ARMNN_REF_ENABLED := 1
endif

#######################
# libarmnn-driver@1.0 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.0
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
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security

ifeq ($(P_OR_LATER),1)
# Required to build with the changes made to the Android ML framework starting from Android P,
# regardless of the HAL version used for the build.
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_P
endif # PLATFORM_VERSION == 9

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTECL_ENABLED
endif # ARMNN_COMPUTE_CL_ENABLED == 1

ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTENEON_ENABLED
endif # ARMNN_COMPUTE_NEON_ENABLED == 1

ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNREF_ENABLED
endif # ARMNN_REF_ENABLED == 1

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
        libboost_filesystem \
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

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_STATIC_LIBRARY)

ifeq ($(P_OR_LATER),1)
# The following target is available starting from Android P

#######################
# libarmnn-driver@1.1 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.1
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
#PRODUCT_PACKAGES_ENG := libarmnn-driver@1.1
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
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_P \
        -DARMNN_ANDROID_NN_V1_1

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTECL_ENABLED
endif # ARMNN_COMPUTE_CL_ENABLED == 1

ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTENEON_ENABLED
endif # ARMNN_COMPUTE_NEON_ENABLED == 1

ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNREF_ENABLED
endif # ARMNN_REF_ENABLED == 1

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
        libboost_filesystem \
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
        android.hidl.memory@1.0

ifeq ($(Q_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hardware.neuralnetworks@1.2
endif # PLATFORM_VERSION == Q

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_STATIC_LIBRARY)

endif # PLATFORM_VERSION == 9

ifeq ($(Q_OR_LATER),1)
# The following target is available starting from Android Q

#######################
# libarmnn-driver@1.2 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.2
LOCAL_MODULE_TAGS := optional
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
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_Q \
        -DARMNN_ANDROID_NN_V1_2

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTECL_ENABLED
endif # ARMNN_COMPUTE_CL_ENABLED == 1

ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTENEON_ENABLED
endif # ARMNN_COMPUTE_NEON_ENABLED == 1

ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNREF_ENABLED
endif # ARMNN_REF_ENABLED == 1

LOCAL_SRC_FILES := \
        1.0/ArmnnDriverImpl.cpp \
        1.0/HalPolicy.cpp \
        1.1/ArmnnDriverImpl.cpp \
        1.1/HalPolicy.cpp \
        1.2/ArmnnDriverImpl.cpp \
        1.2/HalPolicy.cpp \
        ArmnnDevice.cpp \
        ArmnnDriverImpl.cpp \
        ArmnnPreparedModel.cpp \
        ArmnnPreparedModel_1_2.cpp \
        ConversionUtils.cpp \
        DriverOptions.cpp \
        ModelToINetworkConverter.cpp \
        RequestThread.cpp \
        Utils.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        libboost_filesystem \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := libarmnn

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libutils \
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_STATIC_LIBRARY)

endif # PLATFORM_VERSION == Q

#####################################################
# android.hardware.neuralnetworks@1.0-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.0-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-service-armnn.rc
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        libboost_filesystem \
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

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

ifeq ($(P_OR_LATER),1)
# The following target is available starting from Android P

#####################################################
# android.hardware.neuralnetworks@1.1-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.1-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.1-service-armnn.rc
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_1
ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_program_options \
        libboost_system \
        libboost_thread \
        libboost_filesystem \
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
        android.hidl.memory@1.0

ifeq ($(Q_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hardware.neuralnetworks@1.2
endif # PLATFORM_VERSION == Q

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == 9

ifeq ($(Q_OR_LATER),1)
# The following target is available starting from Android Q

#####################################################
# android.hardware.neuralnetworks@1.2-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.2-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.2-service-armnn.rc
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_2 \
        -DBOOST_NO_AUTO_PTR
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
        libboost_filesystem \
        armnn-arm_compute

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.2

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
        libnativewindow \
        libui \
        libfmq \
        libcutils \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == Q

##########################
# armnn module and tests #
##########################
# Note we use ANDROID_NN_DRIVER_LOCAL_PATH rather than LOCAL_PATH because LOCAL_PATH will be overwritten
# when including other .mk files that set it.
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/armnn/Android.mk
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/test/Android.mk
