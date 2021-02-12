#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

ANDROID_NN_DRIVER_LOCAL_PATH := $(call my-dir)
LOCAL_PATH := $(ANDROID_NN_DRIVER_LOCAL_PATH)

P_OR_LATER := 0
Q_OR_LATER := 0
R_OR_LATER := 0
S_OR_LATER := 0
ANDROID_R  := 0
ANDROID_S  := 0

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

ifeq ($(PLATFORM_VERSION),R)
P_OR_LATER := 1
Q_OR_LATER := 1
R_OR_LATER := 1
ANDROID_R  := 1
endif # PLATFORM_VERSION == R

ifeq ($(PLATFORM_VERSION),11)
P_OR_LATER := 1
Q_OR_LATER := 1
R_OR_LATER := 1
ANDROID_R  := 1
endif # PLATFORM_VERSION == 11

ifeq ($(PLATFORM_VERSION),S)
P_OR_LATER := 1
Q_OR_LATER := 1
R_OR_LATER := 1
S_OR_LATER := 1
ANDROID_R  := 0
ANDROID_S  := 1
endif # PLATFORM_VERSION == S

ifeq ($(PLATFORM_VERSION),12)
P_OR_LATER := 1
Q_OR_LATER := 1
R_OR_LATER := 1
S_OR_LATER := 1
ANDROID_R  := 0
ANDROID_S  := 1
endif # PLATFORM_VERSION == 12

CPP_VERSION := c++14

ifeq ($(Q_OR_LATER),1)
CPP_VERSION := c++17
endif

# Configure these paths if you move the source or Khronos headers
ARMNN_HEADER_PATH := $(LOCAL_PATH)/armnn/include
ARMNN_THIRD_PARTY_PATH := $(LOCAL_PATH)/armnn/third-party
ARMNN_UTILS_HEADER_PATH := $(LOCAL_PATH)/armnn/src/armnnUtils
ARMNN_THIRD_PARTY_PATH := $(LOCAL_PATH)/armnn/third-party
OPENCL_HEADER_PATH := $(LOCAL_PATH)/clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../frameworks/ml/nn/runtime/include
ifeq ($(S_OR_LATER),1)
NN_HEADER_PATH := $(LOCAL_PATH)/../../../packages/modules/NeuralNetworks/runtime/include
endif

# Variables to control CL/NEON/reference backend support
# Set them to '0' to disable support for a specific backend
ARMNN_COMPUTE_CL_ENABLED := 1
ARMNN_COMPUTE_NEON_ENABLED := 1
ARMNN_REF_ENABLED := 1
ARMNN_ETHOSN_ENABLED := 1

ifeq ($(ARMNN_COMPUTE_CL_ENABLE),0)
ARMNN_COMPUTE_CL_ENABLED := 0
endif

ifeq ($(ARMNN_COMPUTE_NEON_ENABLE),0)
ARMNN_COMPUTE_NEON_ENABLED := 0
endif

ifeq ($(ARMNN_REF_ENABLE),0)
ARMNN_REF_ENABLED := 0
endif

ifeq ($(ARMNN_ETHOSN_ENABLE),0)
ARMNN_ETHOSN_ENABLED := 0
endif

# Variable to control inclusion of libOpenCL shared library
ARMNN_INCLUDE_LIBOPENCL := $(ARMNN_COMPUTE_CL_ENABLED)
ifeq ($(ARMNN_LIBOPENCL),0)
ARMNN_INCLUDE_LIBOPENCL := 0
endif

# Variable to control retire rate of priority queue
RETIRE_RATE := 3

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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

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

ifeq ($(ARMNN_ETHOSN_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNETHOSN_ENABLED
endif # ARMNN_ETHOSN_ENABLED == 1

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
        libflatbuffers-framework \
        arm_compute_library

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
endif # Q or later

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_NN_V1_1

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

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

ifeq ($(ARMNN_ETHOSN_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNETHOSN_ENABLED
endif # ARMNN_ETHOSN_ENABLED == 1

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
        libflatbuffers-framework \
        arm_compute_library

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

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_NN_V1_2

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

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

ifeq ($(ARMNN_ETHOSN_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNETHOSN_ENABLED
endif # ARMNN_ETHOSN_ENABLED == 1

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
        libflatbuffers-framework \
        arm_compute_library

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

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # R or later

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_STATIC_LIBRARY)

endif # PLATFORM_VERSION == Q

ifeq ($(R_OR_LATER),1)
# The following target is available starting from Android R

#######################
# libarmnn-driver@1.3 #
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver@1.3
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(ARMNN_UTILS_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Werror \
        -Wno-format-security \
        -DARMNN_ANDROID_NN_V1_3 \

ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS+= \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

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

ifeq ($(ARMNN_ETHOSN_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNETHOSN_ENABLED
endif # ARMNN_ETHOSN_ENABLED == 1

LOCAL_CFLAGS += \
        -DRETIRE_RATE=$(RETIRE_RATE)

LOCAL_SRC_FILES := \
        1.0/ArmnnDriverImpl.cpp \
        1.0/HalPolicy.cpp \
        1.1/ArmnnDriverImpl.cpp \
        1.1/HalPolicy.cpp \
        1.2/ArmnnDriverImpl.cpp \
        1.2/HalPolicy.cpp \
        1.3/ArmnnDriverImpl.cpp \
        1.3/HalPolicy.cpp \
        ArmnnDevice.cpp \
        ArmnnDriverImpl.cpp \
        ArmnnPreparedModel.cpp \
        ArmnnPreparedModel_1_2.cpp \
        ArmnnPreparedModel_1_3.cpp \
        ConversionUtils.cpp \
        DriverOptions.cpp \
        ModelToINetworkConverter.cpp \
        RequestThread.cpp \
        RequestThread_1_3.cpp \
        Utils.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libflatbuffers-framework \
        arm_compute_library

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
        libsync \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2 \
        android.hardware.neuralnetworks@1.3

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_STATIC_LIBRARY)

endif # PLATFORM_VERSION == R

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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libflatbuffers-framework \
        arm_compute_library

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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_1

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libflatbuffers-framework \
        arm_compute_library

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

ifeq ($(R_OR_LATER),1)
LOCAL_SHARED_LIBRARIES+= \
        libsync \
        android.hardware.neuralnetworks@1.3
endif # PLATFORM_VERSION == R

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
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
        $(ARMNN_THIRD_PARTY_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_2 \

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

# Required to build with the changes made to the Android ML framework specific to Android R
ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libflatbuffers-framework \
        arm_compute_library

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

#####################################################
# android.hardware.neuralnetworks@1.3-service-armnn #
#####################################################
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.3-service-armnn
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.3-service-armnn.rc
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_THIRD_PARTY_PATH) \
        $(NN_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -DARMNN_ANDROID_NN_V1_3 \

ifeq ($(ANDROID_R),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_R
endif

ifeq ($(ANDROID_S),1)
LOCAL_CFLAGS+= \
        -DARMNN_ANDROID_S
endif

ifeq ($(ARMNN_DRIVER_DEBUG),1)
LOCAL_CFLAGS += \
        -UNDEBUG
endif # ARMNN_DRIVER_DEBUG == 1

LOCAL_SRC_FILES := \
        service.cpp

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libflatbuffers-framework \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := \
        libarmnn-driver@1.3

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
        libsync \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0 \
        android.hardware.neuralnetworks@1.0 \
        android.hardware.neuralnetworks@1.1 \
        android.hardware.neuralnetworks@1.2 \
        android.hardware.neuralnetworks@1.3

ifeq ($(ARMNN_INCLUDE_LIBOPENCL),1)
LOCAL_SHARED_LIBRARIES+= \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)

endif # PLATFORM_VERSION == R

##########################
# armnn module and tests #
##########################
# Note we use ANDROID_NN_DRIVER_LOCAL_PATH rather than LOCAL_PATH because LOCAL_PATH will be overwritten
# when including other .mk files that set it.
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/armnn/Android.mk
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/test/Android.mk
