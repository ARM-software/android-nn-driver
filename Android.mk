#
# Copyright © 2017 ARM Ltd. All rights reserved.
# See LICENSE file in the project root for full license information.
#

ANDROID_NN_DRIVER_LOCAL_PATH := $(call my-dir)
LOCAL_PATH := $(ANDROID_NN_DRIVER_LOCAL_PATH)

# Configure these paths if you move the source or Khronos headers
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../mali/product/khronos/original
NN_HEADER_PATH := $(LOCAL_PATH)/../../../frameworks/ml/nn/runtime/include

###################
# libarmnn-driver #
###################
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn-driver
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES :=	 \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH)

LOCAL_CFLAGS := \
	-std=c++14 \
	-fexceptions \
	-Werror \
	-Wno-format-security
ifeq ($(ARMNN_DRIVER_DEBUG),1)
	LOCAL_CFLAGS+= -UNDEBUG
endif

LOCAL_SRC_FILES :=	\
	ArmnnDriver.cpp \
	ArmnnPreparedModel.cpp \
	ModelToINetworkConverter.cpp \
	RequestThread.cpp \
	Utils.cpp

LOCAL_STATIC_LIBRARIES := \
	libneuralnetworks_common \
	libarmnn \
	libboost_log \
	libboost_program_options \
	libboost_system \
	libboost_thread \
	armnn-arm_compute \

LOCAL_SHARED_LIBRARIES :=  \
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

include $(BUILD_STATIC_LIBRARY)

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

LOCAL_C_INCLUDES :=	 \
	$(NN_HEADER_PATH)

LOCAL_CFLAGS := \
	-std=c++14 \
	-fexceptions
ifeq ($(ARMNN_DRIVER_DEBUG),1)
	LOCAL_CFLAGS+= -UNDEBUG
endif

LOCAL_SRC_FILES :=	\
	service.cpp

LOCAL_STATIC_LIBRARIES := \
	libarmnn-driver \
	libneuralnetworks_common \
	libarmnn \
	libboost_log \
	libboost_program_options \
	libboost_system \
	libboost_thread \
	armnn-arm_compute

LOCAL_SHARED_LIBRARIES :=  \
	libbase \
	libhidlbase \
	libhidltransport \
	libhidlmemory \
	libdl \
	libhardware \
	libtextclassifier \
	libtextclassifier_hash \
	liblog \
	libutils \
	android.hardware.neuralnetworks@1.0 \
	android.hidl.allocator@1.0 \
	android.hidl.memory@1.0 \
	libOpenCL

include $(BUILD_EXECUTABLE)

##########################
# armnn module and tests #
##########################
# Note we use ANDROID_NN_DRIVER_LOCAL_PATH rather than LOCAL_PATH because LOCAL_PATH will be overwritten
# when including other .mk files that set it.
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/armnn/Android.mk
include $(ANDROID_NN_DRIVER_LOCAL_PATH)/test/Android.mk