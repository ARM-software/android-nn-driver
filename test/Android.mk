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

include $(CLEAR_VARS)

LOCAL_C_INCLUDES :=	 \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH) \
	$(ARMNN_HEADER_PATH) \
	$(ARMNN_DRIVER_HEADER_PATH)

LOCAL_CFLAGS := \
	-std=c++14 \
	-fexceptions \
	-Werror \
	-UNDEBUG

LOCAL_SRC_FILES :=	\
	Tests.cpp \
	UtilsTests.cpp

LOCAL_STATIC_LIBRARIES := \
	libarmnn-driver \
	libneuralnetworks_common \
	libarmnn \
	libboost_log \
	libboost_system \
	libboost_unit_test_framework \
	libboost_thread \
	armnn-arm_compute

LOCAL_SHARED_LIBRARIES :=  \
	libbase \
	libhidlbase \
	libhidltransport \
	libhidlmemory \
	libtextclassifier \
	libtextclassifier_hash \
	liblog \
	libutils \
	android.hardware.neuralnetworks@1.0 \
	android.hidl.allocator@1.0 \
	android.hidl.memory@1.0 \
	libOpenCL

LOCAL_MODULE := armnn-driver-tests

LOCAL_MODULE_TAGS := eng optional

LOCAL_ARM_MODE := arm

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

include $(BUILD_EXECUTABLE)



