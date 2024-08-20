#!/bin/bash

#
# Copyright © 2018,2020-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo "$1"
    echo "+++ Command exited with code $EXITCODE. Please fix the above errors and re-run"
    exit 1
  fi
}

BUILD_DIR=build-x86_64
FLATBUFFERS_DIR=$PWD/flatbuffers

function BuildCmake {
  CMAKE_VERSION=3.22.1
  CMAKE_DIST="cmake-3.22.1"
  # Note that the Cmake version with and without the patch release are used in this URL.
  CMAKE_URL="https://cmake.org/files/v3.22/${CMAKE_DIST}.tar.gz"
  CMAKE_ZIP=$PWD/${CMAKE_DIST}.tar.gz
  CMAKE_PKG=$PWD/${CMAKE_DIST}

  # If the correct package doesn't exist then download and unpack it.
  if [[ ! -d "$CMAKE_PKG" ]]; then
      # Download the version of CMake we want
      echo "+++ Downloading CMake"
      mkdir -p cmake
      wget -O $CMAKE_ZIP $CMAKE_URL
      AssertZeroExitCode "Downloading CMake failed"
      echo "+++ Unpacking CMake"
      tar -xzf "$CMAKE_ZIP" -C "$PWD"
      AssertZeroExitCode "Unpacking CMake failed"
  fi
  # This version of cmake will be used in devenv only.
  echo "+++ Building CMake"
  cd "$CMAKE_PKG" || exit 1
  ./bootstrap --prefix=$PWD/${CMAKE_DIST}/ || exit 1
  make || exit 1
  make install || exit 1
  echo "+++ CMake Successfully Installed in $PWD/${CMAKE_DIST}/"

}

function BuildFlatbuffers {
  pushd flatbuffers
  rm -rf $BUILD_DIR
  rm -f CMakeCache.txt
  FLATBUFFERS_DIR=$PWD

  mkdir -p $BUILD_DIR
  cd $BUILD_DIR

  echo "+++ Building Google Flatbufers"
  CMD="$PWD/${CMAKE_DIST}/bin/cmake -DFLATBUFFERS_BUILD_FLATC=1 -DCMAKE_INSTALL_PREFIX:PATH=$FLATBUFFERS_DIR .."
  # Force -fPIC to allow relocatable linking.
  CXXFLAGS="-fPIC" $CMD
  AssertZeroExitCode "cmake Google Flatbuffers failed. command was: ${CMD}"
  make all install
  AssertZeroExitCode "Building Google Flatbuffers failed"
  mkdir -p $FLATBUFFERS_DIR/bin
  cp -f flatc $FLATBUFFERS_DIR/bin
  AssertZeroExitCode "Failed to copy the Flatbuffers Compiler"
  popd
}

if [ ! -d flatbuffers ]; then
  echo "++ Downloading FlatBuffers v23.5.26"

  FLATBUFFERS_PKG=v23.5.26.tar.gz

  curl -LOk https://github.com/google/flatbuffers/archive/${FLATBUFFERS_PKG}
  AssertZeroExitCode "Downloading FlatBuffers failed"
  mkdir -p flatbuffers
  tar xzf $FLATBUFFERS_PKG
  AssertZeroExitCode "Unpacking FlatBuffers failed"

  BuildCmake
  BuildFlatbuffers

  rm -rf $FLATBUFFERS_PKG
fi

if [ ! -d armnn ]; then
  echo "++ Downloading armnn"

  git clone git@github.com:ARM-software/armnn armnn
  AssertZeroExitCode "Cloning armnn failed"
fi

if [ ! -d clframework ]; then
  echo "++ Downloading clframework"

  git clone git@github.com:ARM-software/ComputeLibrary clframework
  AssertZeroExitCode "Cloning clframework failed"
fi

# Get scons to create the generated source code which clframework needs to compile.
# This is required for the Android build system to build clframework (see below)
pushd clframework
scons os=android build=embed_only neon=0 opencl=1 embed_kernels=1 validation_tests=0 \
    arch=arm64-v8.2-a build_dir=android-arm64v8.2-a benchmark_tests=0 -j16 \
    build/android-arm64v8.2-a/src/core/arm_compute_version.embed build/android-arm64v8.2-a/src/core/CL/cl_kernels
AssertZeroExitCode "Precompiling clframework failed for v82.a"

scons os=android build=embed_only neon=0 opencl=1 embed_kernels=1 validation_tests=0 \
    arch=arm64-v8a build_dir=android-arm64v8a benchmark_tests=0 -j16 \
    build/android-arm64v8a/src/core/arm_compute_version.embed build/android-arm64v8a/src/core/CL/cl_kernels
AssertZeroExitCode "Precompiling clframework failed for v8a."
popd

if [ ! -d armnn/generated ]; then
  mkdir -p armnn/generated
fi

echo "+++ Generating new ArmnnSchema_generated.h"
$FLATBUFFERS_DIR/bin/flatc -o armnn/generated --cpp armnn/src/armnnSerializer/ArmnnSchema.fbs
AssertZeroExitCode "Generating ArmnnSchema_generated.h failed."
