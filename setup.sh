#!/bin/bash

function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo "$1"
    echo "+++ Command exited with code $EXITCODE. Please fix the above errors and re-run"
    exit 1
  fi
}

if [ ! -d v1.12.0 ]; then
  echo "++ Downloading FlatBuffers"

  FLATBUFFERS_PKG=v1.12.0.tar.gz

  curl -LOk https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
  AssertZeroExitCode "Downloading FlatBuffers failed"

  tar xzf $FLATBUFFERS_PKG
  AssertZeroExitCode "Unpacking FlatBuffers failed"

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

