#!/bin/bash

function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo "$1"
    echo "+++ Command exited with code $EXITCODE. Please fix the above errors and re-run"
    exit 1
  fi
}

if [ ! -d boost_1_64_0 ]; then
  echo "++ Downloading Boost"

  BOOST_PKG=boost_1_64_0.tar.gz

  # There is a problem with downloading boost from the external. Issue can be found here:https://github.com/boostorg/boost/issues/299.
  # Using a mirror link to download boost.
  curl -LOk https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
  # curl -LOk https://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.gz # had switched to this mirror as we were not able to download boost from boostorg.
  AssertZeroExitCode "Downloading Boost failed"

  tar xzf $BOOST_PKG
  AssertZeroExitCode "Unpacking Boost failed"

  rm -rf $BOOST_PKG
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
    arch=arm64-v8a build_dir=android-arm64v8a benchmark_tests=0 -j16 \
    build/android-arm64v8a/src/core/arm_compute_version.embed build/android-arm64v8a/src/core/CL/cl_kernels
AssertZeroExitCode "Precompiling clframework failed"
popd

