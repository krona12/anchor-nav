#!/usr/bin/env bash

LOCAL_NVIDIA_ROOT="${LOCAL_NVIDIA_ROOT:-/home/chenlin/krona/anchor-nav/local_nvidia_580_126_09/root}"
LOCAL_NVIDIA_LIB="${LOCAL_NVIDIA_ROOT}/usr/lib/x86_64-linux-gnu"
LOCAL_NVIDIA_EGL_JSON="${LOCAL_NVIDIA_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

if [ ! -d "${LOCAL_NVIDIA_LIB}" ]; then
  echo "local NVIDIA lib dir not found: ${LOCAL_NVIDIA_LIB}" >&2
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "${LOCAL_NVIDIA_EGL_JSON}" ]; then
  echo "local NVIDIA EGL vendor JSON not found: ${LOCAL_NVIDIA_EGL_JSON}" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${LOCAL_NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${LOCAL_NVIDIA_EGL_JSON}"
export PATH="${LOCAL_NVIDIA_ROOT}/usr/bin${PATH:+:${PATH}}"
export LOCAL_NVIDIA_ROOT

echo "Using local NVIDIA 580.126.09 user-space libraries: ${LOCAL_NVIDIA_LIB}"
