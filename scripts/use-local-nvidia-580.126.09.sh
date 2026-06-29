#!/usr/bin/env bash

LOCAL_NVIDIA_ROOT="${LOCAL_NVIDIA_ROOT:-/home/chenlin/krona/anchor-nav/local_nvidia_580_126_09/root}"
LOCAL_NVIDIA_LIB="${LOCAL_NVIDIA_ROOT}/usr/lib/x86_64-linux-gnu"
LOCAL_NVIDIA_EGL_JSON="${LOCAL_NVIDIA_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
LOCAL_NVIDIA_VERSION="${LOCAL_NVIDIA_VERSION:-580.126.09}"

_remove_local_nvidia_path_entries() {
  local value="$1"
  local cleaned=""
  local entry
  local old_ifs="${IFS}"
  IFS=':'
  for entry in ${value}; do
    case "${entry}" in
      *local_nvidia_580_126_09*) ;;
      "")
        if [ -z "${cleaned}" ]; then cleaned="${entry}"; fi
        ;;
      *)
        if [ -z "${cleaned}" ]; then
          cleaned="${entry}"
        else
          cleaned="${cleaned}:${entry}"
        fi
        ;;
    esac
  done
  IFS="${old_ifs}"
  printf '%s' "${cleaned}"
}

if [ "${FORCE_LOCAL_NVIDIA:-0}" != "1" ] && [ -r /proc/driver/nvidia/version ]; then
  if ! grep -q "${LOCAL_NVIDIA_VERSION}" /proc/driver/nvidia/version; then
    echo "Skipping local NVIDIA ${LOCAL_NVIDIA_VERSION} user-space libraries; kernel driver is:" >&2
    sed -n '1p' /proc/driver/nvidia/version >&2
    export LD_LIBRARY_PATH="$(_remove_local_nvidia_path_entries "${LD_LIBRARY_PATH:-}")"
    export PATH="$(_remove_local_nvidia_path_entries "${PATH:-}")"
    unset __EGL_VENDOR_LIBRARY_FILENAMES
    unset LOCAL_NVIDIA_ROOT
    return 0 2>/dev/null || exit 0
  fi
fi

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
