"""TensorFlow GPU check with WSL2 / old-driver workarounds.

Two problems break TensorFlow 2.21 in this environment:

1. The TensorFlow wheel's RUNPATH does not contain
   ``site-packages/nvidia/cusolver/lib``, so ``libcusolver.so.11`` cannot be
   dlopen'ed and TensorFlow skips all GPU devices.
2. The Windows NVIDIA driver is 528.49 (CUDA 12.0), which does not implement
   ``CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`` (132).  TensorFlow 2.21 queries
   that attribute unconditionally during ``CudaExecutor::Init()`` and treats
   the resulting ``CUDA_ERROR_INVALID_VALUE`` as fatal.

Both fixes must be active before the process starts, so re-exec once with
``LD_LIBRARY_PATH`` (for libcusolver) and ``LD_PRELOAD`` (for the shim that
reports unsupported attributes as 0).

See docs/ for the full debugging write-up.
"""
import glob
import os
import sys
import sysconfig

ROOT = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(ROOT, "gpu_compat", "libcuda_compat.so")


def find_cusolver_dir():
    """Locate site-packages/nvidia/cusolver/lib for the active interpreter."""
    candidates = []
    try:
        candidates.append(
            os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cusolver", "lib")
        )
    except KeyError:
        pass
    # Fall back to a .venv sitting next to this script.
    candidates += glob.glob(
        os.path.join(ROOT, ".venv", "lib*", "python*", "site-packages",
                     "nvidia", "cusolver", "lib")
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def bootstrap():
    if os.environ.get("TF_WSL_GPU_COMPAT") == "1":
        return
    if not os.path.exists(SHIM):
        print(
            "warning: %s not found; run gpu_compat/build.sh first" % SHIM,
            file=sys.stderr,
        )
        return

    cusolver = find_cusolver_dir()
    if cusolver is None:
        print("warning: nvidia/cusolver/lib not found in site-packages",
              file=sys.stderr)

    env = os.environ.copy()
    env["TF_WSL_GPU_COMPAT"] = "1"
    env["LD_PRELOAD"] = SHIM + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    if cusolver:
        env["LD_LIBRARY_PATH"] = cusolver + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    os.execve(sys.executable,
              [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
              env)


bootstrap()

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
