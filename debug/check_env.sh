#!/bin/sh
# Environment diagnostic for WSL2 + TensorFlow GPU issues.
#
#   ./debug/check_env.sh [path-to-venv]
#
# Prints the facts that actually matter when TensorFlow reports
# "Cannot dlopen some GPU libraries" or fails to create a GPU context.
set -u

VENV="${1:-.venv}"
PY="$VENV/bin/python"

hr() { echo; echo "=== $* ==="; }

hr "host"
uname -r
[ -r /etc/os-release ] && . /etc/os-release && echo "$PRETTY_NAME"

hr "nvidia-smi"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 \
    || echo "nvidia-smi not available"

hr "wsl device nodes"
ls -l /dev/dxg 2>&1
ls -l /usr/lib/wsl/lib/libcuda.so.1 2>&1

hr "python / tensorflow"
if [ -x "$PY" ]; then
    "$PY" - <<'EOF' 2>/dev/null
import sys
print("python:", sys.version.split()[0])
try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
except Exception as e:
    print("tensorflow: import failed:", e)
EOF
else
    echo "no interpreter at $PY"
fi

hr "nvidia pip packages"
if [ -x "$PY" ]; then
    "$PY" - <<'EOF'
import importlib.metadata as m
names = {
    (d.metadata["Name"], d.version)
    for d in m.distributions()
    if (d.metadata["Name"] or "").startswith("nvidia")
}
for n, v in sorted(names):
    print(f"{n}=={v}")
EOF
fi

hr "libtensorflow RUNPATH (nvidia dirs)"
FW=$(ls "$VENV"/lib*/python*/site-packages/tensorflow/libtensorflow_framework.so.2 2>/dev/null | head -1)
if [ -n "${FW:-}" ] && command -v eu-readelf >/dev/null 2>&1; then
    eu-readelf -d "$FW" | grep -o 'nvidia/[a-z_]*/lib' | sort -u
    echo "---"
    if eu-readelf -d "$FW" | grep -q 'nvidia/cusolver/lib'; then
        echo "cusolver in RUNPATH: yes"
    else
        echo "cusolver in RUNPATH: NO  <- root cause #1"
    fi
else
    echo "skipped (need eu-readelf from elfutils)"
fi

hr "driver CUDA version vs attribute 132"
"$PY" - <<'EOF' 2>/dev/null
try:
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    if cu.cuInit(0) != 0:
        print("cuInit failed")
        raise SystemExit
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    val = ctypes.c_int()
    for attr, label in ((1, "MAX_THREADS_PER_BLOCK"),
                        (75, "COMPUTE_CAPABILITY_MAJOR"),
                        (132, "MULTICAST_SUPPORTED")):
        r = cu.cuDeviceGetAttribute(ctypes.byref(val), attr, dev)
        print(f"attr {attr:3d} ({label}): rc={r} value={val.value}")
    print("rc=1 for attr 132 means the driver predates CUDA 12.5"
          " <- root cause #2")
except Exception as e:
    print("probe failed:", e)
EOF

hr "GPU visible to tensorflow (without shim)"
if [ -x "$PY" ]; then
    "$PY" -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))' 2>&1 \
        | grep -E "Cannot dlopen|Could not load|dlerror|Skipping|\[|RuntimeError" | head -20
fi
