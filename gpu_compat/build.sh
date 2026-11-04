#!/bin/sh
# Rebuild the LD_PRELOAD compatibility shim.
#
# Requires gcc and patchelf.  If they are not on PATH (e.g. openSUSE
# Tumbleweed without gcc installed), use nix:
#
#   nix shell nixpkgs#gcc nixpkgs#patchelf --command ./build.sh
#
# The patchelf --remove-rpath step is required when building with nix's gcc:
# without it the .so carries a nix-store glibc RUNPATH and will crash when
# preloaded into a process linked against the system glibc.
set -eu
cd "$(dirname "$0")"

for tool in gcc patchelf; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found in PATH" >&2
        echo "       try: nix shell nixpkgs#gcc nixpkgs#patchelf --command ./build.sh" >&2
        exit 1
    fi
done

gcc -O2 -Wall -Wextra -shared -fPIC -o libcuda_compat.so cuda_compat.c -ldl
patchelf --remove-rpath libcuda_compat.so

echo "built: $(pwd)/libcuda_compat.so"
