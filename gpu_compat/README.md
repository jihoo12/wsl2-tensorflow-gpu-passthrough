# cuda_compat — LD_PRELOAD shim

TensorFlow 2.21 이 구형 NVIDIA 드라이버에서 GPU 초기화에 실패하는 문제를
우회하는 아주 작은 shim 입니다. 빌드하면 `libcuda_compat.so` 하나가 나옵니다.

## 무엇을 하는가

`CudaExecutor::Init()` 은 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`(132) 를
무조건 조회합니다. CUDA 12.5 에서 추가된 attribute 라, CUDA 12.0 세대 드라이버
(예: 528.49)는 `CUDA_ERROR_INVALID_VALUE`(1) 를 돌려주고 TensorFlow 는 이를
치명적 오류로 처리합니다.

shim 은 TensorFlow 가 `dlsym("cuDeviceGetAttribute")` 로 함수 포인터를 얻는
순간을 가로채, 래퍼 함수를 대신 돌려줍니다. 래퍼는 원본을 호출한 뒤

- 반환값이 `CUDA_ERROR_INVALID_VALUE` 이고
- attribute 가 `132` 이상이고
- 출력 포인터가 유효하면

`*pi = 0` (미지원) 으로 바꿔 `CUDA_SUCCESS` 를 반환합니다.

## 왜 심볼을 직접 덮어쓰지 않는가

`LD_PRELOAD` 로 `cuDeviceGetAttribute` 심볼을 정의하면 TensorFlow 호출은 잡히지만,
WSL 의 `libcuda.so.1` 래퍼가 내부적으로 같은 심볼을 `dlsym` 으로 찾기 때문에
무한 재귀(`SIGSEGV`)가 납니다.

그래서 **호출자를 구분**합니다.

```c
if (name && strcmp(name, "cuDeviceGetAttribute") == 0 &&
    caller_is_tensorflow(__builtin_return_address(0)))
    return (void *)&compat_cuDeviceGetAttribute;
return real_dlsym_fn(handle, name);
```

`caller_is_tensorflow()` 는 `dladdr()` 로 호출자의 모듈 경로를 보고
`libtensorflow` / `lib_pywrap_tf` 인지 확인합니다.

## 빌드

```sh
# gcc, patchelf 가 PATH 에 있을 때
./build.sh

# 이 환경처럼 nix 로 받아야 할 때
nix shell nixpkgs#gcc nixpkgs#patchelf --command ./build.sh
```

`build.sh` 는 빌드 후 `patchelf --remove-rpath` 를 실행합니다.
nix 의 gcc 로 빌드하면 glibc RUNPATH 가 박혀서, 시스템 파이썬에 preload 했을 때
glibc 가 중복 로드되어 죽기 때문입니다.

```sh
$ ldd libcuda_compat.so
	linux-vdso.so.1
	libdl.so.2 => /lib64/libdl.so.2
	libc.so.6 => /lib64/libc.so.6
	/lib64/ld-linux-x86-64.so.2
```

## 사용

프로세스 시작 전에 넣어야 합니다.

```sh
LD_PRELOAD=$PWD/gpu_compat/libcuda_compat.so \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cusolver/lib \
  .venv/bin/python your_script.py
```

`../test.py` 는 이 두 줄을 자동으로 처리합니다.

## 주의

- 드라이버를 업데이트하면 이 shim 을 **제거**하세요. attribute 132 이상을
  전부 0으로 보고하므로, 새 드라이버의 기능을 불필요하게 끄게 됩니다.
- 단일 GPU 환경 전용입니다. multicast / NVLink 기반 기능은 사용할 수 없습니다.
