# 02. 원인 1 — wheel RUNPATH 에 cusolver 가 없다

## 로그

`Cannot dlopen some GPU libraries` 는 TensorFlow의 DSO 로더가 GPU 관련
라이브러리를 하나라도 못 열면 출력합니다. 실제로 못 연 라이브러리는
바로 위 줄에 찍힙니다.

```
I0000 ... dso_loader.cc:144] Could not load dynamic library 'libcusolver.so.11';
      dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory
```

`libcusolver.so.11` 은 **pip 패키지 `nvidia-cusolver-cu12` 안에 존재합니다.**

```
$ ls .venv/lib/python3.13/site-packages/nvidia/cusolver/lib/
__init__.py  libcusolver.so.11  libcusolverMg.so.11
```

파일은 있는데 못 찾는다? 그럼 탐색 경로 문제입니다.

## RUNPATH 확인

TensorFlow wheel 의 네이티브 라이브러리는 `nvidia/*` 디렉터리들을 `$ORIGIN`
기준 상대경로로 RUNPATH 에 넣어둡니다. 어떤 디렉터리가 들어있는지 뽑아봅니다.

```sh
eu-readelf -d \
  .venv/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.so.2 \
  | grep -o 'nvidia/[a-z_]*/lib' | sort -u
```

```
nvidia/cublas/lib
nvidia/cuda_cupti/lib
nvidia/cuda_runtime/lib
nvidia/cudnn/lib
nvidia/cufft/lib
nvidia/cusparse/lib
nvidia/nccl/lib
nvidia/nvshmem/lib
```

**`nvidia/cusolver/lib` 가 없습니다.** `cusparse`, `cufft`, `cublas` 는 있는데
`cusolver` 만 빠져 있습니다. (같은 wheel 의 `libtensorflow_cc.so.2` 도 동일)

## 왜 이게 치명적인가

TensorFlow는 초기화 시 GPU에 필요한 DSO 목록을 순서대로 dlopen 합니다.
하나라도 실패하면 `gpu_device.cc` 에서 GPU 등록을 **전부** 건너뜁니다.

```
W0000 gpu_device.cc:2365] Cannot dlopen some GPU libraries.
Skipping registering GPU devices...
```

그래서 `list_physical_devices("GPU")` 가 빈 리스트가 됩니다.
`libcusolver` 자체를 쓰지 않는 연산이어도, 로딩 단계에서 막히면 끝입니다.

## 최소 재현 / 확인

`LD_LIBRARY_PATH` 로 cusolver 경로만 넣고 다시 실행:

```sh
CS="$PWD/.venv/lib/python3.13/site-packages/nvidia/cusolver/lib"
LD_LIBRARY_PATH="$CS" .venv/bin/python -c \
  "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

GPU가 **등록**됩니다. 여기서 문제가 끝났다면 좋았겠지만, 실제 연산을 돌리면
두 번째 문제가 나옵니다. → [03-root-cause-attribute132.md](03-root-cause-attribute132.md)

## 영구 대안 (root 권한이 있을 때)

`LD_LIBRARY_PATH` 대신 시스템 로더 설정으로 해결할 수도 있습니다.

```sh
# /etc/ld.so.conf.d/nvidia-pip.conf
/home/<user>/.../.venv/lib/python3.13/site-packages/nvidia/cusolver/lib
```

```sh
sudo ldconfig
```

또는 wheel 자체를 패치:

```sh
patchelf --add-rpath '$ORIGIN/../../nvidia/cusolver/lib' \
  .venv/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.so.2
```

이 저장소는 root 없이도 동작하도록 `LD_LIBRARY_PATH` 방식을 씁니다.
