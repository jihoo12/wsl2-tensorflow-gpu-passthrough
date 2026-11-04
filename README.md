# WSL2 + TensorFlow GPU passthrough (구 드라이버 대응)

WSL2(openSUSE Tumbleweed) + TensorFlow 2.21 + **구형 Windows NVIDIA 드라이버(528.49, CUDA 12.0)**
환경에서 `tf.config.list_physical_devices("GPU")` 가 `[]` 를 반환하던 문제의
**디버깅 과정과 해결 방법**을 정리한 저장소입니다.

```
TensorFlow: 2.21.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## TL;DR

원인이 **두 개** 겹쳐 있었습니다.

| # | 원인 | 증상 |
|---|------|------|
| 1 | TensorFlow wheel의 `RUNPATH` 에 `nvidia/cusolver/lib` 가 없음 | `libcusolver.so.11` dlopen 실패 → "Cannot dlopen some GPU libraries" → GPU 등록 자체를 건너뜀 |
| 2 | Windows 드라이버가 CUDA 12.0 세대라 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`(132) 미지원 | GPU는 잡히지만 `TFE_NewContext` 가 `CUDA_ERROR_INVALID_VALUE` 로 실패 |

해결:

1. `LD_LIBRARY_PATH` 에 `site-packages/nvidia/cusolver/lib` 추가 → 원인 1
2. `LD_PRELOAD` shim 으로, TensorFlow가 구 드라이버에 없는 attribute(≥132)를 조회하면
   "미지원(0)" 으로 응답 → 원인 2
3. 두 환경변수는 **프로세스 시작 전**에만 유효하므로 `test.py` 가 `os.execve()` 로 자기 자신을 재실행

근본 해결은 **Windows 쪽 NVIDIA 드라이버 업데이트**입니다. 이 저장소의 workaround 는
드라이버를 건드릴 수 없을 때 쓰는 임시 조치입니다. → [docs/05-native-fix.md](docs/05-native-fix.md)

---

## 검증 환경

| 항목 | 값 |
|------|-----|
| 배포판 | openSUSE Tumbleweed 20261030 |
| 커널 | `6.18.33.2-microsoft-standard-WSL2` |
| GPU | NVIDIA RTX A4000 16GB (compute capability 8.6) |
| Windows 드라이버 | **528.49** (`nvidia-smi` 기준 CUDA 12.0) |
| Python | 3.13.16 |
| TensorFlow | 2.21.0 (CUDA 12.5 / cuDNN 9.3 빌드) |
| `nvidia-*` pip 패키지 | 12.5 계열, `nvidia-cudnn-cu12==9.3.0.75` |

> WSL2는 Windows에 설치된 드라이버를 그대로 씁니다. 컨테이너/배포판 안에서
> 드라이버를 업데이트해도 `nvidia-smi` 버전은 바뀌지 않습니다.

---

## 사용법 (이 저장소 그대로)

```sh
# 1) shim 빌드 (gcc/patchelf 필요; 없으면 nix로)
nix shell nixpkgs#gcc nixpkgs#patchelf --command ./gpu_compat/build.sh

# 2) venv 준비 (CUDA 12.5 세대 pin)
uv venv .venv
uv pip install --python .venv/bin/python --no-deps -r requirements/nvidia-cu125.txt
uv pip install --python .venv/bin/python tensorflow==2.21.0

# 3) 확인
.venv/bin/python test.py
```

`test.py` 는 shim 과 `LD_LIBRARY_PATH` 를 스스로 설정한 뒤 재실행하므로,
환경변수를 직접 export 할 필요가 없습니다.

실제 연산까지 확인하려면:

```sh
.venv/bin/python -c "
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
with tf.device('/GPU:0'):
    c = tf.matmul(tf.random.normal((1500,1500)), tf.random.normal((1500,1500)))
    print(c.device, float(tf.reduce_sum(c)))
"
```

```
device: /job:localhost/replica:0/task:0/device:GPU:0
sum: 4443.8935546875
```

---

## 진단 스크립트

원인 1·2 를 한 번에 확인할 수 있습니다.

```sh
./debug/check_env.sh .venv
```

```
=== libtensorflow RUNPATH (nvidia dirs) ===
nvidia/cublas/lib
...
cusolver in RUNPATH: NO  <- root cause #1

=== driver CUDA version vs attribute 132 ===
attr   1 (MAX_THREADS_PER_BLOCK): rc=0 value=1024
attr  75 (COMPUTE_CAPABILITY_MAJOR): rc=0 value=8
attr 132 (MULTICAST_SUPPORTED): rc=1 value=8
rc=1 for attr 132 means the driver predates CUDA 12.5 <- root cause #2
```

---

## 저장소 구조

```
README.md                      이 문서
test.py                        재실행 bootstrap + GPU 확인
gpu_compat/
  cuda_compat.c                LD_PRELOAD shim (dlsym 후킹)
  build.sh                     빌드 스크립트 (patchelf로 RUNPATH 제거)
  README.md                    shim 설계 설명
requirements/nvidia-cu125.txt  TF 2.21에 맞는 nvidia pip 핀
docs/
  01-symptom.md                최초 증상과 로그
  02-root-cause-cusolver.md    원인 1: RUNPATH 누락
  03-root-cause-attribute132.md 원인 2: attribute 132 (gdb 추적)
  04-workaround.md             shim 설계, 재실행 bootstrap, 함정
  05-native-fix.md             근본 해결 / 다른 대안들
debug/
  check_env.sh                 환경 진단 스크립트
  trace_attr.gdb               cuDeviceGetAttribute 인자 + 실패 백트레이스 추적
  trace_cuda.gdb               libcuda 진입점 전체 호출 순서 추적
```

---

## 디버깅 요약

### 1단계: GPU가 아예 등록되지 않음

`nvidia-smi` 는 정상, `/dev/dxg` 도 존재, `libcuda.so.1` 도 로드 가능한데
TensorFlow만 `[]` 를 반환했습니다.

```
W0000 gpu_device.cc:2365] Cannot dlopen some GPU libraries.
I0000 dso_loader.cc:144] Could not load dynamic library 'libcusolver.so.11';
      dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory
Skipping registering GPU devices...
```

wheel 안의 `RUNPATH` 를 까보면 `cublas`, `cudnn`, `cufft`, `cusparse`, `nccl`,
`cuda_cupti`, `cuda_runtime` 은 있는데 **`cusolver` 만 빠져 있습니다.**

```sh
eu-readelf -d .venv/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.so.2 \
  | grep -o 'nvidia/[a-z_]*/lib' | sort -u
# nvidia/cublas/lib
# nvidia/cuda_cupti/lib
# nvidia/cuda_runtime/lib
# nvidia/cudnn/lib
# nvidia/cufft/lib
# nvidia/cusparse/lib
# nvidia/nccl/lib
# nvidia/nvshmem/lib      <- cusolver 없음
```

`LD_LIBRARY_PATH` 로 cusolver 경로만 넣어주면 GPU가 **등록은** 됩니다.
그런데 여기서 끝이 아니었습니다.

### 2단계: GPU는 잡히는데 컨텍스트 생성이 실패

```
tensorflow/core/common_runtime/gpu/gpu_device.cc ... 
RuntimeError: Bad StatusOr access: INTERNAL: CUDA error: : CUDA_ERROR_INVALID_VALUE: invalid argument
```

`ld_preload` 로 libcuda 호출을 전부 로깅하고, `cuGetErrorName(1)` 이 불리는
순간의 백트레이스를 잡아보니:

```
#0 cuGetErrorName () from /usr/lib/wsl/lib/libcuda.so.1
#1 stream_executor::cuda::internal::ToStatusSlow(cudaError_enum, ...)
#2 stream_executor::gpu::CudaExecutor::Init()
...
#10 TFE_NewContext ()
```

`cuDeviceGetAttribute` 인자를 덤프하면 마지막 두 번이 `attr 132` 입니다.

```
attr 118 dev 0
attr 119 dev 0
attr 121 dev 0
attr 125 dev 0
attr 120 dev 0
attr 129 dev 0
attr 102 dev 0
attr 110 dev 0
attr 132 dev 0   <- CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
attr 132 dev 0
cuGetErrorName code 1
```

`132 = CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` 는 CUDA 12.5 드라이버에 추가된
attribute 입니다. 528.49 드라이버는 이 값을 모르므로 `CUDA_ERROR_INVALID_VALUE`(1)를
돌려주고, TensorFlow 2.21의 `CudaExecutor::Init()` 은 이를 **치명적 오류로 처리**합니다.

자세한 과정: [docs/03-root-cause-attribute132.md](docs/03-root-cause-attribute132.md)

---

## 왜 이 방식인가

- `LD_PRELOAD` 는 프로세스 시작 시점에만 적용되므로, 파이썬 스크립트 안에서
  `os.environ` 을 수정하는 것으로는 부족합니다. → `os.execve()` 재실행
- 인터셉트는 `cuDeviceGetAttribute` **자체**가 아니라 `dlsym("cuDeviceGetAttribute")`
  에 겁니다. WSL의 `libcuda.so.1` 래퍼가 내부적으로 같은 심볼을 `dlsym` 으로
  찾기 때문에, 심볼을 통째로 갈아치우면 **무한 재귀**로 죽습니다.
  → 호출자가 TensorFlow일 때만 shim을 돌려주고, 그 외에는 원본을 그대로 반환
- shim 은 `TensorFlow` 가 dlsym 한 포인터로만 호출되므로, 다른 라이브러리
  (PyTorch 등)의 정상 동작에는 영향을 주지 않습니다.

자세한 설계와 함정: [docs/04-workaround.md](docs/04-workaround.md)

---

## 주의

- 이 workaround 는 `CUDA_ERROR_INVALID_VALUE` 를 **"미지원"으로 위장**하는 것입니다.
  attribute 132가 실제로 필요한 기능(multicast / NVLink 기반 collectives)은
  사용할 수 없습니다. 단일 GPU 워크로드에서는 문제되지 않습니다.
- attribute 값 `>= 132` 를 전부 0으로 처리하므로, 나중에 드라이버를 업데이트하면
  **shim 을 제거**해야 합니다. (불필요하게 새 기능을 끄게 됨)
- 근본 해결: [docs/05-native-fix.md](docs/05-native-fix.md)
