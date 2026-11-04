# 03. 원인 2 — attribute 132 와 드라이버 528.49

원인 1(cusolver RUNPATH)을 고치면 GPU는 등록되지만, 이번엔 컨텍스트 생성에서
죽습니다.

```
RuntimeError: Bad StatusOr access: INTERNAL: CUDA error: : CUDA_ERROR_INVALID_VALUE: invalid argument
```

`CUDA_ERROR_INVALID_VALUE` 는 "인자 값이 잘못됨" 이라 원인을 특정하기 어렵습니다.
어떤 CUDA 호출이 어떤 인자로 실패했는지 직접 봐야 합니다.

## gdb 로 libcuda 진입점 추적

`libcuda.so.1` 의 exported symbol 중 `cu[A-Z]...` 를 전부 브레이크포인트로 걸고
호출 이름을 로그로 남깁니다. (`debug/trace_cuda.gdb`)

핵심은 **`cuInit` 이 실제로 호출된 뒤에** 브레이크포인트를 설치하는 것입니다.
프로세스 시작 전에 `rbreak` 를 걸면 아직 로드되지 않은 심볼은 잡히지 않습니다.

```gdb
set pagination off
set confirm off
set breakpoint pending on
break cuInit
run
python
import gdb
out = open('/tmp/cuda_calls.log', 'w', buffering=1)
gdb.execute('rbreak ^cu[A-Z]', to_string=True)
...
end
continue
```

결과 (요약):

```
$ sort cuda_calls.log | uniq -c | sort -rn | head
  11179 cuModuleGetFunction
    413 cuGetProcAddress_v2
    381 cuDeviceGetAttribute
    330 cuLibraryUnload
    ...
      4 cuInit
      2 cuGetErrorName    <- 실패 지점
```

## 실패한 호출의 인자 보기

`cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)` 의
두 번째 인자는 SysV ABI 에서 `%esi` 입니다. 이걸 덤프합니다.
(`debug/trace_attr.gdb`)

```gdb
class AttrBP(gdb.Breakpoint):
    def stop(self):
        attr = int(gdb.parse_and_eval('$esi')) & 0xffffffff
        dev  = int(gdb.parse_and_eval('$edx')) & 0xffffffff
        out.write('attr %d dev %d\n' % (attr, dev))
        return False
```

`cuGetErrorName` 이 호출되는 순간 백트레이스도 같이 남깁니다.

## 결과

```
attr 118 dev 0
attr 119 dev 0
attr 121 dev 0
attr 125 dev 0
attr 120 dev 0
attr 129 dev 0
attr 102 dev 0
attr 110 dev 0
attr 132 dev 0      <-- 여기
attr 132 dev 0
cuGetErrorName code 1
--- backtrace ---
#0  cuGetErrorName () from /usr/lib/wsl/lib/libcuda.so.1
#1  stream_executor::cuda::internal::ToStatusSlow(cudaError_enum, ...)
#2  stream_executor::gpu::CudaExecutor::Init()
#3  stream_executor::gpu::CudaPlatform::GetUncachedExecutor(int)
#4  ... ExecutorForDevice(int)::$_0 ...
#5  stream_executor::ExecutorCache::GetOrCreate(...)
#6  stream_executor::gpu::CudaPlatform::ExecutorForDevice(int)
#7  tensorflow::BaseGPUDeviceFactory::GetInterconnectMaps(...)
#8  tensorflow::BaseGPUDeviceFactory::CreateDevices(...)
#9  tensorflow::DeviceFactory::AddDevices(...)
#10 TFE_NewContext ()
#11 pybind11 ... _pywrap_tfe.so
```

두 가지가 확인됩니다.

1. 실패 직전 마지막 `cuDeviceGetAttribute` 인자가 **132**
2. 에러를 `CudaExecutor::Init()` 이 `ToStatusSlow` 로 승격 → `TFE_NewContext` 실패

## 132 는 무엇인가

`cuda.h` (CUDA 12.5 헤더, `nvidia-cuda-runtime-cu12` 패키지) 에서:

```c
CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
```

CUDA 12.5 드라이버에 추가된 attribute 입니다.
`nvidia-smi` 가 보고한 드라이버는 **528.49 (CUDA 12.0)** — 이 attribute 를 모릅니다.
모르는 attribute 번호를 넘기면 드라이버는 `CUDA_ERROR_INVALID_VALUE`(1) 를 반환합니다.

```python
import ctypes
cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
d = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(d), 0)
v = ctypes.c_int()
cu.cuDeviceGetAttribute(ctypes.byref(v), 132, d)   # -> 1
cu.cuDeviceGetAttribute(ctypes.byref(v), 0, d)     # -> 0 (정상 attribute)
```

## 왜 TensorFlow 2.21 이 문제인가

`CudaExecutor::Init()` 은 이 attribute 를 **조건 없이** 조회하고, 반환값이
`CUDA_SUCCESS` 가 아니면 초기화 실패로 처리합니다. 구 드라이버에서는
"이 기능은 지원 안 함" 이 아니라 "잘못된 인자" 로 보이기 때문에
복구 로직이 개입할 여지가 없습니다.

정리하면:

- 원인 1: cusolver 경로 → GPU **등록** 실패
- 원인 2: attribute 132 → GPU **초기화** 실패

둘 다 고쳐야 실제 연산이 됩니다. → [04-workaround.md](04-workaround.md)
