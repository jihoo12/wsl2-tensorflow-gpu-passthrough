# 04. Workaround 설계

두 원인에 각각 대응합니다.

| 원인 | 대응 | 적용 시점 |
|------|------|-----------|
| RUNPATH 에 cusolver 없음 | `LD_LIBRARY_PATH` 에 `nvidia/cusolver/lib` 추가 | 프로세스 시작 전 |
| attribute 132 미지원 | `LD_PRELOAD` shim 으로 "미지원(0)" 응답 | 프로세스 시작 전 |

## 왜 재실행(execve)이 필요한가

`LD_PRELOAD` 는 **로더가 프로세스를 시작할 때** 읽습니다.
`LD_LIBRARY_PATH` 도 마찬가지로, 이미 로드된 라이브러리의 탐색에는 소급 적용되지
않습니다. 파이썬 스크립트 안에서

```python
os.environ["LD_PRELOAD"] = ...
```

라고 해봐야 아무 효과가 없습니다. (자식 프로세스에는 적용됩니다.)

그래서 `test.py` 는 최상단에서 조건을 확인하고, 필요하면 **환경변수를 채운 뒤
자기 자신을 exec** 합니다.

```python
if os.environ.get("TF_WSL_GPU_COMPAT") != "1" and os.path.exists(SHIM):
    cusolver = find_cusolver_dir()          # site-packages/nvidia/cusolver/lib
    env = os.environ.copy()
    env["TF_WSL_GPU_COMPAT"] = "1"
    env["LD_PRELOAD"] = SHIM + ":" + env.get("LD_PRELOAD", "")
    if cusolver:
        env["LD_LIBRARY_PATH"] = cusolver + ":" + env.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable,
              [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
              env)
```

`TF_WSL_GPU_COMPAT=1` 가드가 없으면 무한 재실행이 됩니다.
`execve` 는 PID 를 유지하므로 프로세스가 늘어나지는 않습니다.

## shim: 어디를 후킹할 것인가

### 시도 1 (실패) — 심볼 자체를 덮어쓰기

`libcuda.so.1` 보다 먼저 로드되는 `LD_PRELOAD` 라이브러리에서
`cuDeviceGetAttribute` 를 정의하면 TF의 호출을 가로챌 수 있습니다.
문제는 **WSL의 `libcuda.so.1` 래퍼도 같은 심볼을 `dlsym` 으로 찾는다**는 점입니다.
우리 심볼을 다시 잡아 무한 재귀 → `SIGSEGV`.

```
Program received signal SIGSEGV
#0  compat_cuDeviceGetAttribute ()
#1  compat_cuDeviceGetAttribute ()
#2  compat_cuDeviceGetAttribute ()
...
```

### 시도 2 (성공) — `dlsym` 을 후킹하되 호출자를 구분

TensorFlow는 CUDA driver API 를 `dlsym` 으로 동적 해석합니다.
그 지점만 가로채면, "누가 물어봤는지" 를 알 수 있습니다.

```c
void *dlsym(void *handle, const char *name) {
    if (name && strcmp(name, "cuDeviceGetAttribute") == 0 &&
        caller_is_tensorflow(__builtin_return_address(0)))
        return (void *)&compat_cuDeviceGetAttribute;
    return real_dlsym_fn(handle, name);
}

static int caller_is_tensorflow(void *caller) {
    Dl_info info;
    if (dladdr(caller, &info) == 0 || info.dli_fname == NULL)
        return 0;
    return strstr(info.dli_fname, "libtensorflow") != NULL ||
           strstr(info.dli_fname, "lib_pywrap_tf") != NULL;
}
```

- 호출자가 `libtensorflow*` / `lib_pywrap_tf*` → shim 반환
- 호출자가 `libcuda.so.1` 내부 → 원본 그대로 반환 (재귀 차단)

`__builtin_return_address(0)` 을 `dlsym` 안에서 쓰면 "dlsym 을 호출한 쪽" 주소가
나옵니다. 다만 인라인/최적화에 영향받지 않도록 별도 함수로 넘겨 처리합니다.

### shim 본체

```c
static CUresult compat_cuDeviceGetAttribute(int *pi, int attrib, int dev) {
    /* 원본은 dlopen("libcuda.so.1") 으로 직접 얻는다 */
    if (real_cuDeviceGetAttribute == NULL) { ... }

    CUresult r = real_cuDeviceGetAttribute(pi, attrib, dev);

    /* CUDA 12.5+ 에서 추가된 attribute 를 구 드라이버가 모르면
       CUDA_ERROR_INVALID_VALUE(1) 이 온다. "미지원(0)" 으로 바꿔준다. */
    if (r == 1 && attrib >= 132 && pi != NULL) {
        *pi = 0;
        return 0;
    }
    return r;
}
```

원본 함수 포인터는 `dlopen("libcuda.so.1", RTLD_LAZY|RTLD_GLOBAL)` 로 얻습니다.
`real_dlsym_fn` 을 경유해야 우리 `dlsym` 후킹에 다시 걸리지 않습니다.

## 함정 정리

### 1) `dlsym` 을 얻는 방법

`dlsym(RTLD_NEXT, "dlsym")` 은 이 상황에서 불안정했습니다.
glibc 2.34 부터 `dlsym`/`dlopen`/`dlvsym` 이 `libdl` 에서 `libc` 로 이동했기
때문입니다. `libc.so.6` 을 `RTLD_NOLOAD` 로 잡고 버전 지정 심볼을 얻는 방식이
확실합니다.

```c
void *libc = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
real_dlsym_fn = (void *(*)(void *, const char *))dlvsym(libc, "dlsym", "GLIBC_2.34");
```

### 2) nix 로 빌드하면 RPATH 가 박힌다

`nix shell nixpkgs#gcc` 로 빌드하면 산출물에 nix store 의 glibc 경로가
RUNPATH 로 들어갑니다. 그러면 시스템 파이썬(openSUSE glibc 2.44)에 preload 했을 때
glibc 가 두 개 로드되어 죽습니다.

```sh
patchelf --remove-rpath libcuda_compat.so
```

빌드 후 반드시 제거합니다. (`build.sh` 에 포함)

### 3) gdb 로 추적할 때의 함정

- `rbreak ^cu[A-Z]` 를 `run` **이전에** 걸면, 아직 매핑되지 않은 `libcuda.so.1`
  심볼은 잡히지 않습니다. `break cuInit` → `run` → 그 다음 `rbreak`.
- `break main` 은 파이썬에서 의미가 없습니다 (`libpython` 이 `main` 을 안 가짐).
- 브레이크포인트마다 `continue` 를 파이썬으로 처리하면 13,000 콜에도 충분히 빠릅니다.

### 4) attribute 를 전부 0으로 만들면 안 된다

`attrib >= 132` 조건이 없으면 compute capability 같은 핵심 attribute 까지
0이 되어 GPU가 잘못 초기화됩니다. 반드시 하한을 둡니다.

## 결과

```
$ .venv/bin/python test.py
TensorFlow: 2.21.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

$ # 실제 연산
device: /job:localhost/replica:0/task:0/device:GPU:0
sum: 4443.8935546875
```

다음: [05-native-fix.md](05-native-fix.md)
