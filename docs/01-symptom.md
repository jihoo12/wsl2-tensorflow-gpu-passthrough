# 01. 증상

## 환경

```
$ uname -r
6.18.33.2-microsoft-standard-WSL2

$ cat /etc/os-release | head -2
NAME="openSUSE Tumbleweed"
# VERSION="20261030"

$ nvidia-smi
NVIDIA-SMI 525.89.02    Driver Version: 528.49    CUDA Version: 12.0
|   0  NVIDIA RTX A4000    On   | 00000000:01:00.0  On |                  Off |
```

드라이버가 `528.49` 라는 점을 기억해 둡니다. WSL2는 **Windows 호스트의 드라이버**를
그대로 사용하므로, 배포판 안에서 무엇을 설치하든 이 숫자는 바뀌지 않습니다.

## 확인 스크립트

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
```

결과:

```
TensorFlow: 2.21.0
GPU: []
```

stderr:

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 ... oneDNN custom operations are on.
I0000 ... cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions...
W0000 ... gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing
       libraries mentioned above are installed properly if you would like to use GPU.
       Follow the guide at https://www.tensorflow.org/install/gpu ...
Skipping registering GPU devices...
[]
```

## 이 시점에서 배제한 것들

WSL2 GPU passthrough 에서 흔히 나오는 원인들은 모두 정상이었습니다.

| 확인 | 결과 |
|------|------|
| `/dev/dxg` 존재 | 있음 |
| `nvidia-smi` 동작 | 정상 (GPU, 드라이버, 메모리 모두 인식) |
| `/usr/lib/wsl/lib/libcuda.so.1` | 존재, 로드 가능 |
| `libnvidia-ml.so.1` | 로드 가능 |
| WSL 커널 모듈 | `dxgkrnl` 정상 |
| `CUDA_VISIBLE_DEVICES` | 미설정 (제약 없음) |

즉 **드라이버 passthrough 자체는 되어 있고**, TensorFlow 쪽에서 무언가를
못 찾거나 거부하고 있는 상황입니다. 로그의 `Cannot dlopen some GPU libraries`
가 결정적 힌트였습니다.

다음: [02-root-cause-cusolver.md](02-root-cause-cusolver.md)
