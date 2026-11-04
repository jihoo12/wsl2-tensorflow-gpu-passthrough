# 05. 근본 해결과 대안

## 근본 해결 — Windows NVIDIA 드라이버 업데이트

WSL2의 CUDA는 **Windows 호스트 드라이버를 그대로** 사용합니다.
배포판 안의 `nvidia-*` pip 패키지는 유저 공간 라이브러리(cublas, cudnn, ...)일
뿐이고, `/usr/lib/wsl/lib/libcuda.so.1` 은 Windows 드라이버에서 내려온
스텁입니다. 따라서 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`(132) 를 지원하는
드라이버(= CUDA 12.5 이상, R555 계열 이후)로 올리면 workaround 가 필요 없습니다.

절차:

1. Windows에서 NVIDIA 공식 사이트에서 최신 Game Ready / Studio 드라이버 다운로드
   (RTX A4000 은 Studio 드라이버 권장)
2. 설치 후 **Windows 재부팅**
3. WSL 재시작: PowerShell 에서 `wsl --shutdown`
4. 확인

```sh
nvidia-smi          # Driver Version 이 555 이상인지
.venv/bin/python test.py
```

업데이트 후에는 다음을 되돌립니다.

- `test.py` 의 재실행 bootstrap 블록 제거
- `LD_PRELOAD` 로 `gpu_compat/libcuda_compat.so` 를 넣던 설정 제거
- `gpu_compat/` 디렉터리 삭제

> attribute 132 조건이 `>= 132` 이므로, 드라이버를 올린 뒤 shim 을 그대로 두면
> 새 드라이버가 지원하는 기능까지 "미지원" 으로 보고됩니다. 반드시 제거하세요.

## 대안들

### A. TensorFlow 버전 낮추기

attribute 132 를 조회하지 않는 버전을 쓰면 원인 2를 피할 수 있습니다.
다만 원인 1(RUNPATH)은 wheel 에 따라 여전할 수 있습니다.

```sh
uv pip install --python .venv/bin/python "tensorflow==2.16.*"
```

TensorFlow 2.16 은 CUDA 12.3 / cuDNN 8.9 빌드라, 528.49 드라이버(CUDA 12.0)와의
궁합이 더 좋습니다. 학습 코드가 특정 버전에 묶여 있지 않다면 이쪽이 더 깔끔합니다.

### B. PyTorch 사용

PyTorch 는 attribute 132 를 무조건 조회하지 않으므로, 같은 드라이버에서
cu121/cu124 wheel 이 바로 동작하는 경우가 많습니다.

```sh
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu121 torch
```

### C. 컨테이너 사용

NVIDIA Container Toolkit + `nvidia/cuda:12.0` 이미지로 호스트 드라이버에 맞춘
환경을 고정할 수 있습니다. 다만 TensorFlow wheel 자체의 RUNPATH 문제는
컨테이너 안에서도 동일하게 발생합니다.

### D. wheel RUNPATH 패치 (root 불필요)

```sh
patchelf --add-rpath '$ORIGIN/../../nvidia/cusolver/lib' \
  .venv/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.so.2
```

원인 1만 해결됩니다. 원인 2는 별도 대응이 필요합니다.

### E. 시스템 로더 설정 (root 필요)

```sh
echo "$PWD/.venv/lib/python3.13/site-packages/nvidia/cusolver/lib" \
  | sudo tee /etc/ld.so.conf.d/nvidia-pip.conf
sudo ldconfig
```

역시 원인 1만 해결됩니다.

## 어떤 방법을 선택할까

| 상황 | 권장 |
|------|------|
| Windows 드라이버를 올릴 수 있다 | **드라이버 업데이트** (workaround 전부 제거) |
| 드라이버를 못 올린다, 최신 TF 가 필요하다 | 이 저장소의 shim |
| TF 버전을 고를 수 있다 | TF 2.16 + cuDNN 8.9 |
| root 가 없다 | `LD_LIBRARY_PATH` + shim |

## 이 저장소의 한계

- attribute 132 를 "미지원" 으로 위장하는 것이므로, 해당 기능에 의존하는
  멀티 GPU / multicast 통신은 사용할 수 없습니다. 단일 GPU 학습·추론에는 영향이 없습니다.
- 드라이버 업데이트 없이 최신 TF 를 쓰기 위한 **임시 조치**입니다.
  장기적으로는 드라이버를 올리는 것이 맞습니다.
