# Exact v12 technical pod log

```text
START V12_EXACT_TECHNICAL 2026-07-12T01:50:43+00:00
EXPECTED_COMMIT=d82831309af180e76baa7d7beb2b14c922de761a MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
MODE=exact-technical-only DTYPE=bfloat16 ATTENTION_BACKEND=eager
NVIDIA A100 80GB PCIe, GPU-64d7d74d-6b9a-2e2c-bf1f-ae06c1b6b3bc, 580.126.20, 81920
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Downloading cpython-3.12.11-linux-x86_64-gnu (download) (31.1MiB)
 Downloaded cpython-3.12.11-linux-x86_64-gnu (download)
Installed Python 3.12.11 in 2.30s
 + cpython-3.12.11-linux-x86_64-gnu (python3.12)
warning: `/root/.local/bin` is not on your PATH. To use installed Python executables, run `export PATH="/root/.local/bin:$PATH"` or `uv python update-shell`.
Using CPython 3.12.11
Creating virtual environment at: /workspace/v12-venv
Activate with: source /workspace/v12-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/v12-venv
Resolved 57 packages in 572ms
Downloading numpy (15.9MiB)
Downloading pygments (1.2MiB)
Downloading sentencepiece (1.3MiB)
Downloading networkx (2.0MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading hf-xet (4.3MiB)
Downloading setuptools (1.0MiB)
Downloading sympy (6.0MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading cuda-bindings (6.3MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading torch (507.6MiB)
Downloading transformers (9.7MiB)
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading tokenizers (3.1MiB)
Downloading nvidia-nvjitlink (38.8MiB)
Downloading triton (188.6MiB)
Downloading nvidia-curand (56.8MiB)
Downloading nvidia-cudnn-cu13 (349.2MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
 Downloaded nvidia-cufile
 Downloaded sentencepiece
 Downloaded pygments
 Downloaded nvidia-cuda-runtime
 Downloaded setuptools
 Downloaded tokenizers
 Downloaded networkx
 Downloaded hf-xet
 Downloaded cuda-bindings
 Downloaded sympy
 Downloaded nvidia-cuda-cupti
 Downloaded transformers
 Downloaded numpy
 Downloaded nvidia-nvjitlink
 Downloaded nvidia-curand
 Downloaded nvidia-nvshmem-cu13
 Downloaded nvidia-cuda-nvrtc
 Downloaded nvidia-cusparse
 Downloaded nvidia-cusparselt-cu13
 Downloaded triton
 Downloaded nvidia-cusolver
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cufft
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cublas
 Downloaded torch
Prepared 57 packages in 22.71s
Installed 57 packages in 447ms
 + accelerate==1.14.0
 + annotated-doc==0.0.4
 + anyio==4.14.1
 + certifi==2026.6.17
 + click==8.4.2
 + cuda-bindings==13.3.1
 + cuda-pathfinder==1.5.6
 + cuda-toolkit==13.0.2
 + filelock==3.29.7
 + fsspec==2026.6.0
 + h11==0.16.0
 + hf-xet==1.5.1
 + httpcore==1.0.9
 + httpx==0.28.1
 + huggingface-hub==1.22.0
 + idna==3.18
 + jinja2==3.1.6
 + markdown-it-py==4.2.0
 + markupsafe==3.0.3
 + mdurl==0.1.2
 + mpmath==1.3.0
 + networkx==3.6.1
 + numpy==2.5.1
 + nvidia-cublas==13.1.1.3
 + nvidia-cuda-cupti==13.0.85
 + nvidia-cuda-nvrtc==13.0.88
 + nvidia-cuda-runtime==13.0.96
 + nvidia-cudnn-cu13==9.20.0.48
 + nvidia-cufft==12.0.0.61
 + nvidia-cufile==1.15.1.6
 + nvidia-curand==10.4.0.35
 + nvidia-cusolver==12.0.4.66
 + nvidia-cusparse==12.6.3.3
 + nvidia-cusparselt-cu13==0.8.1
 + nvidia-nccl-cu13==2.29.7
 + nvidia-nvjitlink==13.0.88
 + nvidia-nvshmem-cu13==3.4.5
 + nvidia-nvtx==13.0.85
 + packaging==26.2
 + psutil==7.2.2
 + pygments==2.20.0
 + pyyaml==6.0.3
 + regex==2026.7.10
 + rich==15.0.0
 + safetensors==0.8.0
 + sentencepiece==0.2.1
 + setuptools==81.0.0
 + shellingham==1.5.4
 + sympy==1.14.0
 + tokenizers==0.22.2
 + torch==2.12.1
 + tqdm==4.68.4
 + transformers==5.0.0
 + triton==3.7.1
 + typer==0.26.8
 + typer-slim==0.24.0
 + typing-extensions==4.16.0
SETUP python 3.12.11 (main, Oct  7 2025, 15:34:39) [Clang 20.1.4 ]
SETUP dependencies {'torch': '2.12.1', 'transformers': '5.0.0', 'accelerate': '1.14.0', 'safetensors': '0.8.0', 'huggingface-hub': '1.22.0'}
SETUP torch_cuda 13.0 cuda_available True device_count 1
SETUP gpu NVIDIA A100 80GB PCIe
RUN V12_EXACT_TECHNICAL 2026-07-12T01:51:19+00:00
RUN coherent-canary-v12-technical subject=exact-subject -> /workspace/repo/results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_20260712T015124276346Z.json
{"correct_history_N_repeat": "PASS", "fresh_repeat": "PASS", "fresh_self_replacement_cells": 9, "identity": "PASS", "natural_calibration": "ADVERSE", "path_attempt_count": 3, "path_control": "PASS"}
PASS -> results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T015043Z.json
```
