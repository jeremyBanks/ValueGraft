# Exact v12 technical attempt 2 — preserved pre-forward failures

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `lyt5x3popltrgx` (`NVIDIA A100 80GB PCIe`)

**Pinned subject:** `Qwen/Qwen3-30B-A3B-Instruct-2507` at
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`

The first launch on this rental was rejected by the repository interlock because
the coordinator supplied an incorrectly expanded expected commit. The exact
committed SHA was then read directly with `git rev-parse HEAD` and the same
already-rented pod was restarted. That corrected launch cloned the right
commit and installed the pinned packages, but the host's NVIDIA 550.90.12
driver could not initialize the CUDA 13 runtime installed by pinned
`torch==2.12.1`.

Both failures occurred before model snapshot download, model load, subject
forward, generation, cache comparison, Phase A, treatment, or semantic scoring.
The receipt pointer was therefore absent; the puller preserved the job log and
failed closed as designed. The pod was then terminated.

## Wrong expected-commit launch

```text
START V12_EXACT_TECHNICAL 2026-07-12T01:02:49+00:00
EXPECTED_COMMIT=a3b2c28bb42a5e50528c9f7fbc1c2c341583118a MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
MODE=exact-technical-only DTYPE=bfloat16 ATTENTION_BACKEND=eager
NVIDIA A100 80GB PCIe, GPU-d6c728a3-3685-7995-ce48-50b83cf66d83, 550.90.12, 81920
FATAL: origin/trunk a3b2c280e17a589f0c459c4ee0bfc0b444e9bdeb != expected a3b2c28bb42a5e50528c9f7fbc1c2c341583118a
```

## Correct-commit launch on incompatible host driver

```text
START V12_EXACT_TECHNICAL 2026-07-12T01:03:47+00:00
EXPECTED_COMMIT=a3b2c280e17a589f0c459c4ee0bfc0b444e9bdeb MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
MODE=exact-technical-only DTYPE=bfloat16 ATTENTION_BACKEND=eager
NVIDIA A100 80GB PCIe, GPU-d6c728a3-3685-7995-ce48-50b83cf66d83, 550.90.12, 81920
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Downloading cpython-3.12.11-linux-x86_64-gnu (download) (31.1MiB)
 Downloaded cpython-3.12.11-linux-x86_64-gnu (download)
Installed Python 3.12.11 in 1.57s
 + cpython-3.12.11-linux-x86_64-gnu (python3.12)
warning: `/root/.local/bin` is not on your PATH. To use installed Python executables, run `export PATH="/root/.local/bin:$PATH"` or `uv python update-shell`.
Using CPython 3.12.11
Creating virtual environment at: /workspace/v12-venv
Activate with: source /workspace/v12-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/v12-venv
Resolved 57 packages in 326ms
Downloading nvidia-cudnn-cu13 (349.2MiB)
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading setuptools (1.0MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading numpy (15.9MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
Downloading transformers (9.7MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading cuda-bindings (6.3MiB)
Downloading nvidia-curand (56.8MiB)
Downloading sentencepiece (1.3MiB)
Downloading sympy (6.0MiB)
Downloading triton (188.6MiB)
Downloading nvidia-nvjitlink (38.8MiB)
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading hf-xet (4.3MiB)
Downloading networkx (2.0MiB)
Downloading tokenizers (3.1MiB)
Downloading pygments (1.2MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading torch (507.6MiB)
 Downloaded nvidia-cufile
 Downloaded sentencepiece
 Downloaded nvidia-cuda-runtime
 Downloaded pygments
 Downloaded tokenizers
 Downloaded setuptools
 Downloaded hf-xet
 Downloaded networkx
 Downloaded cuda-bindings
 Downloaded nvidia-cuda-cupti
 Downloaded numpy
 Downloaded sympy
 Downloaded nvidia-nvjitlink
 Downloaded nvidia-curand
 Downloaded nvidia-nvshmem-cu13
 Downloaded transformers
 Downloaded nvidia-cuda-nvrtc
 Downloaded nvidia-cusparse
 Downloaded nvidia-cusparselt-cu13
 Downloaded nvidia-cusolver
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cufft
 Downloaded triton
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cublas
 Downloaded torch
Prepared 57 packages in 26.12s
Installed 57 packages in 545ms
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
/workspace/v12-venv/lib/python3.12/site-packages/torch/cuda/__init__.py:187: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
SETUP python 3.12.11 (main, Oct  7 2025, 15:34:39) [Clang 20.1.4 ]
SETUP dependencies {'torch': '2.12.1', 'transformers': '5.0.0', 'accelerate': '1.14.0', 'safetensors': '0.8.0', 'huggingface-hub': '1.22.0'}
SETUP torch_cuda 13.0 cuda_available False device_count 1
FATAL: exact gate requires exactly one visible CUDA device
```

This artifact is infrastructure evidence only. It contains no scientific
observation and cannot satisfy or weaken the exact technical gate.
