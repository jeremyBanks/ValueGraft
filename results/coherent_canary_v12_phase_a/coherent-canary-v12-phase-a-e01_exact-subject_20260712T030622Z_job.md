# Exact v12 e01 Phase-A pod log

```text
START V12_PHASE_A_E01 2026-07-12T03:06:22+00:00
EXPECTED_COMMIT=139d621e206518b88d7553b5166a0123ee1cdd55 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
CASE=data/coherent_canary_v12/revision2/session_d/e01.json CASE_SHA=6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090 TECHNICAL_REPORT=results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json
NVIDIA A100 80GB PCIe, GPU-0396c7e5-6997-2154-b2cf-90b57c6f05ea, 580.159.04, 81920
data/coherent_canary_v12/revision2/session_d/e01.json: OK
results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json: OK
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Python 3.12.11 is already installed
Using CPython 3.12.11
Creating virtual environment at: /workspace/v12-phasea-venv
Activate with: source /workspace/v12-phasea-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/v12-phasea-venv
Resolved 57 packages in 205ms
Installed 57 packages in 299ms
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
SETUP platform Linux-6.8.0-100-generic-x86_64-with-glibc2.35
SETUP dependencies {'torch': '2.12.1', 'transformers': '5.0.0', 'accelerate': '1.14.0', 'safetensors': '0.8.0', 'huggingface-hub': '1.22.0'}
SETUP torch_cuda 13.0 cuda_available True device_count 1
SETUP gpu NVIDIA A100 80GB PCIe
V12 FROZEN VERIFIED 86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1
RUN V12_PHASE_A_E01 2026-07-12T03:06:33+00:00
RUN coherent-canary-v12-phase-a subject=exact-subject case=data/coherent_canary_v12/revision2/session_d/e01.json -> /workspace/repo_phasea/results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json
{"case_id": "e01", "error": null, "status": "PASS", "treatment_scores_present": false}
PRETREATMENT_PASS -> results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json
```
