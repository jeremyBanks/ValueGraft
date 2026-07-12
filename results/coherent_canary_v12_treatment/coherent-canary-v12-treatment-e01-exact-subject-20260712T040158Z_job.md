# Exact v12 e01 treatment pod log

```text
START V12_TREATMENT_E01 2026-07-12T04:01:58+00:00
EXPECTED_COMMIT=cbdfa481fe08de62bf8178d09ca040716d610f21 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
CASE=data/coherent_canary_v12/revision2/session_d/e01.json CASE_SHA=6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090
TECHNICAL_REPORT=results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json PHASE_A_REPORT=results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json
NVIDIA A100 80GB PCIe, GPU-b2a992b5-2caf-55c6-51d9-3de66c44468e, 580.159.04, 81920
V12 TREATMENT HOST ADMITTED NVIDIA A100 80GB PCIe GPU-b2a992b5-2caf-55c6-51d9-3de66c44468e 580.159.04 81920
data/coherent_canary_v12/revision2/session_d/e01.json: OK
COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md: OK
results/coherent_canary_validation/coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_20260711T205951Z.json: OK
results/coherent_canary_reviews/reviews/revision4_blind_singleton_independent_codex_20260711T210013Z.json: OK
results/coherent_canary_reviews/reviews/revision4_paired_diversity_independent_codex_20260711T210013Z.json: OK
results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json: OK
results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json: OK
results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json: OK
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Downloading cpython-3.12.11-linux-x86_64-gnu (download) (31.1MiB)
 Downloaded cpython-3.12.11-linux-x86_64-gnu (download)
Installed Python 3.12.11 in 1.80s
 + cpython-3.12.11-linux-x86_64-gnu (python3.12)
warning: `/root/.local/bin` is not on your PATH. To use installed Python executables, run `export PATH="/root/.local/bin:$PATH"` or `uv python update-shell`.
Using CPython 3.12.11
Creating virtual environment at: /workspace/v12-treatment-venv
Activate with: source /workspace/v12-treatment-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/v12-treatment-venv
Resolved 57 packages in 409ms
Downloading hf-xet (4.3MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-curand (56.8MiB)
Downloading sentencepiece (1.3MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading transformers (9.7MiB)
Downloading nvidia-nvjitlink (38.8MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading pygments (1.2MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading torch (507.6MiB)
Downloading networkx (2.0MiB)
Downloading tokenizers (3.1MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading nvidia-cudnn-cu13 (349.2MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading numpy (15.9MiB)
Downloading sympy (6.0MiB)
Downloading setuptools (1.0MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading cuda-bindings (6.3MiB)
Downloading triton (188.6MiB)
 Downloaded nvidia-cufile
 Downloaded sentencepiece
 Downloaded nvidia-cuda-runtime
 Downloaded pygments
 Downloaded tokenizers
 Downloaded setuptools
 Downloaded networkx
 Downloaded hf-xet
 Downloaded cuda-bindings
 Downloaded nvidia-cuda-cupti
 Downloaded sympy
 Downloaded numpy
 Downloaded transformers
 Downloaded nvidia-nvjitlink
 Downloaded nvidia-curand
 Downloaded nvidia-nvshmem-cu13
 Downloaded nvidia-cuda-nvrtc
 Downloaded nvidia-cusparse
 Downloaded nvidia-cusparselt-cu13
 Downloaded nvidia-cusolver
 Downloaded triton
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cufft
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cublas
 Downloaded torch
Prepared 57 packages in 21.06s
Installed 57 packages in 354ms
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
SETUP torch_cuda 13.0 cuda_available True
SETUP gpu NVIDIA A100 80GB PCIe GPU-b2a992b5-2caf-55c6-51d9-3de66c44468e 580.159.04 81920
V12 FROZEN VERIFIED 86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1
EXPECTED RELEASE RUNTIME b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484
RUN V12_TREATMENT_E01 2026-07-12T04:02:35+00:00
RUN coherent-canary-v12-treatment subject=exact-subject case=data/coherent_canary_v12/revision2/session_d/e01.json -> /workspace/repo_treatment/results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z.json
{"case_id": "e01", "error": null, "placebo_control_count": 3, "primary_arm_count": 31, "status": "PASS"}
HARVEST case=e01 treatment=results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z.json -> results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json
{"case_id": "e01", "output": "results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json", "treatment_sha256": "f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082"}
```
