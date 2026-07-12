# Precision probe p02 pod log

```text
START PRECISION_PROBE_P02 2026-07-12T10:30:11+00:00
EXPECTED_COMMIT=a40b1dffe8d7bf5e310d308e22d26fef126a73b7 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
PROTOCOL=precision-probe-p02 ORDER=NF4-then-bf16 ATTENTION=eager KV_DTYPE=bfloat16
NVIDIA A100 80GB PCIe, GPU-8a42830e-71ab-fb23-351d-125ccbd5bdb2, 580.159.03, 81920
PROVIDER_BUDGET rate=1.39 start_epoch=1783852147 scientific_cap_seconds=9000 scientific_usd=3.90 absolute_usd=4.00
PRECISION-PROBE-P02-PREREGISTRATION.md: OK
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Downloading cpython-3.12.11-linux-x86_64-gnu (download) (31.1MiB)
 Downloaded cpython-3.12.11-linux-x86_64-gnu (download)
Installed Python 3.12.11 in 2.39s
 + cpython-3.12.11-linux-x86_64-gnu (python3.12)
warning: `/root/.local/bin` is not on your PATH. To use installed Python executables, run `export PATH="/root/.local/bin:$PATH"` or `uv python update-shell`.
Using CPython 3.12.11
Creating virtual environment at: /workspace/p02-venv
Activate with: source /workspace/p02-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/p02-venv
Resolved 47 packages in 489ms
Downloading setuptools (1.0MiB)
Downloading sympy (6.0MiB)
Downloading numpy (15.9MiB)
Downloading hf-xet (4.3MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
Downloading torch (507.6MiB)
Downloading tokenizers (3.1MiB)
Downloading nvidia-curand (56.8MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading triton (188.6MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading bitsandbytes (57.8MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading nvidia-nvjitlink (38.8MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading nvidia-cudnn-cu13 (349.2MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading transformers (11.4MiB)
Downloading cuda-bindings (6.3MiB)
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading networkx (2.0MiB)
 Downloaded nvidia-cufile
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
 Downloaded bitsandbytes
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
Prepared 47 packages in 22.42s
Installed 47 packages in 463ms
 + accelerate==1.14.0
 + bitsandbytes==0.49.2
 + certifi==2026.6.17
 + charset-normalizer==3.4.9
 + cuda-bindings==13.3.1
 + cuda-pathfinder==1.5.6
 + cuda-toolkit==13.0.2
 + filelock==3.29.7
 + fsspec==2026.6.0
 + hf-xet==1.5.1
 + huggingface-hub==0.36.2
 + idna==3.18
 + jinja2==3.1.6
 + markupsafe==3.0.3
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
 + pyyaml==6.0.3
 + regex==2026.7.10
 + requests==2.34.2
 + safetensors==0.8.0
 + setuptools==81.0.0
 + sympy==1.14.0
 + tokenizers==0.22.2
 + torch==2.12.1
 + tqdm==4.68.4
 + transformers==4.57.6
 + triton==3.7.1
 + typing-extensions==4.16.0
 + urllib3==2.7.0
RUNTIME_ATTESTED results/precision_probe_p02/precision-probe-p02-runtime_Qwen3-30B-A3B-Instruct-2507_20260712T103011764456Z.json
RUN precision-probe-p02 model=Qwen/Qwen3-30B-A3B-Instruct-2507 revision=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe output=results/precision_probe_p02 elapsed=115 allowance=8675
{"completion_marker_path": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-completion_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.json", "event": "RUN", "model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "protocol_id": "precision-probe-p02", "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "run_directory": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z", "run_manifest_path": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-run-manifest_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.json", "scientific_cap_seconds": 9000}
`torch_dtype` is deprecated! Use `dtype` instead!
/workspace/p02-venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
/workspace/p02-venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
{"completion_marker_path": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-completion_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.json", "exit_code": 0, "protocol_id": "precision-probe-p02", "run_manifest_path": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-run-manifest_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.json", "status": "COMPLETE"}
{
  "chunk_count": 1,
  "compressed_sha256": "72290c9ad42e04732e9077c34011f9dca9b9285f6ee66c332d1dfae34fe52893",
  "compressed_size_bytes": 40359,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "0fdf62986a0bf94c26d1a47108d1a97ddfa57d7757abca111e4df7fe9b7d9795",
  "original_size_bytes": 128481,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index01-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "c0b6a2b37362f919a0ba1c5eb5610c9851d4d92cfad47116b1e1f197836fd0ca",
  "compressed_size_bytes": 40404,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "5029c2b1181702fc1f7836d1d4b2aeaab05b968f33937b9c4f691d9091834a56",
  "original_size_bytes": 128552,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index02-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "cd7c68d5174900d98afe7c66aa06ea03eb23f911e205b30188630ba1bd5694c0",
  "compressed_size_bytes": 43857,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "8a20adf0d1d587ebdd473e429235741f770675fed6e58f7985d00fba8bffa9c1",
  "original_size_bytes": 128559,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index03-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "759daf751782e93521d2ec51d2ecfecb4eb77783d151cbb3889e677e1e42c4c3",
  "compressed_size_bytes": 43899,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "4af07b277ff2c688d09a3b886e99021f41fc939b8d4931500295763963549f13",
  "original_size_bytes": 128556,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index04-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "18cfd3a94bb2913379287288985dd7e20e30386242e93cff9ff24d8985918884",
  "compressed_size_bytes": 40353,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "f96f7eeecd0f9543f5434d621ea0720e59ee224d3ea8612838b0a08ba2f3d9d5",
  "original_size_bytes": 128552,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index05-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "b1702361f461a545d2bf52d4539e538896b4eb163d5ff41289fb38eb0d37108b",
  "compressed_size_bytes": 40416,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "6b633b5a8cd74cfdd8f12b540d350f5fe0192f7e04a7fb75ef078da4dfbc508a",
  "original_size_bytes": 128553,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index06-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "ae389c3f0df7f3f6ed56287119a927c2cdf0b26e244d85a2ecaaca6d90b38b48",
  "compressed_size_bytes": 1930,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "b7563bc5af950044c29418e00a4db3af7a2ecd397d3ecb7965f57358994e5529",
  "original_size_bytes": 5475,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat1-index07-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "fb3d6848cec49e1834418764ce27030cf1c0d0caf835e663a1b48ef951a3cacf",
  "compressed_size_bytes": 40355,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "f2ccbd5a04278dc714e191c350981ae8de67e6afacc6cf1db30278392f2b2663",
  "original_size_bytes": 128480,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index01-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "a0d806015eaf38156d684344f59df74957435692628bbf85350e21856fe2c816",
  "compressed_size_bytes": 40413,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "cd0133631f1e20f43668cbca5bbad80a4f993956e1e0cbee38b068e490ac0a4d",
  "original_size_bytes": 128551,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index02-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "dbb2c1fc4c463802a2df08823c68543a04368fcc6a92a1092392f1a258b5213b",
  "compressed_size_bytes": 43862,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "c0786eb42b829c75bc268a7f37a433fcde853b9dc7aefe8a54889e8f8bcabc94",
  "original_size_bytes": 128559,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index03-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "71ee6150fb54cd53c1724fad82478f7523a897f8b6f593f26b8908b5ab4494fc",
  "compressed_size_bytes": 43886,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "28b0c53d02f00284eedb768e508491068ee7268e7f5527a7da99ad4d120efd4e",
  "original_size_bytes": 128555,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index04-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "715371d86324052bf446221bd5d018f2686a6e2ef930b836329d643046cb797d",
  "compressed_size_bytes": 40356,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "15e095ceb5126ba71afbe5b17852e77239349392bc4f0ea7a7d9b72d31dac5e4",
  "original_size_bytes": 128551,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index05-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "00d44029f83e0a013823c7a77f7627ae9e6356f17a1596650a8fdb491a1ff52d",
  "compressed_size_bytes": 40414,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "b29e79b84ea7cf2a83859fd2daec325b990adfc0de35570de20a7df9ef2399d8",
  "original_size_bytes": 128553,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-bf16-e01-repeat2-index06-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "ac2989c1d4e6fca41996078429bfdb8512322ff8a9b6b897c2b8936a2a384603",
  "compressed_size_bytes": 40294,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "054fc7f707ca196a22a26a8edc142c6263bcf1ec3687a9e3470dabb19e4b986f",
  "original_size_bytes": 128410,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index01-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "3d27f7b3f3c100d7146c239681bbc5fc40e0fb1425b5cb540b5122b9a75c5f70",
  "compressed_size_bytes": 40372,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "60be866bfb056dd5c5d5ce9895a1e63a37b567b4437570af23af33d37a498c00",
  "original_size_bytes": 128510,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index02-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "47ee84633eda5f964c004c42a87904d9f714165204e9a5b0a581232fd0517c31",
  "compressed_size_bytes": 43871,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "7cc70bee6040150f48e9976b581653929542aca01e59f9685c7e4a55bbe444e2",
  "original_size_bytes": 128495,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index03-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "9cdae7e269b0973d19647e6be5a74bd943078fe6cc482a1e55720720ea03da52",
  "compressed_size_bytes": 43849,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "d572d976edd40aa04a429a359e0529abf74268942f61f13b2c78753791cfcf79",
  "original_size_bytes": 128479,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index04-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "370c1b039d59d54b9a8916e0ac95c9158c96fbb1076a19988623aad0373ec3e9",
  "compressed_size_bytes": 40464,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "d4b59422fd8a08bb3245837587b6b4477a271968fc7ceb4eb8126fd4cb3216df",
  "original_size_bytes": 128722,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index05-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "4975bcdb9effae8d2a133e538bdf0990105f1db4a1b06e236f9e93d94c109dc4",
  "compressed_size_bytes": 40403,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "468e3ad4b37256f46ada58a2f4692be7165a46e225f041b35736e245dc2ae2d6",
  "original_size_bytes": 128508,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index06-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "6df79ee3a686f7609dd8deb9ed96875192057f93bd4f3a28e54a5fec475ad44e",
  "compressed_size_bytes": 1935,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "063c8e8bc89ef5e16c38406f820e3f10b290f4e0df7c6ab6ec8994c9479663f9",
  "original_size_bytes": 5464,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat1-index07-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "a9b4d58f6951ddd1fa1917836c43a8fd20949a03c27c86582d7d9d2130a05908",
  "compressed_size_bytes": 40290,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "eb4dfac47692f19c60d583746f2712dd222d2bdaf26cfb056e1af7ea583831bc",
  "original_size_bytes": 128411,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index01-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "32e0d6be67cd417fde33d33eddfdf1d77f24daba9a74afeda006ae416c313128",
  "compressed_size_bytes": 40371,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "74a813d66863ba31b0017b85da5ae0a1b798b230c100710802f919c4ebb187a4",
  "original_size_bytes": 128510,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index02-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "f3ed696056f8b24f33b19e6d2a3e7505ca3276b23bd3d469b0c01f7cf0a9f9c4",
  "compressed_size_bytes": 43865,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "df3a5f36897a69836f53c77266813b8d478b9e36bbd6f6f9b25da26c99e7eb6a",
  "original_size_bytes": 128494,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index03-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "02d7ea32dd2393131d58bad2f14572b2e0b405fecafc16bbfbd44e5ee6c0f141",
  "compressed_size_bytes": 43853,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "0d36d79d701b0518defd14a0afb1ce83915d03b168080efaf3120769a7563a77",
  "original_size_bytes": 128480,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index04-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "2ce7b9c12afd2135c5da76ede5fae40e63452ff68852c17e390cd5582e377905",
  "compressed_size_bytes": 40458,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "873ad1b9cd7fa038d2b6aea7802b7149dce37f4d1adc963870485f1760b2aae9",
  "original_size_bytes": 128721,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index05-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "772abdc63c5c43ca474dfe383c2ddbb32db2d91c9b5df862ec29dd8aeee69087",
  "compressed_size_bytes": 40406,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "5f66d8bccc854c8b1530e57809af3f34da441c6bfe807423411e7c411d7527a8",
  "original_size_bytes": 128508,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-arm-nf4-e01-repeat2-index06-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "aa487ff507a74e43c1814ec17254b90f8fa58013a1f6f777915553503b466aef",
  "compressed_size_bytes": 184633,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "149efd8ab31c70ea8e08ce7200b5b7875e9a187bdc13ed40545ce1c23d40a5b8",
  "original_size_bytes": 939691,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-foundation-bf16-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "5f18eb63162451c8e926e0ca2ea51cb22d39f077fe8c2f86eae85b3f369c3ee9",
  "compressed_size_bytes": 184634,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "10423f8448085d9eff169f870b801c14653f7cbafa1bff9dd3fe21f4fbf36bcd",
  "original_size_bytes": 939691,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-foundation-bf16-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "d1092af0efdd68acd13dc48d6d1d73421bb1074435f8e58c969609450c3e9671",
  "compressed_size_bytes": 185084,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "96d72f2861c7ff3e4118f9615d63ade2a8f5604b0f175b06f97fde7e27987f1c",
  "original_size_bytes": 940763,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-foundation-nf4-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "8287156b32f10cd18ac68302aae57440413446567e7c384ca1adaa13887f0823",
  "compressed_size_bytes": 185085,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "dd808852bee32f41727d841167fa1f6cf701455df4901859bf78c0dfbcb0ee35",
  "original_size_bytes": 940763,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-foundation-nf4-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "d4221b11b431fbcddbafb6f38adeb5b4c7d6b70a3210d204df1d04178d3ef6d9",
  "compressed_size_bytes": 600047,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "3f126fa1269d81751ce5f083f5e9d29df29b008bde5a89c4e254976daa2ad091",
  "original_size_bytes": 2603917,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-outcome-bf16-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "8921c0592db760dd6dab1d4a938f22b4db3862b26bd5ccc699081db171de63a2",
  "compressed_size_bytes": 599137,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "a03e08b82c0636d773cef47c103c5b9a6b7102a6af725829f0f88ff4680c3895",
  "original_size_bytes": 2600684,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-outcome-bf16-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "f6d315a6b7bc44e11a614a381c08541d656eea24bdf200beb864a594411eed9a",
  "compressed_size_bytes": 602045,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "d471135c7aa09a9cab47c268dbc773cfa74581179281ae15554c0bf7cf6acd4a",
  "original_size_bytes": 2611241,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-outcome-nf4-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "705dd10e9502a9fc407e50a41b7068ba7658e61b9f40fde79b7f5725bf408d29",
  "compressed_size_bytes": 601084,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "672a94e24d595eedc7ba56d9e125326cd21a0e7198820146b599f9861faae82d",
  "original_size_bytes": 2608018,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-outcome-nf4-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "0c30823e5e41dd368e0a4d80e5eaaffd55de1e313780c7e87f1ef30dd1e140d9",
  "compressed_size_bytes": 150227,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "0a1fe5f8352562b1ac833d30346770f7512ee62386b3703e54fd17412a8cec53",
  "original_size_bytes": 816708,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-phase-a-bf16-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "bfd35d7efd6c1d931b084a2727b14736b4af544289678931ff68e91ff4ccc0a5",
  "compressed_size_bytes": 150227,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "29adec3bb394c01bdba74843781bb4ce56d820d4430d20b9815ceb86cd97e532",
  "original_size_bytes": 816708,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-phase-a-bf16-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "3627e59b5a719c4240df581dea1a1fd4e1f5c1d3bd12728d9d4bec2c78324054",
  "compressed_size_bytes": 150446,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "e427e25b4affaf332607a6485b78ae4f1ca7c561999c9021eb3a38493df57842",
  "original_size_bytes": 817312,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-phase-a-nf4-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "4beb175437df36c4d70c6162f218dd60e70078187ddf09613e228b85faef33b7",
  "compressed_size_bytes": 150446,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "fa226fd05c658e97ce946052b1f5496ef33cb993ff42e567bf30f50b83f8560f",
  "original_size_bytes": 817312,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-phase-a-nf4-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "8927001f2524c8178f2d3108106aafc9c67419616166ae835d80fd7c3b904851",
  "compressed_size_bytes": 16899,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "717a1f6ede7f95533939118ac0ccc89a58431a50c95b0abb5e3ad1236cceb0b2",
  "original_size_bytes": 188167,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-technical-bf16-identity-checkpoint-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "649e5d4c0fb4cd4efe6aca4a20382d7067f10e6257ea2766f460623bde3bb86b",
  "compressed_size_bytes": 185605,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "5911a5571690fbe6034aeed73fbf0bf951fd424bfc74a781a5173438cddfc694",
  "original_size_bytes": 1051931,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-technical-bf16-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "419d9e1760aae09140d779f82fee77d79174d64cfba45b5c14aa1e0d10b2369b",
  "compressed_size_bytes": 18338,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "fe6fb51c5462c3b81f524e0089b2a5ea3ce252820cce55456099c8d7a07bceb3",
  "original_size_bytes": 193303,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-technical-nf4-identity-checkpoint-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "074b2e5e75a78cc967fadc77c838bcf29e4038b933acbefccb9f5aabab13f9e8",
  "compressed_size_bytes": 187215,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "2dfba643d78a8942fe58a06af8a3490026e049d7352e84e4477b1b306d638f12",
  "original_size_bytes": 1057063,
  "output": null,
  "package": "/workspace/repo_precision_probe_p02/results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-technical-nf4-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
```
