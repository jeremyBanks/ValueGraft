# Precision probe p01 pod log

```text
START PRECISION_PROBE_P01 2026-07-12T07:54:02+00:00
EXPECTED_COMMIT=f67d631e8a31177118c62eb0da8b8acfc059de41 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
PROTOCOL=precision-probe-p01 ORDER=NF4-then-bf16 ATTENTION=eager KV_DTYPE=bfloat16
NVIDIA A100 80GB PCIe, GPU-470c3e18-9f87-0a04-f3bd-e974d59e1903, 580.159.04, 81920
PROVIDER_BUDGET rate=1.39 start_epoch=1783842808 cap_seconds=7200 max_usd=4.00
PRECISION-PROBE-P01-PREREGISTRATION.md: OK
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Downloading cpython-3.12.11-linux-x86_64-gnu (download) (31.1MiB)
 Downloaded cpython-3.12.11-linux-x86_64-gnu (download)
Installed Python 3.12.11 in 1.40s
 + cpython-3.12.11-linux-x86_64-gnu (python3.12)
warning: `/root/.local/bin` is not on your PATH. To use installed Python executables, run `export PATH="/root/.local/bin:$PATH"` or `uv python update-shell`.
Using CPython 3.12.11
Creating virtual environment at: /workspace/p01-venv
Activate with: source /workspace/p01-venv/bin/activate
Using Python 3.12.11 environment at: /workspace/p01-venv
Resolved 47 packages in 231ms
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading nvidia-curand (56.8MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading hf-xet (4.3MiB)
Downloading sympy (6.0MiB)
Downloading transformers (11.4MiB)
Downloading tokenizers (3.1MiB)
Downloading networkx (2.0MiB)
Downloading setuptools (1.0MiB)
Downloading torch (507.6MiB)
Downloading nvidia-nvjitlink (38.8MiB)
Downloading cuda-bindings (6.3MiB)
Downloading numpy (15.9MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading triton (188.6MiB)
Downloading bitsandbytes (57.8MiB)
Downloading nvidia-cudnn-cu13 (349.2MiB)
 Downloaded nvidia-cufile
 Downloaded nvidia-cuda-runtime
 Downloaded tokenizers
 Downloaded setuptools
 Downloaded hf-xet
 Downloaded networkx
 Downloaded cuda-bindings
 Downloaded nvidia-cuda-cupti
 Downloaded numpy
 Downloaded sympy
 Downloaded nvidia-nvjitlink
 Downloaded nvidia-nvshmem-cu13
 Downloaded nvidia-curand
 Downloaded bitsandbytes
 Downloaded transformers
 Downloaded nvidia-cuda-nvrtc
 Downloaded nvidia-cusparse
 Downloaded nvidia-cusparselt-cu13
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cusolver
 Downloaded nvidia-cufft
 Downloaded triton
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cublas
 Downloaded torch
Prepared 47 packages in 16.89s
Installed 47 packages in 517ms
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
RUNTIME_ATTESTED results/precision_probe_p01/precision-probe-p01-runtime_Qwen3-30B-A3B-Instruct-2507_20260712T075402Z.json
RUN precision-probe-p01 model=Qwen/Qwen3-30B-A3B-Instruct-2507 revision=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe output=results/precision_probe_p01 elapsed=258 allowance=6762
{"completion_marker_path": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-completion_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.json", "event": "RUN", "model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "protocol_id": "precision-probe-p01", "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "run_directory": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z", "run_manifest_path": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-run-manifest_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.json"}
`torch_dtype` is deprecated! Use `dtype` instead!
/workspace/p01-venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
/workspace/p01-venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Traceback (most recent call last):
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 1749, in <module>
    main()
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 1744, in main
    _, _, code = run(parse_args())
                 ^^^^^^^^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 1692, in run
    raise fatal_interrupt
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 1603, in run
    receipt = execute_outcome(
              ^^^^^^^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 1089, in execute_outcome
    raise fatal_interrupt
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 991, in execute_outcome
    phase = _stage(
            ^^^^^^^
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 358, in _stage
    value = function()
            ^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/scripts/run_precision_probe_p01.py", line 993, in <lambda>
    lambda: run_phase_a_case(
            ^^^^^^^^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/src/coherent_canary_case.py", line 186, in run_phase_a_case
    c_result = execute_replay_plan(model, plans["C_N"])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/src/coherent_canary_runtime.py", line 336, in execute_replay_plan
    return _run_events(model, plan.token_ids, plan.events, stop_at=stop,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/src/coherent_canary_runtime.py", line 304, in _run_events
    cache, last_logits = _forward(
                         ^^^^^^^^^
  File "/workspace/repo_precision_probe_p01/src/coherent_canary_runtime.py", line 261, in _forward
    result = model(
             ^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py", line 650, in forward
    outputs: MoeModelOutputWithPast = self.model(
                                      ^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py", line 487, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py", line 359, in forward
    hidden_states = self.mlp(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/p01-venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py", line 252, in forward
    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
{
  "chunk_count": 1,
  "compressed_sha256": "9584f81fbd29d8564a4a9d9844cdf6535b90ffe9118411f00eb6dd9eb8159d7a",
  "compressed_size_bytes": 1486368,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "bc61a8c9fccb51fed18ee8cc4f3f54354fa2e9aef715f0634fb404ee82931694",
  "original_size_bytes": 5319768,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-outcome-bf16-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "a55129aa34cbf69529a326fdea7186ce1c78ad2c41e34cf8929f1e489747f282",
  "compressed_size_bytes": 5447,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "5a3394a1c76305d08bfe51abcd1693aa402beee0fa0a82e6f40263530a651599",
  "original_size_bytes": 29097,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-outcome-bf16-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "5ebf6facf449c0bdfe182d62e7288fe27ad4574b6db7f3e24d2a39f4db21f699",
  "compressed_size_bytes": 1490218,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "bc057e2fd39c8131a5cbfcf181fa11d4569a04dd17b2be9b3292a99ca2ee6990",
  "original_size_bytes": 5327477,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-outcome-nf4-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "4cb23ffceb02a3a8aab62fc5ca17bd84d27f6925ae5a9437858047bc388529d2",
  "compressed_size_bytes": 1490236,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "d00d92029674d0ded6c5ff7e5bec855b827204c0d62966d5038fd1dbe739f759",
  "original_size_bytes": 5327474,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-outcome-nf4-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "c61e59ada2c2484c85e5974f00f66efa43ed6a0cca69f6b28321f58c57107e0b",
  "compressed_size_bytes": 155135,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "970459882c681894da9ada2a0f80e108cd15a83c0edd5c5f5886a84d4f68f512",
  "original_size_bytes": 840194,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-phase-a-checkpoint-bf16-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "7e2d803d304cb2f4c96052800e9ca723e320fc1584b5303796298c58a60a7f32",
  "compressed_size_bytes": 156651,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "6550c89cb583a73d8befbf1e9849ad1e86a501ec1ded35999da3361d0eaf6212",
  "original_size_bytes": 845459,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-phase-a-checkpoint-nf4-e01-repeat1-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "79ef256424ec61f48a9ffc203e0ad66e758b3d7844a173c7cfb0cea2064f45ee",
  "compressed_size_bytes": 156654,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "c11ca36e98c188cc69df8eef1ed220d5039c3618585176bef1e02be78e101eb0",
  "original_size_bytes": 845458,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-phase-a-checkpoint-nf4-e01-repeat2-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "34546f0eb4dd214567651d1a81badb308bff8d3fbb3bc908b70e565d86da3e05",
  "compressed_size_bytes": 16869,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "4f06b6f1b59fe10ae105722ef5f411a3167622e036527f5905be2a73e2f81e9f",
  "original_size_bytes": 187977,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-technical-bf16-identity-checkpoint-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "77d2a2f51f0bcb2af7694d649686772eadc9add97e2cf0419f78f81127e4462b",
  "compressed_size_bytes": 185619,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "6762486bd56a9a806e446e17b9c920fe098ce2b3bec37985490bf61f7c7a3d32",
  "original_size_bytes": 1051834,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-technical-bf16-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "68c9542de92e49b9523c5a68a1522d2e179bc54898d78951f0019865d10bf948",
  "compressed_size_bytes": 18309,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "970d097ebbdf4952e4a605317c80ad576eb2f5c041707dd9bc8776a39ea47b8b",
  "original_size_bytes": 193112,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-technical-nf4-identity-checkpoint-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
{
  "chunk_count": 1,
  "compressed_sha256": "d342b8dad7033b96a3dae1262efdcc97c6cf3fba847c8e472b118542845d2c61",
  "compressed_size_bytes": 187212,
  "current_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "original_sha256": "cc55d61a6de589487a9de8ebbbb50aab3287d310cb3cafe02568aade32b398c2",
  "original_size_bytes": 1056967,
  "output": null,
  "package": "/workspace/repo_precision_probe_p01/results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-technical-nf4-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package",
  "packaging_tool_matches_current": true,
  "packaging_tool_sha256": "6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714",
  "status": "VERIFIED"
}
```
