# STATE.md — direct-local successor

**Updated:** 2026-07-13T00:03:20Z. This is the authoritative live state. The
powered-v13 cloud-era version is archived in `notes/` under the slug
`powered-v13-pre-local-state`.

## Objective and live gap

Produce a conversation-level, replicated, controlled upper confidence bound on
history-specific write-time value grafting, then use surplus for the retained-
tail and doubled-summary tests before writing the paper.

- Direct-local successor valid N: **0/48; gap 48**. The c01 run is technical
  N=0. Historical e01 remains N=1 for the old bf16 study and does not enter this
  successor.
- Active goal elapsed at the route change: about **7.3 hours**.
- Paid compute on the replacement path: **$0**. No RunPod is active. Last
  verified post-cleanup balance was `$57.0272577933` against the existing `$80`
  provider spend limit; no additional funding is assumed.

## Working route

The remote powered-v13 release/lifecycle stack is retired. Do not repair or
retry it. The successor runs directly on the local Apple MLX device with the
fully cached model
`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`, resolved revision
`e9675aa3ca5f900ccef55267914466d55ab325fa`.

Observed direct-path gates:

- L0 identity passed: token-identical continuation, zero logit difference, and
  stored keys matched post-RoPE rather than unrotated keys.
- L3 identities passed: alpha-zero graft equaled fresh and alpha-one old-state
  reconstruction equaled the source.
- The literal 8,736-token c01 canary completed all old arms and 70 probes in
  345 seconds with no swap. Its relevant continuation arms took 1–3 seconds;
  probe generation caused nearly all latency. Its E−B movement was
  `+0.00671875` nat/token, exactly repeating an old result and adding no N.
- L5 applied sham passed on 1,973 rows across all 48 layers: fresh B and B
  routed through the exact value-assignment path had bit-exact caches and equal
  mean logprob `-1.515625`.
- Runtime provenance now records the exact revision, four weight-blob SHA-256s,
  config/tokenizer/template hashes, 4-bit/8-bit mixed quantization, dtype
  inventory, and 48-layer/4-KV-head/128-head-dimension geometry.

## Immediate execution

1. Use frozen `LOCAL-COHERENT-STATE-N48-V3.md`. V1 terminated on its carrier
   prompt/filter. V2's single treatment-blind development canary exposed
   inherited eligibility gates that mismatched the correct-target endpoint; it
   is excluded from v3 and both earlier protocols remain archived intact.
2. The outcome-blind frame, literal order, 80 ranked candidates, and complete-
   history reviews are frozen. Validate the two fixed external v2 carrier
   conditions mechanically and independently against all 80 fixtures before
   any v2 forward pass.
3. Force both exact carrier conditions under C/W and fresh B. Screen only
   oracle/fresh damage and validity; select the first six eligible per stratum
   by frozen rank.
4. Run the five scientific states locally: fresh B, correct-history value
   graft, wrong-history value graft, applied displacement-matched V-row
   placebo, and oracle A; retain the sham identity gate. Teacher-force the
   declared C and W targets. Do not run the old 160-token answer probes.
5. Checkpoint each conversation/render atomically and report valid N, elapsed
   wall time, and resource state every four completed conversations.
6. Average the two fixed carrier conditions within conversation and compute the
   frozen one-sided bound.
   Then run retained-tail followed by doubled-summary/second-copy-only surplus
   tests.

## Hard boundaries

- Conversations are N. Renders, targets, plants, layers, heads, arms, schedules,
  and reruns never increase N.
- No outcome-visible selection or replacement. Invalid candidates advance only
  to the next already-frozen rank.
- The local result is a scoped MLX 4-bit value-only result, not the abandoned
  v13 bf16/full-KV result.
- Missing or unavailable applied controls remain missing; they are never
  relabeled null.
- One local MLX model process at a time. Unique outputs; persist before the next
  case. No provider allocation is needed for the active path.
