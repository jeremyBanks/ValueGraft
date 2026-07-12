# Legacy per-head map derivation lineage audit

**Author:** OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12

## Verdict

The evaluated legacy per-head map is preserved exactly inside the headline result, but its claimed derivation cannot be reproduced from the committed profile that the repository currently cites. The evaluated map contains 109 `(layer, KV-head)` slots. Reapplying the recorded positive-mean rule to all ten committed `results/tune_head_30b_bf16/` rows selects 121 slots. Only 80 overlap: 29 evaluated slots are nonpositive under the committed profile, and 41 positive committed-profile slots are absent from the evaluated map. Exhaustively applying the rule to all 1,023 nonempty subsets of the ten committed rows reproduces the evaluated map zero times.

This is a provenance failure, not evidence that the stored validation scores were numerically miscomputed. The exact embedded maps and their evaluated outputs survive. Those outputs remain descriptive measurements of the fixed interventions that actually ran. What does not survive is a reproducible account of how the head map was selected. The head, intersection, and union rows therefore cannot be described as reproducibly selected from the cited committed profile.

The per-layer lineage is intact: the tracked 27-layer alpha map equals the map embedded in the evaluated layer result. The embedded intersection and union maps also reconstruct exactly from that 27-layer map plus the evaluated 109-slot head map (74 and 143 slots respectively). Their broken link is upstream: the 109-slot head map's fitting profile.

## Likely loss mechanism, explicitly an inference

Git records the committed layer and head profiles at commits `84bcee776da63983c14d3d6b6af00c37c332c349` and `2db7c6d88037ae3c983b7862be95a1550c61b08d` on July 5. Both predate commit `317a0dd1a13a25b614e867a39270ee176e73926f`, which introduced 4,096-token chunked prefill later that night; the tracked profiles therefore used the earlier single-call prefill path. The later head-scan job explicitly reran `src/run_tune_hf.py` on its pod before selecting a map. `scripts/launch_pod.sh` uploaded source, data, and configs but did not upload `results/`, so the job would have produced a fresh pod-local profile under the later chunked code rather than reused the committed July 5 rows. The July 10 validation commit retained result JSONs with the selected maps embedded, but no corresponding new profile or standalone head configs. This is a coherent explanation for the discrepancy, but the missing pod-local files mean it cannot be proven or repaired.

Consequently, the layer map reconstructs from a preserved single-call profile, while the head/intersection/union maps plausibly derive from an unpreserved chunked profile and were evaluated under chunked prefill. The job intended to fit on c01–c06+n01–n04 before testing c07–c24, but without the originating profile/config/log, the exact derivation and held-out separation of those three variants are not independently auditable.

## Reproducible audit

Machine-readable result:

`results/legacy_head_map_lineage/legacy-head-map-lineage_Qwen3-30B-A3B-Instruct-2507_20260712T055444Z.json`

Recompute:

```bash
uv run python scripts/analyze_legacy_head_map_lineage.py \
  --timestamp 2026-07-12T05:54:44Z \
  --output /tmp/legacy-head-map-lineage.json
uv run pytest -q tests/test_analyze_legacy_head_map_lineage.py
```

## Paper disposition

The paper should retain the exact observed estimates but disclose the broken selection lineage. It should describe the embedded head/intersection/union maps as fixed evaluated interventions whose derivation profile was not preserved, not as maps reproducibly selected from the committed `tune_head_30b_bf16` rows.
