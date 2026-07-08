"""Generic cross-architecture value-graft gap-closure harness (single model).

Runs the paper's value-graft benefit measurement -- the teacher-forced
gap-closure metric (E-B)/(A-B) on the GOLD continuation -- across MANY ~30B
model architectures, not just Qwen, to test whether the effect generalizes.

This is a GENERALIZATION of ``src/gap_closure_cat.py``. The A(full) / B(compacted)
/ E(value-graft) construction, the exact-twin alignment, and the metric are
UNCHANGED -- only the model-specific parts are made architecture-agnostic:

  * KV-cache class: we FORCE a plain ``DynamicCache`` for every prefill and every
    rebuild, driving attention purely through explicit ``position_ids`` + the
    standard ``past_key_values`` interface. Snapshots are read generically
    (``kvlib_hf.snapshot_cache``: ``cache.layers[i].keys/values`` on
    transformers>=5, legacy ``key_cache``/``value_cache`` otherwise) -- no model
    type is hardcoded. The value-graft (``blend_values``) is pure tensor surgery
    on those snapshots and is already model-agnostic.

  * Chat template: family-aware via ``arms_common`` (qwen ChatML / gemma /
    mistral). Non-Qwen templates use prefix-diff message boundaries and a
    family-appropriate B-context builder.

  * KV geometry (num_key_value_heads / head_dim / rope base): read from
    ``config`` (or nested ``config.text_config`` for multimodal wrappers) rather
    than hardcoded -- see ``detect_kv_geometry`` and ``arms_hf.rope_base``.

PER-MODEL SMOKE GATE (critical -- numbers are NOT trusted unless it passes):
  (a) alpha_V=0 graft reproduces B's teacher-forced logprobs within a tight
      tolerance (proves the graft plumbing is bit-identical to B), AND
  (b) alpha_V=0.75 graft actually CHANGES the teacher-forced logprobs vs B
      (proves the graft is real, not a no-op on this cache class).
Also checked: the snapshot's stored token count matches the input length -- a
padded/HybridCache snapshot (Gemma-2/3 sliding-window) has T != n_tokens and is
flagged, and the per-layer head/dim shape matches the config geometry.

If ANYTHING fails -- OOM, cache API incompatible, HybridCache padded snapshot,
template not prefix-stable, smoke gate -- the model is written out with status
UNSUPPORTED (or ERROR) + a reason and NO gap-closure numbers. One bad model
NEVER crashes the sweep.

FIXED SUMMARY (critical confound fix -- else the sweep is uninterpretable):
Different models write different-quality compaction summaries, so the baseline
gap (meaning lost, lp_A-lp_B) would vary per model BEFORE any grafting,
confounding "grafting works less" with "this model lost less to recover". So the
summary TEXT is a SHARED INPUT, held FIXED across every model: generated ONCE by
a single designated summarizer (``--make-summaries``, SC_SUMMARIZER_MODEL) into a
shared file (``results/cross_arch/_summaries.json``), then fed verbatim to every
model. Each model still re-tokenizes + re-prefills that SAME text to produce its
own write-time value snapshot (values are necessarily per-model) -- but the
compaction content is identical, so ARCHITECTURE is the only variable.

REPORTING (ROBUST -- the mean-of-ratio (E-B)/(A-B) is Cauchy-unstable with small
denominators and is NO LONGER the primary metric): the HEADLINE per-category
quantity is now raw_EB = lp_E - lp_B (bounded: the graft's logprob lift over
compaction) with a pure-Python percentile bootstrap 95% CI over the plants
(N=10000, seed=42). We also report %_helped (fraction with raw_EB > 0) with its
bootstrap CI, and -- as a scale-free SECONDARY only -- the MEDIAN of the ratio
(E-B)/(A-B) CONDITIONED on |lp_A - lp_B| > 0.5 (tiny-denominator probes dropped;
n_kept reported). The unconditioned mean ratio is retired/deprecated (kept only
under clearly-labeled legacy fields for continuity). This robust block lives in
``by_category_robust``; a per-model ``verdict`` marks SIGNIFICANT when the raw_EB
CI excludes 0 for referent OR sense. Per-model results are DIRECTIONAL (small-N
subset -> bootstrap CI will often overlap 0); the INFERENTIAL claim is the
aggregate ACROSS models, not any single noisy per-model number. This is stated in
the output (``interpretation`` field).

Env:
  SC_HF_MODEL         : HF repo id (REQUIRED for the real run)
  SC_SUMMARIZER_MODEL : model used by --make-summaries (default Qwen/Qwen3.6-27B)
  SC_CONV_LIMIT       : first N conversations (default 4)
  SC_GC_ALPHA         : value-graft strength alpha_V (default 0.75)
  SC_ALPHA0_TOL       : max |lp_E0 - lp_B| for smoke (a) (default 5e-3, TIGHT)
  SC_CHANGE_TOL       : min max|lp_E - lp_B| for smoke (b) (default 1e-3)
  SC_MAX_GOLD_TOK     : cap on gold continuation tokens (default 80)
  SC_TRUST_REMOTE     : "1"/"0" trust_remote_code (default 1; needed e.g. GLM)
  SC_STRONG_PRIOR     : "1"/"0" FEATURE #2 signed codename disambiguation for
                        strong_prior plants (default 1, additive block)
  SC_CHAMPION_SCAN    : FEATURE #3 per-layer champion scan into N fractional-depth
                        regions (0=off default; >=2 sets N; else uses default 6)
  SC_CHAMPION_REGIONS : FEATURE #3 rescue test -- comma-separated region indices
                        to graft together, e.g. "4,5" (alpha=0 elsewhere)
  SC_ABLATE_QK_NORM   : "1"/"0" WITHIN-MODEL QK-NORM ABLATION (default 0). When 1,
                        after load every QK-norm module (q_norm/k_norm/query|key
                        layernorm|norm) is replaced with Identity so the forward
                        pass runs WITHOUT QK-norm -- the clean causal H1 test.
                        FAILs LOUD (status=ERROR) if no QK-norm modules are found.
                        Everything else stays IDENTICAL (within-model comparison).

FEATURE #1 (multi-probe averaging) is ALWAYS ON: each plant's raw_EB is the MEAN
over its paraphrased ``probes`` of the teacher-forced gold lift (de-noises probe
wording); per-probe values are kept in the trace.

Build the shared fixed summaries ONCE (designated summarizer), then run models:
  SC_SUMMARIZER_MODEL=Qwen/Qwen3.6-27B python3 src/cross_arch_probe.py --make-summaries
  SC_HF_MODEL=google/gemma-4-27b-it     python3 src/cross_arch_probe.py

Self-test (NO torch / model / pod):
  python3 src/cross_arch_probe.py --dry-run
constructs every case, prints N plants for SC_CONV_LIMIT, exercises the
config-based KV-geometry detection on synthetic configs of several families, and
asserts the generic snapshot/graft code paths are model-type-agnostic.

Out: results/cross_arch/<model-slug>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms_common import (  # noqa: E402  (stdlib-only, import-safe on CPU box)
    SUMMARY_REQUEST,
    _find_exact_subblock,
    build_alignment,
    build_alignment_direct,
    build_b_messages,
    build_b_messages_gemma,
    canonical_ids_any,
    detect_template_family,
    message_starts_any,
    render_hf,
    strip_reasoning_block,
)

# The FIXED summaries are produced EXTERNALLY (coordinator uses Sonnet -- a
# realistic mid-tier summarizer; a frontier model would write an
# unrealistically good summary and invalidate the test) as a plain
# {conv_id: summary_text} JSON. This is the required input to every model run.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
DEFAULT_SUMMARIES = str(
    Path(__file__).resolve().parent.parent / "data" / "fixed_summaries.json")
DEFAULT_SUMMARIZER = "Qwen/Qwen3.6-27B"

# All continuity categories (skip contaminated). Full dissociation + power.
# strong_prior = famous-name codenames (each such plant carries `keywords` = the
# conversation meaning and `anti_keywords` = the famous prior meaning); it gets
# the standard gold raw_EB like every other category AND, additively, the SIGNED
# disambiguation readout (FEATURE #2).
CATS = ("sense", "referent", "stance", "ruled_out", "evicted_fact",
        "strong_prior")
# full gradient: sense (pure meaning) → referent/ruled_out (evicted decisions) →
# evicted_fact (precise verbatim, tests meaning-vs-verbatim claim) → strong_prior
# (codename vs famous prior); stance = null anchor.

INTERPRETATION = (
    "PRIMARY metric is raw_EB = lp_E - lp_B (bounded logprob lift of the graft "
    "over compaction) with a bootstrap 95% CI -- NOT the mean of the ratio "
    "(E-B)/(A-B), which is Cauchy-unstable with small denominators. The ratio is "
    "reported only as a scale-free SECONDARY, as a MEDIAN conditioned on "
    "|lp_A-lp_B| > 0.5. Per-model results are DIRECTIONAL only: the "
    "sense+referent subset is small, so the raw_EB CI will often straddle 0 -- do "
    "NOT read a single model's overlap-with-0 as a 'null' (see verdict). The "
    "inferential claim is the AGGREGATE across models. Read raw_EB together with "
    "pre_graft_gap (lp_A-lp_B): a small lift on a small available gap is not the "
    "same as a graft failure."
)


def bootstrap_ci_95(values, n_boot=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean (pure python, no numpy)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "lo": vals[0], "hi": vals[0], "n": 1}
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"mean": sum(vals) / n, "lo": lo, "hi": hi, "n": n}


def bootstrap_ci_95_cluster(clusters, n_boot=10000, seed=0):
    """CLUSTER (block) percentile bootstrap 95% CI of the mean.

    ``clusters`` is a list of lists: one inner list per CONVERSATION holding that
    conversation's per-probe values. Resampling is over CONVERSATIONS (whole
    clusters, with replacement); the probes inside a resampled conversation are
    kept together and pooled. Because probes within a conversation are positively
    correlated (same context, same summary), this honest cluster CI is WIDER than
    the naive probe-level ``bootstrap_ci_95`` that treats every probe as
    independent. Returns the SAME keys as bootstrap_ci_95 plus ``n_clusters``.

    The point estimate (``mean``) is the pooled grand mean over all probes -- the
    SAME number bootstrap_ci_95 would report -- so only the interval width
    changes, never the headline value."""
    clusters = [[v for v in c if v is not None] for c in clusters]
    clusters = [c for c in clusters if c]
    flat = [v for c in clusters for v in c]
    n_c = len(clusters)
    if not flat:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    mean = sum(flat) / len(flat)
    if n_c == 1:
        return {"mean": mean, "lo": mean, "hi": mean,
                "n": len(flat), "n_clusters": 1}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        tot = 0.0
        cnt = 0
        for _ in range(n_c):
            c = clusters[rng.randrange(n_c)]
            tot += sum(c)
            cnt += len(c)
        means.append(tot / cnt)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(flat), "n_clusters": n_c}


# Fixed bootstrap params for the ROBUST reporting block (raw_EB, %_helped).
ROBUST_N_BOOT = 10000
ROBUST_SEED = 42

# ---------------------------------------------------------------------------
# CONTROLS (all env-guarded; default run is unaffected). See module docstring.
# ---------------------------------------------------------------------------
# Alpha dose-response sweep values (SC_ALPHA_SWEEP=1). alpha_v (0.75) is in-set
# so the sweep's 0.75 column reproduces the primary raw_EB.
ALPHA_SWEEP_VALUES = (0.25, 0.5, 0.75, 1.0, 2.0)
# Placebo-graft corruption modes (SC_PLACEBO=<mode>). See _placebo_index_plan.
PLACEBO_MODES = ("shuffle_pos", "shuffle_probe", "gauss", "mean")
# FEATURE #3 default region count when the champion scan is enabled but no
# explicit N (>=2) is given (SC_CHAMPION_SCAN=1 -> this default).
CHAMPION_SCAN_DEFAULT_N = 6


def _placebo_index_plan(mode, old_idx, prev_old_idx, seed):
    """PURE (no torch) source-index remap for the position-shuffling placebos.

    Returns a NEW list of OLD-cache source indices (same length as old_idx) to be
    paired with the unchanged destination indices, OR None for the
    tensor-corruption placebos (gauss/mean) that instead build a corrupted source
    snapshot with the ORIGINAL pairing.

      shuffle_pos   : permute the source values ACROSS the grafted positions --
                      right values, wrong slots (a within-probe scramble).
      shuffle_probe : draw source indices (with replacement) from ANOTHER probe's
                      grafted positions (prev conversation's snapshot) -- a
                      different probe's structured write-state. Falls back to
                      shuffle_pos on the first conversation (no prev available).
      gauss / mean  : returns None -> caller corrupts the source VALUE tensor
                      (random Gaussian matched to per-layer value norm; or the
                      mean value vector) keeping the original index pairing.

    Same seed -> same plan (deterministic, reproducible)."""
    rng = random.Random(seed)
    n = len(old_idx)
    if mode == "shuffle_pos" or (mode == "shuffle_probe" and not prev_old_idx):
        perm = list(range(n))
        rng.shuffle(perm)
        return [old_idx[p] for p in perm]
    if mode == "shuffle_probe":
        m = len(prev_old_idx)
        return [prev_old_idx[rng.randrange(m)] for _ in range(n)]
    if mode in ("gauss", "mean"):
        return None
    raise ValueError(f"unknown SC_PLACEBO mode {mode!r}; "
                     f"expected one of {PLACEBO_MODES}")


def _corrupt_source_values(old_snap, old_idx, mode, seed):
    """Build a source snapshot whose VALUE rows at ``old_idx`` are corrupted
    (torch path; keys untouched, original index pairing preserved).

      mean  : each grafted row -> the per-layer mean over the grafted rows.
      gauss : each grafted row -> Gaussian noise rescaled so its per-row L2 norm
              matches the mean per-row norm of the real grafted block IN THAT
              LAYER (matched energy, destroyed structure).

    Returns a new [(K, V'), ...] usable as ``old_snap`` in blend_values with the
    UNCHANGED pairs."""
    import torch  # noqa: PLC0415
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.tensor(list(old_idx))
    out = []
    for k, v in old_snap:
        v2 = v.clone()
        block = v2[..., idx, :].float()            # [B, H, n, D]
        if mode == "mean":
            rep = block.mean(dim=-2, keepdim=True).expand_as(block)
            v2[..., idx, :] = rep.to(v2.dtype)
        elif mode == "gauss":
            noise = torch.randn(block.shape, generator=g).to(block.device)
            tgt_norm = block.norm(dim=-1, keepdim=True).mean()   # per-layer scale
            cur = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            noise = noise / cur * tgt_norm
            v2[..., idx, :] = noise.to(v2.dtype)
        else:
            raise ValueError(f"_corrupt_source_values: bad mode {mode!r}")
        out.append((k, v2))
    return out


# Leaf module names that implement QK-norm in the attention block (Qwen3/OLMo-2/
# Gemma use q_norm/k_norm; other families name them query/key layernorm|norm).
# SHARED by detect_model_hparams (detection) and ablate_qk_norm (removal) so the
# two can never drift apart.
_QK_NORM_LEAF_NAMES = ("q_norm", "k_norm", "query_layernorm", "key_layernorm",
                       "query_norm", "key_norm")


def ablate_qk_norm(model) -> tuple:
    """WITHIN-MODEL QK-NORM ABLATION (SC_ABLATE_QK_NORM) -- the clean causal test
    of H1 (QK-norm presence -> positive referent graft sign). Replaces every
    QK-norm submodule (matching the SAME leaf names detect_model_hparams detects,
    ``_QK_NORM_LEAF_NAMES``) in the attention blocks with ``torch.nn.Identity()``
    so the forward pass runs WITHOUT QK-norm, holding everything else fixed.

    Returns ``(n_ablated, sorted_module_names)``. Already-Identity modules are not
    counted (idempotent). The CALLER is responsible for FAILing LOUD when
    ``n_ablated == 0`` (ablation requested but no QK-norm modules found) -- this
    function does not decide policy, it just reports what it changed."""
    import torch  # noqa: PLC0415
    # Collect FIRST, then mutate -- do not mutate the module tree while iterating.
    targets = [n for n, m in model.named_modules()
               if n.rsplit(".", 1)[-1] in _QK_NORM_LEAF_NAMES
               and m is not None
               and not isinstance(m, torch.nn.Identity)]
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, torch.nn.Identity())
    return len(targets), sorted(targets)


def detect_model_hparams(config, model=None) -> dict:
    """Architecture hyper-parameters for the sign-of-graft regression. Reads the
    (possibly nested text_config) HF config, model-type agnostic. ``gqa_ratio`` =
    n_heads / n_kv (1.0 == full MHA, >1 == GQA, == n_heads == MQA).

    ``qk_norm``: config-key detection is UNRELIABLE (Qwen3/Gemma-3,4/OLMo-2 apply
    QK-norm as q_norm/k_norm *modules* in the attention block, NOT a config flag).
    When a loaded ``model`` is passed we detect it robustly from the module tree
    (presence of a q_norm/k_norm/query|key-layernorm submodule); the config keys
    are only a fallback for the no-model (pure) path."""
    cfg = _cfg_text(config)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_heads
    head_dim = getattr(cfg, "head_dim", None)
    hidden = getattr(cfg, "hidden_size", None)
    if head_dim is None and n_heads and hidden:
        head_dim = hidden // n_heads
    gqa = (n_heads / n_kv) if (n_heads and n_kv) else None
    qk_keys = ("use_qk_norm", "qk_norm", "q_norm", "k_norm", "qk_layernorm",
               "use_qk_layernorm", "attention_qk_norm", "query_key_layernorm")
    qk_norm = (any(bool(getattr(cfg, k, None)) for k in qk_keys)
               or any(bool(getattr(config, k, None)) for k in qk_keys))
    qk_norm_source = "config" if qk_norm else None
    # ROBUST: detect q_norm/k_norm modules on the loaded model (the real signal).
    if model is not None:
        for _n, _m in model.named_modules():
            leaf = _n.rsplit(".", 1)[-1]
            if leaf in _QK_NORM_LEAF_NAMES and _m is not None:
                qk_norm = True
                qk_norm_source = "module:" + leaf
                break
    rope_theta = getattr(cfg, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(cfg, "rotary_emb_base", None)
    return {
        "model_type": getattr(config, "model_type", None)
        or getattr(cfg, "model_type", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "gqa_ratio": gqa,
        "qk_norm": bool(qk_norm),
        "qk_norm_source": qk_norm_source,
        "rope_theta": rope_theta,
        "hidden_size": hidden,
        "vocab_size": getattr(cfg, "vocab_size", None)
        or getattr(config, "vocab_size", None),
    }


def _group_by_conv(rows):
    """[{conversation_id, raw_EB}, ...] -> [[raw_EB, ...] per conversation].

    Preserves conversation grouping for the cluster bootstrap (CONTROL #6);
    insertion-ordered so the grouping is deterministic."""
    groups: dict = {}
    for r in rows:
        cid = r.get("conversation_id")
        groups.setdefault(cid, []).append(r["raw_EB"])
    return list(groups.values())


def _mean(values):
    """Mean (pure python, no numpy). Skips None; returns None on empty input.

    Used for FEATURE #1 multi-probe averaging (mean raw_EB over a plant's
    paraphrased probes) and for the strong_prior meaning-phrase aggregation."""
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _mass_shift(conv_lp_B, conv_lp_E, prior_lp_B, prior_lp_E):
    """FEATURE #2 signed disambiguation for a strong_prior plant.

    mass_shift = (conv_lp_E - prior_lp_E) - (conv_lp_B - prior_lp_B).
    POSITIVE => the graft moved logprob mass toward the CONVERSATION meaning and
    away from the famous PRIOR meaning (i.e. the graft disambiguated the
    codename). Pure arithmetic so it is unit-testable without a model."""
    return (conv_lp_E - prior_lp_E) - (conv_lp_B - prior_lp_B)


def partition_layers(n_layers, n_groups):
    """FEATURE #3: partition ``n_layers`` into up to ``n_groups`` CONTIGUOUS
    layer regions [(lo, hi), ...] by RELATIVE (fractional) DEPTH.

    Region ``k`` spans the layers whose depth fraction falls in
    ``[k/n_groups, (k+1)/n_groups)`` -- i.e. lo = floor(k*L/N), hi =
    floor((k+1)*L/N). This makes "region k" the SAME computational stage across
    models of different layer counts (region 3 of 6 is the same relative depth in
    a 48-layer and a 64-layer model, unlike an absolute layer index). The regions
    tile ``[0, n_layers)`` exactly once (contiguous, gapless, full cover). Empty
    regions (when n_layers < n_groups) are dropped, giving n_layers singletons.
    Pure (no torch)."""
    n_layers = int(n_layers)
    n_groups = max(1, int(n_groups))
    regions = []
    for k in range(n_groups):
        lo = (k * n_layers) // n_groups
        hi = ((k + 1) * n_layers) // n_groups
        if hi > lo:
            regions.append((lo, hi))
    return regions


def _champion_configs(n_regions, custom_regions=None):
    """FEATURE #3: the list of graft configs the champion scan scores, as
    ``(label, frozenset_of_region_indices)``. Pure (no torch), unit-testable.

      * ``region_k``  : each single region alone (the descriptive per-region scan).
      * ``all``       : ALL regions together -- MUST reproduce the uniform raw_EB
                        (miswiring sanity check). NOTE: the sum of the single
                        region raw_EBs will NOT equal this / the uniform value --
                        the graft composes NONLINEARLY over depth, so that is
                        EXPECTED, not a bug.
      * ``regions_..``: an ARBITRARY custom subset (from SC_CHAMPION_REGIONS),
                        grafted together with alpha=0 elsewhere -- the "rescue
                        test" (e.g. graft only the late/positive regions on a
                        net-negative model and see if raw_EB flips positive).
    """
    cfgs = [(f"region_{r}", frozenset([r])) for r in range(n_regions)]
    cfgs.append(("all", frozenset(range(n_regions))))
    if custom_regions:
        valid = frozenset(r for r in custom_regions if 0 <= r < n_regions)
        if valid:
            label = "regions_" + "_".join(str(r) for r in sorted(valid))
            if label not in {lbl for lbl, _ in cfgs}:
                cfgs.append((label, valid))
    return cfgs


def _median(values):
    """Median (pure python, no numpy). Returns None on empty input."""
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def robust_category_stats(raw_eb_vals, ratio_gap_pairs, raw_eb_clusters=None):
    """ROBUST per-category summary. Retires the unstable mean-of-ratio.

    raw_eb_vals    : [lp_E - lp_B, ...] over the plants (PRIMARY, bounded).
    ratio_gap_pairs: [(ratio, |lp_A - lp_B|), ...] for the scale-free secondary;
                     ratio may be None when the denominator was ~0.
    raw_eb_clusters: OPTIONAL list-of-lists grouping the raw_EB values BY
                     CONVERSATION (one inner list per conv). When given, an
                     honest CLUSTER (conversation-level) bootstrap CI is added as
                     ``raw_EB_ci_cluster`` alongside the probe-level ``raw_EB_ci``,
                     plus ``n_conversations``. Probes within a conversation are
                     correlated, so the probe-level CI understates uncertainty and
                     the cluster CI is the one to trust.

    Returns {n, raw_EB_mean, raw_EB_ci:[lo,hi], pct_helped, pct_helped_ci:[lo,hi],
             median_ratio_cond, n_cond[, raw_EB_ci_cluster, n_conversations]}.
    pct_helped is a FRACTION in [0,1] (share of plants with raw_EB > 0); its CI is
    a bootstrap over the same plants. median_ratio_cond drops tiny-denominator
    probes (|lp_A-lp_B| <= 0.5) and reports how many survived as n_cond."""
    raw = [v for v in raw_eb_vals if v is not None]
    n = len(raw)
    eb = bootstrap_ci_95(raw, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    helped_flags = [1.0 if v > 0 else 0.0 for v in raw]
    ph = bootstrap_ci_95(helped_flags, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    # scale-free secondary: median ratio, tiny-denominator probes dropped.
    cond = [r for (r, gap) in ratio_gap_pairs
            if r is not None and gap is not None and abs(gap) > 0.5]
    out = {
        "n": n,
        "raw_EB_mean": eb["mean"],
        "raw_EB_ci": [eb["lo"], eb["hi"]],
        "pct_helped": ph["mean"],
        "pct_helped_ci": [ph["lo"], ph["hi"]],
        "median_ratio_cond": _median(cond),
        "n_cond": len(cond),
    }
    if raw_eb_clusters is not None:
        cl = bootstrap_ci_95_cluster(
            raw_eb_clusters, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
        out["raw_EB_ci_cluster"] = [cl["lo"], cl["hi"]]
        out["n_conversations"] = cl["n_clusters"]
    return out


# ---------------------------------------------------------------------------
# Config-based KV geometry (pure, no torch). Works off any HF PretrainedConfig
# or a plain object exposing the same attributes; unwraps multimodal text_config.
# ---------------------------------------------------------------------------
def _cfg_text(config):
    """Unwrap a nested text_config (multimodal wrappers) if present."""
    return getattr(config, "text_config", None) or config


def detect_kv_geometry(config) -> dict:
    """Return {model_type, num_attention_heads, num_key_value_heads, head_dim}
    from an HF config, model-type-agnostically. NO hardcoded architecture names.

    head_dim: explicit config.head_dim if present, else hidden_size //
    num_attention_heads. num_key_value_heads: explicit (GQA/MQA) else falls back
    to num_attention_heads (full MHA)."""
    cfg = _cfg_text(config)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_heads  # full multi-head attention (no grouped-query)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and n_heads:
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = (hidden // n_heads) if hidden else None
    return {
        "model_type": getattr(config, "model_type", None)
        or getattr(cfg, "model_type", None),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Case construction (pure, no torch) -- SUBSET of the corpus.
# ---------------------------------------------------------------------------
def conversation_paths(data_dir: Path):
    return sorted(Path(data_dir).glob("c*.json"))


def select_plants(conv: dict) -> list[dict]:
    """sense+referent, non-contaminated, with a non-empty gold continuation."""
    out = []
    for pl in conv.get("plants", []):
        if pl.get("category") not in CATS:
            continue
        if pl.get("contaminated_early") or pl.get("contaminated_tail"):
            continue
        if not str(pl.get("gold", "")).strip():
            continue
        out.append(pl)
    return out


def collect_specs(data_dir: Path, conv_limit: int):
    """(conv_dict, [plant,...]) for the first ``conv_limit`` convs that have >=1
    usable plant. Pure -- no tokenizer, no torch."""
    specs = []
    for p in conversation_paths(data_dir)[:conv_limit]:
        conv = json.loads(p.read_text())
        plants = select_plants(conv)
        if plants:
            specs.append((conv, plants))
    return specs


# ===========================================================================
# PER-MODEL IN-CONTEXT (NATIVE) RENDERING -- CROSS-ARCH DESIGN v2 / v2.1
# ---------------------------------------------------------------------------
# The graft re-injects the model's OWN write-time value vectors, and that only
# works on NATIVE context (the model's own assistant replies). So each model must
# be measured on conversations IT would produce. Instead of reading pre-rendered
# data/synthetic/*.json (which are Qwen-4B replies -- FOREIGN to every other
# model), we SHARE only the semantic SCAFFOLD (data/scenarios.json: system + user
# turns + plants) and have the TEST MODEL generate its OWN assistant elaborations
# in-context. This is a port of src/compose.py's growing-KV-cache reply gen onto
# the HF/transformers path already used here.
#
# LOAD-BEARING (design v2.1): the GOLD CONTINUATION stays SHARED -- it comes from
# each plant's `gold` field in the scaffold, NOT from any model-generated text.
# The metric raw_EB = lp_E - lp_B is teacher-forced on that shared gold; only the
# difference over a SHARED target cancels per-model continuation-nativeness. The
# native rendering below fills ONLY the assistant elaboration turns; it NEVER
# touches gold. (select_plants pulls `gold` straight from the scaffold plant.)
#
# The pure turn-ordering / trimming / covariate / gate logic lives here (CPU
# self-testable); the growing-cache generation itself is torch and lives in the
# heavy section (native_render_specs), reasoned-through but GPU-UNVERIFIED.
# ===========================================================================
def scenario_turn_plan(scenario: dict, seed: int) -> dict:
    """PURE (no model/torch) ordered turn plan for ONE scaffold scenario.

    Replicates src/compose.py's proven interleaving EXACTLY so a native render
    is structurally identical to the validated synthetic corpus recipe:
      * early   : each early_user_turn, in order;
      * middle  : plants (deterministically shuffled by ``seed``) interleaved
                  with the middle fillers, ~3 plants between each filler;
      * tail    : tail_filler[0], the plants' tail_user_fragments (shuffled,
                  same rng continued), tail_filler[1].
    Each turn becomes a user message + a model-generated assistant reply (2
    messages), on top of the system message (index 0).

    Returns {turns, early_end_msg, middle_end_msg, n_messages, n_early,
    n_middle, n_tail}. ``turns`` is a list of {kind, text[, plant_id]} with kind
    in {early, filler, plant, tail}. Message indices count the system message as
    0 and 2 messages per turn -- middle_end_msg is the index of the FIRST tail
    message (what run_model uses as ``tail_start_msg``)."""
    rng = random.Random(seed)
    turns: list[dict] = []
    for t in scenario["early_user_turns"]:
        turns.append({"kind": "early", "text": t})
    n_early = len(scenario["early_user_turns"])

    plants = list(scenario["plants"])
    rng.shuffle(plants)
    fillers = list(scenario["middle_filler_user_turns"])
    seq: list[tuple] = []
    pi, fi = 0, 0
    while pi < len(plants) or fi < len(fillers):
        if fi < len(fillers):
            seq.append(("filler", fillers[fi])); fi += 1
        for _ in range(3):  # ~3 plants between fillers
            if pi < len(plants):
                seq.append(("plant", plants[pi])); pi += 1
    for kind, item in seq:
        if kind == "plant":
            turns.append({"kind": "plant", "text": item["middle_user"],
                          "plant_id": item.get("id")})
        else:
            turns.append({"kind": "filler", "text": item})
    n_middle = len(seq)

    tail_frags = [p["tail_user_fragment"] for p in scenario["plants"]
                  if p.get("tail_user_fragment")]
    rng.shuffle(tail_frags)
    tail_seq = [scenario["tail_filler_user_turns"][0], *tail_frags,
                scenario["tail_filler_user_turns"][1]]
    for t in tail_seq:
        turns.append({"kind": "tail", "text": t})
    n_tail = len(tail_seq)

    return {
        "turns": turns,
        "n_early": n_early,
        "n_middle": n_middle,
        "n_tail": n_tail,
        "early_end_msg": 1 + 2 * n_early,
        "middle_end_msg": 1 + 2 * (n_early + n_middle),
        "n_messages": 1 + 2 * (n_early + n_middle + n_tail),
    }


def trim_capped_reply(text: str) -> str:
    """PURE: trim a length-capped reply back to its last complete sentence /
    paragraph (mirrors compose.py). If no sentence boundary lands past the first
    third, the text is returned unchanged (better a hard cut than an empty
    reply)."""
    cut = max(text.rfind("\n\n"), text.rfind(". "),
              text.rfind("! "), text.rfind("? "))
    if cut > len(text) // 3:
        return text[: cut + 1].rstrip()
    return text


def reply_covariates(records: list) -> dict:
    """PURE (design v2.1 residual-confound covariate): aggregate per-model
    assistant-reply statistics for the sign~geometry+covariates regression.

    ``records`` = [{"n_tokens": int, "logprob_sum": float|None}, ...] over the
    model's OWN generated elaboration replies. Returns:
      * mean_reply_len_tokens        -- reply LENGTH covariate.
      * mean_reply_logprob_per_token -- reply INFO-CONTENT proxy: the model's mean
        per-token logprob of its OWN replies under itself (write-time). Higher
        (closer to 0) = more confident/predictable replies; this stands in for the
        richness of the value payload the graft re-injects. logprob_sum may be
        None (not captured) -> that reply is skipped for the info-content mean but
        still counts for length.
    Pure arithmetic so it is unit-testable on fake numbers with no model."""
    recs = [r for r in records if r.get("n_tokens")]
    n = len(recs)
    if not n:
        return {"n_replies": 0, "mean_reply_len_tokens": None,
                "total_reply_tokens": 0, "mean_reply_logprob_per_token": None,
                "n_replies_with_logprob": 0}
    total_tok = sum(r["n_tokens"] for r in recs)
    lp_recs = [r for r in recs if r.get("logprob_sum") is not None]
    lp_tok = sum(r["n_tokens"] for r in lp_recs)
    lp_sum = sum(r["logprob_sum"] for r in lp_recs)
    return {
        "n_replies": n,
        "total_reply_tokens": total_tok,
        "mean_reply_len_tokens": total_tok / n,
        "mean_reply_logprob_per_token": (lp_sum / lp_tok) if lp_tok else None,
        "n_replies_with_logprob": len(lp_recs),
    }


# Gate defaults (design v2.1). All configurable via env / CLI; chosen permissive
# so the validated pre-rendered path (SC_NATIVE_RENDER=0) is unaffected.
HEADROOM_FLOOR_DEFAULT = 0.3     # min per-category A-B gap to be interpretable
TASK_LPA_FLOOR_DEFAULT = -8.0    # min lp_A per-token; below = model can't do task
HEADROOM_EPS = 1e-3              # denominator floor for raw_EB normalization


def category_headroom(raw_EB_mean, pre_gaps, floor=HEADROOM_FLOOR_DEFAULT,
                      eps=HEADROOM_EPS) -> dict:
    """PURE (design v2.1 gate #2 HEADROOM): per-category continuity headroom and
    headroom-normalized raw_EB.

    headroom = mean(lp_A - lp_B) over the category's plants = how much meaning the
    compaction actually EVICTED (and thus how much is even recoverable). If a
    model's native summary already PRESERVES the referent there is no A-B gap ->
    the graft has nothing to recover -> a ~0 raw_EB is a CEILING/FLOOR artifact,
    NOT "geometry says the graft harms". So:
      * raw_EB_normalized = raw_EB_mean / max(headroom, eps) puts the lift on a
        per-unit-evicted-meaning scale (comparable across models with different
        summary quality);
      * floor = headroom < ``floor`` flags the category as UNINTERPRETABLE (no
        evicted meaning = nothing to recover) so it is EXCLUDED from the sign
        verdict -- it is not counted as harm.
    Pure arithmetic; unit-testable on fake logprobs."""
    gaps = [g for g in pre_gaps if g is not None]
    headroom = (sum(gaps) / len(gaps)) if gaps else None
    if raw_EB_mean is None or headroom is None:
        norm = None
    else:
        norm = raw_EB_mean / max(headroom, eps)
    return {
        "headroom": headroom,
        "raw_EB_normalized": norm,
        "floor": bool(headroom is not None and headroom < floor),
        "floor_threshold": floor,
    }


def task_competence_ok(la, floor=TASK_LPA_FLOOR_DEFAULT) -> bool:
    """PURE (design v2.1 gate #3 TASK-COMPETENCE): a plant is scorable only if the
    model can do the task WITH full context -- i.e. its per-token gold logprob
    under A (``la``, already a per-token mean from ``tf``) clears ``floor``. A
    model whose native replies never establish the referent (very low lp_A) yields
    a degenerate plant that should be excluded, not read as a graft signal."""
    return la is not None and la >= floor


# ---------------------------------------------------------------------------
# Family-aware B-context: builder + (summary_msg_idx, tail_msg_idx) so the
# alignment regions can be located regardless of template family.
# ---------------------------------------------------------------------------
def build_b_and_indices(family: str, msgs, summary_text: str, tsm: int):
    """Return (b_msgs, summary_idx, tail_idx).

    qwen/mistral/unknown : [system, assistant(note)@1, *tail@2]      -> (1, 2)
    gemma                : [system, user(note)@1, assistant(ack)@2,
                            *tail@3]                                  -> (1, 3)
    """
    if family == "gemma":
        return build_b_messages_gemma(msgs, summary_text, tsm), 1, 3
    return build_b_messages(msgs, summary_text, tsm), 1, 2


# ---------------------------------------------------------------------------
# Model status container
# ---------------------------------------------------------------------------
def _slug(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def write_result(out_dir: Path, model: str, doc: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    of = out_dir / f"{_slug(model)}.json"
    json.dump(doc, open(of, "w"), indent=1)
    print(f"WROTE {of}  status={doc.get('status')}", flush=True)
    return of


class Unsupported(Exception):
    """Graceful bail-out: model can't be handled; record reason, don't crash."""


# ---------------------------------------------------------------------------
# PURE (tokenizer-only, NO torch / NO model) token layout for the alignment.
# Factored out of the old run_model.build_summary_snapshot + inline region
# construction so the EXACT same token geometry can be exercised on a CPU box
# (--smoke-align) without a GPU. run_model adds the force_prefill snapshots on
# top of this; the smoke path needs none.
# ---------------------------------------------------------------------------
def summary_token_layout(tok, family, msgs, summary_text):
    """Write-time summary token layout WITHOUT the force_prefill snapshot.

    The subtlety is a ``<think>...</think>`` reasoning block in a self-generated
    summary, and it is handled by MATCHING what the authoritative gap_closure_cat
    (F1) harness does -- which is where the known-good self-gen numbers
    (referent +0.136 / +0.156) come from:

      * ``text`` is the FULL self-gen text, reasoning block INCLUDED, and it is
        handed VERBATIM to build_b. The Qwen chat template strips the reasoning
        block from the assistant-turn history -- and, because the "[Context
        note] ..." preamble that build_b prepends sits BEFORE the block, the
        template swallows that preamble too. So the compacted (B) summary turn
        holds exactly the think-free summary with NO preamble. This is the B the
        known-good result was measured on; re-introducing the preamble (by
        pre-stripping the reasoning here) changes B's cache and FLIPS the effect
        (self-gen referent collapsed from +0.136 to a null +0.01 with the
        preamble present). For a fixed/foreign summary with no reasoning block
        there is nothing to strip, so the preamble survives -- exactly the
        verified fixed-summary layout, unchanged.

      * WRITE-TIME PREFILL (``old_ids``) is req_ids + the FULL text, so the
        grafted summary tokens' VALUE vectors are the ones the model computed
        with its reasoning in context.

      * ALIGNMENT REGION (``s_start``:``s_end``) is only the think-free CLEAN
        summary -- exactly what survives in B -- located as an EXACT contiguous
        sub-block of the full write-time span. build_alignment_direct maps it 1:1
        onto B's summary turn (no raise, no dropped tokens); the reasoning tokens
        are present in the value context but never grafted. This is the exact,
        auditable equivalent of what the old difflib matcher did implicitly (drop
        the unmatched reasoning tokens, graft the summary)."""
    clean_text = strip_reasoning_block(summary_text)
    conv_ids = canonical_ids_any(tok, msgs, render_hf)
    if family == "gemma":
        req_msgs = list(msgs)
        if req_msgs and req_msgs[-1]["role"] == "user":
            req_msgs = req_msgs + [
                {"role": "assistant", "content": "Understood."}]
        req_msgs = req_msgs + [{"role": "user", "content": SUMMARY_REQUEST}]
        req_ids = render_hf(tok, req_msgs, True)
    else:
        req_ids = render_hf(
            tok, msgs + [{"role": "user", "content": SUMMARY_REQUEST}], True)
    if req_ids[:len(conv_ids)] != conv_ids:
        raise Unsupported("summary-request render is not a prefix of conv "
                          "(template not prefix-stable)")
    full_summ_ids = tok(summary_text, add_special_tokens=False).input_ids
    clean_summ_ids = tok(clean_text, add_special_tokens=False).input_ids
    # Locate the clean summary as an exact contiguous sub-block of the full
    # write-time span (keeps the reasoning tokens in context, grafts only the
    # clean summary). Fall back to a think-prefix + clean split if a boundary
    # token merge means the standalone clean tokens are not a verbatim sub-block.
    off = _find_exact_subblock(list(full_summ_ids), list(clean_summ_ids))
    if off is None:
        idx = summary_text.rfind(clean_text) if clean_text else -1
        think_prefix = summary_text[:idx] if idx > 0 else ""
        think_ids = (tok(think_prefix, add_special_tokens=False).input_ids
                     if think_prefix else [])
        full_summ_ids = list(think_ids) + list(clean_summ_ids)
        off = len(think_ids)
    old_ids = list(req_ids) + list(full_summ_ids)
    s_start = len(req_ids) + off
    s_end = s_start + len(clean_summ_ids)
    return {
        "text": summary_text,        # FULL text -> build_b (template strips think
                                     # + preamble); matches F1's known-good B
        "gen_ids": clean_summ_ids,   # the grafted summary tokens
        "conv_end": len(conv_ids),
        "s_start": s_start,
        "s_end": s_end,
        "old_ids": old_ids,          # req + FULL summary (reasoning kept in ctx)
    }


def build_token_context(tok, family, msgs, summary_text, tsm):
    """PURE construction of the full alignment geometry -- EXACTLY what run_model
    builds, minus the force_prefill snapshots. Returns ids/starts (A side),
    b_msgs/b_ids/b_starts (B side), the summary token layout, and the two
    alignment regions. Shared by run_model and --smoke-align so the smoke guards
    the identical code path the real run uses."""
    ids = canonical_ids_any(tok, msgs, render_hf)
    starts = message_starts_any(tok, msgs, ids, render_hf)
    summ = summary_token_layout(tok, family, msgs, summary_text)
    b_msgs, summ_idx, tail_idx = build_b_and_indices(
        family, msgs, summ["text"], tsm)
    b_ids = canonical_ids_any(tok, b_msgs, render_hf)
    b_starts = message_starts_any(tok, b_msgs, b_ids, render_hf)
    regions = [
        ((b_starts[tail_idx], len(b_ids)), (starts[tsm], summ["conv_end"])),
        ((b_starts[summ_idx], b_starts[summ_idx + 1]),
         (summ["s_start"], summ["s_end"])),
    ]
    return {
        "ids": ids, "starts": starts, "b_msgs": b_msgs, "b_ids": b_ids,
        "b_starts": b_starts, "summ": summ, "regions": regions,
        "summ_idx": summ_idx, "tail_idx": tail_idx,
    }


# ===========================================================================
# The heavy path (torch/transformers) lives entirely below, imported lazily so
# --dry-run runs on a CPU-only box with no ML deps.
# ===========================================================================
# Native-render generation defaults (CROSS-ARCH DESIGN v2). Reply gen is GREEDY
# by default (temp 0) so the corpus is REPRODUCIBLE and each reply is the model's
# most-likely = maximally-native elaboration; low-temp sampling is available via
# SC_NATIVE_TEMP for diversity if wanted.
NATIVE_MAX_REPLY_DEFAULT = 320
NATIVE_TEMP_DEFAULT = 0.0
NATIVE_TOP_P = 0.8
# THROUGHPUT (07-08): batch the per-reply DECODE across a model's conversations
# (batch dim = the convs). The expensive part of native render is the token-by-
# token decode loop; batching it is a ~10x forward-pass win with ZERO validity
# cost -- the decoded KV is discarded and re-prefilled canonically anyway, so
# batched decode only needs to reproduce the same reply TOKEN IDS. Greedy (temp
# 0, the validated anchor setting) is BYTE-IDENTICAL to the per-token path
# (CPU-verified in _self_test_batched_decode). Guarded by SC_BATCHED_RENDER
# (default on); SC_BATCHED_RENDER=0 falls back to the validated per-token path.
NATIVE_BATCHED_RENDER_DEFAULT = (
    os.environ.get("SC_BATCHED_RENDER", "1") not in ("0", "false", "False"))
# MEMORY CAP (07-08): turn-major batching keeps EVERY conv's KV cache alive at
# once -- on a 30B model that busts an 80GB A100. So decode the model's convs in
# CHUNKS of SC_NATIVE_BATCH (default 4): each chunk of <=N convs renders fully
# (its caches freed) before the next starts, bounding peak memory to N conv-caches
# + the model, not all convs. If a chunk still OOMs, _native_render_group catches
# it, splits the chunk in half, and retries (deterministic greedy -> identical
# output regardless of chunk boundaries), degrading gracefully instead of failing.
NATIVE_BATCH_DEFAULT = max(1, int(os.environ.get("SC_NATIVE_BATCH", "4")))


def _native_pick_token(logits_row, temp, gen):
    """Choose ONE token from a single [vocab] logits row. Greedy (temp 0): argmax
    -- deterministic and byte-identical whether called per-token or per batch row.
    temp>0: seeded top-p nucleus using the per-conv generator ``gen``. Module-level
    so both decode paths AND the self-test exercise the identical sampler."""
    import torch  # noqa: PLC0415
    if temp and temp > 0:
        probs = torch.softmax(logits_row.float() / temp, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) - sp < NATIVE_TOP_P
        keep[0] = True
        sp, si = sp[keep], si[keep]
        return int(si[torch.multinomial(
            sp.cpu() / sp.sum().cpu(), 1, generator=gen)].item())
    return int(torch.argmax(logits_row, dim=-1).item())


def _native_per_token_decode(model, cache, first_logits, next_position, *,
                             max_reply_tokens, temp, eos_ids, seed, dev):
    """Per-token (batch=1) decode -- the VALIDATED path. Returns (reply_ids,
    lp_sum); lp_sum = SUM of chosen tokens' write-time logprobs. EOS excluded from
    both ids and lp_sum. Mutates ``cache`` (grows it by the reply)."""
    import torch  # noqa: PLC0415
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    toks: list[int] = []
    lp_sum = 0.0
    logits = first_logits
    pos = next_position
    for _ in range(max_reply_tokens):
        logp = torch.log_softmax(logits.float(), dim=-1)[0]
        t = _native_pick_token(logits[0], temp, gen)
        if t in eos_ids:
            break
        lp_sum += float(logp[t].item())
        toks.append(t)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[t]], device=dev),
                        past_key_values=cache,
                        position_ids=torch.tensor([[pos]], device=dev),
                        use_cache=True)
        logits = out.logits[:, -1, :]
        pos += 1
    return toks, lp_sum


def _native_batched_decode(model, caches, first_logits_list, next_positions, *,
                           max_reply_tokens, temp, eos_ids, seeds, dev, pad_id):
    """BATCHED decode across a model's convs (batch dim = the convs). Builds a
    transient LEFT-PADDED batched KV cache from the per-conv caches, then decodes
    every reply concurrently -- ONE batched model forward per token -- with
    per-sequence EOS/length handling. Only the FORWARD is batched; each row's token
    choice reuses _native_pick_token, so GREEDY output is byte-identical to
    _native_per_token_decode and seeded sampling uses one generator per conv.

    The batched cache is DISCARDED and the per-conv ``caches`` are NOT mutated
    (native_render throws the reply KV away and re-prefills it canonically), so
    batching only has to reproduce the same reply TOKEN IDS. Left-padding + a
    per-row attention_mask that zeroes the pad prefix + explicit per-row
    position_ids (each conv's TRUE absolute position) make each row's forward
    mathematically equal to its unbatched forward (float32 CPU: exact, verified in
    _self_test_batched_decode; bf16 GPU: up to rounding -- GPU-UNVERIFIED).

    Returns [(reply_ids, lp_sum), ...] aligned to the input order."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812
    from transformers import DynamicCache  # noqa: PLC0415

    from kvlib_hf import snapshot_cache  # noqa: PLC0415

    B = len(caches)
    if B == 0:
        return []
    snaps = [snapshot_cache(c) for c in caches]     # clones -- caches untouched
    n_layers = len(snaps[0])
    lens = [s[0][0].shape[-2] for s in snaps]        # per-conv cache length
    assert list(lens) == [int(p) for p in next_positions], \
        "batched-decode: cache length must equal next decode position"
    lmax = max(lens)
    bcache = DynamicCache()                          # transient, discarded below
    for li in range(n_layers):
        ks, vs = [], []
        for s, L in zip(snaps, lens):
            k, v = s[li]
            pad = lmax - L                           # LEFT-pad along the T axis
            ks.append(F.pad(k, (0, 0, pad, 0)))
            vs.append(F.pad(v, (0, 0, pad, 0)))
        bcache.update(torch.cat(ks, 0), torch.cat(vs, 0), li)
    # [B, lmax] mask: 0 over the left pad, 1 over each conv's real prefix.
    attn = torch.zeros(B, lmax, dtype=torch.long, device=dev)
    for bi, L in enumerate(lens):
        attn[bi, lmax - L:] = 1
    gens = []
    for sd in seeds:
        g = torch.Generator(device="cpu")
        if sd is not None:
            g.manual_seed(sd)
        gens.append(g)
    toks: list[list[int]] = [[] for _ in range(B)]
    lp_sums = [0.0] * B
    done = [False] * B
    logits = torch.stack([fl[0] for fl in first_logits_list], 0)   # [B, vocab]
    pos = torch.tensor([int(p) for p in next_positions], device=dev)  # [B]
    for _step in range(max_reply_tokens):
        logp = torch.log_softmax(logits.float(), dim=-1)              # [B, V]
        step_tok = torch.full((B,), pad_id, dtype=torch.long, device=dev)
        for bi in range(B):
            if done[bi]:
                continue
            t = _native_pick_token(logits[bi], temp, gens[bi])
            if t in eos_ids:
                done[bi] = True
                continue
            lp_sums[bi] += float(logp[bi, t].item())
            toks[bi].append(t)
            step_tok[bi] = t
        if all(done):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=dev)], 1)
        with torch.no_grad():
            out = model(input_ids=step_tok[:, None], position_ids=pos[:, None],
                        past_key_values=bcache, attention_mask=attn,
                        use_cache=True)
        logits = out.logits[:, -1, :]
        pos = pos + 1                    # done rows advance too (output ignored)
    del bcache
    return list(zip(toks, lp_sums))


def native_render_specs(model, tok, family, scenarios, conv_limit, *,
                        max_reply_tokens, temp, seed_base,
                        batched_render=NATIVE_BATCHED_RENDER_DEFAULT,
                        native_batch=NATIVE_BATCH_DEFAULT):
    """PER-MODEL IN-CONTEXT (NATIVE) RENDER of the shared scaffold (design v2).

    GPU path (torch). For each of the first ``conv_limit`` scaffold scenarios,
    grow ONE KV cache and have the TEST MODEL generate its OWN assistant reply
    after every scaffold user turn (early / middle-plant+filler / tail), producing
    a conversation NATIVE to this model. Returns:
      (specs, reply_records)
    where ``specs`` is [(conv_dict, plants), ...] in the EXACT shape collect_specs
    returns (so the rest of run_model is unchanged) and ``reply_records`` is the
    flat per-reply [{n_tokens, logprob_sum}] list for reply_covariates().

    LOAD-BEARING: conv_dict["plants"] = scenario["plants"] VERBATIM -> select_plants
    pulls each plant's SHARED ``gold`` continuation from the scaffold, never from
    generated text. Only the assistant elaboration turns are model-filled.

    This is a direct port of compose.py's ConversationBuilder onto HF primitives,
    including the Qwen canonical-prefix handling: after each sampled/greedy reply
    the growing cache is truncated back to the canonical (user-final) prefix and
    the assistant block is re-prefilled in its canonical (think-free, non-final)
    form, so the stored value vectors match the compacted rendering the graft will
    later align against. GPU-UNVERIFIED (no pod run here).

    THROUGHPUT (``batched_render``, default on / SC_BATCHED_RENDER): the reply
    DECODE is batched ACROSS this model's conversations (batch dim = the convs).
    The loops are inverted to be TURN-MAJOR: at each turn index every still-active
    conv prefills its user block (per-conv), then ALL active convs decode their
    replies CONCURRENTLY in one batched forward per token, with per-sequence
    EOS/length handling. Greedy (temp 0, the validated anchor setting) is
    BYTE-IDENTICAL to the per-token path -- the batched decode's KV is discarded
    and the reply re-prefilled canonically, so batching only has to reproduce the
    same reply TOKEN IDS, and it does (CPU-verified, _self_test_batched_decode).
    SC_BATCHED_RENDER=0 restores the validated per-token, conv-major path.

    MEMORY CAP (``native_batch`` / SC_NATIVE_BATCH, default 4): because turn-major
    batching keeps EVERY conv's KV cache alive at once, the convs are processed in
    CHUNKS of ``native_batch`` -- each chunk of <=N convs renders fully (its caches
    freed) before the next, so peak memory is N conv-caches + the model, not all
    convs. Chunk boundaries do NOT affect any conv's output (each row's decode is
    independent of batch composition). If a chunk still OOMs, it is split in half
    and retried, so a large model degrades to smaller batches instead of failing."""
    import torch  # noqa: PLC0415
    from transformers import DynamicCache  # noqa: PLC0415

    from kvlib_hf import (  # noqa: PLC0415
        prefill,
        rebuild_cache,
        snapshot_cache,
    )

    eos = model.config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    dev = model.device
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = next(iter(eos_ids)) if eos_ids else 0

    def _ids(seq):
        return torch.tensor([list(seq)], device=dev)

    def _pos(lo, hi):
        return torch.arange(lo, hi, device=dev)[None]

    def _truncate(cache, keep):
        snap = snapshot_cache(cache)
        sl = [(k[..., :keep, :].contiguous(), v[..., :keep, :].contiguous())
              for k, v in snap]
        return rebuild_cache(sl, DynamicCache)

    # ---- per-conv mutable state (both paths share the per-turn bookkeeping) ----
    # Results are collected by scenario index so a chunk can be rebuilt fresh on
    # OOM-retry without disturbing already-finished convs.
    results_by_idx: dict = {}          # idx -> (conv, plants)
    reply_records_by_idx: dict = {}    # idx -> [reply record, ...]

    def _build_state(idx, scenario):
        return {
            "idx": idx,
            "scenario": scenario,
            "plan": scenario_turn_plan(scenario, seed=seed_base + idx),
            "conv_seed": seed_base + idx,
            "msgs": [{"role": "system", "content": scenario["system"]}],
            "cache": DynamicCache(),
            "stream": [],
            "truncated": 0, "empty": 0, "early_tokens": 0, "middle_tokens": 0,
            "reply_records": [], "result": None,
        }

    def _prefill_user_turn(st, ti):
        """Append user turn ``ti``, prefill its (user + generation-prompt) block
        into the conv's cache. Returns (cache, first_logits, next_position)."""
        scenario, plan = st["scenario"], st["plan"]
        msgs, stream = st["msgs"], st["stream"]
        msgs.append({"role": "user", "content": plan["turns"][ti]["text"]})
        # canonical prefix THROUGH the user turn (plain render is canonical here --
        # the final message is a USER, no think injection). Mirrors compose r_canon.
        r_canon_user = render_hf(tok, msgs, False)
        r_gen = render_hf(tok, msgs, True)
        assert r_gen[: len(stream)] == stream, \
            f"{scenario['id']} turn {ti}: canonical prefix broke (gen)"
        assert r_gen[: len(r_canon_user)] == r_canon_user, \
            f"{scenario['id']} turn {ti}: gen render doesn't extend canonical"
        new_ids = r_gen[len(stream):]
        cache, logits = prefill(model, _ids(new_ids), past=st["cache"],
                                position_ids=_pos(len(stream), len(r_gen)))
        st["cache"] = cache
        st["_r_canon_user"] = r_canon_user
        return cache, logits, len(r_gen)

    def _finalize_reply(st, ti, reply_ids, lp_sum):
        """Commit one decoded reply: text/trim/covariate, append assistant msg,
        then re-canonicalize the cache (truncate to the user-final prefix, re-
        prefill the assistant block in canonical non-final form). On the conv's
        LAST turn, finalize the conv (build its dict, free the cache)."""
        scenario, plan = st["scenario"], st["plan"]
        msgs = st["msgs"]
        capped = len(reply_ids) >= max_reply_tokens
        reply_text = tok.decode(reply_ids).strip()
        if capped:
            st["truncated"] += 1
            reply_text = trim_capped_reply(reply_text)
        if not reply_text:
            st["empty"] += 1
        st["reply_records"].append({"n_tokens": len(reply_ids),
                                    "logprob_sum": lp_sum})
        msgs.append({"role": "assistant", "content": reply_text})

        r_canon_user = st["_r_canon_user"]
        r_new = canonical_ids_any(tok, msgs, render_hf)
        assert r_new[: len(r_canon_user)] == r_canon_user, \
            f"{scenario['id']} turn {ti}: assistant block changed the prefix"
        cache = _truncate(st["cache"], len(r_canon_user))
        if len(r_new) > len(r_canon_user):
            cache, _ = prefill(
                model, _ids(r_new[len(r_canon_user):]), past=cache,
                position_ids=_pos(len(r_canon_user), len(r_new)))
        st["cache"] = cache
        st["stream"] = r_new
        if ti + 1 == plan["n_early"]:
            st["early_tokens"] = len(r_new)
        if ti + 1 == plan["n_early"] + plan["n_middle"]:
            st["middle_tokens"] = len(r_new)
        if ti + 1 == len(plan["turns"]):
            _finalize_conv(st)

    def _finalize_conv(st):
        scenario, plan = st["scenario"], st["plan"]
        msgs, stream = st["msgs"], st["stream"]
        st["cache"] = None                       # free per-conv cache promptly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        conv = {
            "id": scenario["id"],
            "title": scenario.get("title"),
            "messages": msgs,
            "sections": {
                "early_end_msg": plan["early_end_msg"],
                "middle_end_msg": plan["middle_end_msg"],
                "early_end_tokens": st["early_tokens"],
                "middle_end_tokens": st["middle_tokens"],
                "total_tokens": len(stream),
            },
            "plants": scenario["plants"],
            "meta": {"native_render": True, "seed": st["conv_seed"], "temp": temp,
                     "truncated_replies": st["truncated"],
                     "empty_replies": st["empty"]},
        }
        plants = select_plants(conv)
        assert len(msgs) == plan["n_messages"], \
            f"{scenario['id']}: built {len(msgs)} msgs != planned {plan['n_messages']}"
        print(f"  [native] {scenario['id']}: {len(msgs)} msgs, "
              f"{len(stream)} tokens (early {st['early_tokens']}, middle "
              f"{st['middle_tokens']}), truncated={st['truncated']} "
              f"empty={st['empty']}, {len(plants)} usable plants", flush=True)
        st["result"] = (conv, plants)

    def _decode_group_turnmajor(group):
        """TURN-MAJOR batched render of ONE chunk of convs (their caches are alive
        together). At each turn index, every still-active conv in the chunk
        prefills its user block, then all active convs decode concurrently in one
        batched forward per token. Concurrency == len(group) <= native_batch."""
        max_turns = max(len(st["plan"]["turns"]) for st in group)
        for ti in range(max_turns):
            active = [st for st in group if ti < len(st["plan"]["turns"])]
            caches, firsts, nps, seeds = [], [], [], []
            for st in active:
                c, l, npos = _prefill_user_turn(st, ti)
                caches.append(c); firsts.append(l); nps.append(npos)
                seeds.append(st["conv_seed"])
            results = _native_batched_decode(
                model, caches, firsts, nps, max_reply_tokens=max_reply_tokens,
                temp=temp, eos_ids=eos_ids, seeds=seeds, dev=dev, pad_id=pad_id)
            for st, (reply_ids, lp_sum) in zip(active, results):
                _finalize_reply(st, ti, reply_ids, lp_sum)
            print(f"  [native-batched] chunk turn {ti + 1}/{max_turns}: "
                  f"{len(active)} convs decoded", flush=True)

    def _render_chunk(items):
        """Render one chunk (list of (idx, scenario)) as ONE alive-together group.
        On CUDA OOM, free the partial group, split the chunk in half, and render
        each half sequentially (fewer caches alive). Greedy decode is deterministic
        and each conv is independent, so splitting a chunk yields IDENTICAL output.
        A single-conv chunk that OOMs re-raises (surfaced UNSUPPORTED upstream)."""
        group: list = []
        try:
            group = [_build_state(idx, sc) for idx, sc in items]
            _decode_group_turnmajor(group)
            for st in group:
                results_by_idx[st["idx"]] = st["result"]
                reply_records_by_idx[st["idx"]] = st["reply_records"]
        except torch.cuda.OutOfMemoryError:
            if len(items) <= 1:
                raise
            for st in group:                       # drop partial caches, reclaim
                st["cache"] = None
            del group
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = (len(items) + 1) // 2
            print(f"  [native-batched] OOM on chunk of {len(items)} convs -> "
                  f"splitting into {mid} + {len(items) - mid} and retrying",
                  flush=True)
            _render_chunk(items[:mid])
            _render_chunk(items[mid:])

    items = list(enumerate(scenarios[:conv_limit]))
    if batched_render:
        # Process convs in CHUNKS of native_batch so peak memory is bounded by N
        # conv-caches + the model (not all convs); each chunk frees before the next.
        for lo in range(0, len(items), native_batch):
            _render_chunk(items[lo: lo + native_batch])
    else:
        # CONV-MAJOR per-token fallback (the validated path; one cache at a time).
        for idx, scenario in items:
            st = _build_state(idx, scenario)
            for ti in range(len(st["plan"]["turns"])):
                c, l, npos = _prefill_user_turn(st, ti)
                reply_ids, lp_sum = _native_per_token_decode(
                    model, c, l, npos, max_reply_tokens=max_reply_tokens,
                    temp=temp, eos_ids=eos_ids, seed=st["conv_seed"], dev=dev)
                _finalize_reply(st, ti, reply_ids, lp_sum)
            results_by_idx[idx] = st["result"]
            reply_records_by_idx[idx] = st["reply_records"]

    # assemble in scenario order (stats are order-independent, but keep it stable)
    specs: list = []
    reply_records: list = []
    for idx, _scenario in items:
        reply_records.extend(reply_records_by_idx[idx])
        conv, plants = results_by_idx[idx]
        if plants:
            specs.append((conv, plants))
    return specs, reply_records


def run_model(model_id: str, data_dir: Path, out_dir: Path,
              fixed_summaries: dict | None, *,
              conv_limit: int, alpha_v: float, alpha0_tol: float,
              change_tol: float, max_gold_tok: int,
              trust_remote_code: bool,
              placebo_mode: str | None = None, alpha_sweep: bool = False,
              seed: int = ROBUST_SEED,
              strong_prior: bool = True, champion_scan: int = 0,
              champion_regions: list | None = None,
              native_render: bool = False, scenarios: list | None = None,
              native_max_reply: int = NATIVE_MAX_REPLY_DEFAULT,
              native_temp: float = NATIVE_TEMP_DEFAULT,
              headroom_floor: float = HEADROOM_FLOOR_DEFAULT,
              task_lpa_floor: float = TASK_LPA_FLOOR_DEFAULT,
              ablate_qk_norm_flag: bool = False) -> dict:
    """fixed_summaries: {conv_id: summary_text} loaded from the shared external
    file (Sonnet-written, held IDENTICAL across models). If None, no fixed file
    was present and we fall back to per-model self-generated summaries (results
    are NOT cross-model comparable -- flagged in ``summary_source``).

    native_render (CROSS-ARCH DESIGN v2, SC_NATIVE_RENDER): when True, the corpus
    is RENDERED IN-CONTEXT by THIS model from the shared ``scenarios`` scaffold
    (native_render_specs) instead of read pre-rendered from data/synthetic, and
    the summary is ALWAYS this model's own self-gen (fixed_summaries ignored) --
    the graft re-injects the model's own write-time values, so it must be measured
    on this model's own native conversation. The shared plant ``gold`` continuation
    is UNCHANGED (comes from the scaffold, teacher-forced). Adds per-model reply
    covariates + the headroom / task-competence gates (design v2.1)."""
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DynamicCache,
    )

    from arms_common import SUMMARY_REQUEST as _REQ  # noqa: PLC0415
    from arms_hf import generate_summary_hf, rope_base  # noqa: PLC0415
    from kvlib_hf import (  # noqa: PLC0415
        blend_values,
        prefill,
        rebuild_cache,
        snapshot_cache,
        tf_logprobs,
    )

    doc: dict = {
        "model": model_id,
        "architecture": None,
        "status": "ERROR",
        "reason": None,
        "alpha_v": alpha_v,
        "conv_limit": conv_limit,
        "native_render": native_render,   # DESIGN v2: per-model in-context corpus
        "summary_source": (
            "PER-MODEL SELF-GEN on NATIVE in-context corpus (design v2)"
            if native_render else
            "FIXED external (shared across models)"
            if fixed_summaries is not None
            else "PER-MODEL fallback (NOT cross-model comparable)"),
        "reply_covariates": None,  # v2.1 covariate: reply length + info-content
        "gates": None,             # v2.1 gate summary (headroom / task-competence)
        "skipped_convs": [],
        "n_plants": 0,
        "qk_norm_ablated": False,       # SC_ABLATE_QK_NORM: was QK-norm removed?
        "n_qk_modules_ablated": 0,      # how many q_norm/k_norm modules replaced
        "kv_geometry": None,
        "pre_graft_gap": None,     # mean lp_A - lp_B (meaning lost to compaction)
        "raw_EB": None,            # PRIMARY aggregate: mean(lp_E-lp_B) + boot CI
        "gap_closure": None,       # DEPRECATED/unstable mean (E-B)/(A-B) + boot CI
        "by_category": {},         # legacy block (kept for continuity)
        "by_category_robust": {},  # ROBUST block -- the one that matters
        "verdict": None,           # SIGNIFICANT | null/underpowered (directional)
        "interpretation": INTERPRETATION,
        "model_hparams": None,     # arch regression: predicts the graft's SIGN
        "traces": [],              # raw per-probe traces (offline recompute)
        "placebo": None,           # CONTROL #1 (SC_PLACEBO): corrupted-source graft
        "alpha_sweep": None,       # CONTROL #3 (SC_ALPHA_SWEEP): dose-response
        "strong_prior_signed": None,  # FEATURE #2: signed codename disambiguation
        "champion_scan": None,     # FEATURE #3: per-layer-region raw_EB fingerprint
        "smoke": {"alpha0_ok": None, "graft_changes_output": None,
                  "graft_direction_ok": None, "alpha0_max_abs_diff": None,
                  "graft_max_abs_diff": None, "graft_mean_signed_diff": None,
                  "identity_ok": None, "identity_max_abs_diff": None},
    }
    if placebo_mode is not None and placebo_mode not in PLACEBO_MODES:
        doc.update(status="ERROR",
                   reason=f"bad SC_PLACEBO={placebo_mode!r}; expected one of "
                          f"{PLACEBO_MODES}")
        return doc

    # ---- config first (cheap; gives architecture + geometry even if load fails)
    try:
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        doc["architecture"] = getattr(config, "model_type", None)
        doc["kv_geometry"] = detect_kv_geometry(config)
        doc["model_hparams"] = detect_model_hparams(config)
    except Exception as e:  # noqa: BLE001
        doc["status"] = "ERROR"
        doc["reason"] = f"config load failed: {type(e).__name__}: {e}"
        return doc

    # ---- load model
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=trust_remote_code)
        model.eval()
        # ROBUST qk_norm: re-detect from the LOADED model's modules (config-key
        # detection misses Qwen3/Gemma/OLMo-2 which use q_norm/k_norm submodules).
        try:
            _hp = detect_model_hparams(config, model)
            doc["model_hparams"]["qk_norm"] = _hp["qk_norm"]
            doc["model_hparams"]["qk_norm_source"] = _hp["qk_norm_source"]
        except Exception:  # noqa: BLE001
            pass
        # WITHIN-MODEL QK-NORM ABLATION (SC_ABLATE_QK_NORM) -- clean causal H1 test.
        # Run a QK-norm model with QK-norm DISABLED, everything else IDENTICAL.
        if ablate_qk_norm_flag:
            n_ablated, ablated_names = ablate_qk_norm(model)
            if n_ablated == 0:
                doc.update(
                    status="ERROR",
                    reason="ablation requested but no QK-norm modules found "
                           "(SC_ABLATE_QK_NORM=1 on a model with no "
                           "q_norm/k_norm/query|key layernorm modules); refusing "
                           "to silently run un-ablated")
                return doc
            doc["qk_norm_ablated"] = True
            doc["n_qk_modules_ablated"] = n_ablated
            # After ablation the forward pass has NO QK-norm; reflect that in
            # hparams but keep the distinct qk_norm_ablated flag so an ablated
            # run is unambiguous vs a natively-no-QK-norm model.
            doc["model_hparams"]["qk_norm"] = False
            doc["model_hparams"]["qk_norm_source"] = "ablated"
            print(f"SC_ABLATE_QK_NORM=1 -> ablated {n_ablated} QK-norm modules "
                  f"(replaced with Identity); e.g. {ablated_names[:3]} ... "
                  f"forward pass now runs WITHOUT QK-norm.", flush=True)
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED", reason=f"OOM on load: {e}")
        return doc
    except Exception as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED", reason=f"model load failed: {type(e).__name__}: {e}")
        return doc

    try:
        family = detect_template_family(tok)
        rbase = rope_base(model)
    except Exception as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED",
                   reason=f"template/rope introspection failed: {type(e).__name__}: {e}")
        return doc

    geom = doc["kv_geometry"] or {}

    def force_prefill(ids):
        """Prefill through a FORCED DynamicCache (generic across architectures);
        returns a cloned per-layer (K,V) snapshot. Verifies the snapshot's token
        axis matches the input length -- a padded/HybridCache snapshot fails
        here (Gemma sliding-window) and is surfaced as UNSUPPORTED upstream."""
        cache = DynamicCache()
        t = torch.tensor([list(ids)], device=model.device)
        cache, _ = prefill(model, t, past=cache)
        snap = snapshot_cache(cache)
        # geometry / padding checks (generic HybridCache detector)
        k0 = snap[0][0]
        stored_T = k0.shape[-2]
        if stored_T != len(ids):
            raise Unsupported(
                f"padded/HybridCache snapshot: stored T={stored_T} != "
                f"n_tokens={len(ids)} (sliding-window cache not snapshot-safe)")
        if geom.get("num_key_value_heads") and k0.shape[1] != geom["num_key_value_heads"]:
            raise Unsupported(
                f"cache head axis {k0.shape[1]} != config num_key_value_heads "
                f"{geom['num_key_value_heads']}")
        return snap

    def tf(snap, feed, targets, npos):
        cache = rebuild_cache(snap, DynamicCache)
        pos = torch.arange(npos, npos + len(feed), device=model.device)[None]
        lps = tf_logprobs(model, cache, feed, targets, position_ids=pos)
        return sum(lps) / max(1, len(targets))

    def tf_sum(snap, feed_prefix, phrase_ids, npos):
        """FEATURE #2 helper: SUM (not mean) of the teacher-forced logprobs of
        ``phrase_ids`` scored as the continuation IMMEDIATELY AFTER
        ``feed_prefix`` (the probe's answer-eliciting suffix). Summing keeps
        multi-token meaning phrases comparable to single-token ones as a total
        answer-mass. npos is the cache's stored token count (position offset)."""
        cache = rebuild_cache(snap, DynamicCache)
        feed = list(feed_prefix) + list(phrase_ids[:-1])
        pos = torch.arange(npos, npos + len(feed), device=model.device)[None]
        lps = tf_logprobs(model, cache, feed, phrase_ids, position_ids=pos)
        return sum(lps)

    def value_alignment_per_layer(b_snap, old_snap, pairs):
        """FEATURE #3 geometry readout (~free -- tensors already in hand): per
        layer, the mean COSINE SIMILARITY between the grafted WRITE-TIME value
        vectors (summary snapshot at the old/source indices) and the co-located
        READ-TIME value vectors (B-context snapshot at the new/dest indices), the
        two operands blend_values mixes. This is the candidate geometric CAUSE of
        the per-region raw_EB pattern. Returns a python list, one float per layer
        (mean over batch/heads/positions)."""
        new_idx = torch.tensor([n for n, _ in pairs], device=model.device)
        old_idx = torch.tensor([o for _, o in pairs], device=model.device)
        out = []
        for li, (_k, v) in enumerate(b_snap):
            vr = v[..., new_idx, :].float()               # read-time [B,H,n,D]
            vw = old_snap[li][1][..., old_idx, :].float() # write-time [B,H,n,D]
            cos = torch.nn.functional.cosine_similarity(vr, vw, dim=-1)  # [B,H,n]
            out.append(float(cos.mean().item()))
        return out

    # accumulators
    per_cat: dict[str, list] = {c: [] for c in CATS}
    skipped_convs: list[str] = []
    n_plants = 0
    alpha0_diffs: list[float] = []
    graft_diffs: list[float] = []       # |lp_E - lp_B|
    graft_signed: list[float] = []      # lp_E - lp_B (direction toward target)
    pre_gaps: list[float] = []          # lp_A - lp_B (meaning lost)
    all_gc: list[float] = []            # gap_closure over all plants (for CI)
    identity_diffs: list[float] = []    # CONTROL #2: |lp_identity - lp_A| per conv
    traces: list[dict] = []             # CONTROL #4: raw per-probe traces
    # CONTROL #1 placebo: per-category rows of raw_EB_placebo = lp_E_placebo - lp_B
    per_cat_placebo: dict[str, list] = {c: [] for c in CATS}
    prev_summ_snap = None               # prev conv's summary snapshot (shuffle_probe)
    prev_old_idx: list[int] = []        # prev conv's grafted old-cache indices
    # CONTROL #3 alpha sweep: {alpha: {cat: [rows...]}}
    per_cat_alpha: dict[float, dict[str, list]] = {
        a: {c: [] for c in CATS} for a in ALPHA_SWEEP_VALUES} if alpha_sweep else {}
    alpha_list = list(ALPHA_SWEEP_VALUES) if alpha_sweep else [alpha_v]
    # FEATURE #2 strong-prior signed readout: one row per strong_prior plant.
    strong_prior_rows: list[dict] = []
    # FEATURE #3 champion scan: N fractional-depth layer regions; per-config
    # raw_EB rows + separate wall-clock timers so the caller can gate on %
    # overhead. per_config_rows is keyed by config label (region_k / all /
    # regions_..); champion_configs & the per-model region layer ranges are fixed
    # once (constant layer count within a model).
    n_configs = (champion_scan if champion_scan and champion_scan >= 2
                 else (CHAMPION_SCAN_DEFAULT_N if champion_scan else 0))
    per_config_rows: dict[str, list] = {}
    champion_configs: list | None = None
    champion_region_layers: list | None = None
    value_align_sum: list | None = None   # running per-layer cosine-sim sum
    value_align_cnt = 0                    # convs contributing to value_align_sum
    uniform_seconds = 0.0     # wall time of the standard uniform-alpha scoring
    champion_seconds = 0.0    # wall time of the per-config re-blend + scoring

    # ---- corpus: per-model NATIVE in-context render (design v2) or pre-rendered
    reply_records: list = []
    if native_render:
        if not scenarios:
            doc.update(status="ERROR",
                       reason="native_render=1 but no scenarios scaffold provided")
            return doc
        # the graft needs THIS model's own write-time summary -> ignore any fixed
        # summaries file in native mode.
        fixed_summaries = None
        try:
            print(f"  [native-render] rendering {conv_limit} scaffold scenarios "
                  f"in-context (temp={native_temp}, max_reply={native_max_reply})",
                  flush=True)
            specs, reply_records = native_render_specs(
                model, tok, family, scenarios, conv_limit,
                max_reply_tokens=native_max_reply, temp=native_temp,
                seed_base=1000)
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            doc.update(status="UNSUPPORTED", reason=f"OOM during native render: {e}")
            return doc
        except Exception as e:  # noqa: BLE001
            doc.update(status="ERROR",
                       reason=f"native render failed: {type(e).__name__}: {e}\n"
                              f"{traceback.format_exc()}")
            return doc
        doc["reply_covariates"] = reply_covariates(reply_records)
    else:
        specs = collect_specs(data_dir, conv_limit)
    if not specs:
        doc.update(status="ERROR", reason="no usable plants in corpus subset")
        return doc

    # v2.1 task-competence gate accounting (per category).
    task_excluded: dict[str, list] = {c: [] for c in CATS}

    try:
        for _ci, (conv, plants) in enumerate(specs):
            print(f"  [progress] conv {_ci+1}/{len(specs)} ({conv['id']}) "
                  f"n_plants_so_far={n_plants}", flush=True)
            msgs = conv["messages"][:-1]
            tsm = conv["sections"]["middle_end_msg"]

            summary_text = fixed_summaries.get(conv["id"]) if fixed_summaries else None
            if fixed_summaries is not None and not summary_text:
                # Missing fixed summary for THIS conv -> skip the conv (logged),
                # do NOT abort the model. Other convs still contribute.
                skipped_convs.append(conv["id"])
                print(f"  SKIP {conv['id']}: no entry in fixed summaries file",
                      flush=True)
                continue
            if summary_text is None:
                # No fixed file at all: per-model fallback (marked non-comparable).
                summary_text = generate_summary_hf(
                    model, tok, msgs, request=_REQ)["text"]

            # PURE token geometry (A ids/starts, B b_ids/b_starts, summary
            # layout, alignment regions) -- the SAME construction --smoke-align
            # exercises CPU-only. force_prefill then adds this model's value
            # snapshot on top of the write-time summary layout.
            ctx = build_token_context(tok, family, msgs, summary_text, tsm)
            ids, starts = ctx["ids"], ctx["starts"]
            b_msgs, b_ids, b_starts = ctx["b_msgs"], ctx["b_ids"], ctx["b_starts"]
            regions = ctx["regions"]
            summ = ctx["summ"]
            summ["snapshot"] = force_prefill(summ["old_ids"])

            pairs = build_alignment(
                b_ids, summ["old_ids"], set(tok.all_special_ids), regions)
            if not pairs:
                # nothing to graft in this conv -> skip (not a model failure)
                continue

            a_snap = force_prefill(ids)
            b_snap = force_prefill(b_ids)

            # E (treatment) and E0 (alpha=0 plumbing check) grafts, shared by all
            # plants in this conv.
            e_snap = blend_values(b_snap, summ["snapshot"], pairs, alpha_v)
            e0_snap = blend_values(b_snap, summ["snapshot"], pairs, 0.0)

            new_list = [n for n, _ in pairs]
            old_list = [o for _, o in pairs]
            grafted_layer_count = len(summ["snapshot"])
            summary_len_tokens = len(summ["gen_ids"])

            # ---- CONTROL #2: identity-graft integrity (once per conv, cheap) ----
            # Graft A's OWN write-time values back onto the full-context A cache at
            # the aligned tail positions. Source == destination -> blend is exactly
            # (1-a)V + aV = V, so this MUST be a no-op (lp within alpha0_tol of
            # lp_A). If it is NOT, the blend/rebuild/teacher-force position plumbing
            # is corrupting rows for THIS model (e.g. a layer/position mismatch from
            # a different layer count) and every raw_EB reversal is suspect. Handled
            # in the smoke gate. Uses realistic tail indices [starts[tsm], len(ids)).
            id_pairs = [(p, p) for p in range(starts[tsm], len(ids))]
            id_snap = (blend_values(a_snap, a_snap, id_pairs, alpha_v)
                       if id_pairs else None)

            # ---- CONTROL #1: placebo graft (corrupted SOURCE, same machinery) ----
            e_placebo_snap = None
            if placebo_mode is not None:
                plan = _placebo_index_plan(
                    placebo_mode, old_list, prev_old_idx, seed + _ci)
                if plan is not None:                    # shuffle_pos / shuffle_probe
                    use_prev = (placebo_mode == "shuffle_probe" and prev_old_idx)
                    pl_source = prev_summ_snap if use_prev else summ["snapshot"]
                    pl_pairs = list(zip(new_list, plan))
                else:                                    # gauss / mean
                    pl_source = _corrupt_source_values(
                        summ["snapshot"], old_list, placebo_mode, seed + _ci)
                    pl_pairs = pairs
                e_placebo_snap = blend_values(b_snap, pl_source, pl_pairs, alpha_v)

            # ---- CONTROL #3: alpha dose-response snapshots (opt-in) ----
            e_alpha_snaps = ({a: (e_snap if a == alpha_v
                                  else blend_values(b_snap, summ["snapshot"], pairs, a))
                              for a in alpha_list} if alpha_sweep else {})

            # ---- FEATURE #3: champion-scan config snapshots (opt-in) ----
            # Re-blend alpha_V ONLY in the layers of a fractional-depth region set
            # (alpha=0 elsewhere), reusing the SAME cached b_snap + write-time
            # summary snapshot + pairs. blend_values already accepts a per-layer
            # alpha dict, so a config is just {layer: alpha_v for layer in its
            # regions}. Timed into champion_seconds; the value-alignment geometry
            # readout is accumulated here too (free -- tensors already in hand).
            config_snaps: dict = {}
            if n_configs and grafted_layer_count:
                champ_region_layers = partition_layers(grafted_layer_count, n_configs)
                if champion_configs is None:
                    champion_region_layers = [list(r) for r in champ_region_layers]
                    champion_configs = _champion_configs(
                        len(champ_region_layers), champion_regions)
                    per_config_rows = {lbl: [] for lbl, _ in champion_configs}
                _t_ch = time.perf_counter()
                for lbl, rset in champion_configs:
                    layers = [li for r in rset
                              for li in range(*champ_region_layers[r])]
                    adict = {li: alpha_v for li in layers}
                    config_snaps[lbl] = blend_values(
                        b_snap, summ["snapshot"], pairs, adict)
                champion_seconds += time.perf_counter() - _t_ch
                val = value_alignment_per_layer(b_snap, summ["snapshot"], pairs)
                if value_align_sum is None:
                    value_align_sum = list(val)
                else:
                    for _i, _x in enumerate(val):
                        value_align_sum[_i] += _x
                value_align_cnt += 1

            conv_identity_done = False

            for pl in plants:
                gold = str(pl["gold"]).strip()
                tgt = tok(gold, add_special_tokens=False).input_ids[:max_gold_tok]
                if len(tgt) < 2:
                    continue

                def suffix(mm, probe_text):
                    full = render_hf(
                        tok, mm + [{"role": "user", "content": probe_text}], True)
                    cn = canonical_ids_any(tok, mm, render_hf)
                    return full[len(cn):]

                # ---- FEATURE #1: MULTI-PROBE AVERAGING (default on) ----
                # Score the gold continuation under EACH paraphrased probe and
                # average, so the plant-level la/lb/le/le0 (and hence raw_EB) are
                # de-noised over probe wording. raw_EB is linear in lb/le, so the
                # mean of the per-probe (le-lb) equals mean(le)-mean(lb); the
                # per-probe raw_EBs are kept in the trace. probes[0] == probe.
                probes = pl.get("probes") or [pl["probe"]]
                _t_uni = time.perf_counter()
                per_probe_la, per_probe_lb = [], []
                per_probe_le, per_probe_le0 = [], []
                sa_list, sb_list = [], []
                per_probe_lep = [] if e_placebo_snap is not None else None
                per_probe_alpha = {a: [] for a in alpha_list} if alpha_sweep else {}
                for probe_text in probes:
                    sa_p = suffix(msgs, probe_text)
                    sb_p = suffix(b_msgs, probe_text)
                    sa_list.append(sa_p)
                    sb_list.append(sb_p)
                    la_p = tf(a_snap, sa_p + tgt[:-1], tgt, len(ids))
                    lb_p = tf(b_snap, sb_p + tgt[:-1], tgt, len(b_ids))
                    le_p = tf(e_snap, sb_p + tgt[:-1], tgt, len(b_ids))
                    le0_p = tf(e0_snap, sb_p + tgt[:-1], tgt, len(b_ids))
                    per_probe_la.append(la_p)
                    per_probe_lb.append(lb_p)
                    per_probe_le.append(le_p)
                    per_probe_le0.append(le0_p)
                    if per_probe_lep is not None:
                        per_probe_lep.append(
                            tf(e_placebo_snap, sb_p + tgt[:-1], tgt, len(b_ids)))
                    if alpha_sweep:
                        for a in alpha_list:
                            le_a = (le_p if a == alpha_v
                                    else tf(e_alpha_snaps[a], sb_p + tgt[:-1], tgt,
                                            len(b_ids)))
                            per_probe_alpha[a].append(le_a - lb_p)
                uniform_seconds += time.perf_counter() - _t_uni
                n_probes = len(probes)
                la = _mean(per_probe_la)
                lb = _mean(per_probe_lb)
                le = _mean(per_probe_le)
                le0 = _mean(per_probe_le0)
                per_probe_raw_eb = [e - b for e, b in
                                    zip(per_probe_le, per_probe_lb)]

                # CONTROL #2: identity-graft no-op check (once per conv, reuses
                # this plant's FIRST-probe A-side feed/targets). Run BEFORE the
                # task-competence gate so the machinery control fires even if the
                # first plant is task-excluded.
                if id_snap is not None and not conv_identity_done:
                    la_id = tf(id_snap, sa_list[0] + tgt[:-1], tgt, len(ids))
                    identity_diffs.append(abs(la_id - per_probe_la[0]))
                    conv_identity_done = True

                # ---- v2.1 GATE #3: TASK-COMPETENCE ----
                # If the model cannot do the task even WITH full context (lp_A
                # per-token below the floor), this plant is degenerate -> exclude
                # it from every aggregate (do not read it as a graft signal).
                # Default floor is permissive so the pre-rendered path is
                # unaffected. alpha0/identity machinery checks already ran above.
                if not task_competence_ok(la, task_lpa_floor):
                    task_excluded[pl["category"]].append(
                        {"plant_id": pl["id"], "conversation_id": conv["id"],
                         "lp_A": la})
                    continue

                alpha0_diffs.append(abs(le0 - lb))
                graft_diffs.append(abs(le - lb))
                graft_signed.append(le - lb)      # >0 => toward continuity target
                pre_gaps.append(la - lb)          # meaning lost to compaction

                raw_eb = le - lb          # PRIMARY: bounded logprob lift of graft
                gc = (le - lb) / (la - lb) if abs(la - lb) > 1e-6 else None
                if gc is not None:
                    all_gc.append(gc)
                n_plants += 1
                per_cat[pl["category"]].append({
                    "plant_id": pl["id"], "conversation_id": conv["id"],
                    "lp_A": la, "lp_B": lb, "lp_E": le,
                    "raw_EB": raw_eb, "pre_graft_gap": la - lb,
                    "gap_closure": gc})

                # ---- FEATURE #2: STRONG-PRIOR SIGNED READOUT (default on) ----
                # For a strong_prior codename plant, ALSO measure how much logprob
                # mass the graft moves from the famous PRIOR meaning (anti_keywords)
                # to the CONVERSATION meaning (keywords). We read RIGHT AFTER the
                # probe's answer-eliciting suffix (sb, first probe) -- the position
                # where the model would begin its answer -- scoring each meaning
                # phrase as a short teacher-forced continuation and SUMMING its
                # token logprobs (robust to multi-token phrases). Under B
                # (compacted) and E (graft); mass_shift>0 => graft disambiguated
                # toward the conversation meaning.
                sp_signed = None
                if strong_prior and pl.get("category") == "strong_prior":
                    sb0 = sb_list[0]
                    kw = pl.get("keywords") or []
                    akw = pl.get("anti_keywords") or []

                    def _meaning_lp(snap, phrases):
                        vals = []
                        for ph in phrases:
                            pid = tok(str(ph),
                                      add_special_tokens=False).input_ids
                            if pid:
                                vals.append(tf_sum(snap, sb0, pid, len(b_ids)))
                        return _mean(vals)

                    conv_lp_B = _meaning_lp(b_snap, kw)
                    conv_lp_E = _meaning_lp(e_snap, kw)
                    prior_lp_B = _meaning_lp(b_snap, akw)
                    prior_lp_E = _meaning_lp(e_snap, akw)
                    if None not in (conv_lp_B, conv_lp_E, prior_lp_B, prior_lp_E):
                        sp_signed = {
                            "plant_id": pl["id"],
                            "conversation_id": conv["id"],
                            "conv_meaning_lp_B": conv_lp_B,
                            "conv_meaning_lp_E": conv_lp_E,
                            "prior_meaning_lp_B": prior_lp_B,
                            "prior_meaning_lp_E": prior_lp_E,
                            "mass_shift": _mass_shift(
                                conv_lp_B, conv_lp_E, prior_lp_B, prior_lp_E),
                            # covariate: how dominant the famous prior is under
                            # compaction (no graft), per model.
                            "baseline_prior_strength": prior_lp_B - conv_lp_B,
                        }
                        strong_prior_rows.append(sp_signed)

                # CONTROL #1: placebo raw_EB for this plant (probe-averaged).
                placebo_raw_eb = None
                if e_placebo_snap is not None:
                    placebo_raw_eb = _mean(per_probe_lep) - lb
                    per_cat_placebo[pl["category"]].append({
                        "conversation_id": conv["id"], "raw_EB": placebo_raw_eb})

                # CONTROL #3: alpha dose-response for this plant (probe-averaged).
                alpha_raw_eb = {}
                if alpha_sweep:
                    for a in alpha_list:
                        alpha_raw_eb[a] = _mean(per_probe_alpha[a])
                        per_cat_alpha[a][pl["category"]].append({
                            "conversation_id": conv["id"],
                            "raw_EB": alpha_raw_eb[a]})

                # ---- FEATURE #3: champion-scan per-config raw_EB for this plant
                # (probe-averaged; reuses this plant's per-probe sb + lb). ----
                if config_snaps:
                    _t_ch = time.perf_counter()
                    for lbl, csnap in config_snaps.items():
                        per_probe_c = [
                            tf(csnap, sb_list[pi] + tgt[:-1], tgt, len(b_ids))
                            - per_probe_lb[pi] for pi in range(n_probes)]
                        per_config_rows[lbl].append({
                            "conversation_id": conv["id"],
                            "raw_EB": _mean(per_probe_c),
                            "category": pl["category"]})
                    champion_seconds += time.perf_counter() - _t_ch

                # CONTROL #4: save the raw per-probe trace (offline recompute).
                traces.append({
                    "plant_id": pl["id"], "conversation_id": conv["id"],
                    "category": pl["category"], "distance": pl.get("distance"),
                    "lp_A": la, "lp_B": lb, "lp_E": le, "raw_EB": raw_eb,
                    "gold": gold, "n_gold_tokens": len(tgt), "alpha": alpha_v,
                    "seed": seed, "summary_len_tokens": summary_len_tokens,
                    "grafted_layer_count": grafted_layer_count,
                    "n_pairs": len(pairs),
                    "n_probes": n_probes,
                    "per_probe_raw_EB": per_probe_raw_eb,
                    "placebo_mode": placebo_mode,
                    "placebo_raw_EB": placebo_raw_eb,
                    "alpha_sweep_raw_EB": (alpha_raw_eb or None),
                    "strong_prior_signed": sp_signed,
                })

            # retain THIS conv's summary snapshot for a shuffle_probe placebo on
            # the NEXT conv (only when placebo is active -- else free it).
            if placebo_mode == "shuffle_probe":
                prev_summ_snap = summ["snapshot"]
                prev_old_idx = old_list

            del a_snap, b_snap, e_snap, e0_snap
            del id_snap, e_placebo_snap, e_alpha_snaps, config_snaps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Unsupported as e:
        doc.update(status="UNSUPPORTED", reason=str(e))
        return doc
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED", reason=f"OOM during compute: {e}")
        return doc
    except Exception as e:  # noqa: BLE001
        doc.update(status="ERROR",
                   reason=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return doc

    doc["skipped_convs"] = skipped_convs
    doc["traces"] = traces               # CONTROL #4: always saved (additive)

    # ---- smoke gate ----
    if not alpha0_diffs:
        doc.update(status="ERROR",
                   reason="no plants scored (all convs skipped / empty alignment?)")
        return doc
    alpha0_max = max(alpha0_diffs)
    graft_max = max(graft_diffs)
    graft_mean_signed = sum(graft_signed) / len(graft_signed)
    # CONTROL #2: identity-graft integrity. If no id_pairs ever built (no tail
    # region) identity_max is None and the check is treated as PASSING (nothing to
    # validate); otherwise it must be a near-no-op within the tight alpha0 tol.
    identity_max = max(identity_diffs) if identity_diffs else None
    identity_ok = (identity_max is None) or (identity_max <= alpha0_tol)

    # ---- MACHINERY validity (gates status) vs EFFECT direction (does NOT) ----
    # (a) alpha0 graft is bit-identical to B  -> plumbing is correct.
    # (b) alpha=alpha_v graft actually CHANGES the output -> injection is live,
    #     not a dead no-op. These two are the ONLY smoke conditions that decide
    #     whether the numbers can be TRUSTED (status OK vs UNSUPPORTED).
    alpha0_ok = alpha0_max <= alpha0_tol                 # (a) TIGHT == fresh
    graft_changes = graft_max >= change_tol              # (b) injection live
    # (c) CONTROL #2: identity self-graft is a no-op -> cross-model position/layer
    #     plumbing is intact. A failure here (a "fake reversal" from mismatched
    #     layer counts) is a MACHINERY failure, exactly like (a)/(b).
    machinery_ok = alpha0_ok and graft_changes and identity_ok
    # EFFECT DIRECTION is a SCIENTIFIC RESULT, not a validity condition. A graft
    # that runs correctly but moves the output AWAY from the continuity target
    # (negative mean raw_EB) is a VALID finding to record -- "does the effect
    # travel? maybe it's negative on this architecture" is exactly the question.
    # So graft_direction_ok is INFORMATIONAL only and NEVER gates status.
    graft_direction_ok = graft_mean_signed > 0.0   # == raw_EB mean > 0 (INFO)
    doc["smoke"] = {
        "alpha0_ok": bool(alpha0_ok),
        "graft_changes_output": bool(graft_changes),
        "graft_direction_ok": bool(graft_direction_ok),  # INFO ONLY, not a gate
        "alpha0_max_abs_diff": alpha0_max,
        "graft_max_abs_diff": graft_max,
        "graft_mean_signed_diff": graft_mean_signed,
        "identity_ok": bool(identity_ok),        # CONTROL #2 (gates status)
        "identity_max_abs_diff": identity_max,
        "alpha0_tol": alpha0_tol,
        "change_tol": change_tol,
    }
    doc["n_plants"] = n_plants
    doc["pre_graft_gap"] = bootstrap_ci_95(pre_gaps)
    # PRIMARY aggregate metric: raw_EB = mean(lp_E - lp_B) over all plants, with
    # the fixed robust bootstrap. graft_signed IS the per-plant raw_EB vector.
    doc["raw_EB"] = bootstrap_ci_95(
        graft_signed, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)

    # ---- per-category robust block: ALWAYS populated whenever plants scored ----
    # (populated BEFORE the machinery gate so a negative/null effect -- or even a
    # machinery-failed model -- still carries its measured numbers). This is the
    # block that matters; it was previously left EMPTY on any early return.
    # DEPRECATED/unstable: mean of the ratio (E-B)/(A-B). Kept only for
    # continuity -- Cauchy-unstable with small denominators, NOT the headline.
    doc["gap_closure"] = bootstrap_ci_95(all_gc)  # unstable/deprecated

    # LEGACY per-category block (kept for continuity with prior runs). The
    # unconditioned mean_gc here is explicitly the deprecated unstable metric.
    by_cat = {}
    # ROBUST per-category block -- THE one that matters.
    by_cat_robust = {}
    for cat in CATS:
        rows = per_cat[cat]
        gcs = [r["gap_closure"] for r in rows if r["gap_closure"] is not None]
        gaps = [r["pre_graft_gap"] for r in rows]
        helped = [g for g in gcs if g > 0]
        ci = bootstrap_ci_95(gcs)
        by_cat[cat] = {
            "mean_gc_UNSTABLE_DEPRECATED": ci["mean"],  # do NOT use as headline
            "gc_ci95_UNSTABLE_DEPRECATED": [ci["lo"], ci["hi"]],
            "pct_helped": (100.0 * len(helped) / len(gcs)) if gcs else None,
            "mean_pre_graft_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "n": len(gcs),
        }
        raw_eb_vals = [r["raw_EB"] for r in rows]
        ratio_gap_pairs = [(r["gap_closure"], r["pre_graft_gap"]) for r in rows]
        # CONTROL #6: cluster (conversation-level) bootstrap -- group raw_EB by
        # conversation so resampling is over convs, not correlated probes.
        clusters = _group_by_conv(rows)
        block = robust_category_stats(
            raw_eb_vals, ratio_gap_pairs, raw_eb_clusters=clusters)
        # ---- v2.1 GATE #2: HEADROOM + headroom-normalized raw_EB ----
        # headroom = mean(lp_A - lp_B); raw_EB_normalized = raw_EB/max(headroom,eps);
        # floor=True (headroom<floor) marks the category UNINTERPRETABLE (no evicted
        # meaning to recover) and it is EXCLUDED from the sign verdict below.
        hd = category_headroom(block["raw_EB_mean"], gaps,
                               floor=headroom_floor)
        block.update(headroom=hd["headroom"],
                     raw_EB_normalized=hd["raw_EB_normalized"],
                     floor=hd["floor"], floor_threshold=hd["floor_threshold"])
        by_cat_robust[cat] = block
    doc["by_category"] = by_cat
    doc["by_category_robust"] = by_cat_robust

    # ---- v2.1 gate summary (headroom floors + task-competence exclusions) ----
    doc["gates"] = {
        "headroom_floor": headroom_floor,
        "task_lpa_floor": task_lpa_floor,
        "floored_categories": [c for c in CATS
                               if by_cat_robust.get(c, {}).get("floor")],
        "task_excluded": {c: task_excluded[c] for c in CATS
                          if task_excluded[c]},
        "n_task_excluded": sum(len(v) for v in task_excluded.values()),
        "note": (
            "GATE #2 headroom = per-category mean(lp_A-lp_B); a floored category "
            "(headroom<headroom_floor) has no evicted meaning to recover -> "
            "uninterpretable, EXCLUDED from the sign verdict (NOT counted as harm); "
            "raw_EB_normalized = raw_EB/max(headroom,eps). GATE #3 task-competence: "
            "plants with lp_A per-token < task_lpa_floor were dropped from all "
            "aggregates (model can't do the task even with full context)."),
    }

    # ---- CONTROL #1: placebo report (corrupted-source graft) ----
    # The REAL graft should beat every placebo: a real per-category raw_EB CI
    # above the placebo's proves the lift is STRUCTURED write-time state, not
    # injected energy / a wrong-slot write. Same robust + cluster machinery.
    if placebo_mode is not None:
        placebo_by_cat = {}
        for cat in CATS:
            prows = per_cat_placebo[cat]
            placebo_by_cat[cat] = robust_category_stats(
                [r["raw_EB"] for r in prows],
                [(None, None)] * len(prows),
                raw_eb_clusters=_group_by_conv(prows))
        all_placebo = [r["raw_EB"] for c in CATS for r in per_cat_placebo[c]]
        doc["placebo"] = {
            "mode": placebo_mode,
            "note": ("raw_EB of a corrupted-source graft (E_placebo - B). The "
                     "REAL graft's raw_EB should exceed this; a placebo that "
                     "matches the real effect means the lift is unstructured "
                     "energy, not recovered write-time state."),
            "raw_EB": bootstrap_ci_95(
                all_placebo, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED),
            "by_category_robust": placebo_by_cat,
        }

    # ---- CONTROL #3: alpha dose-response report ----
    if alpha_sweep:
        sweep = {}
        for a in alpha_list:
            per_a = {}
            for cat in CATS:
                arows = per_cat_alpha[a][cat]
                per_a[cat] = robust_category_stats(
                    [r["raw_EB"] for r in arows],
                    [(None, None)] * len(arows),
                    raw_eb_clusters=_group_by_conv(arows))
            all_a = [r["raw_EB"] for c in CATS for r in per_cat_alpha[a][c]]
            sweep[str(a)] = {
                "raw_EB": bootstrap_ci_95(
                    all_a, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED),
                "by_category_robust": per_a,
            }
        doc["alpha_sweep"] = {"alphas": list(alpha_list), "by_alpha": sweep}

    # ---- FEATURE #2: strong-prior signed disambiguation aggregate ----
    if strong_prior and strong_prior_rows:
        ms_vals = [r["mass_shift"] for r in strong_prior_rows]
        base_vals = [r["baseline_prior_strength"] for r in strong_prior_rows]
        doc["strong_prior_signed"] = {
            "n": len(strong_prior_rows),
            "mass_shift": bootstrap_ci_95(
                ms_vals, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED),
            "baseline_prior_strength": bootstrap_ci_95(
                base_vals, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED),
            "per_plant": strong_prior_rows,
            "note": (
                "SIGNED codename disambiguation for strong_prior plants. At the "
                "probe's answer position, logprob mass (summed over each meaning "
                "phrase's tokens) for the CONVERSATION meaning (keywords) vs the "
                "famous PRIOR meaning (anti_keywords), under B (compacted) and E "
                "(graft). mass_shift = (conv_E - prior_E) - (conv_B - prior_B); "
                ">0 => the graft moved mass toward the conversation meaning and "
                "away from the prior. baseline_prior_strength = prior_lp_B - "
                "conv_lp_B is the per-model prior dominance under compaction (no "
                "graft) -- a covariate to de-confound the cross-model comparison, "
                "since models differ in how strongly the famous prior dominates."),
        }

    # ---- FEATURE #3: per-layer champion scan report ----
    # DESCRIPTIVE fingerprint of WHICH fractional-depth layer regions carry the
    # graftable signal -- NOT an optimization (no "best" region is chosen as a
    # headline). NOTE: the sum of the single-region raw_EBs will NOT equal the
    # uniform raw_EB -- the graft composes NONLINEARLY across depth -- so we do
    # NOT assert that. The 'all' config (all regions grafted) SHOULD reproduce the
    # uniform raw_EB (miswiring check); we report both side by side.
    if n_configs and champion_region_layers is not None:
        n_reg = len(champion_region_layers)
        regions_out = []
        for r, (lo, hi) in enumerate(champion_region_layers):
            rows = per_config_rows.get(f"region_{r}", [])
            raw_vals = [x["raw_EB"] for x in rows]
            eb = bootstrap_ci_95(raw_vals, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
            n_layers_r = hi - lo
            # per-layer signal DENSITY: normalize by layers grafted so a THICK
            # region does not read as "hot" merely from more injected alpha-mass.
            density = (eb["mean"] / n_layers_r
                       if (eb["mean"] is not None and n_layers_r) else None)
            per_cat_region = {}
            for cat in CATS:
                crows = [x for x in rows if x["category"] == cat]
                per_cat_region[cat] = robust_category_stats(
                    [x["raw_EB"] for x in crows], [(None, None)] * len(crows),
                    raw_eb_clusters=_group_by_conv(crows))
            regions_out.append({
                "region_index": r,
                "depth_fraction": [r / n_reg, (r + 1) / n_reg],
                "region_layers": [lo, hi],       # ACTUAL per-model layer range
                "n_layers": n_layers_r,
                "raw_EB": eb,
                "raw_EB_per_layer": density,     # signal density (raw_EB/n_layers)
                "by_category_robust": per_cat_region,
            })
        all_rows = per_config_rows.get("all", [])
        champ_all = bootstrap_ci_95(
            [x["raw_EB"] for x in all_rows], n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
        overhead = (100.0 * champion_seconds / uniform_seconds
                    if uniform_seconds > 0 else None)
        champ = {
            "n_configs": n_reg,
            "region_layers": champion_region_layers,
            "regions": regions_out,
            "champion_all_regions_raw_EB": champ_all,   # MUST ~= uniform raw_EB
            "uniform_raw_EB": doc.get("raw_EB"),        # for the sanity compare
            "value_alignment_per_layer": (
                [s / value_align_cnt for s in value_align_sum]
                if value_align_sum is not None and value_align_cnt else None),
            "champion_scan_seconds": champion_seconds,
            "uniform_sweep_seconds": uniform_seconds,
            "overhead_pct": overhead,
            "note": (
                "Fractional-depth region scan: raw_EB when alpha_V is grafted "
                "ONLY within a contiguous relative-depth region (alpha=0 "
                "elsewhere), reusing the cached A/B snapshots. region_layers is "
                "the ACTUAL per-model layer range for each [k/N,(k+1)/N) depth "
                "bin; raw_EB_per_layer normalizes by layers grafted. DESCRIPTIVE "
                "(which layers carry the signal), not an optimization -- no best "
                "config is chosen. 'all' MUST reproduce the uniform raw_EB "
                "(miswiring check); the single-region raw_EBs do NOT sum to it "
                "(nonlinear composition over depth) and that is EXPECTED. "
                "value_alignment_per_layer = per-layer mean cosine similarity of "
                "grafted write-time vs co-located read-time value vectors -- the "
                "candidate geometric cause. overhead_pct = champion_scan_seconds/"
                "uniform_sweep_seconds; enable only if < 25%."),
        }
        # ARBITRARY region-set "rescue test": graft only a custom subset of
        # regions together (SC_CHAMPION_REGIONS). On a net-negative model this
        # expresses "graft only the positive/late regions and see if raw_EB flips
        # positive" without touching the rest.
        custom_lbl = next(
            (lbl for lbl, _ in (champion_configs or []) if lbl.startswith("regions_")),
            None)
        if custom_lbl:
            crows = per_config_rows.get(custom_lbl, [])
            requested = sorted(r for r in (champion_regions or [])
                               if 0 <= r < n_reg)
            champ["custom_region_set"] = {
                "regions": requested,
                "raw_EB": bootstrap_ci_95(
                    [x["raw_EB"] for x in crows],
                    n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED),
                "note": ("rescue test: alpha_V grafted ONLY in these regions, "
                         "alpha=0 elsewhere."),
            }
        doc["champion_scan"] = champ

    # ---- per-model verdict + effect sign (DIRECTIONAL) ----
    # effect_sign is read off the AGGREGATE raw_EB CI (all plants): "positive" if
    # the CI excludes 0 and mean>0, "negative" if CI excludes 0 and mean<0, else
    # "null". per-category significance (referent OR sense CI excludes 0) is also
    # tracked so a directional per-category signal is not lost. n is small so
    # per-model results are DIRECTIONAL -- the cross-model aggregate is the
    # inferential claim.
    def _ci_sign(lo, hi, n):
        """+1 / -1 if CI excludes 0 (both bounds same side), else 0 (spans 0)."""
        if lo is None or hi is None or (n is not None and n < 2):
            return 0
        if lo > 0 and hi > 0:
            return 1
        if lo < 0 and hi < 0:
            return -1
        return 0

    eb = doc["raw_EB"]
    agg_sign = _ci_sign(eb.get("lo"), eb.get("hi"), eb.get("n"))
    effect_sign = "positive" if agg_sign > 0 else "negative" if agg_sign < 0 else "null"
    doc["effect_sign"] = effect_sign

    def _cat_sign(block):
        lo, hi = block.get("raw_EB_ci", [None, None])
        return _ci_sign(lo, hi, block.get("n", 0))

    cat_signs = {c: _cat_sign(by_cat_robust.get(c, {})) for c in CATS}
    any_sig_cat = any(s != 0 for s in cat_signs.values())
    # v2.1 GATE #2: a FLOORED category (headroom<floor) has no evicted meaning to
    # recover -> its sign is uninterpretable and is EXCLUDED from the sign readout.
    floored = [c for c in CATS if by_cat_robust.get(c, {}).get("floor")]
    interpretable_cat_signs = {c: cat_signs[c] for c in CATS if c not in floored}
    doc["interpretable_cat_signs"] = interpretable_cat_signs
    doc["floored_categories"] = floored

    if effect_sign == "positive":
        doc["verdict"] = "SIGNIFICANT_POSITIVE"
    elif effect_sign == "negative":
        doc["verdict"] = "SIGNIFICANT_NEGATIVE"
    else:
        doc["verdict"] = "null/underpowered"
    doc["verdict_note"] = (
        f"aggregate raw_EB CI [{eb.get('lo')}, {eb.get('hi')}] -> effect_sign="
        f"{effect_sign}. per-category CI-excludes-0: {cat_signs} "
        f"(referent/sense any-significant={any_sig_cat}). FLOORED (headroom<"
        f"{headroom_floor}, EXCLUDED from sign): {floored}; interpretable "
        f"per-category signs: {interpretable_cat_signs}. n is small so the "
        f"per-model verdict is DIRECTIONAL; the cross-model aggregate is the "
        f"inferential claim. A SIGNIFICANT_NEGATIVE is a VALID result, not a "
        f"discard.")

    # ---- status: decided ONLY by machinery validity (never by effect sign) ----
    if not machinery_ok:
        why = []
        if not alpha0_ok:
            why.append(
                f"alpha0 graft not bit-identical to B (max|dlp|={alpha0_max:.2e} "
                f"> {alpha0_tol:.0e})")
        if not graft_changes:
            why.append(
                f"alpha={alpha_v} graft did not change output -- dead injection "
                f"(max|dlp|={graft_max:.2e} < {change_tol:.0e})")
        if not identity_ok:
            why.append(
                f"identity self-graft is NOT a no-op (max|dlp|={identity_max:.2e} "
                f"> {alpha0_tol:.0e}) -- cross-model layer/position plumbing "
                f"broken; raw_EB reversals are NOT trustworthy")
        doc.update(status="UNSUPPORTED",
                   reason="machinery check failed: " + "; ".join(why))
        return doc

    doc["status"] = "OK"
    doc["reason"] = None
    return doc


# ---------------------------------------------------------------------------
# Dry-run self-test (NO torch): case construction + geometry detection + a
# model-type-agnosticism assertion over the generic snapshot/graft source.
# ---------------------------------------------------------------------------
class _FakeCfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _dry_run(data_dir: Path, conv_limit: int) -> int:
    print("== cross_arch_probe --dry-run (no torch / no model) ==")
    print(f"data_dir={data_dir}  SC_CONV_LIMIT={conv_limit}  categories={CATS}")

    specs = collect_specs(data_dir, conv_limit)
    total = 0
    for conv, plants in specs:
        print(f"  {conv['id']}: {len(plants)} usable plants "
              f"({', '.join(p['id'] for p in plants)})")
        total += len(plants)
    print(f"TOTAL usable sense+referent plants in first {conv_limit} convs: {total}")
    assert total > 0, "dry-run found no plants -- corpus/selection broken"

    # config-based geometry detection across several architecture families,
    # incl. a nested text_config (multimodal wrapper) and a GLM-style config.
    print("\n== detect_kv_geometry over synthetic configs (model-type-agnostic) ==")
    fakes = {
        "qwen3_moe": _FakeCfg(model_type="qwen3_moe", num_attention_heads=32,
                              num_key_value_heads=4, hidden_size=2048, head_dim=128),
        "gemma2": _FakeCfg(model_type="gemma2", num_attention_heads=32,
                           num_key_value_heads=16, hidden_size=4608, head_dim=128),
        "glm4 (no explicit kv/head_dim)": _FakeCfg(
            model_type="glm4", num_attention_heads=96, hidden_size=12288),
        "llama full-MHA": _FakeCfg(model_type="llama", num_attention_heads=64,
                                   hidden_size=8192),
        "multimodal-wrapped": _FakeCfg(
            model_type="mm_wrapper",
            text_config=_FakeCfg(model_type="mistral", num_attention_heads=40,
                                 num_key_value_heads=8, hidden_size=5120)),
    }
    for name, cfg in fakes.items():
        g = detect_kv_geometry(cfg)
        print(f"  {name:34s} -> {g}")
        assert g["num_key_value_heads"], f"kv heads not resolved for {name}"
        assert g["head_dim"], f"head_dim not resolved for {name}"

    # assert the generic snapshot/graft SOURCE is model-type-agnostic: no
    # architecture name appears in EXECUTABLE code (identifiers / branches).
    # We tokenize and skip STRING (incl. docstrings) + COMMENT tokens, so
    # docstring mentions like "Llama/Qwen rotate_half" are correctly ignored --
    # only real code tokens that would affect behavior are flagged.
    print("\n== model-type-agnosticism check on generic code paths ==")
    import tokenize
    banned = {"qwen", "gemma", "llama", "mistral", "glm", "deepseek", "kimi"}
    # STRING covers docstrings; also skip any f-string sub-tokens where present
    # (Python 3.12+ splits them out) so string CONTENT never trips the check.
    skip_types = {tokenize.STRING, tokenize.COMMENT}
    for attr in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, attr):
            skip_types.add(getattr(tokenize, attr))
    src = Path(__file__).resolve().parent
    for fn in ("kvlib_hf.py", "kv_graft.py"):
        hits = []
        with open(src / fn, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type in skip_types:
                    continue
                if tok.type == tokenize.NAME and tok.string.lower() in banned:
                    hits.append((tok.start[0], tok.string))
        status = "OK (no arch names in executable code)" if not hits else "HITS"
        print(f"  {fn}: {status}")
        for ln, name in hits:
            print(f"     L{ln}: {name}")
        assert not hits, f"{fn} has architecture names in executable code: {hits}"

    print("\n== family-aware B-context indices (template dispatch) ==")
    for fam in ("qwen", "gemma", "mistral", "unknown"):
        # build against a tiny synthetic msgs list; only indices matter here.
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"}]
        b, si, ti = build_b_and_indices(fam, msgs, "SUMMARY", 3)
        print(f"  family={fam:8s} b_len={len(b)} summary_idx={si} tail_idx={ti} "
              f"roles={[m['role'] for m in b]}")

    print("\nDRY-RUN OK")
    return 0


def _self_test_controls() -> int:
    """CPU-only unit tests for the new controls (NO torch, NO model, NO pod)."""
    print("== cross_arch_probe --self-test (controls; no torch/model) ==")

    # ---- CONTROL #6: cluster bootstrap is WIDER than probe bootstrap on
    #      CORRELATED data (probes clustered by conversation). 3 convs x 5 probes;
    #      strong between-conv spread, tiny within-conv noise.
    conv_a = [0.9, 1.0, 1.1, 1.0, 1.0]
    conv_b = [-0.1, 0.0, 0.1, 0.0, 0.0]
    conv_c = [-1.1, -1.0, -0.9, -1.0, -1.0]
    clusters = [conv_a, conv_b, conv_c]
    flat = conv_a + conv_b + conv_c
    probe = bootstrap_ci_95(flat, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    clust = bootstrap_ci_95_cluster(clusters, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    probe_w = probe["hi"] - probe["lo"]
    clust_w = clust["hi"] - clust["lo"]
    print(f"  probe-level  CI=[{probe['lo']:+.3f},{probe['hi']:+.3f}] "
          f"width={probe_w:.3f}  n={probe['n']}")
    print(f"  cluster (conv) CI=[{clust['lo']:+.3f},{clust['hi']:+.3f}] "
          f"width={clust_w:.3f}  n_clusters={clust['n_clusters']}")
    assert abs(probe["mean"] - clust["mean"]) < 1e-9, \
        "point estimate must be identical (only interval width changes)"
    assert clust_w > probe_w, \
        f"cluster CI ({clust_w:.3f}) must be WIDER than probe CI ({probe_w:.3f})"
    print(f"  OK: cluster CI is {clust_w / probe_w:.1f}x wider than probe CI")

    # ---- CONTROL #1: placebo source-corruption index plans (fake tensors) ----
    # Fake per-position value "tensor": row at position o is the marker [o,o,o].
    fake_src = {o: [o, o, o] for o in range(200)}
    old_idx = [10, 11, 12, 13, 14]

    def sel(idxs):  # mimic blend_values reading old_snap value rows at old_idx
        return [fake_src[o] for o in idxs]

    plan_pos = _placebo_index_plan("shuffle_pos", old_idx, [], seed=ROBUST_SEED)
    assert sorted(plan_pos) == sorted(old_idx), "shuffle_pos must be a permutation"
    assert plan_pos != old_idx, "shuffle_pos must actually scramble (this seed)"
    real_rows = sel(old_idx)
    plac_rows = sel(plan_pos)
    assert sorted(map(tuple, real_rows)) == sorted(map(tuple, plac_rows)), \
        "shuffle_pos keeps the SAME source values (right values, wrong slots)"
    assert real_rows != plac_rows, "shuffle_pos must land them in different slots"
    n_moved = sum(1 for a, b in zip(real_rows, plac_rows) if a != b)
    print(f"  shuffle_pos: {n_moved}/{len(old_idx)} positions got a wrong-slot "
          f"source value (right values, scrambled)")

    prev_old = [100, 101, 102]
    plan_probe = _placebo_index_plan("shuffle_probe", old_idx, prev_old,
                                     seed=ROBUST_SEED)
    assert len(plan_probe) == len(old_idx)
    assert all(o in prev_old for o in plan_probe), \
        "shuffle_probe must draw from ANOTHER probe's positions"
    print(f"  shuffle_probe: drew {plan_probe} from prev-conv positions {prev_old}")

    plan_fallback = _placebo_index_plan("shuffle_probe", old_idx, [],
                                        seed=ROBUST_SEED)
    assert sorted(plan_fallback) == sorted(old_idx), \
        "shuffle_probe with no prev conv must fall back to shuffle_pos"
    print("  shuffle_probe (no prev) -> shuffle_pos fallback OK")

    assert _placebo_index_plan("gauss", old_idx, [], 0) is None
    assert _placebo_index_plan("mean", old_idx, [], 0) is None
    print("  gauss/mean plans -> None (tensor-corruption path, pairs unchanged)")
    try:
        _placebo_index_plan("bogus", old_idx, [], 0)
        raise AssertionError("bad mode must raise")
    except ValueError:
        print("  bad placebo mode raises ValueError OK")

    # ---- CONTROL #5: model_hparams populates from config (incl. text_config) ----
    cfg = _FakeCfg(model_type="qwen3_moe", num_hidden_layers=48,
                   num_attention_heads=32, num_key_value_heads=4,
                   hidden_size=2048, head_dim=128, rope_theta=1_000_000.0,
                   vocab_size=151936, use_qk_norm=True)
    hp = detect_model_hparams(cfg)
    print(f"  hparams(qwen3_moe): {hp}")
    assert hp["gqa_ratio"] == 8.0, "gqa_ratio = n_heads/n_kv"
    assert hp["qk_norm"] is True, "qk_norm detected from use_qk_norm"
    assert all(hp[k] is not None for k in hp), "every hparam field populated"

    mm = _FakeCfg(model_type="mm_wrapper", vocab_size=200000,
                  text_config=_FakeCfg(model_type="mistral",
                                       num_hidden_layers=40,
                                       num_attention_heads=40,
                                       num_key_value_heads=8, hidden_size=5120,
                                       rope_theta=1e6, vocab_size=131072))
    hp2 = detect_model_hparams(mm)
    assert hp2["num_hidden_layers"] == 40 and hp2["gqa_ratio"] == 5.0, \
        "hparams must unwrap nested text_config"
    assert hp2["qk_norm"] is False, "no qk-norm keys -> False"
    print(f"  hparams(mm_wrapper->mistral): layers={hp2['num_hidden_layers']} "
          f"gqa={hp2['gqa_ratio']} qk_norm={hp2['qk_norm']} OK")

    # ---- CONTROL #4: trace dict shape (mock a scored probe) ----
    fake_plant = {"id": "c01-referent-1", "category": "referent",
                  "distance": None}
    trace = {
        "plant_id": fake_plant["id"], "conversation_id": "c01",
        "category": fake_plant["category"], "distance": fake_plant.get("distance"),
        "lp_A": -1.0, "lp_B": -2.0, "lp_E": -1.5, "raw_EB": 0.5,
        "gold": "some gold text", "n_gold_tokens": 12, "alpha": 0.75,
        "seed": ROBUST_SEED, "summary_len_tokens": 87, "grafted_layer_count": 48,
        "n_pairs": 30, "n_probes": 3, "per_probe_raw_EB": [0.4, 0.5, 0.6],
        "placebo_mode": None, "placebo_raw_EB": None,
        "alpha_sweep_raw_EB": None, "strong_prior_signed": None,
    }
    expected_keys = {
        "plant_id", "conversation_id", "category", "distance", "lp_A", "lp_B",
        "lp_E", "raw_EB", "gold", "n_gold_tokens", "alpha", "seed",
        "summary_len_tokens", "grafted_layer_count", "n_pairs", "n_probes",
        "per_probe_raw_EB", "placebo_mode", "placebo_raw_EB",
        "alpha_sweep_raw_EB", "strong_prior_signed"}
    assert set(trace) == expected_keys, \
        f"trace keys mismatch: {set(trace) ^ expected_keys}"
    print(f"  trace dict has all {len(expected_keys)} required fields OK")

    # ---- FEATURE #1: multi-probe averaging (mean identity) ----
    assert _mean([]) is None and _mean([None]) is None
    assert abs(_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-12
    assert abs(_mean([0.5, None, 1.5]) - 1.0) < 1e-12  # None skipped
    per_probe_le = [-1.0, -2.0, -1.5]
    per_probe_lb = [-2.0, -2.5, -2.0]
    per_probe_raw = [e - b for e, b in zip(per_probe_le, per_probe_lb)]
    plant_raw_via_means = _mean(per_probe_le) - _mean(per_probe_lb)
    assert abs(plant_raw_via_means - _mean(per_probe_raw)) < 1e-12, \
        "plant raw_EB (mean le - mean lb) must equal mean of per-probe raw_EB"
    print(f"  FEATURE #1 multi-probe: mean(per-probe raw_EB)="
          f"{_mean(per_probe_raw):+.3f} == mean(le)-mean(lb) OK")

    # ---- FEATURE #2: mass_shift formula on fake logprobs ----
    # conv_B,conv_E,prior_B,prior_E: graft lifts conv (+1) and drops prior (-1.5)
    ms = _mass_shift(-3.0, -2.0, -1.0, -2.5)
    # (conv_E - prior_E) - (conv_B - prior_B) = (-2.0 - -2.5) - (-3.0 - -1.0)
    #                                         = 0.5 - (-2.0) = 2.5
    assert abs(ms - 2.5) < 1e-12, f"mass_shift formula wrong: {ms}"
    # sign sanity: a graft that does NOTHING gives mass_shift 0
    assert abs(_mass_shift(-3.0, -3.0, -1.0, -1.0)) < 1e-12
    # a graft moving mass toward the PRIOR (conv down, prior up) is negative
    assert _mass_shift(-2.0, -3.0, -2.0, -1.0) < 0
    print(f"  FEATURE #2 mass_shift: (conv_E-prior_E)-(conv_B-prior_B)={ms:+.3f} "
          f"OK (0 when graft is a no-op; <0 toward prior)")

    # ---- FEATURE #3: fractional-depth layer partitioning ----
    for L, N in [(48, 6), (64, 6), (50, 6), (40, 6), (7, 6), (5, 6), (1, 1),
                 (13, 4), (30, 30), (12, 5)]:
        regs = partition_layers(L, N)
        assert regs[0][0] == 0, f"partition must start at 0 (L={L},N={N})"
        assert regs[-1][1] == L, f"partition must cover to L (L={L},N={N})"
        for i in range(1, len(regs)):
            assert regs[i][0] == regs[i - 1][1], \
                f"regions must be contiguous (L={L},N={N})"
        assert all(lo < hi for lo, hi in regs), "no empty regions"
        assert sum(hi - lo for lo, hi in regs) == L, \
            f"regions must cover all L layers exactly once (L={L},N={N})"
        assert len(regs) == min(N, L), \
            f"expected min(N,L) regions (L={L},N={N} -> {len(regs)})"
        # fractional-depth: each region's lo/hi are the floor of k*L/N boundaries
        # so region k maps to the SAME relative depth across differing L.
        assert regs == [((k * L) // N, ((k + 1) * L) // N)
                        for k in range(N) if ((k + 1) * L) // N > (k * L) // N]
    # region k spans the SAME relative depth for different layer counts:
    r48 = partition_layers(48, 6)
    r64 = partition_layers(64, 6)
    for k in range(6):
        f48 = (r48[k][0] / 48, r48[k][1] / 48)
        f64 = (r64[k][0] / 64, r64[k][1] / 64)
        assert abs(f48[0] - k / 6) < 1.0 / 48 and abs(f64[0] - k / 6) < 1.0 / 64
    print(f"  FEATURE #3 partition: 48L->{r48}  64L->{r64} "
          f"(same fractional depth, contiguous, full cover) OK")

    # ---- FEATURE #3: arbitrary region-set masking configs ----
    cfgs = _champion_configs(6, custom_regions=[4, 5])
    labels = [lbl for lbl, _ in cfgs]
    assert labels[:6] == [f"region_{r}" for r in range(6)], \
        "must have one single-region config per region"
    assert ("all", frozenset(range(6))) in cfgs, "must include the all-regions config"
    custom = dict(cfgs)["regions_4_5"]
    assert custom == frozenset({4, 5}), "custom set must be exactly {4,5}"
    # 'all' union covers every region (reproduces uniform); a single region is a
    # strict subset (so its raw_EB need NOT sum to uniform -- nonlinear).
    assert dict(cfgs)["all"] == frozenset(range(6))
    # out-of-range custom indices are dropped
    cfgs2 = _champion_configs(3, custom_regions=[1, 9, -1])
    assert dict(cfgs2)["regions_1"] == frozenset({1}), \
        "custom set must drop out-of-range region indices"
    # no custom set requested -> only single regions + all
    cfgs3 = _champion_configs(4)
    assert [lbl for lbl, _ in cfgs3] == ["region_0", "region_1", "region_2",
                                         "region_3", "all"]
    print(f"  FEATURE #3 region-set configs: {labels} (single + all + custom) OK")

    # ---- robust_category_stats gains cluster CI without dropping old keys ----
    rows = [{"conversation_id": "c01", "raw_EB": 0.4},
            {"conversation_id": "c01", "raw_EB": 0.5},
            {"conversation_id": "c02", "raw_EB": -0.2}]
    st = robust_category_stats([r["raw_EB"] for r in rows],
                               [(None, None)] * len(rows),
                               raw_eb_clusters=_group_by_conv(rows))
    for k in ("n", "raw_EB_mean", "raw_EB_ci", "pct_helped", "pct_helped_ci",
              "median_ratio_cond", "n_cond", "raw_EB_ci_cluster",
              "n_conversations"):
        assert k in st, f"robust_category_stats missing {k}"
    assert st["n_conversations"] == 2, "two distinct convs"
    print(f"  robust_category_stats: n_conversations={st['n_conversations']} "
          f"cluster_ci={st['raw_EB_ci_cluster']} OK")

    _self_test_native()
    _self_test_batched_decode()
    _self_test_qk_ablation()

    print("\nSELF-TEST OK")
    return 0


def _self_test_qk_ablation() -> int:
    """CPU test for WITHIN-MODEL QK-NORM ABLATION (SC_ABLATE_QK_NORM, H1).

    On a TINY model that HAS q_norm/k_norm (Qwen3): (a) ablation replaces every
    QK-norm module with torch.nn.Identity (verified by module type), (b) the
    forward pass runs and produces DIFFERENT logits than the non-ablated model
    (proving the ablation actually changes computation), and (c) the
    zero-modules-found case (a model with NO QK-norm) returns count 0 so the
    caller can FAIL LOUD. Skips cleanly if torch/Qwen3 are unavailable."""
    print("\n== QK-NORM ABLATION self-test (tiny Qwen3, CPU) ==")
    try:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            LlamaConfig,
            LlamaForCausalLM,
            Qwen3Config,
            Qwen3ForCausalLM,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] no torch/Qwen3 available: {type(e).__name__}: {e}")
        return 0

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128)
    model = Qwen3ForCausalLM(cfg).eval()
    # Give the QK-norm RMSNorm weights a non-identity scale so removing them
    # actually changes the computation (fresh weights init to 1.0 -> RMSNorm
    # still normalizes, so it is NOT a no-op even at unit weight).
    with torch.no_grad():
        for n, m in model.named_modules():
            if n.rsplit(".", 1)[-1] in ("q_norm", "k_norm"):
                m.weight.add_(torch.randn_like(m.weight) * 0.5)

    ids = torch.tensor([[3, 5, 7, 9, 11]])
    with torch.no_grad():
        logits_before = model(ids).logits.clone()

    # detect_model_hparams must see QK-norm on the loaded (un-ablated) model.
    hp_before = detect_model_hparams(cfg, model)
    assert hp_before["qk_norm"] is True, \
        "tiny Qwen3 must be detected as having QK-norm before ablation"
    assert str(hp_before["qk_norm_source"]).startswith("module:"), \
        f"qk_norm_source should be module:* , got {hp_before['qk_norm_source']}"

    # (a) ablation replaces the modules with Identity.
    n_ablated, names = ablate_qk_norm(model)
    assert n_ablated == 4, f"expected 4 QK-norm modules ablated, got {n_ablated}"
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        assert isinstance(getattr(parent, leaf), torch.nn.Identity), \
            f"{name} was not replaced with Identity"
    # After ablation detect_model_hparams no longer sees QK-norm (Identity leaves
    # named q_norm/k_norm are still present, so hparams source stays module:*, but
    # the CALLER sets qk_norm=False + qk_norm_ablated=True; we assert the module
    # objects are now Identity, which is the load-bearing fact).
    print(f"  (a) ablated {n_ablated} modules -> all torch.nn.Identity OK")

    # (b) forward pass runs and logits DIFFER (ablation changed computation).
    with torch.no_grad():
        logits_after = model(ids).logits
    max_abs_diff = (logits_after - logits_before).abs().max().item()
    assert max_abs_diff > 1e-4, \
        f"ablation did not change logits (max|Δ|={max_abs_diff:.2e}) -- QK-norm " \
        "was not actually removed from the forward pass"
    print(f"  (b) forward pass runs; logits DIFFER (max|Δ|={max_abs_diff:.3e}) OK")

    # (b2) idempotent: re-ablating finds nothing left to replace.
    n_again, _ = ablate_qk_norm(model)
    assert n_again == 0, f"re-ablation must find 0 (idempotent), got {n_again}"
    print("  (b2) re-ablation finds 0 (idempotent) OK")

    # (c) zero-modules-found case: a model with NO QK-norm returns count 0 so the
    # caller FAILs LOUD. Tiny Llama has no q_norm/k_norm modules.
    lcfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=128,
                       rope_theta=10000.0)
    llama = LlamaForCausalLM(lcfg).eval()
    assert detect_model_hparams(lcfg, llama)["qk_norm"] is False, \
        "tiny Llama must have no QK-norm"
    n_none, names_none = ablate_qk_norm(llama)
    assert n_none == 0 and names_none == [], \
        f"no-QK-norm model must ablate 0 modules, got {n_none}"
    print("  (c) no-QK-norm model -> ablate count 0 (caller fails loud) OK")

    print("  QK-NORM ABLATION OK")
    return 0


def _self_test_batched_decode() -> int:
    """CPU byte-identity check for the THROUGHPUT optimizations (design v2, 07-08).

    (1) BATCHED DECODE == PER-TOKEN DECODE: the left-padded batched decoder
        (_native_batched_decode) must reproduce the per-token greedy decoder
        (_native_per_token_decode) TOKEN-FOR-TOKEN and lp_sum-for-lp_sum, over a
        batch of DIFFERENT-length conversations, both without EOS (length-capped)
        and WITH an EOS that fires mid-decode for one row (per-sequence stopping).
    (2) SNAPSHOT-REPLAY == FRESH-RENDER: rebuild_cache(snapshot_cache(prefill))
        must give teacher-forced logprobs identical to a fresh prefill -- the
        fidelity that lets run_model render the shared prefix ONCE and replay every
        arm/control graft from the snapshot (render-once/replay-all-arms).

    Uses a tiny RANDOM Llama built in memory (RoPE + GQA + DynamicCache, no
    download). Skips cleanly if torch/transformers are unavailable."""
    print("\n== BATCHED-DECODE + SNAPSHOT-REPLAY self-test (tiny torch model) ==")
    try:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            DynamicCache,
            LlamaConfig,
            LlamaForCausalLM,
        )

        from kvlib_hf import (  # noqa: PLC0415
            prefill,
            rebuild_cache,
            snapshot_cache,
            tf_logprobs,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] no torch/transformers available: {type(e).__name__}: {e}")
        return 0

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512,
                      rope_theta=10000.0)
    model = LlamaForCausalLM(cfg).eval()
    dev = model.device
    maxt = 12
    # DIFFERENT-length conversations -> exercises the left-padding + per-row pos.
    seqs = [[3, 5, 7, 9, 11], [13, 15, 17], [19, 21, 23, 25, 27, 29, 31, 33]]

    def fresh_prefill(s):
        c, lg = prefill(model, torch.tensor([s], device=dev), past=DynamicCache(),
                        position_ids=torch.arange(len(s), device=dev)[None])
        return c, lg

    def run_both(eos_ids):
        cs, fs, ns = [], [], []
        for s in seqs:
            c, lg = fresh_prefill(s)
            cs.append(c); fs.append(lg); ns.append(len(s))
        # batched does NOT mutate cs (it snapshots/clones) -> reuse cs per-token.
        bat = _native_batched_decode(
            model, cs, fs, ns, max_reply_tokens=maxt, temp=0.0,
            eos_ids=eos_ids, seeds=[0] * len(seqs), dev=dev, pad_id=0)
        ref = [_native_per_token_decode(
            model, c, lg, n, max_reply_tokens=maxt, temp=0.0,
            eos_ids=eos_ids, seed=0, dev=dev)
            for c, lg, n in zip(cs, fs, ns)]
        return bat, ref

    # (1a) no EOS: length-capped, all rows run to maxt.
    bat, ref = run_both(set())
    assert [b[0] for b in bat] == [r[0] for r in ref], \
        f"batched tokens != per-token tokens:\n  bat={[b[0] for b in bat]}\n  ref={[r[0] for r in ref]}"
    assert all(abs(b[1] - r[1]) < 1e-6 for b, r in zip(bat, ref)), \
        "batched lp_sum != per-token lp_sum"
    assert all(len(b[0]) == maxt for b in bat), "no-EOS run must be length-capped"
    print(f"  (1a) no-EOS greedy: {len(seqs)} convs, lens={[len(b[0]) for b in bat]}"
          f" -> batched == per-token (ids + lp_sum) OK")

    # (1c) CHUNKED decode (SC_NATIVE_BATCH cap): decoding the convs in sub-batches
    # of 2 must not change ANY conv's output vs decoding all at once / per-token.
    def run_chunked(eos_ids, chunk):
        cs, fs, ns = [], [], []
        for s in seqs:
            c, lg = fresh_prefill(s)
            cs.append(c); fs.append(lg); ns.append(len(s))
        out = []
        for i in range(0, len(seqs), chunk):       # separate batched call / chunk
            out += _native_batched_decode(
                model, cs[i:i + chunk], fs[i:i + chunk], ns[i:i + chunk],
                max_reply_tokens=maxt, temp=0.0, eos_ids=eos_ids,
                seeds=[0] * len(cs[i:i + chunk]), dev=dev, pad_id=0)
        ref = [_native_per_token_decode(
            model, c, lg, n, max_reply_tokens=maxt, temp=0.0,
            eos_ids=eos_ids, seed=0, dev=dev)
            for c, lg, n in zip(cs, fs, ns)]
        return out, ref

    chk, refc = run_chunked(set(), 2)
    assert [c[0] for c in chk] == [r[0] for r in refc], \
        "chunk-size-2 tokens differ from per-token (chunk boundary changed output!)"
    assert all(abs(c[1] - r[1]) < 1e-6 for c, r in zip(chk, refc)), \
        "chunk-size-2 lp_sum differs from per-token"
    assert chk[0][0] == [b[0] for b in bat][0], \
        "chunk-size-2 conv 0 differs from all-at-once conv 0"
    print(f"  (1c) chunk-size-2 (3 convs -> [2]+[1]): batched == per-token "
          f"(ids + lp_sum), boundary-independent OK")

    # (1b) EOS fires mid-decode for conv 0 -> per-sequence stopping.
    eos_tok = ref[0][0][3]                 # 4th greedy token of conv 0
    bat2, ref2 = run_both({eos_tok})
    assert [b[0] for b in bat2] == [r[0] for r in ref2], \
        "batched != per-token under mid-decode EOS"
    assert len(bat2[0][0]) == 3, \
        f"conv 0 must stop BEFORE the EOS token (3 kept), got {len(bat2[0][0])}"
    assert eos_tok not in bat2[0][0], "EOS token must be excluded from the reply"
    print(f"  (1b) mid-decode EOS (tok={eos_tok}): conv0 stopped at 3 toks, "
          f"batched == per-token OK")

    # (2) snapshot -> rebuild teacher-forced logprobs == fresh prefill.
    s = seqs[0]
    src, _ = fresh_prefill(s)
    replay = rebuild_cache(snapshot_cache(src), DynamicCache)
    fresh, _ = fresh_prefill(s)
    feed_prefix, targets = [2, 4, 6], [8, 10, 12]
    feed = feed_prefix + targets[:-1]
    pos = torch.arange(len(s), len(s) + len(feed), device=dev)[None]
    lp_fresh = tf_logprobs(model, fresh, feed, targets, position_ids=pos)
    lp_replay = tf_logprobs(model, replay, feed, targets, position_ids=pos)
    md = max(abs(a - b) for a, b in zip(lp_fresh, lp_replay))
    assert md < 1e-6, f"snapshot-replay tf logprobs differ from fresh by {md}"
    print(f"  (2) snapshot->rebuild tf logprobs == fresh prefill "
          f"(max|Δ|={md:.2e}) OK")

    print("  BATCHED-DECODE + SNAPSHOT-REPLAY OK")
    return 0


def _self_test_native(scenarios_path: Path | None = None) -> int:
    """CPU-only unit tests for the NATIVE-RENDER pure parts (design v2/v2.1):
    scenario->turn-ordering, reply trimming, reply covariates, headroom
    normalization + gate, task-competence gate. NO torch / model / pod."""
    print("\n== NATIVE-RENDER pure-parts self-test (no torch/model) ==")

    # ---- scenario_turn_plan: ordering + section-index math (real scaffold) ----
    sp = scenarios_path or (Path(__file__).resolve().parent.parent
                            / "data" / "scenarios.json")
    scenarios = json.loads(Path(sp).read_text())
    sc = scenarios[0]
    plan = scenario_turn_plan(sc, seed=1000)
    plan2 = scenario_turn_plan(sc, seed=1000)
    assert plan == plan2, "turn plan must be deterministic for a fixed seed"
    kinds = [t["kind"] for t in plan["turns"]]
    n_early = len(sc["early_user_turns"])
    assert kinds[:n_early] == ["early"] * n_early, "early turns come first"
    # every plant's middle_user appears exactly once as a plant turn
    plant_turn_ids = sorted(t["plant_id"] for t in plan["turns"]
                            if t["kind"] == "plant")
    scaffold_ids = sorted(p["id"] for p in sc["plants"])
    assert plant_turn_ids == scaffold_ids, "all plants planted exactly once"
    plant_texts = {t["text"] for t in plan["turns"] if t["kind"] == "plant"}
    assert plant_texts == {p["middle_user"] for p in sc["plants"]}
    # every non-empty tail_user_fragment appears as a tail turn
    frags = [p["tail_user_fragment"] for p in sc["plants"]
             if p.get("tail_user_fragment")]
    tail_texts = {t["text"] for t in plan["turns"] if t["kind"] == "tail"}
    for f in frags:
        assert f in tail_texts, "tail fragment missing from tail turns"
    # section-index math: system(0) + 2 msgs/turn; middle_end = first tail msg
    n_turns = len(plan["turns"])
    assert plan["n_messages"] == 1 + 2 * n_turns
    assert plan["early_end_msg"] == 1 + 2 * plan["n_early"]
    assert plan["middle_end_msg"] == 1 + 2 * (plan["n_early"] + plan["n_middle"])
    assert plan["n_early"] + plan["n_middle"] + plan["n_tail"] == n_turns
    # a different seed reorders the middle plants (not the early turns)
    plan_b = scenario_turn_plan(sc, seed=1001)
    mid_a = [t.get("plant_id") for t in plan["turns"] if t["kind"] == "plant"]
    mid_b = [t.get("plant_id") for t in plan_b["turns"] if t["kind"] == "plant"]
    assert sorted(mid_a) == sorted(mid_b), "same plant set regardless of seed"
    print(f"  scenario_turn_plan({sc['id']}): {n_turns} turns "
          f"(early={plan['n_early']} middle={plan['n_middle']} "
          f"tail={plan['n_tail']}), middle_end_msg={plan['middle_end_msg']}, "
          f"n_messages={plan['n_messages']} OK (deterministic, all plants placed)")

    # ---- trim_capped_reply ----
    long = "First sentence. Second sentence. " + "x" * 50  # boundary past 1/3
    trimmed = trim_capped_reply(long)
    assert trimmed == "First sentence. Second sentence.", trimmed
    # no boundary past the first third -> unchanged
    nb = "x" * 40 + ". y"
    assert trim_capped_reply("no boundary here at all yet") == \
        "no boundary here at all yet"
    para = "Para one body text here.\n\nPara two starts and is long enough xx"
    assert trim_capped_reply(para) == "Para one body text here."
    print("  trim_capped_reply: sentence/paragraph trim + no-op fallback OK")

    # ---- reply_covariates on fake numbers ----
    cov = reply_covariates([
        {"n_tokens": 100, "logprob_sum": -50.0},
        {"n_tokens": 200, "logprob_sum": -60.0},
        {"n_tokens": 0, "logprob_sum": -1.0},        # dropped (no tokens)
        {"n_tokens": 150, "logprob_sum": None},       # length only, no logprob
    ])
    assert cov["n_replies"] == 3, cov
    assert cov["total_reply_tokens"] == 450
    assert abs(cov["mean_reply_len_tokens"] - 150.0) < 1e-9
    # info-content = total_lp / total_tok over the 2 replies WITH logprob
    assert abs(cov["mean_reply_logprob_per_token"] - (-110.0 / 300.0)) < 1e-9
    assert cov["n_replies_with_logprob"] == 2
    assert reply_covariates([])["n_replies"] == 0
    print(f"  reply_covariates: mean_len={cov['mean_reply_len_tokens']:.0f} "
          f"info={cov['mean_reply_logprob_per_token']:+.4f} nats/tok OK")

    # ---- category_headroom + normalization + floor gate ----
    # healthy: A-B gap ~0.9, raw_EB 0.18 -> normalized 0.2, NOT floored
    h1 = category_headroom(0.18, [1.0, 0.8, 0.9])
    assert abs(h1["headroom"] - 0.9) < 1e-9
    assert abs(h1["raw_EB_normalized"] - 0.2) < 1e-9
    assert h1["floor"] is False
    # floored: tiny A-B gap 0.1 < 0.3 -> flagged, normalization uses real gap
    h2 = category_headroom(0.05, [0.1, 0.1, 0.1])
    assert h2["floor"] is True, "headroom below threshold must floor the category"
    # eps guards a near-zero denominator from exploding the normalized value
    h3 = category_headroom(0.05, [0.0, 0.0])
    assert h3["raw_EB_normalized"] == 0.05 / HEADROOM_EPS
    assert category_headroom(None, [1.0])["raw_EB_normalized"] is None
    assert category_headroom(0.2, [])["headroom"] is None
    print(f"  category_headroom: healthy norm={h1['raw_EB_normalized']:.2f} "
          f"floor={h1['floor']}; low-gap floor={h2['floor']} OK")

    # ---- task_competence_ok gate ----
    assert task_competence_ok(-1.5, floor=-8.0) is True
    assert task_competence_ok(-9.0, floor=-8.0) is False
    assert task_competence_ok(None) is False
    assert task_competence_ok(-8.0, floor=-8.0) is True  # boundary inclusive
    print("  task_competence_ok: lp_A floor gate (inclusive) OK")

    print("  NATIVE-RENDER pure-parts OK")
    return 0


# ---------------------------------------------------------------------------
# --smoke-align : CPU-only, TOKENIZER-only alignment preflight. Catches the
# write-time-vs-compacted summary token divergence (the <think>-block bug) in
# SECONDS, with no model / no pod / no GPU. Run BEFORE every pod launch.
# ---------------------------------------------------------------------------
# A LONG (~600-900 token) representative summary WITH a leading <think> reasoning
# block -- because self-gen 30B summaries ARE long and DO carry a think block,
# and that length+block is exactly what broke c01 (899 write tokens vs 538
# compacted). A short fixed summary would NOT catch this. summary_token_layout
# strips the block (matching the Qwen template) so both spans match; on the
# pre-fix code the block survives into the write-time span and the alignment
# raises -- which is precisely what this preflight guards.
_SMOKE_THINK = (
    "<think>\nOkay, the user wants a thorough context note summarizing the "
    "conversation. Let me recall the key decisions, the open threads, the "
    "definitions we introduced, and the constraints each side stated so I can "
    "write something a fresh reader could continue from. I should be redundant "
    "and specific, use retrieval-friendly wording, and avoid any commentary "
    "before or after the note itself. Let me organize it into decisions, open "
    "threads, definitions, and constraints.\n</think>\n\n")
_SMOKE_SUMMARY_BODY = (
    "**Context Note: Project Planning Summary**\n\n"
    "**Decisions Made and What Was Chosen Over What**\n"
    "- **Pricing Model**: Chose usage-based metering with prepaid credits (not "
    "flat monthly fees, per-seat tiers, or other models). This aligns with cost "
    "scaling and user flexibility, and was preferred after weighing predictable "
    "revenue against customer fairness.\n"
    "- **Onboarding Flow (Nimbus)**: Chose a self-serve signup funnel (landing "
    "page then first login) with no friction, no tutorials, and no forced "
    "steps. Rejected guided tours, pop-ups, and webinars because early testers "
    "found them intrusive and low-value.\n"
    "- **Launch Strategy**: Chose an invite-only beta with 30 design partners "
    "(not a public splash). Rejected press outreach, cold email campaigns, and "
    "public announcements as premature given the product's maturity.\n"
    "- **Marketing Approach**: Chose low-effort, high-impact content such as "
    "user stories and visual demos over traditional sales pitches. Rejected "
    "countdown timers, urgency language, and webinar-style content as "
    "inconsistent with the product's values.\n\n"
    "**Open Threads and Next Steps**\n"
    "- **Nimbus Copy**: Finalize updates to remove friction, jargon, and forced "
    "steps (for example, change 'Click here to begin' to 'Try a chart by "
    "uploading a file').\n"
    "- **Beta Coordination**: Define criteria for design partners (each must "
    "ship one real report) and set the timeline for opening the beta.\n"
    "- **Metrics Dashboard**: Finalize five core metrics (active users, "
    "first-chart conversion, time to first chart, report shipments, and credit "
    "usage) and set thresholds for success.\n"
    "- **Post-Launch Metrics**: Determine when metrics become meaningful (for "
    "example, two to four weeks post-launch, after the initial noise subsides).\n\n"
    "**Definitions, Names, and Terms**\n"
    "- **Nimbus**: The self-serve signup and onboarding flow (landing page to "
    "first login). It is not related to cloud infrastructure, which is called "
    "CloudOps or Infra-Relay.\n"
    "- **Trademark**: Filed as TM-88214 on April 9, 2024; expected twelve to "
    "eighteen months for approval.\n"
    "- **Cloud Budget**: Fixed at 6,410 dollars per month and non-negotiable.\n"
    "- **Design Partners**: 30 invitees in the beta who must each ship one real "
    "report before the product opens to the public.\n\n"
    "**Constraints and Preferences**\n"
    "- **No Webinars**: Rejected due to low engagement and negative user "
    "perception in prior tests.\n"
    "- **No Countdown Timers**: Rejected as scammy and inconsistent with the "
    "product's values and tone.\n"
    "- **No Cold Outreach**: Rejected after failed pilot campaigns produced "
    "spam complaints and near-zero conversion. Both sides prefer inbound, "
    "content-led growth and word-of-mouth from satisfied design partners.")
_SMOKE_SUMMARY = _SMOKE_THINK + _SMOKE_SUMMARY_BODY

# Default preflight tokenizers. Qwen covers the family the bug lives in; add a
# gemma tokenizer id (env SC_SMOKE_TOKENIZERS, comma-separated) to also cover
# the gemma B-context builder. Gemma tokenizers are gated on HF, so we do NOT
# hard-require one -- but we DO require at least one qwen-family tokenizer.
_SMOKE_TOKENIZERS_DEFAULT = ("Qwen/Qwen3-0.6B",)


def _smoke_check_conv(tok, family, conv, summary_text):
    """Build the pure token geometry for ONE conv and assert build_alignment
    succeeds AND covers the FULL summary span (no raise, no silent drop).
    Returns (ok, detail)."""
    msgs = conv["messages"][:-1]
    tsm = conv["sections"]["middle_end_msg"]
    ctx = build_token_context(tok, family, msgs, summary_text, tsm)
    summ, regions, b_ids = ctx["summ"], ctx["regions"], ctx["b_ids"]
    s_start, s_end = summ["s_start"], summ["s_end"]
    # This RAISES ValueError on a write-vs-compacted token divergence (the bug).
    pairs = build_alignment_direct(
        b_ids, summ["old_ids"], set(tok.all_special_ids), regions)
    # FULL-coverage check for the summary region: every write-time summary token
    # must map to a compacted position (clean prose has no special tokens / sinks
    # in this span, so coverage must be the whole contiguous range).
    covered = sorted(o for _n, o in pairs if s_start <= o < s_end)
    expected = list(range(s_start, s_end))
    if covered != expected:
        missing = sorted(set(expected) - set(covered))
        return False, (f"summary span NOT fully covered: {len(covered)}/"
                       f"{len(expected)} tokens mapped; missing "
                       f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    return True, f"{len(covered)}/{len(expected)} summary tokens mapped 1:1"


def _smoke_align(data_dir: Path, tokenizer_ids) -> int:
    """CPU/tokenizer-only alignment preflight over EVERY corpus conv (and every
    provided tokenizer/family), using a LONG think-containing summary. Exits 0
    iff every conv aligns with full summary coverage; nonzero on any failure."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    print("== cross_arch_probe --smoke-align (CPU/tokenizer-only) ==")
    print(f"data_dir={data_dir}  tokenizers={list(tokenizer_ids)}")
    print(f"smoke summary: {len(_SMOKE_SUMMARY)} chars "
          f"(has <think> block: {'<think>' in _SMOKE_SUMMARY})")

    convs = [json.loads(p.read_text())
             for p in conversation_paths(data_dir)]
    if not convs:
        print("FAIL: no conversations found in corpus", file=sys.stderr)
        return 2

    n_fail = 0
    n_ok = 0
    families_covered = set()
    loaded_any = False
    for tid in tokenizer_ids:
        try:
            tok = AutoTokenizer.from_pretrained(tid)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip tokenizer {tid}]: load failed "
                  f"({type(e).__name__}: {e})")
            continue
        loaded_any = True
        family = detect_template_family(tok)
        families_covered.add(family)
        # measure the summary token length for THIS tokenizer (informational).
        raw_len = len(tok(_SMOKE_SUMMARY, add_special_tokens=False).input_ids)
        clean_len = len(tok(strip_reasoning_block(_SMOKE_SUMMARY),
                            add_special_tokens=False).input_ids)
        print(f"\n  tokenizer={tid} family={family} "
              f"summary tokens raw(with think)={raw_len} "
              f"clean(stripped)={clean_len}")
        for conv in convs:
            try:
                ok, detail = _smoke_check_conv(
                    tok, family, conv, _SMOKE_SUMMARY)
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"{type(e).__name__}: {e}"
            tag = "OK  " if ok else "FAIL"
            print(f"    [{tag}] {conv['id']}: {detail}")
            if ok:
                n_ok += 1
            else:
                n_fail += 1

    if not loaded_any:
        print("FAIL: no tokenizer could be loaded (need a qwen-family "
              "tokenizer, e.g. Qwen/Qwen3-0.6B)", file=sys.stderr)
        return 2
    if "qwen" not in families_covered:
        print("FAIL: no qwen-family tokenizer exercised (the bug lives in the "
              "qwen think-stripping template)", file=sys.stderr)
        return 2
    print(f"\nSMOKE-ALIGN: {n_ok} ok, {n_fail} failed "
          f"(families: {sorted(families_covered)})")
    if n_fail:
        print("SMOKE-ALIGN FAILED -- write-time vs compacted summary token "
              "spans diverge; do NOT launch the pod run.", file=sys.stderr)
        return 1
    print("SMOKE-ALIGN OK")
    return 0


def make_summaries(summarizer: str, data_dir: Path, summaries_path: Path,
                   conv_limit: int, trust_remote_code: bool):
    """Generate the SHARED fixed summary text ONCE with a single designated
    summarizer, write {conv_id: text, _summarizer: id} to summaries_path. This
    file is then a required INPUT to every per-model run so the compaction
    content is identical across architectures."""
    import torch  # noqa: PLC0415
    from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                              AutoTokenizer)
    from arms_common import SUMMARY_REQUEST  # noqa: PLC0415
    from arms_hf import generate_summary_hf  # noqa: PLC0415

    print(f"MAKE_SUMMARIES summarizer={summarizer} limit={conv_limit}", flush=True)
    tok = AutoTokenizer.from_pretrained(summarizer, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        summarizer, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=trust_remote_code)
    model.eval()

    out = {"_summarizer": summarizer}
    # merge with any existing (idempotent / resumable)
    if summaries_path.exists():
        out.update(json.loads(summaries_path.read_text()))
        out["_summarizer"] = summarizer
    for conv, _plants in collect_specs(data_dir, conv_limit):
        cid = conv["id"]
        if out.get(cid):
            print(f"  {cid}: cached", flush=True)
            continue
        msgs = conv["messages"][:-1]
        summ = generate_summary_hf(model, tok, msgs, request=SUMMARY_REQUEST)
        out[cid] = summ["text"]
        print(f"  {cid}: {len(summ['gen_ids'])} summary tokens", flush=True)
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(summaries_path, "w"), indent=1)
    print(f"WROTE {summaries_path} ({len(out) - 1} summaries)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("SC_HF_MODEL"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default="results/cross_arch")
    ap.add_argument("--summaries", default=DEFAULT_SUMMARIES,
                    help="shared FIXED summaries file (input to every model)")
    ap.add_argument("--summarizer",
                    default=os.environ.get("SC_SUMMARIZER_MODEL", DEFAULT_SUMMARIZER))
    ap.add_argument("--make-summaries", action="store_true",
                    help="generate the shared fixed summaries ONCE, then exit")
    ap.add_argument("--conv-limit", type=int,
                    default=int(os.environ.get("SC_CONV_LIMIT", "4")))
    ap.add_argument("--alpha-v", type=float,
                    default=float(os.environ.get("SC_GC_ALPHA", "0.75")))
    ap.add_argument("--alpha0-tol", type=float,
                    default=float(os.environ.get("SC_ALPHA0_TOL", "5e-3")))
    ap.add_argument("--change-tol", type=float,
                    default=float(os.environ.get("SC_CHANGE_TOL", "1e-3")))
    ap.add_argument("--max-gold-tok", type=int,
                    default=int(os.environ.get("SC_MAX_GOLD_TOK", "80")))
    ap.add_argument("--placebo", default=(os.environ.get("SC_PLACEBO") or None),
                    help="CONTROL #1: corrupted-source placebo graft mode "
                         f"({'|'.join(PLACEBO_MODES)}); default off")
    ap.add_argument("--alpha-sweep", action="store_true",
                    default=os.environ.get("SC_ALPHA_SWEEP", "0")
                    not in ("0", "", "false", "False"),
                    help="CONTROL #3: run E at alpha in "
                         f"{ALPHA_SWEEP_VALUES}; default off")
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("SC_SEED", str(ROBUST_SEED))))
    ap.add_argument("--strong-prior", dest="strong_prior",
                    action="store_true",
                    default=os.environ.get("SC_STRONG_PRIOR", "1")
                    not in ("0", "", "false", "False"),
                    help="FEATURE #2: signed codename disambiguation readout for "
                         "strong_prior plants; default ON")
    ap.add_argument("--no-strong-prior", dest="strong_prior",
                    action="store_false",
                    help="disable FEATURE #2 strong-prior signed readout")
    ap.add_argument("--champion-scan", type=int,
                    default=int(os.environ.get("SC_CHAMPION_SCAN", "0")),
                    help="FEATURE #3: per-layer champion scan into N "
                         "fractional-depth regions (0=off; a value >=2 sets N; "
                         f"any other positive value uses the default "
                         f"{CHAMPION_SCAN_DEFAULT_N}); default off")
    ap.add_argument("--champion-regions",
                    default=(os.environ.get("SC_CHAMPION_REGIONS") or None),
                    help="FEATURE #3 rescue test: comma-separated region indices "
                         "to graft TOGETHER (e.g. '4,5'); alpha=0 elsewhere")
    # ---- CROSS-ARCH DESIGN v2: per-model in-context (native) render ----
    ap.add_argument("--native-render", dest="native_render",
                    action="store_true",
                    default=os.environ.get("SC_NATIVE_RENDER", "1")
                    not in ("0", "", "false", "False"),
                    help="DESIGN v2: render the corpus IN-CONTEXT per model from "
                         "the shared scenarios scaffold (default ON). The gold "
                         "continuation stays SHARED; only assistant replies + the "
                         "self-gen summary are model-filled.")
    ap.add_argument("--no-native-render", dest="native_render",
                    action="store_false",
                    help="use PRE-RENDERED data/synthetic convs (validated legacy "
                         "path; SC_NATIVE_RENDER=0)")
    ap.add_argument("--ablate-qk-norm", dest="ablate_qk_norm",
                    action="store_true",
                    default=os.environ.get("SC_ABLATE_QK_NORM", "0")
                    not in ("0", "", "false", "False"),
                    help="WITHIN-MODEL QK-NORM ABLATION (SC_ABLATE_QK_NORM=1, "
                         "default OFF): after load, replace every QK-norm module "
                         "(q_norm/k_norm/query|key layernorm|norm) with Identity "
                         "so the forward pass runs WITHOUT QK-norm -- the clean "
                         "causal test of H1. FAILs LOUD (status=ERROR) if the "
                         "model has no QK-norm modules to ablate. Everything else "
                         "stays IDENTICAL for a clean within-model comparison.")
    ap.add_argument("--scenarios",
                    default=os.environ.get(
                        "SC_SCENARIOS",
                        str(Path(__file__).resolve().parent.parent
                            / "data" / "scenarios.json")),
                    help="shared scaffold for native render (data/scenarios.json)")
    ap.add_argument("--native-max-reply", type=int,
                    default=int(os.environ.get("SC_NATIVE_MAX_REPLY",
                                               str(NATIVE_MAX_REPLY_DEFAULT))),
                    help="cap on model-generated assistant reply tokens")
    ap.add_argument("--native-temp", type=float,
                    default=float(os.environ.get("SC_NATIVE_TEMP",
                                                 str(NATIVE_TEMP_DEFAULT))),
                    help="native reply gen temperature (0=greedy, reproducible)")
    ap.add_argument("--headroom-floor", type=float,
                    default=float(os.environ.get("SC_HEADROOM_FLOOR",
                                                 str(HEADROOM_FLOOR_DEFAULT))),
                    help="v2.1 GATE #2: min per-category A-B headroom to be "
                         "interpretable (below = floored, excluded from sign)")
    ap.add_argument("--task-lpa-floor", type=float,
                    default=float(os.environ.get("SC_TASK_LPA_FLOOR",
                                                 str(TASK_LPA_FLOOR_DEFAULT))),
                    help="v2.1 GATE #3: min lp_A per-token to score a plant "
                         "(below = model can't do the task, plant excluded)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="run the CPU-only controls self-tests, then exit")
    ap.add_argument("--self-test-native", action="store_true",
                    help="run ONLY the native-render pure-parts self-test "
                         "(turn-ordering/gates/covariates), then exit")
    ap.add_argument("--smoke-align", action="store_true",
                    help="CPU/tokenizer-only alignment preflight over the corpus "
                         "with a long think-containing summary; exits nonzero on "
                         "any write-vs-compacted token divergence. Run before a "
                         "pod launch.")
    ap.add_argument("--smoke-tokenizers",
                    default=(os.environ.get("SC_SMOKE_TOKENIZERS") or None),
                    help="comma-separated tokenizer ids for --smoke-align "
                         "(default Qwen/Qwen3-0.6B; add a gemma id to cover that "
                         "family)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test_controls())

    if args.self_test_native:
        sys.exit(_self_test_native())

    data_dir = Path(args.data_dir)
    if args.dry_run:
        sys.exit(_dry_run(data_dir, args.conv_limit))

    if args.smoke_align:
        toks = (tuple(t.strip() for t in args.smoke_tokenizers.split(",")
                      if t.strip())
                if args.smoke_tokenizers else _SMOKE_TOKENIZERS_DEFAULT)
        sys.exit(_smoke_align(data_dir, toks))

    trust = os.environ.get("SC_TRUST_REMOTE", "1") not in ("0", "false", "False")
    summaries_path = Path(args.summaries)

    if args.make_summaries:
        make_summaries(args.summarizer, data_dir, summaries_path,
                       args.conv_limit, trust)
        print("CROSS_ARCH_DONE", flush=True)
        return

    if not args.model:
        print("FATAL: SC_HF_MODEL (or --model) is required for the real run",
              file=sys.stderr)
        sys.exit(2)

    # CORPUS + SUMMARY SOURCE.
    # DESIGN v2 (SC_NATIVE_RENDER=1, default): the corpus is rendered IN-CONTEXT by
    # THIS model from the shared data/scenarios.json scaffold, and the summary is
    # ALWAYS this model's own self-gen -- so fixed summaries are ignored entirely.
    # The graft re-injects the model's own write-time values, so it must be measured
    # on the model's own native conversation. (The plant gold stays SHARED.)
    scenarios = None
    fixed_summaries = None
    force_selfgen = os.environ.get("SC_SELFGEN", "").strip() in ("1", "true", "yes")
    if args.native_render:
        scenarios = json.loads(Path(args.scenarios).read_text())
        print(f"SC_NATIVE_RENDER=1 -> PER-MODEL IN-CONTEXT native render from "
              f"{args.scenarios} ({len(scenarios)} scenarios); self-gen summary; "
              f"gold stays SHARED.", flush=True)
    # Legacy pre-rendered path (SC_NATIVE_RENDER=0): the REDESIGNED sweep still uses
    # PER-MODEL SELF-GENERATED summaries (SC_SELFGEN=1): a FIXED foreign summary was
    # found to SUPPRESS the graft to null -- the mechanism needs the model's OWN
    # write-time summarization act. Set SC_SELFGEN=1 (or delete the summaries file)
    # to force self-gen; a fixed summaries file is loaded only when SC_SELFGEN unset.
    elif force_selfgen:
        print("SC_SELFGEN=1 -> PER-MODEL SELF-GENERATED summaries (redesign default; "
              "the graft needs the model's own summary).", flush=True)
    elif summaries_path.exists():
        fixed_summaries = json.loads(summaries_path.read_text())
        # tolerate an optional "_summarizer" provenance key if present
        fixed_summaries = {k: v for k, v in fixed_summaries.items()
                           if not k.startswith("_")}
        print(f"LOADED fixed summaries: {summaries_path} "
              f"({len(fixed_summaries)} convs)", flush=True)
    else:
        print(f"WARNING: fixed summaries file {summaries_path} not found -- "
              f"falling back to PER-MODEL self-generated summaries (results NOT "
              f"cross-model comparable). Build the shared file with Sonnet, or: "
              f"SC_SUMMARIZER_MODEL={args.summarizer} "
              f"python3 src/cross_arch_probe.py --make-summaries", file=sys.stderr)

    out_dir = Path(args.out_dir)

    # top-level guard: even an unexpected crash writes an ERROR record rather
    # than taking down a multi-model sweep.
    champion_regions = None
    if args.champion_regions:
        champion_regions = [int(x) for x in str(args.champion_regions).split(",")
                            if x.strip() != ""]

    try:
        doc = run_model(
            args.model, data_dir, out_dir, fixed_summaries,
            conv_limit=args.conv_limit, alpha_v=args.alpha_v,
            alpha0_tol=args.alpha0_tol, change_tol=args.change_tol,
            max_gold_tok=args.max_gold_tok, trust_remote_code=trust,
            placebo_mode=args.placebo, alpha_sweep=args.alpha_sweep,
            seed=args.seed, strong_prior=args.strong_prior,
            champion_scan=args.champion_scan, champion_regions=champion_regions,
            native_render=args.native_render, scenarios=scenarios,
            native_max_reply=args.native_max_reply, native_temp=args.native_temp,
            headroom_floor=args.headroom_floor, task_lpa_floor=args.task_lpa_floor,
            ablate_qk_norm_flag=args.ablate_qk_norm)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        doc = {
            "model": args.model, "architecture": None, "status": "ERROR",
            "reason": f"uncaught {type(e).__name__}: {e}\n{traceback.format_exc()}",
            "n_plants": 0, "by_category": {},
            "smoke": {"alpha0_ok": None, "graft_changes_output": None},
        }
    write_result(out_dir, args.model, doc)
    print("CROSS_ARCH_DONE", flush=True)


if __name__ == "__main__":
    main()
