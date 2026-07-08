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
import traceback
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms_common import (  # noqa: E402  (stdlib-only, import-safe on CPU box)
    build_alignment,
    build_b_messages,
    build_b_messages_gemma,
    canonical_ids_any,
    detect_template_family,
    message_starts_any,
    render_hf,
)

# The FIXED summaries are produced EXTERNALLY (coordinator uses Sonnet -- a
# realistic mid-tier summarizer; a frontier model would write an
# unrealistically good summary and invalidate the test) as a plain
# {conv_id: summary_text} JSON. This is the required input to every model run.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
DEFAULT_SUMMARIES = str(
    Path(__file__).resolve().parent.parent / "data" / "fixed_summaries.json")
DEFAULT_SUMMARIZER = "Qwen/Qwen3.6-27B"

# All 5 continuity categories (skip contaminated). Full dissociation + power.
CATS = ("sense", "referent", "stance", "ruled_out", "evicted_fact")
# full gradient: sense (pure meaning) → referent/ruled_out (evicted decisions) →
# evicted_fact (precise verbatim, tests meaning-vs-verbatim claim); stance = null anchor.
# 115 probes/model (~72% more power than sense+referent alone).

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


def detect_model_hparams(config) -> dict:
    """Architecture hyper-parameters for the sign-of-graft regression (pure, no
    torch). Reads the (possibly nested text_config) HF config, model-type
    agnostic. ``qk_norm`` is detected from any of several config spellings used
    across families; ``gqa_ratio`` = n_heads / n_kv (1.0 == full MHA, >1 == GQA,
    == n_heads == MQA)."""
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


# ===========================================================================
# The heavy path (torch/transformers) lives entirely below, imported lazily so
# --dry-run runs on a CPU-only box with no ML deps.
# ===========================================================================
def run_model(model_id: str, data_dir: Path, out_dir: Path,
              fixed_summaries: dict | None, *,
              conv_limit: int, alpha_v: float, alpha0_tol: float,
              change_tol: float, max_gold_tok: int,
              trust_remote_code: bool,
              placebo_mode: str | None = None, alpha_sweep: bool = False,
              seed: int = ROBUST_SEED) -> dict:
    """fixed_summaries: {conv_id: summary_text} loaded from the shared external
    file (Sonnet-written, held IDENTICAL across models). If None, no fixed file
    was present and we fall back to per-model self-generated summaries (results
    are NOT cross-model comparable -- flagged in ``summary_source``)."""
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
        "summary_source": ("FIXED external (shared across models)"
                           if fixed_summaries is not None
                           else "PER-MODEL fallback (NOT cross-model comparable)"),
        "skipped_convs": [],
        "n_plants": 0,
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

    def build_summary_snapshot(msgs, summary_text):
        """Write-time snapshot of the FIXED summary text in THIS model's cache.

        Mirrors arms_hf.generate_summary_hf's layout, but the summary tokens are
        the shared fixed text (teacher-forced, NOT model-generated), so the
        compaction content is identical across models. Returns the same dict
        shape the alignment/graft code expects.

        Family-aware (``family`` captured from run_model's enclosing scope,
        detect_template_family(tok) @ ~L390): the qwen/mistral path is
        unchanged; gemma needs an alternation-safe [conversation +
        summary-request] because Gemma's template (a) has no system role
        (the template folds system into the first user turn) and (b) forbids
        consecutive same-role messages. If the conversation already ends in a
        user turn, appending the user summary-request directly raises
        TemplateError ("roles must alternate"); we insert a minimal assistant
        turn first (mirroring build_b_messages_gemma's user-note/assistant-ack
        structure). Prefix-stability (verified for gemma via
        message_token_starts_prefix) keeps conv_ids a prefix of req_ids, and
        the summary tokens still land in a contiguous span [s_start, s_end) at
        the very end (summ_ids appended after req_ids exactly as before)."""
        conv_ids = canonical_ids_any(tok, msgs, render_hf)
        if family == "gemma":
            req_msgs = list(msgs)
            if req_msgs and req_msgs[-1]["role"] == "user":
                req_msgs = req_msgs + [
                    {"role": "assistant", "content": "Understood."}]
            req_msgs = req_msgs + [{"role": "user", "content": _REQ}]
            req_ids = render_hf(tok, req_msgs, True)
        else:
            req_ids = render_hf(
                tok, msgs + [{"role": "user", "content": _REQ}], True)
        if req_ids[:len(conv_ids)] != conv_ids:
            raise Unsupported("summary-request render is not a prefix of conv "
                              "(template not prefix-stable)")
        summ_ids = tok(summary_text, add_special_tokens=False).input_ids
        old_ids = list(req_ids) + list(summ_ids)
        snap = force_prefill(old_ids)
        return {
            "text": summary_text,
            "gen_ids": summ_ids,
            "conv_end": len(conv_ids),
            "s_start": len(req_ids),
            "s_end": len(req_ids) + len(summ_ids),
            "old_ids": old_ids,
            "snapshot": snap,
        }

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

    specs = collect_specs(data_dir, conv_limit)
    if not specs:
        doc.update(status="ERROR", reason="no usable plants in corpus subset")
        return doc

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

            ids = canonical_ids_any(tok, msgs, render_hf)
            starts = message_starts_any(tok, msgs, ids, render_hf)

            summ = build_summary_snapshot(msgs, summary_text)

            b_msgs, summ_idx, tail_idx = build_b_and_indices(
                family, msgs, summ["text"], tsm)
            b_ids = canonical_ids_any(tok, b_msgs, render_hf)
            b_starts = message_starts_any(tok, b_msgs, b_ids, render_hf)

            regions = [
                ((b_starts[tail_idx], len(b_ids)), (starts[tsm], summ["conv_end"])),
                ((b_starts[summ_idx], b_starts[summ_idx + 1]),
                 (summ["s_start"], summ["s_end"])),
            ]
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

            conv_identity_done = False

            for pl in plants:
                gold = str(pl["gold"]).strip()
                tgt = tok(gold, add_special_tokens=False).input_ids[:max_gold_tok]
                if len(tgt) < 2:
                    continue

                def suffix(mm):
                    full = render_hf(
                        tok, mm + [{"role": "user", "content": pl["probe"]}], True)
                    cn = canonical_ids_any(tok, mm, render_hf)
                    return full[len(cn):]

                sa, sb = suffix(msgs), suffix(b_msgs)
                la = tf(a_snap, sa + tgt[:-1], tgt, len(ids))
                lb = tf(b_snap, sb + tgt[:-1], tgt, len(b_ids))
                le = tf(e_snap, sb + tgt[:-1], tgt, len(b_ids))
                le0 = tf(e0_snap, sb + tgt[:-1], tgt, len(b_ids))

                # CONTROL #2: identity-graft no-op check (once per conv, reuses
                # this plant's A-side feed/targets).
                if id_snap is not None and not conv_identity_done:
                    la_id = tf(id_snap, sa + tgt[:-1], tgt, len(ids))
                    identity_diffs.append(abs(la_id - la))
                    conv_identity_done = True

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

                # CONTROL #1: placebo raw_EB for this plant (same gold/feed).
                placebo_raw_eb = None
                if e_placebo_snap is not None:
                    lep = tf(e_placebo_snap, sb + tgt[:-1], tgt, len(b_ids))
                    placebo_raw_eb = lep - lb
                    per_cat_placebo[pl["category"]].append({
                        "conversation_id": conv["id"], "raw_EB": placebo_raw_eb})

                # CONTROL #3: alpha dose-response for this plant.
                alpha_raw_eb = {}
                if alpha_sweep:
                    for a in alpha_list:
                        le_a = (le if a == alpha_v
                                else tf(e_alpha_snaps[a], sb + tgt[:-1], tgt,
                                        len(b_ids)))
                        alpha_raw_eb[a] = le_a - lb
                        per_cat_alpha[a][pl["category"]].append({
                            "conversation_id": conv["id"], "raw_EB": le_a - lb})

                # CONTROL #4: save the raw per-probe trace (offline recompute).
                traces.append({
                    "plant_id": pl["id"], "conversation_id": conv["id"],
                    "category": pl["category"], "distance": pl.get("distance"),
                    "lp_A": la, "lp_B": lb, "lp_E": le, "raw_EB": raw_eb,
                    "gold": gold, "n_gold_tokens": len(tgt), "alpha": alpha_v,
                    "seed": seed, "summary_len_tokens": summary_len_tokens,
                    "grafted_layer_count": grafted_layer_count,
                    "n_pairs": len(pairs),
                    "placebo_mode": placebo_mode,
                    "placebo_raw_EB": placebo_raw_eb,
                    "alpha_sweep_raw_EB": (alpha_raw_eb or None),
                })

            # retain THIS conv's summary snapshot for a shuffle_probe placebo on
            # the NEXT conv (only when placebo is active -- else free it).
            if placebo_mode == "shuffle_probe":
                prev_summ_snap = summ["snapshot"]
                prev_old_idx = old_list

            del a_snap, b_snap, e_snap, e0_snap
            del id_snap, e_placebo_snap, e_alpha_snaps
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
        by_cat_robust[cat] = robust_category_stats(
            raw_eb_vals, ratio_gap_pairs, raw_eb_clusters=clusters)
    doc["by_category"] = by_cat
    doc["by_category_robust"] = by_cat_robust

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

    if effect_sign == "positive":
        doc["verdict"] = "SIGNIFICANT_POSITIVE"
    elif effect_sign == "negative":
        doc["verdict"] = "SIGNIFICANT_NEGATIVE"
    else:
        doc["verdict"] = "null/underpowered"
    doc["verdict_note"] = (
        f"aggregate raw_EB CI [{eb.get('lo')}, {eb.get('hi')}] -> effect_sign="
        f"{effect_sign}. per-category CI-excludes-0: {cat_signs} "
        f"(referent/sense any-significant={any_sig_cat}). n is small so the "
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
        "n_pairs": 30, "placebo_mode": None, "placebo_raw_EB": None,
        "alpha_sweep_raw_EB": None,
    }
    expected_keys = {
        "plant_id", "conversation_id", "category", "distance", "lp_A", "lp_B",
        "lp_E", "raw_EB", "gold", "n_gold_tokens", "alpha", "seed",
        "summary_len_tokens", "grafted_layer_count", "n_pairs", "placebo_mode",
        "placebo_raw_EB", "alpha_sweep_raw_EB"}
    assert set(trace) == expected_keys, \
        f"trace keys mismatch: {set(trace) ^ expected_keys}"
    print(f"  trace dict has all {len(expected_keys)} required fields OK")

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

    print("\nSELF-TEST OK")
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="run the CPU-only controls self-tests, then exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test_controls())

    data_dir = Path(args.data_dir)
    if args.dry_run:
        sys.exit(_dry_run(data_dir, args.conv_limit))

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

    # Prefer the shared FIXED summaries file (external, Sonnet-written,
    # {conv_id: summary_text}). If absent, fall back to per-model self-generated
    # summaries (a convenience for testing -- results are NOT cross-model
    # comparable and are flagged as such in the output).
    fixed_summaries = None
    if summaries_path.exists():
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
    try:
        doc = run_model(
            args.model, data_dir, out_dir, fixed_summaries,
            conv_limit=args.conv_limit, alpha_v=args.alpha_v,
            alpha0_tol=args.alpha0_tol, change_tol=args.change_tol,
            max_gold_tok=args.max_gold_tok, trust_remote_code=trust,
            placebo_mode=args.placebo, alpha_sweep=args.alpha_sweep,
            seed=args.seed)
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
