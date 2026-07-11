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
compaction) with a pure-Python percentile bootstrap 95% CI resampled over
CONVERSATIONS (the honest inferential unit -- plants within a conversation are
correlated, so a plant-level bootstrap is anti-conservative; the plant-level
interval is retained only as raw_EB_ci_plant for reference) (N=10000, seed=42). We also report %_helped (fraction with raw_EB > 0) with its
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
  SC_CONV_LIMIT       : N conversations to score (default 4)
  SC_CONV_START       : 0-based OFFSET into the sorted conversation list; the run
                        scores convs [SC_CONV_START : SC_CONV_START+SC_CONV_LIMIT]
                        (default 0 == the legacy first-N slice, byte-identical).
                        e.g. SC_CONV_START=12 SC_CONV_LIMIT=24 -> the held-out set
                        c13..c36 (fresh convs the effect was never tuned on). True
                        conv ids are preserved in every result/label.
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
  SC_ABLATE_LAMBDA    : GRADED QK-norm ablation dose-response, DECOUPLED FROM
                        GENERATION (comma-separated lambdas, e.g.
                        "1.0,0.5,0.25,0.0"; default off/1.0 = no change). Each
                        QK-norm module is wrapped as qk_lambda(x)=(1-lambda)*x+
                        lambda*RMSNorm_qk(x): lambda=1.0=clean/original,
                        lambda=0.0=identity (== the full ablation above). The
                        corpus is rendered + summarized ONCE at lambda=1.0 (clean,
                        coherent); ONLY the teacher-forced graft-vs-compacted
                        SCORING read-out (lp_A/lp_B/lp_E) is perturbed per lambda --
                        the degraded model NEVER generates. Reports per lambda: the
                        referent raw_EB (conversation-clustered CI) + n plants
                        clearing the per-model relative competence floor. 1.0 is
                        always added as the sanity anchor (must reproduce the
                        validated referent). Mutually exclusive with
                        SC_ABLATE_QK_NORM (that one breaks generation).

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

Out: results/cross_arch/<model-slug>.json  (the pooled per-model result), AND
     PER-CONVERSATION CHECKPOINTS results/cross_arch/<model-slug>/conv_<NNN>__<cid>.json
     (incident #38). Each per-conv file is a COMPLETE, REUSABLE render artifact,
     written atomically (tmp + os.replace) as the conv is rendered+scored, BEFORE
     the next conv -- so an interruption loses <=1 conversation and the render
     (the expensive generation) is never redone. The final <slug>.json is a pure
     function of these files (pool -> identical numbers). Cross-pod SPLITTING =
     pool the union (scripts/block_analysis.py reads the traces). Schema:
       schema, stage("rendered"|"scored"), window_pos, conv_id,
       fingerprint          : all SCORING params -> gates EXACT score replay/resume,
       render_fingerprint   : GENERATION params only -> gates render+summary REUSE
                              across DIFFERENT scoring configs (alpha/region/floor);
                              a future run reuses the saved text, forward-passes to
                              rebuild KV, and re-grafts/re-scores with NO generation,
       render{conv,plants,reply_records} : the native in-context conversation TEXT
                              (the model's generated replies = the write-time KV),
       summary_text         : the self-gen compaction summary (TEXT),
       scan_lpa             : per-plant lp_A for the pooled relative competence floor,
       payload              : this conv's per-plant traces + raw_EB + every
                              accumulator delta needed to reconstruct the result.
     SC_CHECKPOINT_FRESH=1 ignores + overwrites existing checkpoints (force fresh).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
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


# --------------------------------------------------------------------------- #
# COMPRESSION SWEEP (SC_SUMMARY_LEVEL) -- vary ONLY the self-gen summary
# length/detail budget along a MONOTONIC compression axis, holding model / dtype
# / corpus / intervention / placebo FIXED. Tests whether the value-graft benefit
# CONCENTRATES under aggressive compaction (positives lived under BRIEF; the
# per-layer champion is null under the realistic SUMMARY_REQUEST). Each level is a
# self-gen summary REQUEST; the harness MEASURES the realized compression ratio
# (clean-summary_tokens / full_context_tokens) per conv so "aggressive vs
# realistic" is a NUMBER, reported alongside raw_EB + content-specificity. Level
# 'realistic' == the current SUMMARY_REQUEST null condition (DEFAULT; byte-
# unchanged when SC_SUMMARY_LEVEL is unset). BRIEF and REALISTIC reuse the
# existing arms_common requests; ULTRA and MEDIUM are constructed intermediates
# that vary ONLY the length/detail budget in the same instruction style.
# --------------------------------------------------------------------------- #
_ULTRA_REQUEST = (
    "Please write a ONE-SENTENCE context note (at most 25 words) giving only the "
    "single most important thing about our conversation so far. No lists, no "
    "specific details, no names or numbers. Do not add commentary before or after "
    "the note itself.")
_MEDIUM_REQUEST = (
    "Please write a short context note (about 150 words) summarizing our "
    "conversation so far, for someone who will continue this conversation without "
    "seeing it. Cover the main decisions made, the key terms we introduced, and "
    "the open threads. Include the most important specifics but omit minor "
    "detail. Write it as flowing prose. Do not add commentary before or after the "
    "note itself.")


def compression_levels() -> dict:
    """Ordered compression ladder {name: (request_text, approx_words, regime)},
    MOST-AGGRESSIVE first. BRIEF/REALISTIC/PROD reuse the existing arms_common
    requests (imported lazily so this module stays CPU-import-safe)."""
    from arms_common import (  # noqa: PLC0415
        SUMMARY_REQUEST, SUMMARY_REQUEST_BRIEF, SUMMARY_REQUEST_PROD)
    return {
        "ultra":     (_ULTRA_REQUEST, 25, "very-aggressive (~1 sentence, <=25w)"),
        "brief":     (SUMMARY_REQUEST_BRIEF, 70, "aggressive (3-5 sentences)"),
        "medium":    (_MEDIUM_REQUEST, 150, "moderate (~150 words)"),
        "realistic": (SUMMARY_REQUEST, 400,
                      "realistic (~300-500 words; current null condition)"),
        "prod":      (SUMMARY_REQUEST_PROD, 400,
                      "faithful (~300-500 words, OpenHands-condenser style)"),
    }


def resolve_summary_level(default_request):
    """Resolve SC_SUMMARY_LEVEL -> (name, request_text, approx_words, regime).
    Unset or 'realistic' -> the passed default request (behavior byte-unchanged).
    An unknown level FAILS LOUD rather than silently scoring the wrong regime."""
    name = (os.environ.get("SC_SUMMARY_LEVEL") or "").strip().lower()
    if not name or name == "realistic":
        return ("realistic", default_request, 400,
                "realistic (~300-500 words; current null condition)")
    levels = compression_levels()
    if name not in levels:
        raise ValueError(
            f"SC_SUMMARY_LEVEL={name!r} not in {sorted(levels)}")
    req, words, regime = levels[name]
    return (name, req, words, regime)


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


# --- GRADED QK-NORM ABLATION (lambda dose-response, DECOUPLED FROM GENERATION) --
# Fable's salvage of the causal H1 test: the BINARY full ablation above swaps
# QK-norm for Identity at LOAD, which makes the DEGRADED model GENERATE -> empty
# generation, every plant competence-floored. The graded version NEVER makes the
# perturbed model generate: it WRAPS each original q_norm/k_norm RMSNorm in an
# interpolation module and perturbs ONLY the graft-vs-compacted SCORING read-out
# (run_model forces lambda=1.0 for the native render + self-gen summary, and sets
# lambda<1.0 only around the teacher-forced lp_A/lp_B/lp_E forward passes).
#
#   qk_lambda(x) = (1-lambda)*x + lambda*RMSNorm_qk(x)
#     lambda == 1.0 -> exactly RMSNorm_qk(x)  (CLEAN; reproduces validated behavior)
#     lambda == 0.0 -> exactly x              (IDENTITY; == full QK-norm ablation)
#
# lambda is a plain mutable attribute so it is swept WITHOUT reloading weights (a
# reference to the original RMSNorm is stored; the wrapper recomputes the blend).
_QK_LAMBDA_CLASS = None


def _qk_lambda_class():
    """Lazily define + memoize the wrapper class (torch is imported lazily in this
    module so it stays import-safe on the CPU box)."""
    global _QK_LAMBDA_CLASS
    if _QK_LAMBDA_CLASS is None:
        import torch  # noqa: PLC0415

        class QKLambdaNorm(torch.nn.Module):
            """Interpolate between identity (lambda=0) and the ORIGINAL RMSNorm
            (lambda=1). At lambda==1.0 the forward is byte-for-byte the original
            module call (no arithmetic), so an installed-but-unswept wrapper is a
            no-op; at lambda==0.0 it is a pure identity == the binary ablation."""

            _is_qk_lambda = True

            def __init__(self, orig):
                super().__init__()
                self.orig = orig
                self.qk_lambda = 1.0

            def forward(self, x):  # noqa: D401
                lam = self.qk_lambda
                if lam == 1.0:
                    return self.orig(x)          # CLEAN: identical to original
                if lam == 0.0:
                    return x                      # IDENTITY: full ablation
                return (1.0 - lam) * x + lam * self.orig(x)

        _QK_LAMBDA_CLASS = QKLambdaNorm
    return _QK_LAMBDA_CLASS


def install_qk_lambda(model) -> tuple:
    """Wrap every QK-norm leaf (``_QK_NORM_LEAF_NAMES``) in a QKLambdaNorm, storing
    a ref to the ORIGINAL module. Idempotent (already-wrapped / Identity leaves are
    skipped). All wrappers start at lambda=1.0 (== clean). Returns
    ``(n_wrapped, sorted_names)``; the caller FAILs LOUD on 0 (nothing to graft)."""
    import torch  # noqa: PLC0415
    cls = _qk_lambda_class()
    targets = [n for n, m in model.named_modules()
               if n.rsplit(".", 1)[-1] in _QK_NORM_LEAF_NAMES
               and m is not None
               and not getattr(m, "_is_qk_lambda", False)
               and not isinstance(m, torch.nn.Identity)]
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, cls(getattr(parent, leaf)))
    return len(targets), sorted(targets)


def set_qk_lambda(model, value) -> int:
    """Set ``qk_lambda`` on every installed QKLambdaNorm wrapper (no reload).
    Returns the number of wrappers updated."""
    n = 0
    for _n, m in model.named_modules():
        if getattr(m, "_is_qk_lambda", False):
            m.qk_lambda = float(value)
            n += 1
    return n


def parse_lambda_values(spec) -> list:
    """Parse a comma-separated SC_ABLATE_LAMBDA / --ablate-lambda spec into a
    sorted-descending list of lambdas for the dose-response. Returns [] when the
    sweep is a no-op (empty, or solely 1.0) so the default path is byte-unchanged;
    otherwise ALWAYS includes 1.0 (the clean sanity anchor) and sorts high->low."""
    if not spec:
        return []
    vals = []
    for tok_ in str(spec).split(","):
        tok_ = tok_.strip()
        if tok_ == "":
            continue
        vals.append(float(tok_))
    if not vals:
        return []
    if all(v == 1.0 for v in vals):
        return []                          # no perturbation requested -> no sweep
    vals.append(1.0)                        # always anchor at clean
    return sorted(set(vals), reverse=True)


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


def load_champion_graft_cfg(path):
    """CHAMPION-VALIDATION wiring (per-layer / per-head tuned value graft).

    Loads a champion value-graft config so the SAME placebo-controlled estimator
    that runs at scalar alpha can instead apply the tuned per-LAYER alpha-map (or
    per-HEAD slot mask) to BOTH the real graft E and the placebo graft -- i.e.
    E_champion vs placebo_champion at the SAME layers/positions/alpha. VALUE-ONLY
    by construction: blend_values only ever rewrites V rows (keys untouched), so
    alpha_K=0 is guaranteed regardless of this config.

    Canonical JSON schema (exactly one of alpha_map / head_map):
        {"label": "...",
         "alpha_map": {"3": 0.75, "4": 1.0, ...}}   # per-LAYER alpha (all heads)
      or
        {"label": "...",
         "alpha": 1.0,
         "head_map":  {"4": [0,1], "25": [3], ...}}  # per-LAYER kv-head subset

    Returns the resolved dict (int-keyed maps) or None when ``path`` is falsy.
    """
    if not path:
        return None
    cfg = json.loads(Path(path).read_text())
    label = cfg.get("label") or Path(path).stem
    amap = cfg.get("alpha_map")
    hmap = cfg.get("head_map")
    if amap and hmap:
        raise ValueError(
            f"champion config {path}: give alpha_map OR head_map, not both "
            "(per-layer and per-head tuning are distinct champion families).")
    if amap:
        graft_alpha = {int(k): float(v) for k, v in amap.items()
                       if float(v) != 0.0}
        if not graft_alpha:
            raise ValueError(
                f"champion config {path}: alpha_map has no nonzero layers.")
        return {"label": label, "alpha_map": graft_alpha}
    if hmap:
        graft_head_map = {int(k): [int(h) for h in v]
                          for k, v in hmap.items() if v}
        if not graft_head_map:
            raise ValueError(f"champion config {path}: head_map is empty.")
        return {"label": label, "alpha": float(cfg.get("alpha", 1.0)),
                "head_map": graft_head_map}
    raise ValueError(
        f"champion config {path}: needs an alpha_map or a head_map.")


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
        "raw_EB_ci_method": "plant-level",
    }
    if raw_eb_clusters is not None:
        cl = bootstrap_ci_95_cluster(
            raw_eb_clusters, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
        # PROMOTE the conversation-clustered CI to be the HEADLINE raw_EB_ci.
        # Plants in one conversation share context + summary and are positively
        # correlated, so the plant-level bootstrap treats correlated plants as
        # independent and is ANTI-CONSERVATIVE (intervals too narrow -> false
        # "excludes zero"). The honest inferential unit is the CONVERSATION.
        # The plant-level interval is retained as raw_EB_ci_plant for reference;
        # raw_EB_ci_cluster is kept as an explicit alias. raw_EB_mean (the point
        # estimate) is unchanged -- only the CI resampling unit differs.
        out["raw_EB_ci_plant"] = [eb["lo"], eb["hi"]]
        out["raw_EB_ci_cluster"] = [cl["lo"], cl["hi"]]
        out["raw_EB_ci"] = [cl["lo"], cl["hi"]]
        out["raw_EB_ci_method"] = "conversation-clustered"
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


def collect_specs(data_dir: Path, conv_limit: int, conv_start: int = 0):
    """(conv_dict, [plant,...]) for the ``conv_limit`` convs starting at OFFSET
    ``conv_start`` (default 0 == the first ``conv_limit``) that have >=1 usable
    plant. Pure -- no tokenizer, no torch.

    ``conv_start`` selects a HELD-OUT window of the corpus: e.g. conv_start=12,
    conv_limit=24 selects convs c13..c36 (0-based offset into the sorted
    conversation_paths list). conv_start=0 reproduces the legacy first-N slice
    byte-for-byte. The TRUE conv ids (from each file) are preserved -- nothing is
    relabeled to c01."""
    specs = []
    for p in conversation_paths(data_dir)[conv_start: conv_start + conv_limit]:
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
# GATE #3 RELATIVE variant (pre-registered 2026-07-08). See PREREGISTRATION.md
# "GATE #3 AMENDMENT". K in robust-sigma (MADN) units; MIN_N plants required to
# estimate a per-model floor (below -> no plant is dropped).
TASK_COMPETENCE_MODE_DEFAULT = "absolute"   # {"absolute","relative"}
TASK_COMPETENCE_K_DEFAULT = 3.0
TASK_COMPETENCE_MIN_N = 8


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


def relative_competence_floor(la_values, k=TASK_COMPETENCE_K_DEFAULT,
                              min_n=TASK_COMPETENCE_MIN_N):
    """PURE (design v2.1 gate #3, RELATIVE variant -- pre-registered 2026-07-08,
    PREREGISTRATION.md "GATE #3 AMENDMENT"): a per-MODEL competence floor derived
    from the model's OWN gold-lp_A distribution, so the gate does not confound
    cross-model comparison.

        floor = median(lp_A) - k * MADN(lp_A)
        MADN  = 1.4826 * median(|lp_A - median(lp_A)|)   (normal-consistent MAD)

    WHY (vs the ABSOLUTE -8.0): the absolute floor lives on a per-model logprob
    scale -- on disk it excludes 0 plants on Qwen/Mistral (lp_A runs high) but
    EVERY plant on OLMo-2 (lp_A runs lower), so different models get scored on
    different subsets. This within-model robust-outlier floor adapts to each
    model's scale: it keeps a model's TYPICAL plants regardless of the absolute
    level and drops only plants that are anomalously low FOR THAT MODEL (the
    genuine can't-do-the-task / degenerate plants the gate is meant to remove).
    MAD (not mean/std) is used so the very low-outlier plants we want to exclude
    cannot inflate the spread and mask themselves.

    Returns the per-token lp_A threshold, or None if fewer than ``min_n`` finite
    values are available (too few to estimate a scale -> caller keeps all plants).
    k in {2.5..4} gives identical (zero) exclusions on the observable high-lp_A
    models; k=3.0 is the conventional extreme-outlier cutoff, chosen a priori."""
    xs = [x for x in la_values if x is not None and math.isfinite(x)]
    if len(xs) < min_n:
        return None
    med = statistics.median(xs)
    madn = 1.4826 * statistics.median([abs(x - med) for x in xs])
    return med - k * madn


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


# ---------------------------------------------------------------------------
# PER-CONVERSATION INCREMENTAL CHECKPOINTING (incident #38 fix).
#
# The cross-arch harness used to render ALL conversations in memory and write
# ONE final result JSON at the very end -- any interruption (timeout, OOM, pod
# death, ssh drop) lost the whole multi-hour render. This restores the earlier
# phases' incremental-save discipline (AGENTS.md "Results-in-repo rule"): as soon
# as a conversation is fully RENDERED and SCORED, its full contribution is written
# to a durable per-conversation file BEFORE the next conversation starts. The
# final per-model result is then re-assembled by POOLING those per-conv files, so
# it is a pure function of the checkpoints (byte-identical to the all-in-memory
# path). A killed run RESUMES from the checkpoints, losing <=1 conversation of
# scoring. This also makes cross-pod SPLITTING trivial (each pod checkpoints its
# conv range; pool the union), consistent with scripts/block_analysis.py.
#
# ATOMICITY: each checkpoint is written to a same-directory tmp file and
# os.replace()d into place (atomic rename on POSIX), so a crash mid-write can
# never leave a torn/partial checkpoint -- either the old file or the fully
# written new one is present, never a mix.
# ---------------------------------------------------------------------------
CHECKPOINT_SCHEMA = 1


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (tmp in the SAME directory +
    os.replace, so the rename is atomic and never crosses a filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def checkpoint_dir(out_dir: Path, model: str) -> Path:
    """Per-model checkpoint directory: results/cross_arch/<slug>/."""
    return out_dir / _slug(model)


def checkpoint_path(out_dir: Path, model: str, window_pos: int,
                    conv_id: str) -> Path:
    """Per-conversation checkpoint file. Keyed by BOTH the 0-based window position
    (for deterministic ordering + the position-based render seed) AND the true
    conversation id (audit-legible, stable across conv_limit changes)."""
    return (checkpoint_dir(out_dir, model)
            / f"conv_{window_pos:03d}__{_slug(str(conv_id))}.json")


def render_fingerprint(model_id: str, *, conv_start: int, native_render: bool,
                       native_max_reply: int, native_temp: float,
                       summary_request_sha256: str | None = None) -> dict:
    """Only the parameters that determine the GENERATED TEXT of a conversation --
    the native in-context replies AND (deterministically, fixed seed) the self-gen
    summary. A checkpoint's saved render (messages + summary) is REUSABLE by any
    future run whose render_fingerprint matches, EVEN IF its scoring params (alpha,
    graft region, floor, controls) differ -- that future run skips ALL generation
    and only forward-passes the saved text to rebuild KV + re-graft/re-score. This
    is the expensive part (generation ~16 min/conv); scoring is seconds.

    RANK-3 CAVEAT (Fable review 07-09, note-only while the design-v2 scaffold is
    FROZEN): this hashes the generation PARAMS but NOT the scaffold scenario
    CONTENT. If a scenario's text were edited under a STABLE id, a render-reuse
    would serve the STALE saved conversation. Safe today (scaffold frozen); before
    ANY champion/tuning render-reuse across scaffold edits, add a per-conv content
    hash (scenario system+turns+plants) to this fingerprint."""
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model_id,
        "conv_start": conv_start,
        "native_render": native_render,
        "native_max_reply": native_max_reply,
        "native_temp": native_temp,
        # The self-gen summary IS part of the render, so a compression-level change
        # (different summary request) MUST invalidate a saved summary. Without this
        # a level-2 run in a shared out-dir would silently reuse level-1's summary.
        "summary_request_sha256": summary_request_sha256,
    }


def run_fingerprint(model_id: str, *, conv_start: int, alpha_v: float, seed: int,
                    native_render: bool, native_max_reply: int,
                    native_temp: float, max_gold_tok: int,
                    task_competence_mode: str, task_competence_k: float,
                    task_lpa_floor: float, headroom_floor: float,
                    placebo_mode, alpha_sweep: bool, strong_prior: bool,
                    champion_scan: int, champion_regions,
                    ablate_qk_norm_flag: bool, ablate_lambda_values,
                    alpha0_tol: float, change_tol: float,
                    champion_cfg=None,
                    summary_request_sha256: str | None = None) -> dict:
    """Every parameter that can change a per-conversation NUMBER. A checkpoint is
    only reused when its fingerprint matches the current run's -- so a run with a
    different alpha / seed / gate / control config never silently pools stale or
    incompatible per-conv results (the checkpoint is recomputed instead)."""
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model_id,
        "conv_start": conv_start,
        "alpha_v": alpha_v,
        "seed": seed,
        "native_render": native_render,
        "native_max_reply": native_max_reply,
        "native_temp": native_temp,
        "max_gold_tok": max_gold_tok,
        "task_competence_mode": task_competence_mode,
        "task_competence_k": task_competence_k,
        "task_lpa_floor": task_lpa_floor,
        "headroom_floor": headroom_floor,
        "placebo_mode": placebo_mode,
        "alpha_sweep": bool(alpha_sweep),
        "strong_prior": bool(strong_prior),
        "champion_scan": int(champion_scan or 0),
        "champion_regions": list(champion_regions) if champion_regions else None,
        "ablate_qk_norm_flag": bool(ablate_qk_norm_flag),
        "ablate_lambda_values": (list(ablate_lambda_values)
                                 if ablate_lambda_values else None),
        "alpha0_tol": alpha0_tol,
        "change_tol": change_tol,
        # CHAMPION config changes the per-plant SCORE (which layers/heads/alpha
        # are grafted) but NOT the render/summary -> included in the SCORE
        # fingerprint only (never the render fingerprint), so a champion run
        # reuses the expensive render but never pools uniform-alpha scored rows.
        # Canonicalized as a sorted-key JSON string so it is stable across the
        # save/reload round-trip (int map keys -> str on disk).
        "champion_cfg": (json.dumps(champion_cfg, sort_keys=True)
                         if champion_cfg else None),
        # Compression sweep: the summary request determines the self-gen summary,
        # which changes every per-conv NUMBER -> a level change recomputes.
        "summary_request_sha256": summary_request_sha256,
    }


def _load_checkpoint(path: Path, fingerprint: dict, want_stage: str,
                     fp_key: str = "fingerprint"):
    """Return a loaded checkpoint dict iff it exists, parses, the requested
    fingerprint (``fp_key`` = "fingerprint" for exact SCORE reuse, or
    "render_fingerprint" for GENERATION-only reuse across differing scoring
    params) matches, and its stage is at least ``want_stage`` ("rendered" or
    "scored"). Any mismatch/corruption -> None (recompute), printed loudly. A torn
    tmp file is never seen here (atomic rename)."""
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            ck = json.load(fh)
    except Exception as e:  # noqa: BLE001
        print(f"  [checkpoint] IGNORING unreadable {path.name}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    if ck.get(fp_key) != fingerprint:
        print(f"  [checkpoint] IGNORING {path.name} for {fp_key}: mismatch "
              f"-> will recompute", flush=True)
        return None
    stage_rank = {"rendered": 1, "scored": 2}
    if stage_rank.get(ck.get("stage"), 0) < stage_rank[want_stage]:
        return None
    return ck


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
                        max_reply_tokens, temp, seed_base, conv_start=0,
                        batched_render=NATIVE_BATCHED_RENDER_DEFAULT,
                        native_batch=NATIVE_BATCH_DEFAULT,
                        only_positions=None, on_conv_rendered=None):
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
        raw_reply_text = tok.decode(reply_ids).strip()
        reply_text = raw_reply_text
        if capped:
            st["truncated"] += 1
            reply_text = trim_capped_reply(reply_text)
        if not reply_text:
            st["empty"] += 1
        # SAVE EVERY RENDER: retain the complete generated IDs/text even when the
        # canonical conversation body trims a capped reply to its last sentence.
        # Previously only the trimmed text survived, losing expensive generation.
        st["reply_records"].append({
            "n_tokens": len(reply_ids), "logprob_sum": lp_sum,
            "token_ids": [int(x) for x in reply_ids],
            "raw_text": raw_reply_text,
            "canonical_text": reply_text,
            "hit_token_cap": capped,
            "ended_on_eos": not capped,
            "trimmed_character_count": len(raw_reply_text) - len(reply_text),
            "raw_token_ids_sha256": hashlib.sha256(json.dumps(
                [int(x) for x in reply_ids], separators=(",", ":")).encode()).hexdigest(),
        })
        msgs.append({"role": "assistant", "content": reply_text})

        r_canon_user = st["_r_canon_user"]
        r_new = canonical_ids_any(tok, msgs, render_hf)
        assert r_new[: len(r_canon_user)] == r_canon_user, \
            f"{scenario['id']} turn {ti}: assistant block changed the prefix"
        canonical_block_ids = r_new[len(r_canon_user):]
        st["reply_records"][-1]["canonical_block_ids"] = canonical_block_ids
        st["reply_records"][-1]["canonical_block_ids_sha256"] = hashlib.sha256(
            json.dumps(canonical_block_ids, separators=(",", ":")).encode()).hexdigest()
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

    # HELD-OUT window: select ``conv_limit`` scenarios starting at OFFSET
    # ``conv_start`` (conv_start=0 == the legacy first-N slice, byte-identical).
    # enumerate() re-bases idx to 0..N-1 over the SELECTED window, so the per-conv
    # seed (seed_base + idx) is POSITION-based -- the held-out window is a
    # structural mirror of the tuned window (c13-as-first gets the same seed c01
    # got), and start=0 is unchanged. The true conv id (scenario["id"]) is what
    # flows into every result/label, never this positional idx.
    items = list(enumerate(scenarios[conv_start: conv_start + conv_limit]))
    # INCREMENTAL CHECKPOINTING / RESUME: render only the requested window
    # positions (``only_positions`` = set of 0-based idx into the window). idx is
    # preserved from the FULL enumeration so the position-based seed (seed_base +
    # idx) is UNCHANGED -- rendering a subset yields byte-identical output for each
    # rendered conv (greedy temp-0 decode is independent of batch composition).
    all_items = items
    if only_positions is not None:
        want = set(only_positions)
        items = [(idx, sc) for idx, sc in items if idx in want]

    def _emit_rendered(idx):
        """Persist/notify one freshly rendered conv (checkpoint stage=rendered) so
        a crash after this conv never re-renders it. Bounds render-loss to the
        in-flight chunk (<= native_batch)."""
        if on_conv_rendered is not None:
            conv, plants = results_by_idx[idx]
            on_conv_rendered(idx, conv, plants, reply_records_by_idx[idx])

    if batched_render:
        # Process convs in CHUNKS of native_batch so peak memory is bounded by N
        # conv-caches + the model (not all convs); each chunk frees before the next.
        for lo in range(0, len(items), native_batch):
            chunk = items[lo: lo + native_batch]
            _render_chunk(chunk)
            for idx, _sc in chunk:               # checkpoint each conv as rendered
                _emit_rendered(idx)
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
            _emit_rendered(idx)

    # assemble in scenario order (stats are order-independent, but keep it stable).
    # Only the rendered positions are returned here; the caller merges these with
    # any convs loaded from checkpoints. results_by_idx / reply_records_by_idx are
    # returned position-keyed so the caller can order + pool across both sources.
    specs: list = []
    reply_records: list = []
    for idx, _scenario in all_items:
        if idx not in results_by_idx:
            continue                             # not rendered this run (resumed)
        reply_records.extend(reply_records_by_idx[idx])
        conv, plants = results_by_idx[idx]
        if plants:
            specs.append((conv, plants))
    return specs, reply_records, results_by_idx, reply_records_by_idx


def run_model(model_id: str, data_dir: Path, out_dir: Path,
              fixed_summaries: dict | None, *,
              conv_limit: int, alpha_v: float, alpha0_tol: float,
              change_tol: float, max_gold_tok: int,
              trust_remote_code: bool, conv_start: int = 0,
              placebo_mode: str | None = None, alpha_sweep: bool = False,
              seed: int = ROBUST_SEED,
              strong_prior: bool = True, champion_scan: int = 0,
              champion_regions: list | None = None,
              native_render: bool = False, scenarios: list | None = None,
              native_max_reply: int = NATIVE_MAX_REPLY_DEFAULT,
              native_temp: float = NATIVE_TEMP_DEFAULT,
              headroom_floor: float = HEADROOM_FLOOR_DEFAULT,
              task_lpa_floor: float = TASK_LPA_FLOOR_DEFAULT,
              task_competence_mode: str = TASK_COMPETENCE_MODE_DEFAULT,
              task_competence_k: float = TASK_COMPETENCE_K_DEFAULT,
              ablate_qk_norm_flag: bool = False,
              ablate_lambda_values: list | None = None,
              champion_cfg: dict | None = None,
              selfgen_declared: bool = False) -> dict:
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
    # COMPRESSION SWEEP: SC_SUMMARY_LEVEL overrides the self-gen summary request
    # (unset/'realistic' -> unchanged _REQ). This single override covers EVERY
    # self-gen call site (all use _REQ). The realized compression ratio is
    # MEASURED per conv below and aggregated into doc["compression"].
    (_summ_level, _REQ, _summ_words, _summ_regime) = resolve_summary_level(_REQ)
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
        "conv_start": conv_start,   # HELD-OUT offset into the corpus (0 == first-N)
        "native_render": native_render,   # DESIGN v2: per-model in-context corpus
        # summary_source MUST distinguish DECLARED self-gen (SC_SELFGEN=1, the
        # redesign default -- a fixed foreign summary was found to SUPPRESS the
        # graft to null; FINDINGS 07-08) from the ACCIDENTAL missing-file fallback.
        # Both leave fixed_summaries=None, so the flag is threaded in explicitly --
        # otherwise an intended self-gen run mislabels itself "fallback NOT
        # comparable" (which is exactly what confused the champion-validation read).
        "summary_source": (
            "PER-MODEL SELF-GEN on NATIVE in-context corpus (design v2)"
            if native_render else
            "PER-MODEL SELF-GEN summary (DECLARED SC_SELFGEN=1; redesign default -- "
            "a fixed foreign summary suppresses the graft)"
            if selfgen_declared else
            "FIXED external (shared across models)"
            if fixed_summaries is not None
            else "PER-MODEL fallback (NO fixed_summaries file AND SC_SELFGEN unset "
                 "-- ACCIDENTAL self-gen, NOT cross-model comparable)"),
        "reply_covariates": None,  # v2.1 covariate: reply length + info-content
        "gates": None,             # v2.1 gate summary (headroom / task-competence)
        "skipped_convs": [],
        "n_plants": 0,
        "qk_norm_ablated": False,       # SC_ABLATE_QK_NORM: was QK-norm removed?
        "n_qk_modules_ablated": 0,      # how many q_norm/k_norm modules replaced
        "ablate_lambda": None,          # SC_ABLATE_LAMBDA: graded dose-response block
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
    # SC_LOAD_DTYPE (default bfloat16): compute/activation dtype. For a
    # PRE-QUANTIZED 4-bit repo (GPTQ/AWX/AutoRound) transformers reads the repo's
    # quantization_config and keeps int4 weights; this dtype is only the
    # activation dtype and normally coexists with 4-bit fine. If a specific quant
    # integration rejects an explicit dtype, set SC_LOAD_DTYPE=auto. The KV cache
    # (the graft surface) stays bf16/fp16 regardless -> the value graft is
    # quant-agnostic. VRAM residency is the load positive-control (printed below).
    _load_dtype_env = os.environ.get("SC_LOAD_DTYPE", "bfloat16")
    _load_dtype = torch.bfloat16 if _load_dtype_env == "bfloat16" else _load_dtype_env
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=_load_dtype, device_map="auto",
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
        # BINARY and GRADED ablation are mutually exclusive (the binary one swaps
        # to Identity at load and breaks generation -- exactly what the graded one
        # fixes). Refuse rather than silently produce a broken/empty run.
        if ablate_qk_norm_flag and ablate_lambda_values:
            doc.update(
                status="ERROR",
                reason="SC_ABLATE_QK_NORM=1 and SC_ABLATE_LAMBDA are mutually "
                       "exclusive: the binary ablation makes the model GENERATE "
                       "degraded (the failure the graded lambda dose-response was "
                       "built to avoid). Pick one.")
            return doc
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
        # GRADED QK-NORM ABLATION (SC_ABLATE_LAMBDA) -- decoupled from generation.
        # Wrap every QK-norm module in a QKLambdaNorm at lambda=1.0 (== clean, so
        # the native render + self-gen summary below are byte-clean); the plant
        # loop toggles lambda<1.0 ONLY around the teacher-forced SCORING passes.
        n_qk_wrapped = 0
        if ablate_lambda_values:
            n_qk_wrapped, wrapped_names = install_qk_lambda(model)
            if n_qk_wrapped == 0:
                doc.update(
                    status="ERROR",
                    reason="SC_ABLATE_LAMBDA requested but no QK-norm modules "
                           "found to wrap (model has no q_norm/k_norm/query|key "
                           "layernorm modules); refusing to run a graded ablation "
                           "with nothing perturbed")
                return doc
            set_qk_lambda(model, 1.0)   # generation/render + primary scoring stay clean
            print(f"SC_ABLATE_LAMBDA={ablate_lambda_values} -> wrapped "
                  f"{n_qk_wrapped} QK-norm modules with QKLambdaNorm (lambda=1.0 "
                  f"now; perturbed ONLY around scoring). e.g. {wrapped_names[:3]} "
                  f"... render/self-gen run CLEAN at lambda=1.0.", flush=True)
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED", reason=f"OOM on load: {e}")
        return doc
    except Exception as e:  # noqa: BLE001
        doc.update(status="UNSUPPORTED", reason=f"model load failed: {type(e).__name__}: {e}")
        return doc

    # ---- PROVENANCE MANIFEST (runtime-read: dtype from a real parameter tensor,
    # quantization from the loaded config, git from subprocess) -- so a wrong
    # model/dtype/intervention/condition run is self-evident in every result +
    # per-conv checkpoint + the run-level manifest.json.
    import provenance as prov  # noqa: PLC0415
    _champ_path = os.environ.get("SC_CHAMPION_CONFIG")
    if champion_cfg:
        _alpha_field = champion_cfg.get("alpha_map") or champion_cfg.get("alpha")
        _champ_label = champion_cfg.get("label")
        _champ_hash = (prov.sha256_file(_champ_path)
                       or prov.sha256_text(json.dumps(champion_cfg,
                                                      sort_keys=True)))
        _arm = "E-champion"
    else:
        _alpha_field = alpha_v
        _champ_label = None
        _champ_hash = None
        _arm = "E (scalar value-graft)"
    _mp = prov.capture_model_provenance(model, model_id_hint=model_id)
    manifest = prov.build_manifest(
        model_provenance=_mp,
        dtype_env=os.environ.get("SC_LOAD_DTYPE", "bfloat16"),
        harness="src/cross_arch_probe.py",
        intervention={
            "arm": _arm,
            "graft_type": "value",          # keys neutral (alpha_K=0)
            "alpha": _alpha_field,
            "champion_config_path": _champ_path if champion_cfg else None,
            "champion_config_sha256": _champ_hash,
            "champion_label": _champ_label,
            "placebo_mode": placebo_mode,   # None for the real graft
            "alignment": "difflib positional-within-region (tail+summary regions)",
            "qk_norm_ablated": bool(ablate_qk_norm_flag),
        },
        metric=prov.METRIC_CHAT_RAW_EB,
        condition={
            # COMPRESSION SWEEP: the summary request now varies by SC_SUMMARY_LEVEL
            # (ultra/brief/medium/realistic/prod). 'realistic' == the current null
            # condition; the measured ratio is filled after the conv loop.
            "summary_kind": (f"self-gen level={_summ_level} ({_summ_regime})"
                             if (native_render or selfgen_declared)
                             else doc["summary_source"]),
            "summary_request_sha256": prov.sha256_text(_REQ),
            "summary_source": doc["summary_source"],
            "summary_selfgen_declared": bool(selfgen_declared or native_render),
            "summary_compression_level": _summ_level,
            "summary_compression_regime": _summ_regime,
            "summary_target_words": _summ_words,
            "measured_compression_ratio_mean": None,  # filled after the conv loop
        },
        corpus={
            "name": ("native in-context render (design v2)" if native_render
                     else str(data_dir)),
            "split": f"conv_start={conv_start} conv_limit={conv_limit}",
            "n": conv_limit,
            "instance_ids": None,           # filled after the conv loop
        },
        extra={"seed": seed, "alpha_sweep": bool(alpha_sweep),
               "champion_scan": int(champion_scan or 0)},
    )
    doc["_manifest"] = manifest
    print("MANIFEST model.repo_id=%s dtype=%s quant=%s arm=%s summary=%s git=%s"
          % (manifest["model"]["repo_id"], manifest["load"]["dtype"],
             manifest["load"]["quantization"], _arm, doc["summary_source"],
             manifest["code"]["git_commit"]), flush=True)
    prov.write_run_manifest(out_dir, manifest,
                            name=f"manifest__{_slug(model_id)}.json")

    try:
        family = detect_template_family(tok)
        # LOUD provenance: name the resolved chat-template family and which
        # message-boundary path it takes. "unknown" is NOT an error -- it routes
        # to the template-agnostic prefix-rendering boundary detector (verified
        # correct for OLMo-2, whose <|user|>/<|assistant|> role markers are
        # multi-token and NOT special tokens, so the qwen <|im_start|> scan does
        # not apply) -- but a SILENT "unknown" previously hid which path ran.
        _boundary = ("qwen im_start scan" if family == "qwen"
                     else "prefix-rendering (template-agnostic)")
        print(f"  [template] family={family} boundary={_boundary} "
              f"special_tokens={tok.all_special_tokens}", flush=True)
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
    # DIAGNOSTIC breakdown of WHY plants may fail to score, so the terminal
    # "no plants scored" ERROR names the ACTUAL cause instead of guessing
    # "empty alignment?" (which conflates three unrelated drop reasons and sent a
    # debugging effort chasing a non-existent OLMo-2 region-detection bug: the
    # A<->B alignment is CPU-verified non-empty for OLMo-2's template; the real
    # per-architecture trap is the ABSOLUTE task_lpa_floor excluding every plant
    # when a model's gold logprobs sit on a lower scale). Defined BEFORE the try
    # so they are readable in the terminal reason after the loop.
    empty_align_convs: list[str] = []   # convs skipped: build_alignment -> no pairs
    short_gold_drops = 0                 # plants dropped: gold continuation < 2 tok
    n_plants = 0
    alpha0_diffs: list[float] = []
    graft_diffs: list[float] = []       # |lp_E - lp_B|
    graft_signed: list[float] = []      # lp_E - lp_B (direction toward target)
    pre_gaps: list[float] = []          # lp_A - lp_B (meaning lost)
    all_gc: list[float] = []            # gap_closure over all plants (for CI)
    identity_diffs: list[float] = []    # CONTROL #2: |lp_identity - lp_A| per conv
    traces: list[dict] = []             # CONTROL #4: raw per-probe traces
    # GRADED QK-NORM ABLATION (SC_ABLATE_LAMBDA): per-lambda per-plant readout rows.
    # {lambda: [{conversation_id, category, lp_A, raw_EB, cleared}, ...]}. Only
    # populated for lambda-mode runs; each plant scored on the frozen lambda=1
    # snapshots with lambda toggled ONLY around the tf() scoring passes.
    lambda_values = list(ablate_lambda_values) if ablate_lambda_values else []
    per_lambda_rows: dict[float, list] = {lam: [] for lam in lambda_values}
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

    # ---- INCREMENTAL CHECKPOINTING config (incident #38 fix) ------------------
    # resume_enabled (default): reuse any valid per-conv checkpoints and only
    # render/score the MISSING convs. SC_CHECKPOINT_FRESH=1 forces a fresh run
    # (ignore + overwrite existing checkpoints). The fingerprint gates reuse to
    # runs with byte-identical numeric params; shuffle_probe placebo carries live
    # cross-conv state so resume-SKIP is disabled for it (checkpoints still written).
    checkpoint_fresh = os.environ.get("SC_CHECKPOINT_FRESH", "0") in (
        "1", "true", "True", "yes")
    resume_enabled = not checkpoint_fresh
    ck_fp = run_fingerprint(
        model_id, conv_start=conv_start, alpha_v=alpha_v, seed=seed,
        native_render=native_render, native_max_reply=native_max_reply,
        native_temp=native_temp, max_gold_tok=max_gold_tok,
        task_competence_mode=task_competence_mode,
        task_competence_k=task_competence_k, task_lpa_floor=task_lpa_floor,
        headroom_floor=headroom_floor, placebo_mode=placebo_mode,
        alpha_sweep=alpha_sweep, strong_prior=strong_prior,
        champion_scan=champion_scan, champion_regions=champion_regions,
        ablate_qk_norm_flag=ablate_qk_norm_flag,
        ablate_lambda_values=ablate_lambda_values,
        alpha0_tol=alpha0_tol, change_tol=change_tol,
        champion_cfg=champion_cfg,
        summary_request_sha256=prov.sha256_text(_REQ))

    # ---- CHAMPION graft config (per-layer alpha-map OR per-head slot mask) ----
    # Resolve the tuned config ONCE into the exact args blend_values takes. When
    # no champion config is given this is the historical scalar-alpha behaviour
    # (graft_alpha=alpha_v, graft_head_map=None). VALUE-ONLY either way (keys
    # never touched by blend_values => alpha_K=0). Applied IDENTICALLY to the
    # real graft E and the placebo graft below, so E_champion and
    # placebo_champion differ ONLY in the SOURCE values, never the slots/alpha.
    graft_alpha = alpha_v
    graft_head_map = None
    if champion_cfg:
        if "alpha_map" in champion_cfg:
            graft_alpha = {int(k): float(v)
                           for k, v in champion_cfg["alpha_map"].items()}
            graft_head_map = None
        elif "head_map" in champion_cfg:
            graft_alpha = float(champion_cfg.get("alpha", alpha_v))
            graft_head_map = {int(k): [int(h) for h in v]
                              for k, v in champion_cfg["head_map"].items()}
        doc["champion_cfg"] = champion_cfg
        doc["champion_label"] = champion_cfg.get("label")
        print(f"CHAMPION graft ACTIVE: label={champion_cfg.get('label')} "
              f"mode={'alpha_map' if 'alpha_map' in champion_cfg else 'head_map'} "
              f"(value-only, alpha_K=0); applied to E and placebo alike.",
              flush=True)
    # RENDER fingerprint (generated TEXT only) -- a checkpoint's saved render +
    # self-gen summary is REUSED whenever THIS matches, even if the SCORE
    # fingerprint (ck_fp: alpha/region/floor/controls) differs. So a future run
    # with a different alpha or graft region reuses the expensive generation and
    # only re-forward-passes to re-score. ck_fp gates the exact SCORE replay.
    render_fp = render_fingerprint(
        model_id, conv_start=conv_start, native_render=native_render,
        native_max_reply=native_max_reply, native_temp=native_temp,
        summary_request_sha256=prov.sha256_text(_REQ))
    resume_skip_ok = resume_enabled and placebo_mode != "shuffle_probe"
    # self-gen summaries reused from checkpoints (keyed by conv id) so a resumed /
    # render-reusing run regenerates NO summary. Populated during render assembly
    # and the relative pre-scan; consumed by _process_conv (which regenerates only
    # on a cache miss). Deterministic (fixed seed) so reuse is byte-identical.
    summary_cache: dict = {}
    # RELATIVE-mode pooled-floor correctness (Fable review, 07-09): the relative
    # pre-scan computes lp_A for EVERY plant of EVERY conv INCLUDING empty-alignment
    # convs (it never builds A<->B pairs), but the main loop returns EARLY on an
    # empty-align conv so it contributes ZERO per_cat/task_excluded rows. Deriving a
    # checkpoint's scan_lpa from those rows would therefore DROP empty-align convs'
    # lp_A on resume -> a drifted median-k*MADN floor -> silently different plant
    # gating -> changed raw_EB/CI. So the checkpoint stores the PRE-SCAN's OWN
    # per-conv lp_A here (keyed by window position), independent of the scoring
    # rows, making the resumed floor pool byte-identical to the fresh pool.
    prescan_lpa_by_pos: dict[int, list] = {}

    def _ckpath(pos, cid):
        return checkpoint_path(out_dir, model_id, pos, cid)

    # spec_positions[i] = the 0-based WINDOW position of specs[i]. The window
    # position keys the checkpoint file + (in native mode) the render seed; it is
    # DISTINCT from the enumerate index _ci over specs (which drives the placebo
    # per-conv seed and MUST stay 0..len(specs)-1, unchanged, for numeric identity).
    spec_positions: list[int] = []
    reply_records_by_pos: dict[int, list] = {}

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
        _window = scenarios[conv_start: conv_start + conv_limit]
        _n_window = len(_window)
        # RESUME / RENDER-REUSE: which window positions already have a durable
        # checkpoint whose RENDER fingerprint matches? Those skip ALL generation
        # (native replies AND self-gen summary) -- we reuse the saved conversation
        # text + summary and only re-forward-pass to score. Matching is on
        # render_fp (NOT ck_fp), so a differing alpha/region still reuses the
        # render.
        rendered_ck: dict[int, dict] = {}
        missing_positions: list[int] = []
        for _pos in range(_n_window):
            _cid = _window[_pos].get("id")
            _ck = (_load_checkpoint(_ckpath(_pos, _cid), render_fp, "rendered",
                                    fp_key="render_fingerprint")
                   if resume_enabled else None)
            if _ck is not None:
                rendered_ck[_pos] = _ck
                if _ck.get("summary_text") is not None:   # reuse self-gen summary
                    summary_cache[_cid] = _ck["summary_text"]
            else:
                missing_positions.append(_pos)
        print(f"  [native-render] window={_n_window} convs "
              f"(conv_start={conv_start}); {len(rendered_ck)} render(s) reused "
              f"from checkpoints ({sum(1 for p in rendered_ck.values() if p.get('summary_text') is not None)} "
              f"with saved summary), {len(missing_positions)} to generate "
              f"(temp={native_temp}, max_reply={native_max_reply}, "
              f"resume={resume_enabled})", flush=True)

        # NOTE on byte-identity of RE-rendered convs (batched decode): a conv that
        # ALREADY has a checkpoint is reused from its saved TEXT -> byte-identical
        # forever. A conv LOST before its rendered checkpoint is re-generated on
        # resume; under batched decode (SC_BATCHED_RENDER=1) greedy token ids can
        # depend on a chunk's batch COMPOSITION (padding/accumulation), so a
        # re-render only matches the original if the SAME conv-set renders together
        # (whole-chunk loss -> same grouping -> identical; a partial-chunk loss can
        # differ at the token level). Set SC_BATCHED_RENDER=0 for byte-exact resume
        # of a partially-lost chunk. Checkpointing itself is exact regardless
        # (CPU-validated: unbatched resume/re-render/pool are byte-identical).
        def _on_rendered(idx, conv, plants, rr):
            """Persist one freshly rendered conv as stage=rendered BEFORE the next
            chunk, so a crash never re-renders it (render-loss bound = 1 chunk).
            Carries BOTH fingerprints so it is reusable for render (any scoring
            config) and, once upgraded to scored, for exact score replay."""
            _atomic_write_json(_ckpath(idx, conv.get("id")), {
                "schema": CHECKPOINT_SCHEMA, "stage": "rendered",
                "fingerprint": ck_fp, "render_fingerprint": render_fp,
                "window_pos": idx, "conv_id": conv.get("id"),
                "render": {"conv": conv, "plants": plants, "reply_records": rr}})

        try:
            (_r_specs, _r_reply, r_results_by_idx,
             r_reply_by_idx) = native_render_specs(
                model, tok, family, scenarios, conv_limit,
                max_reply_tokens=native_max_reply, temp=native_temp,
                seed_base=1000, conv_start=conv_start,
                only_positions=missing_positions, on_conv_rendered=_on_rendered)
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            doc.update(status="UNSUPPORTED", reason=f"OOM during native render: {e}")
            return doc
        except Exception as e:  # noqa: BLE001
            doc.update(status="ERROR",
                       reason=f"native render failed: {type(e).__name__}: {e}\n"
                              f"{traceback.format_exc()}")
            return doc
        # Assemble the FULL corpus in window-position order, merging freshly
        # rendered convs with checkpoint-loaded ones. reply_records pools ALL
        # positions (incl. 0-plant convs) so reply_covariates is byte-identical.
        specs = []
        for _pos in range(_n_window):
            if _pos in r_results_by_idx:
                conv, plants = r_results_by_idx[_pos]
                rr = r_reply_by_idx[_pos]
            else:
                _ck = rendered_ck[_pos]
                conv = _ck["render"]["conv"]
                plants = _ck["render"]["plants"]
                rr = _ck["render"]["reply_records"]
            reply_records.extend(rr)
            reply_records_by_pos[_pos] = rr
            if plants:
                specs.append((conv, plants))
                spec_positions.append(_pos)
        doc["reply_covariates"] = reply_covariates(reply_records)
    else:
        specs = collect_specs(data_dir, conv_limit, conv_start=conv_start)
        # pre-rendered corpus: window position == enumerate index (all have plants).
        spec_positions = list(range(len(specs)))
    if not specs:
        doc.update(status="ERROR", reason="no usable plants in corpus subset")
        return doc

    # Record the ACTUAL scored conversation ids in the manifest (+ refresh the
    # run-level manifest.json now that the exact instance set is known).
    try:
        manifest["corpus"]["instance_ids"] = [c.get("id") for c, _ in specs]
        manifest["corpus"]["n_scored"] = len(specs)
        prov.write_run_manifest(out_dir, manifest,
                                name=f"manifest__{_slug(model_id)}.json")
    except Exception:  # noqa: BLE001
        pass

    # v2.1 task-competence gate accounting (per category).
    task_excluded: dict[str, list] = {c: [] for c in CATS}

    # ---- v2.1 GATE #3: RELATIVE (per-model) competence floor -----------------
    # Pre-registered 2026-07-08 (PREREGISTRATION.md "GATE #3 AMENDMENT"). The
    # ABSOLUTE floor (-8.0) confounds cross-model comparison; in RELATIVE mode we
    # PRE-SCAN this model's OWN gold-lp_A distribution and set a robust-outlier
    # floor (median - k*MADN). lp_A depends ONLY on the FULL-context A snapshot
    # (NOT on the summary or the graft), so the scan is well-defined and the value
    # it sees is byte-identical to the one the gate sees in the main loop below.
    # summary_cache carries the (checkpoint-reused or once-computed) per-conv
    # summary into the main loop so it is not regenerated. ABSOLUTE mode (default)
    # runs NO pre-scan -> the existing code path and every already-scored number
    # are byte-unchanged. (summary_cache was initialized above so render-reused
    # summaries are already present for both the pre-scan and the main loop.)
    active_floor = task_lpa_floor
    if task_competence_mode == "relative":
        scan_lpa: list = []
        for _si, (_cv, _pls) in enumerate(specs):
            # RESUME: a scored checkpoint already carries this conv's per-plant
            # lp_A (== what the prescan would recompute -- lp_A is deterministic
            # and independent of the summary); reuse it so a resumed run does not
            # re-render/re-score done convs just to rebuild the pooled floor. The
            # POOLED floor is therefore identical whether or not any conv resumed.
            _pos_si = spec_positions[_si]
            _psck = (_load_checkpoint(_ckpath(_pos_si, _cv["id"]),
                                      ck_fp, "scored") if resume_skip_ok else None)
            if _psck is not None:
                _conv_lpa = list(_psck.get("scan_lpa") or [])
                scan_lpa.extend(_conv_lpa)
                prescan_lpa_by_pos[_pos_si] = _conv_lpa
                continue
            _msgs = _cv["messages"][:-1]
            _tsm = _cv["sections"]["middle_end_msg"]
            # RENDER-REUSE: a checkpoint's saved self-gen summary is already in
            # summary_cache -> reuse it (regenerate NO summary). The summary is
            # deterministic (fixed seed), so reuse is byte-identical to a fresh
            # generation; lp_A does not depend on it either way.
            _st = summary_cache.get(_cv["id"])
            if _st is None:
                _st = fixed_summaries.get(_cv["id"]) if fixed_summaries else None
                if fixed_summaries is not None and not _st:
                    continue                   # skip logged by the main loop
                if _st is None:
                    _st = generate_summary_hf(
                        model, tok, _msgs, request=_REQ)["text"]
            summary_cache[_cv["id"]] = _st
            _ctx = build_token_context(tok, family, _msgs, _st, _tsm)
            _ids = _ctx["ids"]
            _a_snap = force_prefill(_ids)
            _conv_lpa = []
            for _pl in _pls:
                _tgt = tok(str(_pl["gold"]).strip(),
                           add_special_tokens=False).input_ids[:max_gold_tok]
                if len(_tgt) < 2:
                    continue
                _per = []
                for _probe in (_pl.get("probes") or [_pl["probe"]]):
                    _full = render_hf(
                        tok, _msgs + [{"role": "user", "content": _probe}], True)
                    _cn = canonical_ids_any(tok, _msgs, render_hf)
                    _per.append(tf(_a_snap, _full[len(_cn):] + _tgt[:-1],
                                   _tgt, len(_ids)))
                _conv_lpa.append(_mean(_per))
            # store THIS conv's own pre-scan lp_A (all non-short-gold plants,
            # empty-align convs included) so its checkpoint's scan_lpa == the fresh
            # floor contribution -- see prescan_lpa_by_pos note above.
            scan_lpa.extend(_conv_lpa)
            prescan_lpa_by_pos[_pos_si] = _conv_lpa
        _rel = relative_competence_floor(scan_lpa, k=task_competence_k)
        if _rel is not None:
            active_floor = _rel
        print(f"  [gate3-relative] n_scanned={len(scan_lpa)} "
              f"relative_floor={active_floor:.3f} (k={task_competence_k}; "
              f"absolute_would_be={task_lpa_floor}) -- per-model robust-outlier "
              f"floor keeps high-lp_A models' plant sets intact, adapts to "
              f"low-lp_A models (e.g. OLMo).", flush=True)

    # ---- PER-CONVERSATION CHECKPOINT machinery (incident #38) -----------------
    # Every accumulator below is an append-only list (per conv) or a scalar
    # counter, so ONE conversation's entire contribution is the SLICE appended /
    # the amount incremented during its iteration. _snap() records the pre-conv
    # state; _delta() extracts exactly this conv's contribution; _replay_conv()
    # re-applies a loaded contribution WITHOUT re-computing. The scoring code
    # itself is UNTOUCHED -- these only OBSERVE and REPLAY it -- so a resumed /
    # pooled run is byte-identical to the all-in-memory run.
    def _snap():
        return {
            "per_cat": {c: len(per_cat[c]) for c in CATS},
            "task_excluded": {c: len(task_excluded[c]) for c in CATS},
            "placebo": {c: len(per_cat_placebo[c]) for c in CATS},
            "alpha": {a: {c: len(per_cat_alpha[a][c]) for c in CATS}
                      for a in per_cat_alpha},
            "lambda": {lam: len(per_lambda_rows[lam]) for lam in per_lambda_rows},
            "config": {lbl: len(per_config_rows[lbl]) for lbl in per_config_rows},
            "traces": len(traces), "alpha0_diffs": len(alpha0_diffs),
            "graft_diffs": len(graft_diffs), "graft_signed": len(graft_signed),
            "pre_gaps": len(pre_gaps), "all_gc": len(all_gc),
            "identity_diffs": len(identity_diffs),
            "strong_prior_rows": len(strong_prior_rows),
            "skipped_convs": len(skipped_convs),
            "empty_align_convs": len(empty_align_convs),
            "n_plants": n_plants, "short_gold_drops": short_gold_drops,
            "value_align_cnt": value_align_cnt,
            "uniform_seconds": uniform_seconds,
            "champion_seconds": champion_seconds,
            "value_align_sum": (list(value_align_sum)
                                if value_align_sum is not None else None),
            "champion_set": champion_region_layers is not None,
        }

    def _delta(b):
        d = {
            "per_cat": {c: per_cat[c][b["per_cat"][c]:] for c in CATS},
            "task_excluded": {c: task_excluded[c][b["task_excluded"][c]:]
                              for c in CATS},
            "traces": traces[b["traces"]:],
            "alpha0_diffs": alpha0_diffs[b["alpha0_diffs"]:],
            "graft_diffs": graft_diffs[b["graft_diffs"]:],
            "graft_signed": graft_signed[b["graft_signed"]:],
            "pre_gaps": pre_gaps[b["pre_gaps"]:],
            "all_gc": all_gc[b["all_gc"]:],
            "identity_diffs": identity_diffs[b["identity_diffs"]:],
            "strong_prior_rows": strong_prior_rows[b["strong_prior_rows"]:],
            "skipped_convs": skipped_convs[b["skipped_convs"]:],
            "empty_align_convs": empty_align_convs[b["empty_align_convs"]:],
            "n_plants": n_plants - b["n_plants"],
            "short_gold_drops": short_gold_drops - b["short_gold_drops"],
            "value_align_cnt": value_align_cnt - b["value_align_cnt"],
            "uniform_seconds": uniform_seconds - b["uniform_seconds"],
            "champion_seconds": champion_seconds - b["champion_seconds"],
        }
        if placebo_mode is not None:
            d["placebo"] = {c: per_cat_placebo[c][b["placebo"][c]:] for c in CATS}
        if alpha_sweep:
            d["alpha"] = {str(a): {c: per_cat_alpha[a][c][b["alpha"][a][c]:]
                                   for c in CATS} for a in per_cat_alpha}
        if lambda_values:
            d["lambda"] = {str(lam): per_lambda_rows[lam][b["lambda"][lam]:]
                           for lam in per_lambda_rows}
        cfg = {lbl: per_config_rows[lbl][b["config"].get(lbl, 0):]
               for lbl in per_config_rows}
        if any(cfg.values()):
            d["config"] = cfg
        if (value_align_cnt - b["value_align_cnt"]) > 0 and value_align_sum:
            d["value_align"] = ([value_align_sum[i] - b["value_align_sum"][i]
                                 for i in range(len(value_align_sum))]
                                if b["value_align_sum"] is not None
                                else list(value_align_sum))
        if champion_region_layers is not None and not b["champion_set"]:
            d["champion_region_layers"] = [list(r) for r in champion_region_layers]
            d["champion_configs"] = [[lbl, sorted(rset)]
                                     for lbl, rset in champion_configs]
        return d

    def _replay_conv(p):
        nonlocal n_plants, short_gold_drops, value_align_cnt
        nonlocal uniform_seconds, champion_seconds, value_align_sum
        nonlocal champion_configs, champion_region_layers, per_config_rows
        for c in CATS:
            per_cat[c].extend(p["per_cat"][c])
            task_excluded[c].extend(p["task_excluded"][c])
        traces.extend(p["traces"])
        alpha0_diffs.extend(p["alpha0_diffs"])
        graft_diffs.extend(p["graft_diffs"])
        graft_signed.extend(p["graft_signed"])
        pre_gaps.extend(p["pre_gaps"])
        all_gc.extend(p["all_gc"])
        identity_diffs.extend(p["identity_diffs"])
        strong_prior_rows.extend(p["strong_prior_rows"])
        skipped_convs.extend(p["skipped_convs"])
        empty_align_convs.extend(p["empty_align_convs"])
        n_plants += p["n_plants"]
        short_gold_drops += p["short_gold_drops"]
        value_align_cnt += p["value_align_cnt"]
        uniform_seconds += p["uniform_seconds"]
        champion_seconds += p["champion_seconds"]
        if placebo_mode is not None and "placebo" in p:
            for c in CATS:
                per_cat_placebo[c].extend(p["placebo"][c])
        if alpha_sweep and "alpha" in p:
            for a in per_cat_alpha:
                for c in CATS:
                    per_cat_alpha[a][c].extend(p["alpha"][str(a)][c])
        if lambda_values and "lambda" in p:
            for lam in per_lambda_rows:
                per_lambda_rows[lam].extend(p["lambda"][str(lam)])
        if "champion_region_layers" in p and champion_region_layers is None:
            champion_region_layers = [tuple(r)
                                      for r in p["champion_region_layers"]]
            champion_configs = [(lbl, frozenset(rset))
                                for lbl, rset in p["champion_configs"]]
            per_config_rows = {lbl: [] for lbl, _ in champion_configs}
        for lbl, rows in p.get("config", {}).items():
            per_config_rows.setdefault(lbl, []).extend(rows)
        if p.get("value_align") is not None:
            if value_align_sum is None:
                value_align_sum = list(p["value_align"])
            else:
                for i in range(len(value_align_sum)):
                    value_align_sum[i] += p["value_align"][i]

    # COMPRESSION SWEEP: per-conv realized ratio (clean-summary tokens / full-
    # context tokens). Quantifies "aggressive vs realistic" as a NUMBER; aggregated
    # into doc["compression"] + the manifest after the loop.
    compression_rows: list = []

    def _measure_compression(conv, summary_text):
        """Realized compression ratio for one conv: the CLEAN self-gen summary
        (what actually lands in the compacted B context, reasoning block stripped)
        over the FULL pre-compaction context tokens. Best-effort; None on any
        tokenization failure (never blocks the checkpoint)."""
        try:
            full_ctx = len(canonical_ids_any(
                tok, conv["messages"][:-1], render_hf))
            clean = strip_reasoning_block(summary_text) or summary_text
            summ_tok = len(tok(clean, add_special_tokens=False).input_ids)
            ratio = (summ_tok / full_ctx) if full_ctx else None
            return {"conv_id": conv.get("id"), "summary_tokens": summ_tok,
                    "full_ctx_tokens": full_ctx, "ratio": ratio}
        except Exception:  # noqa: BLE001
            return {"conv_id": conv.get("id"), "summary_tokens": None,
                    "full_ctx_tokens": None, "ratio": None}

    def _write_scored_ck(pos, cid, conv, plants, summary_text, base):
        """Atomically persist this conv's FULL contribution (render spec +
        self-gen summary + per-plant lp_A for the pooled floor + the accumulator
        delta) as stage=scored, BEFORE the next conv starts."""
        payload = _delta(base)
        comp = _measure_compression(conv, summary_text)
        compression_rows.append(comp)
        # scan_lpa = the RELATIVE pre-scan's OWN per-conv lp_A (all non-short-gold
        # plants, INCLUDING empty-align convs whose main-loop early-return leaves
        # per_cat/task_excluded empty). Deriving from those rows would silently drop
        # empty-align convs from the resumed floor pool (Fable review 07-09). In
        # ABSOLUTE mode there is no pre-scan and scan_lpa is never read on resume,
        # so [] is correct there. For NON-empty-align convs the pre-scan lp_A equals
        # the main-loop la, so this is byte-identical for them.
        scan = list(prescan_lpa_by_pos.get(pos, []))
        _atomic_write_json(_ckpath(pos, cid), {
            "schema": CHECKPOINT_SCHEMA, "stage": "scored", "fingerprint": ck_fp,
            "render_fingerprint": render_fp, "window_pos": pos, "conv_id": cid,
            "render": {"conv": conv, "plants": plants,
                       "reply_records": reply_records_by_pos.get(pos, [])},
            "summary_text": summary_text, "scan_lpa": scan, "payload": payload,
            "compression": comp, "_manifest": manifest})

    try:
        def _process_conv(conv, plants, _ci):
            nonlocal n_plants, short_gold_drops, value_align_cnt
            nonlocal uniform_seconds, champion_seconds, value_align_sum
            nonlocal champion_configs, champion_region_layers, per_config_rows
            nonlocal prev_summ_snap, prev_old_idx
            print(f"  [progress] conv {_ci+1}/{len(specs)} ({conv['id']}) "
                  f"n_plants_so_far={n_plants}", flush=True)
            msgs = conv["messages"][:-1]
            tsm = conv["sections"]["middle_end_msg"]

            # RELATIVE mode pre-scan already computed (and cached) this conv's
            # summary; reuse it so it is not regenerated. In ABSOLUTE mode the
            # cache is empty and this is the original code path exactly.
            summary_text = summary_cache.get(conv["id"])
            if summary_text is None:
                summary_text = (fixed_summaries.get(conv["id"])
                                if fixed_summaries else None)
                if fixed_summaries is not None and not summary_text:
                    # Missing fixed summary for THIS conv -> skip the conv (logged),
                    # do NOT abort the model. Other convs still contribute.
                    skipped_convs.append(conv["id"])
                    print(f"  SKIP {conv['id']}: no entry in fixed summaries file",
                          flush=True)
                    return summary_text          # conv-level: checkpoint + next
                if summary_text is None:
                    # No fixed file at all: per-model fallback (non-comparable).
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
                # nothing to graft in this conv -> skip (not a model failure).
                # Recorded so the terminal reason can distinguish a genuine
                # alignment miss from competence-gate exclusions.
                empty_align_convs.append(conv["id"])
                return summary_text              # conv-level: checkpoint + next

            a_snap = force_prefill(ids)
            b_snap = force_prefill(b_ids)

            # E (treatment) and E0 (alpha=0 plumbing check) grafts, shared by all
            # plants in this conv.
            e_snap = blend_values(b_snap, summ["snapshot"], pairs, graft_alpha,
                                  head_map=graft_head_map)
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
                e_placebo_snap = blend_values(b_snap, pl_source, pl_pairs,
                                              graft_alpha, head_map=graft_head_map)

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
                    short_gold_drops += 1
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
                if not task_competence_ok(la, active_floor):
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

                # ---- GRADED QK-NORM ABLATION (SC_ABLATE_LAMBDA) dose-response ----
                # Re-score THIS plant's graft (lp_E), compacted baseline (lp_B) and
                # full-context lp_A at each lambda, perturbing ONLY the teacher-
                # forced SCORING read-out (the frozen lambda=1 render/summary/write-
                # time KV snapshots are untouched -- the model NEVER generates while
                # degraded). lambda=1.0 reuses the values above, so it reproduces the
                # validated referent exactly (the sanity anchor asserted after the
                # loop). Restored to 1.0 before the next conv's summary/prefill.
                if lambda_values:
                    for lam in lambda_values:
                        if lam == 1.0:
                            la_l, lb_l, le_l = la, lb, le
                        else:
                            set_qk_lambda(model, lam)
                            _la_l, _lb_l, _le_l = [], [], []
                            for pi in range(n_probes):
                                _la_l.append(
                                    tf(a_snap, sa_list[pi] + tgt[:-1], tgt, len(ids)))
                                _lb_l.append(
                                    tf(b_snap, sb_list[pi] + tgt[:-1], tgt, len(b_ids)))
                                _le_l.append(
                                    tf(e_snap, sb_list[pi] + tgt[:-1], tgt, len(b_ids)))
                            la_l = _mean(_la_l)
                            lb_l = _mean(_lb_l)
                            le_l = _mean(_le_l)
                        per_lambda_rows[lam].append({
                            "conversation_id": conv["id"],
                            "category": pl["category"],
                            "lp_A": la_l, "raw_EB": le_l - lb_l,
                            "cleared": bool(task_competence_ok(la_l, active_floor))})
                    set_qk_lambda(model, 1.0)   # restore CLEAN before next conv/gen

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
            return summary_text

        # ---- resume-aware per-conversation loop ----
        # For each conv: if a matching SCORED checkpoint exists (and resume-skip is
        # allowed), REPLAY it (no re-score); otherwise snapshot -> score -> write
        # its checkpoint atomically BEFORE moving on. Loss on any interruption is
        # thus <= 1 conversation of scoring (the render loss is bounded to 1 chunk
        # by native_render_specs's per-conv rendered checkpoints).
        for _ci, (conv, plants) in enumerate(specs):
            _pos = spec_positions[_ci]
            _cid = conv["id"]
            if resume_skip_ok:
                _sck = _load_checkpoint(_ckpath(_pos, _cid), ck_fp, "scored")
                if _sck is not None:
                    _replay_conv(_sck["payload"])
                    print(f"  [checkpoint] conv {_ci+1}/{len(specs)} ({_cid}) "
                          f"RESUMED from {_ckpath(_pos, _cid).name} "
                          f"(n_plants now {n_plants})", flush=True)
                    continue
            _base = _snap()
            _summary_used = _process_conv(conv, plants, _ci)
            _write_scored_ck(_pos, _cid, conv, plants, _summary_used, _base)
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
        # Report the ACTUAL cause instead of guessing "empty alignment?". The
        # three ways every plant can fail to score are independent; naming which
        # one dominated turns a dead-end ERROR into a self-diagnosing one. A run
        # where task_excluded dominates is a COMPETENCE-FLOOR problem (the model's
        # per-token gold logprobs sit below the ABSOLUTE task_lpa_floor), NOT an
        # alignment/region-detection failure.
        n_task_excl = sum(len(v) for v in task_excluded.values())
        doc.update(
            status="ERROR",
            reason=(
                "no plants scored -- breakdown: convs_seen=%d, "
                "empty_alignment_convs=%d, short_gold_drops=%d, "
                "task_excluded_plants=%d (task_lpa_floor=%s). If "
                "task_excluded dominates, the model's gold logprobs are below the "
                "ABSOLUTE competence floor (a per-architecture lp_A scale issue), "
                "NOT an A<->B alignment failure."
                % (len(specs), len(empty_align_convs), short_gold_drops,
                   n_task_excl, task_lpa_floor)))
        doc["empty_alignment_convs"] = empty_align_convs
        doc["short_gold_drops"] = short_gold_drops
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
    # ---- COMPRESSION SWEEP aggregate: realized ratio at THIS summary level. ----
    # Read authoritatively from the on-disk scored checkpoints (resume-safe: a
    # reused checkpoint contributes its saved "compression" even if this process
    # never re-scored that conv). raw_EB / content_specificity below are then
    # reported AS A FUNCTION of this measured ratio across the sweep's levels.
    _comp_rows = list(compression_rows)
    try:
        _seen = {r.get("conv_id") for r in _comp_rows}
        for _cf in sorted(checkpoint_dir(out_dir, model_id).glob("conv_*.json")):
            try:
                _ck = json.loads(_cf.read_text())
            except Exception:  # noqa: BLE001
                continue
            _c = _ck.get("compression")
            if _c and _c.get("conv_id") not in _seen:
                _comp_rows.append(_c)
                _seen.add(_c.get("conv_id"))
    except Exception:  # noqa: BLE001
        pass
    _ratios = [r["ratio"] for r in _comp_rows
               if r and r.get("ratio") is not None]
    _summ_toks = [r["summary_tokens"] for r in _comp_rows
                  if r and r.get("summary_tokens") is not None]
    _ctx_toks = [r["full_ctx_tokens"] for r in _comp_rows
                 if r and r.get("full_ctx_tokens") is not None]
    _ratio_mean = (sum(_ratios) / len(_ratios)) if _ratios else None
    doc["compression"] = {
        "level": _summ_level,
        "regime": _summ_regime,
        "target_words": _summ_words,
        "summary_request_sha256": prov.sha256_text(_REQ),
        "ratio_mean": _ratio_mean,
        "ratio_min": (min(_ratios) if _ratios else None),
        "ratio_max": (max(_ratios) if _ratios else None),
        "mean_summary_tokens": (sum(_summ_toks) / len(_summ_toks)
                                if _summ_toks else None),
        "mean_full_ctx_tokens": (sum(_ctx_toks) / len(_ctx_toks)
                                 if _ctx_toks else None),
        "n_convs": len(_ratios),
        "per_conv": _comp_rows,
    }
    # backfill the manifest's measured ratio now that it is known.
    try:
        manifest["condition"]["measured_compression_ratio_mean"] = _ratio_mean
        prov.write_run_manifest(out_dir, manifest,
                                name=f"manifest__{_slug(model_id)}.json")
    except Exception:  # noqa: BLE001
        pass

    doc["pre_graft_gap"] = bootstrap_ci_95(pre_gaps)
    # PRIMARY aggregate metric: raw_EB = mean(lp_E - lp_B) over all plants.
    # The CI is CONVERSATION-CLUSTERED (resample whole conversations with
    # replacement, pool their plants) -- the honest inferential unit, since plants
    # within a conversation are correlated. The plant-level interval (which treats
    # every plant as independent and is anti-conservative) is retained as
    # ``ci_plant`` for reference. The point estimate (``mean``) is identical to
    # the plant-level pooled mean -- only the interval width changes.
    _agg_rows = [r for cat in CATS for r in per_cat[cat]]
    _agg_clusters = _group_by_conv(_agg_rows)
    _agg_plant = bootstrap_ci_95(
        graft_signed, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    _agg_cluster = bootstrap_ci_95_cluster(
        _agg_clusters, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
    doc["raw_EB"] = {
        "mean": _agg_cluster["mean"],
        "lo": _agg_cluster["lo"],
        "hi": _agg_cluster["hi"],
        "n": _agg_cluster["n"],                       # n plants pooled
        "n_conversations": _agg_cluster["n_clusters"],
        "ci_method": "conversation-clustered",
        "ci_plant": [_agg_plant["lo"], _agg_plant["hi"]],  # reference only
    }

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
        "task_competence_mode": task_competence_mode,
        "task_competence_k": task_competence_k,
        # the floor ACTUALLY applied: absolute -> task_lpa_floor; relative ->
        # the per-model median-k*MADN value (None-fallback keeps task_lpa_floor).
        "task_competence_active_floor": active_floor,
        "floored_categories": [c for c in CATS
                               if by_cat_robust.get(c, {}).get("floor")],
        "task_excluded": {c: task_excluded[c] for c in CATS
                          if task_excluded[c]},
        "n_task_excluded": sum(len(v) for v in task_excluded.values()),
        # DIAGNOSTIC (audit trail on EVERY run, not just failures): how many
        # convs were skipped for empty A<->B alignment vs plants dropped for a
        # too-short gold. Lets a future zero-score run be attributed without a
        # re-run (the ERROR-path reason mirrors these).
        "empty_alignment_convs": empty_align_convs,
        "short_gold_drops": short_gold_drops,
        "note": (
            "GATE #2 headroom = per-category mean(lp_A-lp_B); a floored category "
            "(headroom<headroom_floor) has no evicted meaning to recover -> "
            "uninterpretable, EXCLUDED from the sign verdict (NOT counted as harm); "
            "raw_EB_normalized = raw_EB/max(headroom,eps). GATE #3 task-competence: "
            "plants with lp_A per-token < task_competence_active_floor were dropped "
            "from all aggregates (model can't do the task even with full context); "
            "mode='absolute' uses the fixed task_lpa_floor, mode='relative' uses the "
            "pre-registered per-model median-k*MADN robust-outlier floor."),
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

        # ---- DECISIVE CONTRAST: E - placebo, PAIRED, conversation-clustered ----
        # The champion-validation headline. Per plant the difference is
        #   (lp_E - lp_B) - (lp_E_placebo - lp_B) = lp_E - lp_E_placebo,
        # i.e. the lift attributable to the REAL write-time CONTENT over a graft
        # that used the SAME champion slots/alpha but CORRUPTED source values.
        # Paired within plant (cancels the plant/gold difficulty), then resampled
        # over CONVERSATIONS (honest clustered CI). raw_EB (E-B) and
        # placebo_raw_EB (E_placebo-B) are stored index-aligned per scored plant
        # in ``traces`` (resume-safe: traces are restored from checkpoints).
        # CONFIRM: e_minus_placebo CI lower bound > 0 => the tuned champion carries
        # genuine CONTENT-SPECIFIC state. REFUTE: CI spans 0 => the tuning only
        # amplifies the content-INDEPENDENT (generic/regularizer) effect and the
        # placebo-controlled null stands.
        cs_pairs = [t for t in traces
                    if t.get("placebo_raw_EB") is not None
                    and t.get("raw_EB") is not None]

        def _cs_ci(rows):
            clusters = _group_by_conv(
                [{"conversation_id": t["conversation_id"],
                  "raw_EB": t["raw_EB"] - t["placebo_raw_EB"]} for t in rows])
            return bootstrap_ci_95_cluster(
                clusters, n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)

        doc["content_specificity"] = {
            "mode": placebo_mode,
            "champion_label": (champion_cfg or {}).get("label"),
            "note": ("PAIRED champion-vs-placebo contrast per plant "
                     "(lp_E - lp_E_placebo), conversation-clustered bootstrap "
                     "95% CI. Same champion layers/positions/alpha for E and "
                     "placebo; only the source values differ. CI lower bound > 0 "
                     "=> content-SPECIFIC champion effect (CONFIRM). CI includes 0 "
                     "=> tuning amplifies the generic effect; null stands (REFUTE)."),
            "e_minus_placebo": _cs_ci(cs_pairs),
            "by_category": {cat: _cs_ci([t for t in cs_pairs
                                         if t.get("category") == cat])
                            for cat in CATS},
            "n_pairs": len(cs_pairs),
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

    # ---- GRADED QK-NORM ABLATION (SC_ABLATE_LAMBDA) dose-response report ----
    # Per lambda: referent raw_EB (conversation-clustered CI) + how many plants
    # still clear the per-model relative competence floor at that lambda. The
    # readout is whether graft-benefit tracks lambda within the window where the
    # model stays competent (n_cleared). lambda=1.0 is the CLEAN sanity anchor.
    if lambda_values:
        def _lam_block(rows):
            cleared = [r for r in rows if r["cleared"]]
            ci = bootstrap_ci_95_cluster(
                _group_by_conv(cleared), n_boot=ROBUST_N_BOOT, seed=ROBUST_SEED)
            return {
                "raw_EB": {
                    "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                    "n": ci["n"], "n_conversations": ci["n_clusters"],
                    "ci_method": "conversation-clustered"},
                "n_cleared": len(cleared),
                "n_candidate": len(rows),
            }
        by_lambda = {}
        for lam in lambda_values:
            rows = per_lambda_rows[lam]
            by_lambda[str(lam)] = {
                "referent": _lam_block([r for r in rows
                                        if r["category"] == "referent"]),
                "aggregate": _lam_block(rows),
                "n_plants_cleared": sum(1 for r in rows if r["cleared"]),
                "n_plants_candidate": len(rows),
                "per_category": {
                    c: _lam_block([r for r in rows if r["category"] == c])
                    for c in CATS},
            }
        # SANITY ANCHOR: lambda=1.0 MUST reproduce the validated relative-mode
        # referent raw_EB (~+0.10 on Qwen3-30B-A3B). A mismatch invalidates the run.
        prim_ref = by_cat_robust.get("referent", {}).get("raw_EB_mean")
        lam1_ref = (by_lambda.get("1.0", {}).get("referent", {})
                    .get("raw_EB", {}).get("mean"))
        sanity_ok = None
        if prim_ref is not None and lam1_ref is not None:
            sanity_ok = abs(prim_ref - lam1_ref) <= 1e-6
        doc["ablate_lambda"] = {
            "lambdas": lambda_values,
            "n_qk_modules_wrapped": n_qk_wrapped,
            "competence_mode": task_competence_mode,
            "competence_active_floor": active_floor,
            "by_lambda": by_lambda,
            "sanity_anchor": {
                "lambda1_referent_raw_EB": lam1_ref,
                "primary_referent_raw_EB": prim_ref,
                "ok": sanity_ok,
                "note": ("lambda=1.0 must reproduce the validated relative-mode "
                         "referent raw_EB (the point estimate equals the primary "
                         "by_category_robust referent). ok=False INVALIDATES the "
                         "run -- the clean anchor did not reproduce."),
            },
            "note": (
                "Graded QK-norm ablation DECOUPLED FROM GENERATION (Fable H1 "
                "salvage). The corpus was rendered + summarized ONCE at lambda=1.0 "
                "(clean, coherent generation); ONLY the teacher-forced graft-vs-"
                "compacted SCORING read-out (lp_A/lp_B/lp_E) was perturbed per "
                "lambda via qk_lambda(x)=(1-lambda)*x+lambda*RMSNorm_qk(x). "
                "lambda=1.0=clean/original, lambda=0.0=identity (full ablation). "
                "The degraded model NEVER generates. Per lambda: referent raw_EB "
                "(conversation-clustered CI) and n plants clearing the per-model "
                "relative competence floor (n_cleared); reads whether graft-benefit "
                "tracks lambda within the competent window."),
        }

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
    # raw_EB CI is now conversation-clustered: guard the sign on the CLUSTER count
    # (n_conversations), so a single-conversation model -- whose cluster CI
    # degenerates to a zero-width point interval -- reads as null, not a spurious
    # "excludes zero".
    agg_sign = _ci_sign(eb.get("lo"), eb.get("hi"),
                        eb.get("n_conversations", eb.get("n")))
    effect_sign = "positive" if agg_sign > 0 else "negative" if agg_sign < 0 else "null"
    doc["effect_sign"] = effect_sign

    def _cat_sign(block):
        lo, hi = block.get("raw_EB_ci", [None, None])
        # conversation-clustered CI -> guard on n_conversations when present.
        return _ci_sign(lo, hi,
                        block.get("n_conversations", block.get("n", 0)))

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


def _dry_run(data_dir: Path, conv_limit: int, conv_start: int = 0) -> int:
    print("== cross_arch_probe --dry-run (no torch / no model) ==")
    print(f"data_dir={data_dir}  SC_CONV_START={conv_start}  "
          f"SC_CONV_LIMIT={conv_limit}  categories={CATS}")

    # Show the FULL sorted corpus window that will be selected (true ids), so the
    # held-out selection is auditable even for convs with 0 usable plants.
    all_paths = conversation_paths(data_dir)
    window = all_paths[conv_start: conv_start + conv_limit]
    window_ids = [p.stem for p in window]
    print(f"SELECTED WINDOW ({len(window_ids)} convs, offset {conv_start}): "
          f"{window_ids}")

    specs = collect_specs(data_dir, conv_limit, conv_start=conv_start)
    total = 0
    for conv, plants in specs:
        print(f"  {conv['id']}: {len(plants)} usable plants "
              f"({', '.join(p['id'] for p in plants)})")
        total += len(plants)
    print(f"TOTAL usable sense+referent plants in convs "
          f"[{conv_start}:{conv_start + conv_limit}]: {total}")
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
              "raw_EB_ci_plant", "raw_EB_ci_method", "n_conversations"):
        assert k in st, f"robust_category_stats missing {k}"
    assert st["n_conversations"] == 2, "two distinct convs"
    # the HEADLINE raw_EB_ci must now BE the conversation-clustered interval,
    # and the point estimate must be unchanged (pooled plant mean).
    assert st["raw_EB_ci_method"] == "conversation-clustered"
    assert st["raw_EB_ci"] == st["raw_EB_ci_cluster"], \
        "headline raw_EB_ci must be the conversation-clustered interval"
    assert abs(st["raw_EB_mean"] - (0.4 + 0.5 - 0.2) / 3) < 1e-9, \
        "point estimate must be the unchanged pooled plant mean"
    # NOTE: the cluster CI is wider IN EXPECTATION for correlated data with
    # adequate/balanced samples (demonstrated on the 15-probe/3-conv fixture
    # above, ~2.5x wider); on a degenerate 3-plant/2-conv toy the discrete cluster
    # resampling can be marginally narrower, so we do NOT assert width here.
    w_cluster = st["raw_EB_ci_cluster"][1] - st["raw_EB_ci_cluster"][0]
    w_plant = st["raw_EB_ci_plant"][1] - st["raw_EB_ci_plant"][0]
    print(f"  robust_category_stats: n_conversations={st['n_conversations']} "
          f"headline(cluster)_ci={st['raw_EB_ci']} plant_ci={st['raw_EB_ci_plant']} "
          f"(width cluster={w_cluster:.4f} plant={w_plant:.4f}) OK")

    _self_test_native()
    _self_test_checkpoint()
    _self_test_batched_decode()
    _self_test_qk_ablation()
    _self_test_qk_lambda()

    print("\nSELF-TEST OK")
    return 0


def _self_test_checkpoint() -> int:
    """CPU/torch-free unit test of the per-conv checkpoint PRIMITIVES (incident
    #38): atomic write round-trip, path/slug naming, fingerprint gating (score vs
    render), and stage gating. The full numbers-identical / resume / render-reuse
    proof is the model-level validation (scratchpad validate_ckpt.py on
    Qwen3-0.6B); this guards the plumbing so a regression fails fast + offline."""
    import tempfile  # noqa: PLC0415
    print("\n== PER-CONV CHECKPOINT primitives self-test (no torch) ==")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        model = "Vendor/Some-Model-30B"
        # path + slug: keyed by window position AND conv id, under <slug>/.
        p = checkpoint_path(out, model, 7, "c13")
        assert p.parent == checkpoint_dir(out, model) == out / _slug(model)
        assert p.name == "conv_007__c13.json", p.name
        # atomic write is durable + parseable; no leftover tmp file.
        obj = {"stage": "scored", "payload": {"x": [1, 2, 3]}, "u": None}
        _atomic_write_json(p, obj)
        assert json.loads(p.read_text()) == obj
        assert not list(p.parent.glob(".*tmp*")), "atomic tmp file leaked"
        print(f"  path={p.name} atomic-write round-trip OK (no tmp leak)")

        full = run_fingerprint(
            model, conv_start=0, alpha_v=0.75, seed=42, native_render=True,
            native_max_reply=320, native_temp=0.0, max_gold_tok=80,
            task_competence_mode="relative", task_competence_k=3.0,
            task_lpa_floor=-8.0, headroom_floor=0.3, placebo_mode=None,
            alpha_sweep=False, strong_prior=True, champion_scan=0,
            champion_regions=None, ablate_qk_norm_flag=False,
            ablate_lambda_values=None, alpha0_tol=5e-3, change_tol=1e-3)
        rend = render_fingerprint(
            model, conv_start=0, native_render=True, native_max_reply=320,
            native_temp=0.0)
        ck = {"stage": "scored", "fingerprint": full, "render_fingerprint": rend}
        _atomic_write_json(p, ck)
        # exact score-fingerprint match loads; a changed SCORING param does not
        # (score replay must be exact) ...
        assert _load_checkpoint(p, full, "scored") is not None
        full_a = dict(full, alpha_v=0.5)
        assert _load_checkpoint(p, full_a, "scored") is None
        # ... but the SAME conv reused for a DIFFERENT alpha still matches on the
        # RENDER fingerprint (generation is reusable across scoring configs).
        assert _load_checkpoint(p, rend, "rendered",
                                fp_key="render_fingerprint") is not None
        # stage gating: a rendered-only checkpoint is not accepted as scored.
        _atomic_write_json(p, {"stage": "rendered", "fingerprint": full,
                               "render_fingerprint": rend})
        assert _load_checkpoint(p, full, "scored") is None
        assert _load_checkpoint(p, rend, "rendered",
                                fp_key="render_fingerprint") is not None
        # missing / corrupt file -> None (recompute), never raises.
        assert _load_checkpoint(out / "nope.json", full, "scored") is None
        (out / "bad.json").write_text("{not json")
        assert _load_checkpoint(out / "bad.json", full, "scored") is None
        # render fingerprint is independent of scoring params (alpha change ->
        # SAME render fp, DIFFERENT score fp).
        assert render_fingerprint(model, conv_start=0, native_render=True,
                                  native_max_reply=320, native_temp=0.0) == rend
        assert full_a != full
        print("  fingerprint gating (score-exact vs render-reusable) + stage "
              "gating + corrupt/missing tolerance OK")
    print("  PER-CONV CHECKPOINT primitives OK")
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


def _self_test_qk_lambda() -> int:
    """CPU test for the GRADED QK-NORM ABLATION (SC_ABLATE_LAMBDA) -- the
    dose-response DECOUPLED FROM GENERATION.

    Asserts: (a) lambda=1.0 is numerically IDENTICAL to the original RMSNorm (both
    at the module level and via full-model logits == un-wrapped), (b) lambda=0.0
    equals Identity (== the binary full ablation's logits), (c) intermediate lambda
    interpolates ((1-l)*x + l*orig(x)), (d) the perturbation toggles WITHOUT reload
    -- lambda can be set to 1.0 (clean/render path), then <1.0 (scoring), then back
    to 1.0 reproducing the clean logits exactly. Plus pure parse_lambda_values
    checks. Skips cleanly if torch/Qwen3 are unavailable."""
    print("\n== GRADED QK-NORM ABLATION (lambda) self-test (tiny Qwen3, CPU) ==")

    # (pure) parse_lambda_values: no-op cases and canonical ordering + anchor.
    assert parse_lambda_values(None) == []
    assert parse_lambda_values("") == []
    assert parse_lambda_values("1.0") == []               # solely 1.0 -> no sweep
    assert parse_lambda_values("1.0,1.0") == []
    assert parse_lambda_values("0.5") == [1.0, 0.5]        # anchor auto-added
    assert parse_lambda_values("1.0,0.5,0.25,0.0") == [1.0, 0.5, 0.25, 0.0]
    assert parse_lambda_values("0.0,0.5,1.0") == [1.0, 0.5, 0.0]  # sorted desc
    print("  (parse) parse_lambda_values no-op/anchor/order OK")

    try:
        import copy  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] no torch/Qwen3 available: {type(e).__name__}: {e}")
        return 0

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, m in model.named_modules():
            if n.rsplit(".", 1)[-1] in ("q_norm", "k_norm"):
                m.weight.add_(torch.randn_like(m.weight) * 0.5)

    ids = torch.tensor([[3, 5, 7, 9, 11]])
    with torch.no_grad():
        logits_clean = model(ids).logits.clone()

    # Reference twins for the two extremes: an untouched clean twin and a
    # binary-ablated (Identity) twin, both weight-identical to `model`.
    clean_twin = copy.deepcopy(model).eval()
    ablated_twin = copy.deepcopy(model).eval()
    n_abl, _ = ablate_qk_norm(ablated_twin)
    assert n_abl == 4, f"twin ablation expected 4, got {n_abl}"
    with torch.no_grad():
        logits_ablated = ablated_twin(ids).logits.clone()

    # Install the graded wrappers on `model`.
    n_wrapped, names = install_qk_lambda(model)
    assert n_wrapped == 4, f"expected 4 QK-norm modules wrapped, got {n_wrapped}"
    # idempotent: re-installing wraps nothing new (already wrapped).
    assert install_qk_lambda(model)[0] == 0, "install_qk_lambda not idempotent"

    # (a) module-level: lambda=1.0 IDENTICAL to the original RMSNorm; and the full
    # model at lambda=1.0 reproduces the un-wrapped logits EXACTLY (the render/gen
    # path stays byte-clean).
    set_qk_lambda(model, 1.0)
    a_wrapper = None
    for _n, m in model.named_modules():
        if getattr(m, "_is_qk_lambda", False):
            a_wrapper = m
            break
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        orig_out = a_wrapper.orig(x)
        assert torch.equal(a_wrapper(x), orig_out), \
            "lambda=1.0 wrapper is not identical to the original RMSNorm"
        logits_l1 = model(ids).logits
    d1 = (logits_l1 - logits_clean).abs().max().item()
    assert d1 == 0.0, f"lambda=1.0 full-model logits differ from clean (max|d|={d1})"
    print(f"  (a) lambda=1.0 == original RMSNorm (module exact; full-model max|d|"
          f"={d1:.1e}) OK")

    # (b) lambda=0.0 == Identity == binary full ablation (module and full model).
    set_qk_lambda(model, 0.0)
    with torch.no_grad():
        assert torch.equal(a_wrapper(x), x), "lambda=0.0 wrapper is not Identity"
        logits_l0 = model(ids).logits
    d0 = (logits_l0 - logits_ablated).abs().max().item()
    assert d0 <= 1e-5, \
        f"lambda=0.0 does not match the binary full ablation (max|d|={d0:.2e})"
    dclean0 = (logits_l0 - logits_clean).abs().max().item()
    assert dclean0 > 1e-4, "lambda=0.0 did not change logits vs clean"
    print(f"  (b) lambda=0.0 == Identity == full ablation (vs ablated max|d|"
          f"={d0:.1e}; vs clean max|d|={dclean0:.3e}) OK")

    # (c) intermediate lambda interpolates: (1-l)*x + l*orig(x).
    for lam in (0.25, 0.5, 0.75):
        set_qk_lambda(model, lam)
        with torch.no_grad():
            expect = (1.0 - lam) * x + lam * a_wrapper.orig(x)
            got = a_wrapper(x)
        di = (got - expect).abs().max().item()
        assert di <= 1e-6, f"lambda={lam} interpolation wrong (max|d|={di:.2e})"
    print("  (c) intermediate lambda interpolates (1-l)*x+l*orig(x) OK")

    # (d) toggling WITHOUT reload: 1.0 (clean render/gen) -> 0.5 (scoring) -> 1.0
    # reproduces the clean logits EXACTLY (models the run's render-clean/score-
    # perturbed lambda cycling; weights are never reloaded).
    set_qk_lambda(model, 1.0)
    with torch.no_grad():
        assert torch.equal(model(ids).logits, logits_clean), \
            "restoring lambda=1.0 did not reproduce clean logits"
    set_qk_lambda(model, 0.5)
    with torch.no_grad():
        mid = model(ids).logits
    assert (mid - logits_clean).abs().max().item() > 1e-4, \
        "lambda=0.5 scoring pass did not perturb logits"
    set_qk_lambda(model, 1.0)
    with torch.no_grad():
        assert torch.equal(model(ids).logits, logits_clean), \
            "second restore to lambda=1.0 did not reproduce clean logits"
    # clean_twin was never perturbed -> still matches clean (sanity on the twin).
    with torch.no_grad():
        assert torch.equal(clean_twin(ids).logits, logits_clean)
    print("  (d) lambda toggles 1.0->0.5->1.0 without reload; render path clean OK")

    print("  GRADED QK-NORM ABLATION (lambda) OK")
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

    # ---- relative_competence_floor (GATE #3 RELATIVE, pre-registered 07-08) ----
    # Too few points -> None (permissive: caller keeps all plants).
    assert relative_competence_floor([-1.0, -2.0, -3.0], min_n=8) is None
    # A HIGH-lp_A model (Qwen/Mistral-like tight distribution): the robust floor
    # sits well below the minimum -> excludes 0 plants (matches absolute -8.0),
    # so already-scored numbers are unchanged. INVARIANT for k in {2.5..4}.
    hi = [-0.3, -0.9, -1.3, -1.7, -1.9, -2.1, -2.5, -3.0, -3.6, -4.0]
    for _k in (2.5, 3.0, 3.5, 4.0):
        f_hi = relative_competence_floor(hi, k=_k)
        assert f_hi < min(hi), (f_hi, min(hi), _k)
        assert all(task_competence_ok(x, f_hi) for x in hi)
    # A LOW-lp_A model (OLMo-like: whole distribution shifted DOWN) -- the absolute
    # -8.0 would drop EVERY plant; the relative floor adapts to the model's own
    # scale and KEEPS its typical plants while still dropping an in-model extreme
    # low outlier (gate not disabled).
    lo = [-9.0, -9.4, -9.6, -10.0, -10.1, -10.3, -10.8, -11.2, -11.9, -18.0]
    assert all(not task_competence_ok(x, -8.0) for x in lo)  # absolute drops all
    f_lo = relative_competence_floor(lo, k=3.0)
    kept = [x for x in lo if task_competence_ok(x, f_lo)]
    assert len(kept) == 9 and -18.0 not in kept, (f_lo, kept)  # keeps 9, drops outlier
    print(f"  relative_competence_floor: hi-model floor<{min(hi):.1f} (0 excluded, "
          f"k-invariant); lo-model floor={f_lo:.2f} keeps {len(kept)}/10 (drops "
          f"outlier, adapts scale) OK")

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
                   conv_limit: int, trust_remote_code: bool, conv_start: int = 0):
    """Generate the SHARED fixed summary text ONCE with a single designated
    summarizer, write {conv_id: text, _summarizer: id} to summaries_path. This
    file is then a required INPUT to every per-model run so the compaction
    content is identical across architectures."""
    import torch  # noqa: PLC0415
    from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                              AutoTokenizer)
    from arms_common import SUMMARY_REQUEST  # noqa: PLC0415
    from arms_hf import generate_summary_hf  # noqa: PLC0415

    print(f"MAKE_SUMMARIES summarizer={summarizer} start={conv_start} "
          f"limit={conv_limit}", flush=True)
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
    for conv, _plants in collect_specs(data_dir, conv_limit, conv_start=conv_start):
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
                    default=int(os.environ.get("SC_CONV_LIMIT") or "4"))
    ap.add_argument("--conv-start", type=int,
                    default=int(os.environ.get("SC_CONV_START") or "0"),
                    help="HELD-OUT selector: 0-based OFFSET into the sorted "
                         "conversation list; the run scores convs "
                         "[conv_start : conv_start+conv_limit]. Default 0 == the "
                         "legacy first-N slice (byte-identical). e.g. "
                         "SC_CONV_START=12 SC_CONV_LIMIT=24 selects c13..c36 (the "
                         "fresh/held-out set the effect was never tuned on).")
    ap.add_argument("--alpha-v", type=float,
                    default=float(os.environ.get("SC_GC_ALPHA") or "0.75"))
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
                    default=int(os.environ.get("SC_CHAMPION_SCAN") or "0"),
                    help="FEATURE #3: per-layer champion scan into N "
                         "fractional-depth regions (0=off; a value >=2 sets N; "
                         f"any other positive value uses the default "
                         f"{CHAMPION_SCAN_DEFAULT_N}); default off")
    ap.add_argument("--champion-regions",
                    default=(os.environ.get("SC_CHAMPION_REGIONS") or None),
                    help="FEATURE #3 rescue test: comma-separated region indices "
                         "to graft TOGETHER (e.g. '4,5'); alpha=0 elsewhere")
    ap.add_argument("--champion-config",
                    default=(os.environ.get("SC_CHAMPION_CONFIG") or None),
                    help="CHAMPION VALIDATION: path to a canonical champion "
                         "value-graft config JSON (per-layer alpha_map OR "
                         "per-head head_map+alpha). When set, the REAL graft E "
                         "and the placebo graft both use this tuned config "
                         "(same layers/positions/alpha) instead of the scalar "
                         "--alpha, so E_champion vs placebo_champion vs B is the "
                         "content-specificity test. Value-only (alpha_K=0).")
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
    ap.add_argument("--ablate-lambda", dest="ablate_lambda",
                    default=(os.environ.get("SC_ABLATE_LAMBDA") or None),
                    help="GRADED QK-NORM ABLATION dose-response, DECOUPLED FROM "
                         "GENERATION (SC_ABLATE_LAMBDA; comma-separated lambdas, "
                         "e.g. '1.0,0.5,0.25,0.0'; default off/1.0 = no change). "
                         "Wraps each QK-norm module in qk_lambda(x)=(1-lambda)*x+"
                         "lambda*RMSNorm_qk(x): lambda=1.0=clean/original, "
                         "lambda=0.0=identity (full ablation). The corpus is "
                         "rendered + summarized ONCE at lambda=1.0 (clean); ONLY "
                         "the teacher-forced graft-vs-compacted SCORING read-out is "
                         "perturbed per lambda (the degraded model NEVER generates). "
                         "1.0 is always added as the sanity anchor. Mutually "
                         "exclusive with --ablate-qk-norm.")
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
                    help="v2.1 GATE #3 (absolute mode): min lp_A per-token to score "
                         "a plant (below = model can't do the task, plant excluded)")
    ap.add_argument("--task-competence-mode",
                    choices=["absolute", "relative"],
                    default=(os.environ.get("SC_TASK_COMPETENCE_MODE")
                             or TASK_COMPETENCE_MODE_DEFAULT),
                    help="v2.1 GATE #3 floor mode. 'absolute' (default) = fixed "
                         "--task-lpa-floor. 'relative' (pre-registered 2026-07-08) = "
                         "per-model robust-outlier floor median-k*MADN(lp_A); "
                         "scale-adaptive, does not confound cross-model comparison "
                         "(unblocks OLMo). Effective only on a fresh scored run.")
    ap.add_argument("--task-competence-k", type=float,
                    default=float(os.environ.get("SC_TASK_COMPETENCE_K",
                                                 str(TASK_COMPETENCE_K_DEFAULT))),
                    help="relative-mode K (robust-sigma/MADN units; default 3.0)")
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
        sys.exit(_dry_run(data_dir, args.conv_limit, conv_start=args.conv_start))

    if args.smoke_align:
        toks = (tuple(t.strip() for t in args.smoke_tokenizers.split(",")
                      if t.strip())
                if args.smoke_tokenizers else _SMOKE_TOKENIZERS_DEFAULT)
        sys.exit(_smoke_align(data_dir, toks))

    trust = os.environ.get("SC_TRUST_REMOTE", "1") not in ("0", "false", "False")
    summaries_path = Path(args.summaries)

    if args.make_summaries:
        make_summaries(args.summarizer, data_dir, summaries_path,
                       args.conv_limit, trust, conv_start=args.conv_start)
        print("CROSS_ARCH_DONE", flush=True)
        return

    if not args.model:
        print("FATAL: SC_HF_MODEL (or --model) is required for the real run",
              file=sys.stderr)
        sys.exit(2)

    # CHAMPION VALIDATION: resolve the tuned graft config (per-layer alpha-map or
    # per-head slot mask) up front so a bad config fails LOUD before model load.
    champion_cfg = load_champion_graft_cfg(args.champion_config)
    if champion_cfg is not None:
        print(f"RUN champion-validation: config={args.champion_config} "
              f"label={champion_cfg.get('label')} "
              f"placebo={args.placebo} model={args.model}", flush=True)

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

    # GRADED QK-NORM ABLATION dose-response: parse the lambda list (no-op unless a
    # value != 1.0 is requested). It takes precedence over the binary ablation (the
    # binary one breaks generation, which the graded one exists to avoid).
    lambda_values = parse_lambda_values(args.ablate_lambda)
    if lambda_values and args.ablate_qk_norm:
        print("SC_ABLATE_LAMBDA set -> ignoring SC_ABLATE_QK_NORM (binary ablation "
              "breaks generation; the graded dose-response replaces it).",
              file=sys.stderr)
        args.ablate_qk_norm = False
    if lambda_values:
        print(f"SC_ABLATE_LAMBDA -> graded QK-norm dose-response at lambdas="
              f"{lambda_values} (render/summary CLEAN at lambda=1.0; only the "
              f"scoring read-out perturbed).", flush=True)

    try:
        doc = run_model(
            args.model, data_dir, out_dir, fixed_summaries,
            conv_limit=args.conv_limit, conv_start=args.conv_start,
            alpha_v=args.alpha_v,
            alpha0_tol=args.alpha0_tol, change_tol=args.change_tol,
            max_gold_tok=args.max_gold_tok, trust_remote_code=trust,
            placebo_mode=args.placebo, alpha_sweep=args.alpha_sweep,
            seed=args.seed, strong_prior=args.strong_prior,
            champion_scan=args.champion_scan, champion_regions=champion_regions,
            native_render=args.native_render, scenarios=scenarios,
            native_max_reply=args.native_max_reply, native_temp=args.native_temp,
            headroom_floor=args.headroom_floor, task_lpa_floor=args.task_lpa_floor,
            task_competence_mode=args.task_competence_mode,
            task_competence_k=args.task_competence_k,
            ablate_qk_norm_flag=args.ablate_qk_norm,
            ablate_lambda_values=lambda_values,
            champion_cfg=champion_cfg,
            selfgen_declared=force_selfgen)
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
