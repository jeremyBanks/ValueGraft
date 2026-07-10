"""Stage 2: SWE-Gym/OpenHands coding-trace evaluation under compaction.

Per trajectory (filtered to fit budget): context = messages up to a cut at
~75% of tokens (cut lands before an assistant turn); target = that next
assistant action. Arms A / B / B-min-pack / H-pack / E-tuned built over the
context (evicting sinks..tail_start like the chat experiments; tail = last
~25% incl. the most recent tool results, per production practice).

Metrics per arm:
  tf: mean teacher-forced logprob of the true next action (primary)
  gen: greedy 200-token generation (for behavioral analysis offline):
       stored raw; plus auto flags — repeats-failed-command (a command line
       that previously produced an error/traceback in the EVICTED region).

Env: SC_HF_MODEL, SC_SWE_TAG, SC_SWE_N (default 75), SC_SHARD, SC_E_ALPHA,
SC_SUMMARY (brief|prod), SC_CHAMPION_CONFIG (adds E-champion arm).
In-domain optimization (score-only extra arms; see the block below the env
constants): SC_E_ALPHAS (alpha-sweep -> E-a{alpha}), SC_SWE_PLACEBO (P-a{alpha}
content-specificity), SC_PROFILE_REGIONS / SC_PROFILE_ALPHA (per-region marginals
R{k}), SC_TRAJ_SPLIT_SEED (deterministic held-out tune|eval split, recorded per
trajectory). Analysis: src/analyze_swegym_tune.py.
Output: results/swegym_<tag>/<idx>.json  (use a UNIQUE stamped <tag> per run).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST_BRIEF,
    SUMMARY_REQUEST_PROD,
    build_alignment,
    build_b_messages,
    bmin_pack_ids,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import (
    answer_hf,
    arm_h_pack_snapshot_hf,
    generate_summary_hf,
    hf_prefill_ids,
    rope_base,
    to_ids,
)
from kvlib_hf import blend_values, rebuild_cache, tf_logprobs
from cross_arch_probe import load_champion_graft_cfg  # reuse the canonical loader
import provenance as prov

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_SWE_TAG", "30b_bf16")
N_TRAJ = int(os.environ.get("SC_SWE_N", "75"))
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))
# Summary condition: SC_SUMMARY=brief -> terse mechanism-isolation summary (the
# condition the +0.0156 anchor result used); default -> SUMMARY_REQUEST_PROD, the
# production-faithful OpenHands-condenser-style summary (task/state/paths/decisions,
# 300-500w) — the std arm we now want as first-class enrichment (Fable's #1).
SUMM_REQ = SUMMARY_REQUEST_BRIEF if os.environ.get("SC_SUMMARY") == "brief" else SUMMARY_REQUEST_PROD
SUMM_TAG = "brief" if os.environ.get("SC_SUMMARY") == "brief" else "prod"
_shard = os.environ.get("SC_SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in _shard.split("/"))
PARQUET = os.environ.get("SC_SWE_DATA", "swegym.parquet")
MAX_TOK, MIN_TOK = 15000, 6000

# CHAMPION arm (SC_CHAMPION_CONFIG): when set, ADD an "E-champion" arm applying the
# TUNED per-LAYER alpha_map (or per-HEAD head_map) value-graft (keys neutral, α_K=0)
# ALONGSIDE the scalar "E-tuned" (flat α=SC_E_ALPHA) arm -- so ONE run yields both
# numbers vs B for a direct paired comparison. Tests whether the conversation-tuned
# champion TRANSFERS to the coding task. None -> champion arm skipped (scalar only).
CHAMPION_CFG_PATH = os.environ.get("SC_CHAMPION_CONFIG") or None
CHAMPION_CFG = load_champion_graft_cfg(CHAMPION_CFG_PATH)

# ---- IN-DOMAIN OPTIMIZATION knobs (the brief-SWE-Gym regime is the only cell the
# graft is net-positive; every prior champion was tuned on the SYNTHETIC corpus and
# transferred here adding nothing). These add CHEAP, SCORE-ONLY arms (no 200-token
# generation) so a single pass yields the whole optimization surface:
#   SC_E_ALPHAS      alpha-sweep: comma list -> arm "E-a{alpha}" per alpha (value
#                    graft at that flat alpha). Answers "is 0.75 optimal on code?".
#   SC_SWE_PLACEBO   =1 -> per swept alpha, a "P-a{alpha}" arm = same slots, SOURCE
#                    values position-shuffled (seeded) -> content-specificity E-P.
#   SC_PROFILE_REGIONS  N>=1 -> partition the model's layers into N contiguous
#                    fractional-depth regions and add a "R{k}" arm per region = the
#                    graft restricted to that region's layers at SC_PROFILE_ALPHA.
#                    A CPU analysis builds a SWE-Gym-tuned per-layer champion from
#                    the per-region marginals ON THE TUNE SPLIT only.
#   SC_PROFILE_ALPHA per-region profiling alpha (default 1.0, single-region marginal).
# HELD-OUT: every trajectory records a deterministic split ("tune"|"eval") so the
# CPU analysis selects alpha / builds the layer champion on TUNE and reports on the
# DISJOINT EVAL set -- never tuning on the eval trajectories. With only ~75 usable
# trajectories this split is thin; the analysis reports the overfitting caveat.
ALPHA_SWEEP = [float(x) for x in os.environ.get("SC_E_ALPHAS", "").split(",")
               if x.strip()]
SWE_PLACEBO = os.environ.get("SC_SWE_PLACEBO", "0") in ("1", "true", "yes")
PLACEBO_SEED = int(os.environ.get("SC_PLACEBO_SEED", "1234"))
PROFILE_REGIONS = int(os.environ.get("SC_PROFILE_REGIONS", "0"))
PROFILE_ALPHA = float(os.environ.get("SC_PROFILE_ALPHA", "1.0"))
SPLIT_SEED = int(os.environ.get("SC_TRAJ_SPLIT_SEED", "20260710"))


def traj_split(idx):
    """Deterministic, stable tune/eval assignment for a trajectory index -- so the
    CPU analysis selects on TUNE and reports on the DISJOINT EVAL set. Hash the
    (seed, idx) pair rather than idx-parity so the split does not correlate with the
    (ordered) trajectory index. ~50/50."""
    import hashlib
    h = hashlib.sha256(f"{SPLIT_SEED}:{idx}".encode()).hexdigest()
    return "tune" if int(h[:8], 16) % 2 == 0 else "eval"


def region_layer_sets(n_layers, n_regions):
    """Partition [0, n_layers) into n_regions contiguous fractional-depth regions,
    same rule as the cross-arch champion scan (region k = layers whose depth
    fraction is in [k/N, (k+1)/N)). Returns [(k, set_of_layers), ...]."""
    out = []
    for k in range(max(1, n_regions)):
        lo = (k * n_layers) // n_regions
        hi = ((k + 1) * n_layers) // n_regions
        if hi > lo:
            out.append((k, set(range(lo, hi))))
    return out
# Resolve the tuned config ONCE into the exact args blend_values takes (mirrors
# cross_arch_probe.run_model): per-layer alpha_map -> alpha is a {layer: α} dict;
# per-head head_map -> scalar alpha + {layer: [kv_head,...]}. VALUE-ONLY either way.
CHAMP_ALPHA = None
CHAMP_HEAD_MAP = None
if CHAMPION_CFG:
    if "alpha_map" in CHAMPION_CFG:
        CHAMP_ALPHA = {int(k): float(v)
                       for k, v in CHAMPION_CFG["alpha_map"].items()}
        CHAMP_HEAD_MAP = None
    elif "head_map" in CHAMPION_CFG:
        CHAMP_ALPHA = float(CHAMPION_CFG.get("alpha", E_ALPHA))
        CHAMP_HEAD_MAP = {int(k): [int(h) for h in v]
                          for k, v in CHAMPION_CFG["head_map"].items()}


def load_trajectories():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    return [list(r) for r in df["messages"]]


def find_cut(tokenizer, msgs):
    """Largest assistant-turn index whose prefix stays within budget and
    leaves >=15% of tokens after it; returns (ctx_msgs, target_msg, meta)."""
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    if not (MIN_TOK <= len(ids) <= MAX_TOK):
        return None
    starts = message_token_starts(tokenizer, ids, len(msgs))
    # candidate cuts: assistant turns in the 60-85% token range
    cands = [i for i in range(4, len(msgs))
             if msgs[i]["role"] == "assistant"
             and 0.60 * len(ids) <= starts[i] <= 0.85 * len(ids)]
    if not cands:
        return None
    cut = cands[len(cands) // 2]
    ctx = msgs[:cut]
    if ctx[-1]["role"] != "user":
        return None
    return ctx, msgs[cut], {"n_tokens_full": len(ids), "cut_msg": cut,
                            "n_ctx_tokens": starts[cut]}


ERR_PAT = re.compile(r"error|traceback|failed|exception", re.I)
CMD_PAT = re.compile(r"<execute_bash>\s*(.+?)\s*</execute_bash>", re.S)


def failed_commands(msgs, lo, hi):
    """Commands in msgs[lo:hi] whose following user turn shows an error."""
    out = set()
    for i in range(lo, min(hi, len(msgs) - 1)):
        if msgs[i]["role"] != "assistant":
            continue
        m = CMD_PAT.search(str(msgs[i]["content"]))
        if m and ERR_PAT.search(str(msgs[i + 1]["content"])[:2000]):
            out.add(m.group(1).strip().split("\n")[0][:120])
    return out


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    base = rope_base(model)
    _tcfg = getattr(model.config, "text_config", model.config)
    N_LAYERS = _tcfg.num_hidden_layers
    REGION_SETS = (region_layer_sets(N_LAYERS, PROFILE_REGIONS)
                   if PROFILE_REGIONS else [])

    trajs = load_trajectories()
    outdir = Path(f"results/swegym_{TAG}_{SUMM_TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- PROVENANCE MANIFEST (read from the RUNTIME, not inferred from TAG). --
    # dtype comes from a real parameter tensor; quantization from the loaded
    # config; git commit from subprocess -- so a wrong-model/dtype/condition run
    # is self-evident in every result file + the run-level manifest.json.
    mp = prov.capture_model_provenance(model, model_id_hint=MODEL)
    manifest = prov.build_manifest(
        model_provenance=mp,
        dtype_env=os.environ.get("SC_LOAD_DTYPE", "bfloat16"),
        harness="src/run_swegym_hf.py",
        intervention={
            # PRIMARY graft arm (kept for continuity); FULL per-arm provenance is
            # in "grafted_arms" and in each result's arms.<name>.intervention.
            "arm": "E-tuned",
            "graft_type": "value",          # keys neutral (alpha_K=0)
            "alpha": E_ALPHA,               # SCALAR alpha -- NOT a tuned champion
            "champion_config_path": CHAMPION_CFG_PATH,
            "champion_config_sha256": prov.sha256_file(CHAMPION_CFG_PATH),
            "champion_label": (CHAMPION_CFG.get("label") if CHAMPION_CFG
                               else None),
            "alignment": "difflib positional-within-region (tail+summary regions)",
            "grafted_arms": [
                {"arm": "E-tuned", "kind": "scalar", "alpha": E_ALPHA,
                 "champion": None,
                 "note": "flat value-graft at alpha; NOT the tuned champion"},
            ] + ([
                {"arm": "E-champion", "kind": "champion",
                 "family": ("alpha_map" if "alpha_map" in CHAMPION_CFG
                            else "head_map"),
                 "alpha": (None if isinstance(CHAMP_ALPHA, dict)
                           else CHAMP_ALPHA),
                 "champion_config_path": CHAMPION_CFG_PATH,
                 "champion_config_sha256": prov.sha256_file(CHAMPION_CFG_PATH),
                 "champion_label": CHAMPION_CFG.get("label")},
            ] if CHAMPION_CFG else []),
            "note": ("E-tuned = SCALAR value-graft at alpha=%.3f (NOT the tuned "
                     "champion). E-champion (only if SC_CHAMPION_CONFIG set) = the "
                     "tuned per-layer/per-head map. Other arms: A(full), "
                     "B(compacted), B-min-pack, H-pack." % E_ALPHA),
            # IN-DOMAIN OPTIMIZATION surface (score-only arms).
            "alpha_sweep": ALPHA_SWEEP or None,
            "placebo": ("position-shuffle per swept alpha (arms P-a{alpha})"
                        if SWE_PLACEBO else None),
            "profile_regions": PROFILE_REGIONS or None,
            "profile_alpha": PROFILE_ALPHA if PROFILE_REGIONS else None,
            "held_out_split": {
                "rule": "sha256(SC_TRAJ_SPLIT_SEED:idx) parity -> tune|eval",
                "seed": SPLIT_SEED,
                "note": ("select alpha / build the layer champion on TUNE, report "
                         "on the DISJOINT EVAL split; ~75 usable trajectories so "
                         "each split is thin -- overfitting caveat in the analysis"),
            },
        },
        metric=prov.METRIC_SWEGYM_TF_LOGPROB,
        condition={
            "summary_kind": SUMM_TAG,       # brief=handicapped | prod=faithful
            "summary_request_sha256": prov.sha256_text(SUMM_REQ),
            "summary_source": "self-gen (model summarizes its own trajectory)",
        },
        corpus={
            "name": "SWE-Gym/OpenHands trajectories",
            "parquet": PARQUET,
            "split": f"shard {_shard}",
            "n_requested": N_TRAJ,
            "instance_ids": None,           # filled in the run manifest at the end
        },
    )
    print("MANIFEST model.repo_id=%s dtype=%s quant=%s summary=%s alpha=%s "
          "git=%s" % (manifest["model"]["repo_id"], manifest["load"]["dtype"],
                      manifest["load"]["quantization"], SUMM_TAG, E_ALPHA,
                      manifest["code"]["git_commit"]), flush=True)
    if CHAMPION_CFG:
        print("CHAMPION arm ACTIVE: E-champion <- %s (label=%s, family=%s, "
              "sha=%s) applied value-only alongside scalar E-tuned(alpha=%.3f)."
              % (CHAMPION_CFG_PATH, CHAMPION_CFG.get("label"),
                 "alpha_map" if "alpha_map" in CHAMPION_CFG else "head_map",
                 (prov.sha256_file(CHAMPION_CFG_PATH) or "")[:16], E_ALPHA),
              flush=True)
    else:
        print("CHAMPION arm INACTIVE (SC_CHAMPION_CONFIG unset): scalar E-tuned "
              "(alpha=%.3f) only." % E_ALPHA, flush=True)
    if ALPHA_SWEEP or PROFILE_REGIONS:
        print("IN-DOMAIN OPT: alpha_sweep=%s placebo=%s profile_regions=%s "
              "(profile_alpha=%.2f) split_seed=%s" % (
                  ALPHA_SWEEP or None, SWE_PLACEBO,
                  PROFILE_REGIONS or None, PROFILE_ALPHA, SPLIT_SEED), flush=True)
    prov.write_run_manifest(outdir, manifest)
    scored_idxs = []
    done = 0
    for idx in range(SHARD_K, len(trajs), SHARD_N):
        if done >= N_TRAJ:
            break
        outfile = outdir / f"t{idx:04d}.json"
        if outfile.exists():
            try:
                json.load(open(outfile)); scored_idxs.append(idx)
                done += 1; continue
            except Exception:
                outfile.unlink()
        msgs = trajs[idx]
        built = None
        try:
            built = find_cut(tokenizer, msgs)
        except AssertionError:
            pass
        if built is None:
            continue
        ctx, target, meta = built
        t0 = time.time()
        try:
            summary = generate_summary_hf(model, tokenizer, ctx,
                                          request=SUMM_REQ)
        except AssertionError as e:
            print(f"t{idx}: summary failed ({e})", flush=True); continue
        ctx_ids = canonical_ids(tokenizer, ctx, renderer=render_hf)
        starts = message_token_starts(tokenizer, ctx_ids, len(ctx))
        # tail = message boundary nearest 75% of ctx
        tgt75 = 0.75 * len(ctx_ids)
        tail_start_msg = min(range(1, len(ctx)),
                             key=lambda i: abs(starts[i] - tgt75))
        b_msgs = build_b_messages(ctx, summary["text"], tail_start_msg)
        b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
        s_leak = False  # no single gold string here
        # target: teacher-force the rendered next assistant turn
        gp_full = render_hf(tokenizer, ctx, True)
        assert gp_full[: len(ctx_ids)] == ctx_ids
        gp_suffix = gp_full[len(ctx_ids):]
        tgt_ids = tokenizer(str(target["content"]),
                            add_special_tokens=False).input_ids[:600]
        failed_evicted = sorted(failed_commands(ctx, 1, tail_start_msg))

        res = {"arms": {}}

        def eval_arm(name, snap, next_pos, intervention=None):
            cache = rebuild_cache(snap, DynamicCache)
            feed = gp_suffix + tgt_ids[:-1]
            pos = torch.arange(next_pos, next_pos + len(feed),
                               device=model.device)[None]
            lps = tf_logprobs(model, cache, feed, tgt_ids, position_ids=pos)
            gen = answer_hf(model, tokenizer, snap, gp_suffix, next_pos,
                            max_tokens=200)
            m = CMD_PAT.search(gen)
            first_cmd = (m.group(1).strip().split("\n")[0][:120] if m else "")
            res["arms"][name] = {
                # BORN-ANNOTATED: each grafted arm records EXACTLY which
                # intervention produced its number (scalar α vs tuned champion).
                "intervention": intervention,
                "tf_mean": sum(lps) / len(lps),
                "gen": gen[:500],
                "gen_first_cmd": first_cmd,
                "repeats_failed": first_cmd in failed_evicted if first_cmd
                                  else False,
            }

        def score_arm(name, snap, next_pos, intervention=None):
            """SCORE-ONLY (no 200-token generation): just the teacher-forced mean
            logprob of the true next action. Used for the alpha-sweep / placebo /
            per-region profiling arms so a single pass yields the whole
            optimization surface cheaply (the metric is tf_mean; generation is only
            for the offline behavioral flags on the primary arms)."""
            cache = rebuild_cache(snap, DynamicCache)
            feed = gp_suffix + tgt_ids[:-1]
            pos = torch.arange(next_pos, next_pos + len(feed),
                               device=model.device)[None]
            lps = tf_logprobs(model, cache, feed, tgt_ids, position_ids=pos)
            res["arms"][name] = {"intervention": intervention,
                                 "tf_mean": sum(lps) / len(lps)}

        # A
        a_snap, _ = hf_prefill_ids(model, ctx_ids)
        eval_arm("A", a_snap, len(ctx_ids), intervention={"arm": "A-full"})
        del a_snap
        # B + E
        b_snap, _ = hf_prefill_ids(model, b_ids)
        eval_arm("B", b_snap, len(b_ids), intervention={"arm": "B-compacted"})
        b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
        regions = [
            ((b_starts[2], len(b_ids)), (starts[tail_start_msg],
                                         summary["conv_end"])),
            ((b_starts[1], b_starts[2]), (summary["s_start"],
                                          summary["s_end"])),
        ]
        pairs = build_alignment(b_ids, summary["old_ids"],
                                set(tokenizer.all_special_ids), regions)
        # SCALAR arm ("E-tuned" -- a misnomer kept for continuity: it is FLAT
        # α=E_ALPHA, NOT the tuned champion).
        e_snap = blend_values(b_snap, summary["snapshot"], pairs, E_ALPHA)
        eval_arm("E-tuned", e_snap, len(b_ids), intervention={
            "arm": "E-tuned", "graft_type": "value", "alpha": E_ALPHA,
            "champion": None,
            "note": "SCALAR value-graft at flat alpha; NOT the tuned champion"})
        del e_snap
        # CHAMPION arm ("E-champion" -- the TUNED per-layer alpha_map / per-head
        # head_map). Only added when SC_CHAMPION_CONFIG is set. Same B snapshot,
        # same pairs -> a clean paired comparison vs both B and the scalar arm.
        if CHAMPION_CFG:
            ec_snap = blend_values(b_snap, summary["snapshot"], pairs,
                                   CHAMP_ALPHA, head_map=CHAMP_HEAD_MAP)
            eval_arm("E-champion", ec_snap, len(b_ids), intervention={
                "arm": "E-champion", "graft_type": "value",
                "alpha": (None if isinstance(CHAMP_ALPHA, dict) else CHAMP_ALPHA),
                "champion": {
                    "config_path": CHAMPION_CFG_PATH,
                    "config_sha256": prov.sha256_file(CHAMPION_CFG_PATH),
                    "label": CHAMPION_CFG.get("label"),
                    "family": ("alpha_map" if "alpha_map" in CHAMPION_CFG
                               else "head_map")}})
            del ec_snap
        # ---- IN-DOMAIN OPTIMIZATION arms (score-only; b_snap still alive) --------
        # ALPHA-SWEEP: value graft at each swept alpha -> "E-a{alpha}".
        import random as _random
        for a in ALPHA_SWEEP:
            sw = blend_values(b_snap, summary["snapshot"], pairs, a)
            score_arm(f"E-a{a:g}", sw, len(b_ids), intervention={
                "arm": f"E-a{a:g}", "graft_type": "value", "alpha": a,
                "champion": None, "kind": "alpha-sweep"})
            del sw
            # PLACEBO at this alpha: SAME slots, source values position-shuffled
            # (seeded, per-trajectory) -> content-specificity = E-a - P-a.
            if SWE_PLACEBO:
                olds = [o for _, o in pairs]
                rng = _random.Random(PLACEBO_SEED + idx)
                perm = olds[:]
                rng.shuffle(perm)
                shuf_pairs = [(n, perm[i]) for i, (n, _) in enumerate(pairs)]
                pb = blend_values(b_snap, summary["snapshot"], shuf_pairs, a)
                score_arm(f"P-a{a:g}", pb, len(b_ids), intervention={
                    "arm": f"P-a{a:g}", "graft_type": "value-placebo",
                    "alpha": a, "placebo": "position-shuffle",
                    "placebo_seed": PLACEBO_SEED + idx, "kind": "placebo"})
                del pb
        # PER-REGION PROFILING: graft restricted to one fractional-depth region at
        # a time -> "R{k}" (marginal next-action recovery of that region). The CPU
        # analysis builds a SWE-Gym-tuned per-layer champion from the TUNE-split
        # marginals; here we only record each region's marginal per trajectory.
        for k, lset in REGION_SETS:
            rr = blend_values(b_snap, summary["snapshot"], pairs, PROFILE_ALPHA,
                              layer_set=lset)
            score_arm(f"R{k}", rr, len(b_ids), intervention={
                "arm": f"R{k}", "graft_type": "value", "alpha": PROFILE_ALPHA,
                "kind": "region-profile", "region_index": k,
                "layers": sorted(lset)})
            del rr
        del b_snap
        # packed arms (gp suffix belongs to the chat frame; packed contexts
        # still get the same generation-prompt tokens after their storage)
        bmp = bmin_pack_ids(summary, ctx_ids)
        bmp_snap, _ = hf_prefill_ids(model, bmp)
        eval_arm("B-min-pack", bmp_snap, len(bmp)); del bmp_snap
        hp = arm_h_pack_snapshot_hf(summary, base)
        eval_arm("H-pack", hp, hp[0][0].shape[2]); del hp

        tmp = outfile.with_suffix(".tmp")
        result = {"idx": idx, "meta": meta, "model": MODEL,
                  "dtype": manifest["load"]["dtype"],  # RUNTIME-read, not "bfloat16"
                  "split": traj_split(idx),   # deterministic held-out tune/eval
                  "summary_tokens": len(summary["gen_ids"]),
                  "failed_evicted_commands": failed_evicted,
                  "n_target_tokens": len(tgt_ids), **res}
        prov.stamp(result, manifest)
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1)
        tmp.rename(outfile)
        scored_idxs.append(idx)
        done += 1
        print(f"== t{idx:04d} done in {time.time()-t0:.0f}s "
              f"[{done}/{N_TRAJ}] A={res['arms']['A']['tf_mean']:.3f} "
              f"B={res['arms']['B']['tf_mean']:.3f}", flush=True)
        del summary
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Finalize the run-level manifest with the ACTUAL scored instance ids.
    manifest["corpus"]["instance_ids"] = sorted(scored_idxs)
    manifest["corpus"]["n_scored"] = len(scored_idxs)
    prov.write_run_manifest(outdir, manifest)
    print("RUN MANIFEST finalized: %d trajectories scored -> %s" % (
        len(scored_idxs), outdir / "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
