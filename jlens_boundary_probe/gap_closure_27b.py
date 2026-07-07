#!/usr/bin/env python3
"""Behavioral gap-closure on Qwen3.6-27B, ported to the J-lens probe machinery.

This is a straight port of ``src/gap_closure_cat.py`` (our (E-B)/(A-B) behavioral
gap-closure metric on teacher-forced GOLD-continuation logprob, stratified by
sense/referent/stance) onto the SAME 27B model + cache-surgery code path used by
the J-lens boundary probes. The point is to establish the behavioral
dissociation on the identical model the lens is fitted to, using the *probe's*
proven graft machinery rather than a second implementation.

Per plant we measure the mean teacher-forced logprob of the GOLD continuation
under three states:

  * ``A`` / ``full``  -- full original context (no compaction),
  * ``B`` / ``fresh`` -- fresh-compacted context (system + model-written summary
                         + retained tail), and
  * ``E`` / ``graft`` -- fresh-compacted context with the aligned summary-token
                         *value*-cache entries V-grafted (alpha, default 0.75)
                         from the write-time summary path.

Gap-closure = ``(lp_graft - lp_fresh) / (lp_full - lp_fresh)``.

The graft path is *exactly* the probe's: ``snapshot_standard_kv`` +
``alignment_pairs_by_exact_tokens`` + ``blend_values`` from ``boundary_probe``
(the hybrid list-cache / layer-cache aware Qwen3.6 V-graft). Teacher forcing
mirrors ``src/kvlib_hf.tf_logprobs`` (last ``len(target)`` logit positions score
the target), rebuilt on top of a probe cache snapshot.

Self-test locally with ``--dry-run`` (no torch): it loads every conversation,
selects the sense/referent/stance plants, and prints the constructed specs
(plant id, category, probe, and the GOLD continuation that will be teacher
forced). The summaries are model-generated on the pod, so they are not shown in
the dry run.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NOTE: ``boundary_probe`` imports torch/transformers, absent on a CPU-only box.
# To keep ``--dry-run`` runnable with no ML deps we do NOT import it (or torch)
# at module scope; the heavy imports happen lazily inside ``main`` past the
# dry-run gate. Only stdlib is needed to construct and print the specs.

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
CATS = ("sense", "referent", "stance")


@dataclass(frozen=True)
class PlantSpec:
    """One teacher-forced GOLD-continuation gap-closure target."""

    conv_id: str
    plant_id: str
    category: str
    probe: str
    gold: str
    keywords: tuple[str, ...]
    contaminated: bool


def _load_conversation(data_dir: Path, path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def conversation_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("c*.json"))


def plant_specs_for(conv: dict[str, Any], include_contaminated: bool) -> list[PlantSpec]:
    out: list[PlantSpec] = []
    conv_id = conv["id"]
    for pl in conv.get("plants", []):
        if pl.get("category") not in CATS:
            continue
        contaminated = bool(pl.get("contaminated_early") or pl.get("contaminated_tail"))
        if contaminated and not include_contaminated:
            continue
        gold = str(pl.get("gold", "")).strip()
        if not gold:
            continue
        out.append(
            PlantSpec(
                conv_id=conv_id,
                plant_id=pl["id"],
                category=pl["category"],
                probe=pl["probe"],
                gold=gold,
                keywords=tuple(str(k) for k in pl.get("keywords", [])),
                contaminated=contaminated,
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default="outputs/qwen36_gap_closure.json")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--max-new-summary-tokens", type=int, default=256)
    p.add_argument("--max-gold-tokens", type=int, default=80)
    p.add_argument(
        "--include-contaminated",
        action="store_true",
        default=False,
        help="include plants flagged contaminated_early/contaminated_tail (default: skip)",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="construct + print gap-closure specs without loading the model")
    return p.parse_args()


def dry_run(data_dir: Path, include_contaminated: bool) -> int:
    print("=" * 72)
    print("GAP-CLOSURE 27B  --  DRY RUN (no model / no torch)")
    print(f"data dir: {data_dir}")
    print(f"categories: {CATS}   include_contaminated={include_contaminated}")
    print("=" * 72)
    paths = conversation_paths(data_dir)
    if not paths:
        print("NO CONVERSATIONS FOUND")
        return 1
    total = 0
    by_cat: dict[str, int] = {c: 0 for c in CATS}
    all_ok = True
    for path in paths:
        conv = _load_conversation(data_dir, path)
        specs = plant_specs_for(conv, include_contaminated)
        print()
        print(f"### CONV {conv['id']}  [{conv.get('title')}]  ->  {len(specs)} plant(s)")
        for s in specs:
            total += 1
            by_cat[s.category] += 1
            gold_ok = len(s.gold) > 0
            all_ok = all_ok and gold_ok
            print(f"  - {s.plant_id}  ({s.category})"
                  + ("  [CONTAMINATED]" if s.contaminated else ""))
            print(f"      probe        : {s.probe}")
            print(f"      GOLD (forced): {s.gold}")
            print(f"      keywords     : {list(s.keywords)}")
            if not gold_ok:
                print("      => PROBLEM: empty gold")
    print()
    print("=" * 72)
    print(f"DRY RUN {'PASSED' if all_ok else 'FAILED'}: {total} gap-closure plant spec(s) "
          f"across {len(paths)} conversation(s).")
    print(f"  per-category: " + ", ".join(f"{c}={by_cat[c]}" for c in CATS))
    print("=" * 72)
    return 0 if all_ok else 1


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.dry_run:
        raise SystemExit(dry_run(data_dir, args.include_contaminated))

    # --- heavy imports only past the dry-run gate ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from boundary_probe import (
        alignment_pairs_by_exact_tokens,
        blend_values,
        build_b_messages,
        dtype_from_name,
        find_subsequence,
        generate_summary,
        model_input_device,
        rebuild_standard_cache,
        render_ids,
        snapshot_standard_kv,
        token_tensor,
    )

    def tf_gold_logprob(model, snap, feed_ids, target_ids, npos) -> float:
        """Mean teacher-forced logprob of ``target_ids``.

        Mirrors ``src/kvlib_hf.tf_logprobs``: feed = prefix_suffix + target[:-1];
        the last ``len(target)`` logit positions score the target. The cache is
        rebuilt from the probe snapshot via the probe's own ``rebuild_standard_cache``.
        """
        n = len(target_ids)
        cache = rebuild_standard_cache(snap, model)
        dev = model_input_device(model)
        pos = torch.arange(npos, npos + len(feed_ids), device=dev)[None]
        with torch.no_grad():
            out = model(
                input_ids=token_tensor(model, feed_ids),
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
                logits_to_keep=n,
            )
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        tgt = torch.tensor(target_ids, device=lp.device)
        vals = lp[torch.arange(n, device=lp.device), tgt]
        return float(vals.sum().item() / max(1, n))

    torch_dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    result: dict[str, Any] = {
        "model": args.model,
        "design": (
            "Behavioral gap-closure (E-B)/(A-B) on teacher-forced GOLD-continuation "
            "logprob, ported from src/gap_closure_cat.py to the J-lens probe's 27B "
            "cache-snapshot + V-graft machinery (boundary_probe.snapshot_standard_kv / "
            "alignment_pairs_by_exact_tokens / blend_values). A=full context, "
            "B=fresh-compacted (system+summary+retained tail), E=B with aligned "
            "summary-token value-cache V-grafted from the write-time summary path. "
            "Teacher forcing mirrors kvlib_hf.tf_logprobs. Stratified by "
            "sense/referent/stance."
        ),
        "alpha": args.alpha,
        "categories": list(CATS),
        "include_contaminated": args.include_contaminated,
        "conversations": {},
        "plants": {},
    }

    paths = conversation_paths(data_dir)
    for path in paths:
        conv = json.loads(path.read_text())
        cid = conv["id"]
        specs = plant_specs_for(conv, args.include_contaminated)
        if not specs:
            continue
        print(f"CONV {cid}: {len(specs)} plant(s)", flush=True)
        try:
            msgs = conv["messages"][:-1]
            tsm = conv["sections"]["middle_end_msg"]

            summary = generate_summary(model, tokenizer, msgs, args.max_new_summary_tokens)
            summary_text = summary["text"]
            b_msgs = build_b_messages(msgs, summary_text, tsm)

            full_ids = render_ids(tokenizer, msgs, False)
            b_ids = render_ids(tokenizer, b_msgs, False)
            summary_text_ids = tokenizer(summary_text, add_special_tokens=False).input_ids
            b_summary_start = find_subsequence(b_ids, summary_text_ids)
            if b_summary_start is None:
                raise ValueError("could not locate summary text in fresh-compacted ids")
            b_summary_range = (b_summary_start, b_summary_start + len(summary_text_ids))

            pairs = alignment_pairs_by_exact_tokens(
                b_ids,
                summary["old_ids"],
                b_summary_range,
                (summary["s_start"], summary["s_end"]),
            )
            if not pairs:
                raise ValueError("no exact summary-token alignment pairs")

            a_snap, _ = snapshot_standard_kv(model, full_ids)
            b_snap, _ = snapshot_standard_kv(model, b_ids)
            old_snap, _ = snapshot_standard_kv(model, summary["old_ids"])
            e_snap = blend_values(b_snap, old_snap, pairs, args.alpha)

            def suffix(mm: list[dict[str, str]], probe: str) -> list[int]:
                gen = render_ids(tokenizer, mm + [{"role": "user", "content": probe}], True)
                cn = render_ids(tokenizer, mm, False)
                if gen[: len(cn)] != cn:
                    raise ValueError("probe generation prompt does not extend prefix")
                return gen[len(cn):]

            conv_res: dict[str, Any] = {
                "title": conv.get("title"),
                "summary_text": summary_text,
                "alignment_pairs": len(pairs),
                "token_counts": {
                    "full": len(full_ids),
                    "fresh_compacted": len(b_ids),
                    "old_with_summary": len(summary["old_ids"]),
                    "summary_tokens": len(summary_text_ids),
                },
                "plants": {},
            }

            for s in specs:
                try:
                    tgt = tokenizer(s.gold, add_special_tokens=False).input_ids[: args.max_gold_tokens]
                    if len(tgt) < 2:
                        conv_res["plants"][s.plant_id] = {
                            "category": s.category,
                            "error": "gold continuation shorter than 2 tokens",
                        }
                        continue
                    sa = suffix(msgs, s.probe)
                    sb = suffix(b_msgs, s.probe)
                    lp_full = tf_gold_logprob(model, a_snap, sa + tgt[:-1], tgt, len(full_ids))
                    lp_fresh = tf_gold_logprob(model, b_snap, sb + tgt[:-1], tgt, len(b_ids))
                    lp_graft = tf_gold_logprob(model, e_snap, sb + tgt[:-1], tgt, len(b_ids))
                    denom = lp_full - lp_fresh
                    gc = (lp_graft - lp_fresh) / denom if abs(denom) > 1e-6 else None
                    row = {
                        "conversation_id": cid,
                        "category": s.category,
                        "probe": s.probe,
                        "gold": s.gold,
                        "gold_token_count": len(tgt),
                        "contaminated": s.contaminated,
                        "lp_full": lp_full,
                        "lp_fresh": lp_fresh,
                        "lp_graft": lp_graft,
                        "gap_closure": gc,
                    }
                    conv_res["plants"][s.plant_id] = {k: v for k, v in row.items()
                                                      if k not in ("conversation_id",)}
                    result["plants"][s.plant_id] = row
                    print(f"  {s.plant_id} ({s.category}): full={lp_full:.4f} "
                          f"fresh={lp_fresh:.4f} graft={lp_graft:.4f} gc={gc}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    conv_res["plants"][s.plant_id] = {
                        "category": s.category,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    print(f"  ERROR {s.plant_id}: {type(exc).__name__}: {exc}", flush=True)

            result["conversations"][cid] = conv_res

            del a_snap, b_snap, old_snap, e_snap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            result["conversations"][cid] = {
                "title": conv.get("title"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR CONV {cid}: {type(exc).__name__}: {exc}", flush=True)

    # Category-level aggregate over completed plants.
    agg: dict[str, dict[str, Any]] = {c: {"n": 0, "gap_closure_sum": 0.0, "gap_closure_n": 0} for c in CATS}
    for row in result["plants"].values():
        c = row.get("category")
        if c not in agg:
            continue
        agg[c]["n"] += 1
        gc = row.get("gap_closure")
        if isinstance(gc, (int, float)):
            agg[c]["gap_closure_sum"] += float(gc)
            agg[c]["gap_closure_n"] += 1
    for c, a in agg.items():
        a["gap_closure_mean"] = (a["gap_closure_sum"] / a["gap_closure_n"]) if a["gap_closure_n"] else None
    result["category_summary"] = agg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
