"""Category-stratified gap-closure (B4): per evicted-content probe, measure
teacher-forced logprob of the GOLD continuation under A (full), B (compacted),
E (grafted). Gap-closure = (E-B)/(A-B). Stratify by sense/referent/stance.
Independent, judge-free corroboration of F1. Runs on the pod (needs model).
Env: SC_HF_MODEL, SC_GC_ALPHA (default 0.75). Out: results/gap_closure_cat/<conv>.json
"""
import json, os, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, "src")
from arms_common import (SUMMARY_REQUEST, build_alignment, build_b_messages,
    canonical_ids, message_token_starts, render_hf)
from arms_hf import generate_summary_hf, hf_prefill_ids
from kvlib_hf import rebuild_cache, tf_logprobs, blend_values

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
ALPHA = float(os.environ.get("SC_GC_ALPHA", "0.75"))
CATS = ("sense", "referent", "stance")

def tf(model, snap, feed, targets, npos):
    cache = rebuild_cache(snap, DynamicCache)
    pos = torch.arange(npos, npos+len(feed), device=model.device)[None]
    return sum(tf_logprobs(model, cache, feed, targets, position_ids=pos)) / max(1,len(targets))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto"); model.eval()
    out = Path("results/gap_closure_cat"); out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p)); cid = c["id"]
        of = out/f"{cid}.json"
        if of.exists(): print(cid,"done"); continue
        msgs = c["messages"][:-1]; tsm = c["sections"]["middle_end_msg"]
        ids = canonical_ids(tok, msgs, renderer=render_hf)
        starts = message_token_starts(tok, ids, len(msgs))
        summ = generate_summary_hf(model, tok, msgs, request=SUMMARY_REQUEST)
        b_msgs = build_b_messages(msgs, summ["text"], tsm)
        b_ids = canonical_ids(tok, b_msgs, renderer=render_hf)
        b_starts = message_token_starts(tok, b_ids, len(b_msgs))
        regions=[((b_starts[2],len(b_ids)),(starts[tsm],summ["conv_end"])),
                 ((b_starts[1],b_starts[2]),(summ["s_start"],summ["s_end"]))]
        pairs = build_alignment(b_ids, summ["old_ids"], set(tok.all_special_ids), regions)
        a_snap,_ = hf_prefill_ids(model, ids)
        b_snap,_ = hf_prefill_ids(model, b_ids)
        e_snap = blend_values(b_snap, summ["snapshot"], pairs, ALPHA)
        res={}
        for pl in c["plants"]:
            if pl["category"] not in CATS: continue
            if pl.get("contaminated_early") or pl.get("contaminated_tail"): continue
            gold = str(pl.get("gold","")).strip()
            if not gold: continue
            tgt = tok(gold, add_special_tokens=False).input_ids[:80]
            if len(tgt)<2: continue
            # continuation appended after the probe; feed = probe_suffix + gold[:-1]
            def suffix(mm):
                full = render_hf(tok, mm+[{"role":"user","content":pl["probe"]}], True)
                cn = canonical_ids(tok, mm, renderer=render_hf)
                return full[len(cn):]
            sa=suffix(msgs); sb=suffix(b_msgs)
            la = tf(model,a_snap, sa+tgt[:-1], tgt, len(ids))
            lb = tf(model,b_snap, sb+tgt[:-1], tgt, len(b_ids))
            le = tf(model,e_snap, sb+tgt[:-1], tgt, len(b_ids))
            gc = (le-lb)/(la-lb) if abs(la-lb)>1e-6 else None
            res[pl["id"]]={"category":pl["category"],"lp_A":la,"lp_B":lb,"lp_E":le,"gap_closure":gc}
        json.dump(res, open(of,"w"), indent=1)
        print(f"== {cid}: {len(res)} probes", flush=True)
        del a_snap,b_snap,e_snap; torch.cuda.empty_cache()
    print("GAP_CLOSURE_DONE", flush=True)

if __name__=="__main__": main()
