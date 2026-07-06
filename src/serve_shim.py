"""E-track serving shim: OpenAI-compatible chat endpoint with server-side
compaction + optional ValueGraft, hosting the bf16 model on a pod.

The agent harness (OpenHands, locally) talks to this like a normal API and
keeps sending the FULL message list every call. The shim decides, per mode,
what the model actually sees:

  mode A : full context, no compaction (oracle)
  mode B : when rendered context exceeds SC_COMPACT_AT tokens, compact:
           summary of everything but the last SC_TAIL_KEEP tokens + tail,
           re-encoded fresh (production compaction)
  mode E : B + ValueGraft (alpha SC_E_ALPHA) from the pre-compaction cache
           at aligned twin positions

Mode is selected by the requested model name suffix: "sc-A", "sc-B", "sc-E".
Compaction state is per-session; session key = SHA1 of the first user
message. Caches are kept on GPU for the active session only (LRU size 1 —
one agent task per shim pod at a time; run one task per pod).

Endpoint: POST /v1/chat/completions (subset: messages, max_tokens,
temperature==0 assumed, stream unsupported). GET /health.

Env: SC_HF_MODEL, SC_COMPACT_AT (default 9000), SC_TAIL_KEEP (default 2500),
SC_E_ALPHA (default 0.75), SC_PORT (default 8000).

Run: python3 -u src/serve_shim.py
"""

import hashlib
import json
import os
import sys
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST_BRIEF,
    build_alignment,
    build_b_messages,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import answer_hf, generate_summary_hf, hf_prefill_ids
from kvlib_hf import blend_values

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
COMPACT_AT = int(os.environ.get("SC_COMPACT_AT", "9000"))
TAIL_KEEP = int(os.environ.get("SC_TAIL_KEEP", "2500"))
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))
PORT = int(os.environ.get("SC_PORT", "8000"))
INCR_CACHE = os.environ.get("SC_INCR_CACHE") == "1"

_lock = threading.Lock()
_model = None
_tok = None
# per-session state: {skey: {"compacted_msgs", "b_snap", "n_compactions"}}
_sessions = {}


def _load():
    global _model, _tok
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    _model.eval()
    print("model loaded;",
          torch.cuda.memory_allocated() // (1 << 20), "MiB", flush=True)


def _skey(messages):
    for m in messages:
        if m["role"] == "user":
            return hashlib.sha1(str(m["content"]).encode()).hexdigest()[:16]
    return "nouser"


def _norm(messages):
    out = []
    for m in messages:
        role = m["role"] if m["role"] in ("system", "user", "assistant") \
            else "user"
        out.append({"role": role, "content": str(m.get("content") or "")})
    # merge consecutive same-role (OpenHands emits tool results as user)
    merged = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)
    return merged


def _generate(msgs, max_tokens, mode, sess, alpha=E_ALPHA, compact_at=COMPACT_AT):
    """Return (text, dbg). msgs = normalized full history from the agent."""
    ids = canonical_ids(_tok, msgs, renderer=render_hf)
    dbg = {"mode": mode, "alpha": alpha, "compact_at": compact_at, "full_tokens": len(ids),
           "n_compactions": sess.get("n_compactions", 0)}
    dbg["alpha_kind"] = "map" if isinstance(alpha, dict) else alpha
    if mode == "A" or len(ids) <= compact_at:
        snap = None
        if INCR_CACHE:
            ic = sess.get("icache")
            if ic and ids[: len(ic["ids"])] == ic["ids"]:
                from kvlib_hf import prefill as _pf, snapshot_cache as _sc, rebuild_cache as _rc
                from transformers import DynamicCache as _DC
                cache = _rc(ic["snap"], _DC)
                new = ids[len(ic["ids"]):]
                if new:
                    pos = torch.arange(len(ic["ids"]), len(ids),
                                       device=_model.device)[None]
                    cache, _ = _pf(_model, torch.tensor([new], device=_model.device),
                                   past=cache, position_ids=pos)
                snap = _sc(cache)
                dbg["icache"] = "HIT"
        if snap is None:
            snap, _ = hf_prefill_ids(_model, ids)
            if INCR_CACHE:
                dbg["icache"] = "MISS"
        if INCR_CACHE:
            for other in list(_sessions):
                _sessions[other].pop("icache", None)
            sess["icache"] = {"ids": list(ids), "snap": snap}
            n_cached = sum(1 for v in _sessions.values() if "icache" in v)
            assert n_cached <= 1, f"icache bound violated: {n_cached}"
        gp = render_hf(_tok, msgs, True)
        text = answer_hf(_model, _tok, snap, gp[len(ids):], len(ids),
                         max_tokens=max_tokens)
        if not INCR_CACHE:
            del snap
        torch.cuda.empty_cache()
        return text, dbg

    # B/E: compact — summary of everything except the last TAIL_KEEP tokens.
    # Boundary FREEZES at first compaction (production semantics) and only
    # rolls forward when the tail regrows past threshold; the summary (and
    # its write-time snapshot, for E) is cached per frozen prefix.
    starts = message_token_starts(_tok, ids, len(msgs))
    cache = sess.get("ccache")
    tsm = None
    if cache and cache["n_msgs"] <= len(msgs) \
            and msgs[: cache["n_msgs"]] == cache["prefix_msgs"] \
            and len(ids) - starts[cache["tsm"]] <= compact_at - min(1000, TAIL_KEEP):
        tsm = cache["tsm"]
        summary = cache["summary"]
        dbg["summary_cache"] = "HIT"
    if tsm is None:
        target = len(ids) - TAIL_KEEP
        tsm = min(range(1, len(msgs)), key=lambda i: abs(starts[i] - target))
        tsm = max(2, min(tsm, len(msgs) - 2))
        summary = generate_summary_hf(_model, _tok, msgs,
                                      request=SUMMARY_REQUEST_BRIEF,
                                      max_tokens=400)
        for other in list(_sessions):
            _sessions[other].pop("ccache", None)
        sess["ccache"] = {"tsm": tsm, "n_msgs": len(msgs),
                          "prefix_msgs": [dict(m) for m in msgs],
                          "summary": summary}
        n_cc = sum(1 for v in _sessions.values() if "ccache" in v)
        assert n_cc <= 1, f"ccache bound violated: {n_cc}"
        dbg["summary_cache"] = "MISS"
    b_msgs = build_b_messages(msgs, summary["text"], tsm)
    b_ids = canonical_ids(_tok, b_msgs, renderer=render_hf)
    b_snap, _ = hf_prefill_ids(_model, b_ids)
    dbg.update(compacted=True, b_tokens=len(b_ids),
               summary_tokens=len(summary["gen_ids"]))
    if mode == "E":
        b_starts = message_token_starts(_tok, b_ids, len(b_msgs))
        regions = [
            ((b_starts[2], len(b_ids)), (starts[tsm], summary["conv_end"])),
            ((b_starts[1], b_starts[2]),
             (summary["s_start"], summary["s_end"])),
        ]
        pairs = build_alignment(b_ids, summary["old_ids"],
                                set(_tok.all_special_ids), regions)
        if alpha == 999.0:
            import random as _r
            olds = [o for _, o in pairs]
            _r.Random(13).shuffle(olds)
            pairs = [(n, o) for (n, _), o in zip(pairs, olds)]
            b_snap = blend_values(b_snap, summary["snapshot"], pairs, 0.75)
            dbg["shuffled_control"] = True
        else:
            b_snap = blend_values(b_snap, summary["snapshot"], pairs, alpha,
                                  head_map=sess.get("cfg_head_map"))
        dbg["grafted_positions"] = len(pairs)
    gp = render_hf(_tok, b_msgs, True)
    text = answer_hf(_model, _tok, b_snap, gp[len(b_ids):], len(b_ids),
                     max_tokens=max_tokens)
    sess["n_compactions"] = sess.get("n_compactions", 0) + 1
    del b_snap
    torch.cuda.empty_cache()
    return text, dbg


def app(environ, start_response):
    path = environ.get("PATH_INFO", "")
    if path == "/health":
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b'{"ok": true}']
    if path != "/v1/chat/completions":
        start_response("404 Not Found", [])
        return [b"{}"]
    try:
        body = environ["wsgi.input"].read(
            int(environ.get("CONTENT_LENGTH") or 0))
        req = json.loads(body)
        mode = "A"
        req_head_map = None
        alpha = E_ALPHA
        compact_at = COMPACT_AT
        m = str(req.get("model", ""))
        if "sc-" in m:
            parts = m.split("sc-", 1)[1].split(":")
            mode = parts[0][:1] if parts[0][:1] in "ABE" else "A"
            for p in parts[1:]:
                if p == "shuf":
                    alpha = 999.0  # sentinel: shuffled-pairs control
                elif p.startswith("a"):
                    alpha = float(p[1:])
                elif p.startswith("cfg="):
                    import json as _j
                    _cfgs = _j.load(open("tune_configs.json"))
                    _c = _cfgs[p[4:]]
                    if "alpha_map" in _c:
                        alpha = {int(k): v for k, v in _c["alpha_map"].items()}
                    else:
                        alpha = _c.get("alpha", 0.75)
                    req_head_map = ({int(k): v for k, v in
                                     _c["head_map"].items()}
                                    if _c.get("head_map") else None)
                elif p.startswith("c"):
                    compact_at = int(p[1:])
        msgs = _norm(req["messages"])
        max_tokens = min(int(req.get("max_tokens") or 1500), 3000)
        skey = _skey(msgs) + "|" + m.split("sc-")[-1]
        with _lock:
            sess = _sessions.setdefault(skey, {})
            sess["cfg_head_map"] = req_head_map
            t0 = time.time()
            text, dbg = _generate(msgs, max_tokens, mode, sess, alpha, compact_at)
        resp = {
            "id": f"sc-{int(time.time()*1000)}",
            "object": "chat.completion",
            "model": m,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": dbg["full_tokens"],
                      "completion_tokens": len(
                          _tok(text, add_special_tokens=False).input_ids)},
            "sc_debug": dbg,
        }
        print(f"[{mode}] {dbg['full_tokens']}tok "
              f"{'C' if dbg.get('compacted') else '-'} "
              f"{time.time()-t0:.0f}s", flush=True)
        out = json.dumps(resp).encode()
        start_response("200 OK", [("Content-Type", "application/json")])
        return [out]
    except Exception as e:
        import traceback
        traceback.print_exc()
        err = json.dumps({"error": {"message": str(e)}}).encode()
        start_response("500 Internal Server Error",
                       [("Content-Type", "application/json")])
        return [err]


def main():
    _load()
    from wsgiref.simple_server import make_server
    print(f"shim listening :{PORT}", flush=True)
    make_server("0.0.0.0", PORT, app).serve_forever()


if __name__ == "__main__":
    main()
