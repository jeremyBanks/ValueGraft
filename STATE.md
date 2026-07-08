# STATE.md — session handoff notes

*Updated 07-08 ~02:30 by Opus 4.8. CROSS-ARCH REDESIGN IN FLUX (Fable) — wide sweep HELD.*

## ⚡ DESIGN IN FLUX (07-08) — do NOT blast the wide sweep until settled
Fable design consult REFRAMED the cross-arch experiment (recorded MASTER-PLAN
"CROSS-ARCH REDESIGN"). KEY DECISIONS PENDING:
- REFRAME (paper-saver): value vectors come from the ATTENTION block; MoE is FFN →
  MoE does NOT touch W_V. "MoE flips the sign" is a REVIEWER TRAP. Organizing
  question = "what ATTENTION-GEOMETRY property predicts the SIGN (help/harm)?"
- DE-CONFOUND: add Qwen3-32B (dense, same gen as Qwen3-30B-A3B MoE) — the clean
  dense-vs-MoE control. Qwen2.5-vs-Qwen3 reversal is confounded (~5 axes).
- DEPTH not just width: Tier1 anchors deep (Qwen3-30B-A3B, Qwen3-32B, Qwen2.5-32B,
  +~100-150 probes/cat), Tier4 breadth shallow. Bootstrap over CONVERSATIONS not probes.
- MUST-CAPTURE CONTROLS: placebo graft, identity-graft check, alpha dose-response,
  own-vs-foreign summary axis, SAVE ALL raw traces + log every attn hyperparam/model.
- OWN-SUMMARY EXPERIMENT (> a 12th model): vary summary source own/other/fixed/degraded/
  PARAPHRASED-OWN (crux). Fund by dropping Yi/Llama-2.
- CORPUS QUESTION (sent to Fable): corpus is SYNTHETIC (we made it up). Remake vs
  augment? Add STRONG-PRIOR referents (Pokémon/Sanderson/LOTR famous names as codenames —
  tests recovering conv-meaning OVER strong prior = harder disambiguation). Awaiting Fable.
- HARNESS CHANGES NEEDED before wide launch: 5 categories (DONE, deployed), placebo,
  identity-check, alpha-sweep, raw-trace capture, hyperparam logging, conv-bootstrap.

## CONFIRMED (the crisis arc, all resolved + recorded in FINDINGS):
- Estimator bug (mean-ratio Cauchy) → robust metric raw E-B + bootstrap CI. Effect REAL.
- Keys CLOSED (neutral, value is operative axis).
- MECHANISTIC FINDING: graft needs model's OWN generated summary (fixed foreign
  suppresses; self-gen reproduces). Sweep uses SELF-GEN.
- MAP so far (self-gen, robust): Qwen3-30B-A3B MoE +0.14/81% (positive); Qwen2.5-32B
  dense −0.30 (REVERSES, real — trusted apparatus agrees). Harness VALIDATED (positive
  control passed on self-gen).

## PODS: 6 up (0rjwbqwpte2rzf:11591, welhpjmbqyd880:12994, +p636wvrnqbzy06, ik8irhx4yhqdj0,
+pods5/6 provisioning). $1.19/hr each = ~$4.76-7/hr. 400G each. WIDE SWEEP HELD pending
design. pods1-2 running OLMo/Mistral (OLD design — partial value, may redo). Idle
watchdog b19m5koqi live. Balance ~$76-78.

## ## PHASE 4 PLAN (autonomous, no review-ask; me+Fable; see MASTER-PLAN addenda 1-7)
Strengthen PRIMARY (grafting) — paper drifted lens-heavy (dud), grafting under-evidenced.
1. CROSS-ARCH breadth sweep (running next) — does effect travel across architectures.
2. MORE EXAMPLES from existing data (cheap, high-value; surface more grafting exhibits).
3. Story-continuation task (src/story_tasks.py, built-never-run) + more domains.
4. REBALANCE paper: grafting=primary, lens proportionate; FIX overloaded "lens" (76 uses,
   22 ambiguous — name logit-lens vs J-lens once, disambiguate).
5. ARCHITECTURE limitation: cross-arch results + MLA block; Fable RESEARCH pass (tools).
6. LOW-PRI: per-model CHAMPION TUNING profiles as architectural fingerprint —
   **ASK FABLE about this at some point (user 07-07), timing my call.**
FABLE: brief on the FOCUS (grafting primary) in every conceptual/edit prompt. FINAL review =
3 tool-less Fable passes different angles + terminology-consistency dimension.

## STANDING RULES added this phase: quiet monitors (silent unless done/error, 2min→24min
backoff) + launch-verify REAL work AND RIGHT MODEL (rules 26,29); unique self-announcing
output names (28); don't suppress deploy stderr (25); verify CUDA (24).

## (superseded) LIVE STATE — Phase 2 K/V sweep RUNNING (resume from here)
PHASE 1 DONE: synthesis paper shipped to trunk (REPORT.md, ~7700w, honest, pushed).
PHASE 2 (task 23) IN PROGRESS — the K/V behavioral sweep is RUNNING NOW:
- POD: lveawgxzfm5xte, COMMUNITY cloud, ssh root@104.255.9.187 -p 11534
  -i ~/.ssh/id_ed25519_runpod. State: .pod_k1_state.json (annotated w/ ip+port).
  A100 80GB. torch pinned 2.6.0+cu124 (driver CUDA 12.5 — do NOT pip -U torch,
  incident 27). CUDA verified available.
- RUN: /workspace/exp/src/kv_sweep.py, log kvsweep3.log, results→
  /workspace/exp/results/kv_sweep/<conv>.json. ~8/12 convs done @15:16, ~4min/conv.
  Watcher task bz4jt51tc pulls to results/kv_sweep/ + signals at 12/12.
- WHAT IT IS: COARSE UNTUNED sweep — 6 global policies (uniform αK/αV across all
  layers/heads): v_only(0,.75) k_only(.75,0) coupled(.75,.75) ind_k50(.5,.75)
  ind_v50(.75,.5) ind_k100(1,.75). NOT per-layer/head tuned (that's next if signal).

## RESUME STEPS when sweep completes (analyzer pre-staged):
1. rsync results/kv_sweep/ from pod (watcher does this). Clear any stale first.
2. Run scratchpad/analyze_kv.py results/kv_sweep — category×policy gap-closure table.
3. SANITY GATE (K-graft is NEW): v_only referent must be POSITIVE (~+0.04, paper's
   30B value — kv_sweep v_only = bit-identical to gap_closure_cat by construction).
   If v_only referent NEGATIVE/off → K-graft or metric broken, STOP+debug, don't
   trust K/coupled. (NB: 0.6B self-test showed v_only referent −0.125 = noise, not
   predictive of 30B.)
4. KEY QUESTION: does k_only/coupled/independent recover REFERENT > v_only? does
   independent beat v_only on SENSE? Inspect implausible #s (rule 19).
5. Record→FINDINGS new F-entry. IF keys help → (a) per-layer tuning of independent
   (αK,αV) [kv_graft.py supports alpha-dicts+head_map], (b) 27B port (where-V-floored
   test; kv_graft is 30B-path, needs boundary_probe 27B snapshot port), (c) fold K/V
   into REPORT.md via Fable conceptual gut-check + Fable readability SUBAGENTS (main
   loop is Opus-flipped; Fable readability MANDATORY). IF negative → honest 'value is
   the operative axis' result. Either way: TERMINATE pod lveawgxzfm5xte, report to user.

## VALIDATED THIS PHASE (committed): kv_graft.py (K-graft w/ RoPE re-rotation,
fp32-exact cosine 0.99999982); kv_sweep.py (V-only bit-identical to baseline). See
FINDINGS Phase-2 entries. Balance ~$85. Fable = conceptual gut-check + readability,
always via subagent.

## PHASE 2 IN PROGRESS — K/V exploration (task 23)
KEY Q: does KEY-grafting recover REFERENT where value-only floored (27B +0.004)?
Keys carry addressing/RoPE = plausibly what retrieval needs. Steps: build
K-graft w/ RoPE re-rotation → smoke small model (verify α0==fresh + re-rotation
correctness) → sweep V/K/coupled/independent per-layer on sense/referent/stance
→ sparse-regime lens testbed. Provision pod. Balance ~$85. 0 pods now.

## 11:55 07-07 — (superseded, timestamp was wrong-ordered)

**HEADLINE RESULT (FINDINGS.md F1, CORROBORATED):** write-time value
grafting recovers ~10-12pp (judged) of compaction damage in the
SEMANTIC-RICHNESS regime (sense/referent), null on stance (summary already
preserves it) — effect TRACKS the damage. Corroborated by an independent
judge-free gap-closure metric (direction agrees: sense/referent>stance;
weaker magnitude = predicted "recovers meaning not verbatim form" = 2nd
signature). Two-method finding. F2 honesty, F3 tuning, F4 scope-boundary
(SWE-bench too hard, chains/tau-banking too short) also banked.

**RUNNING NOW:** 4B gap-closure (src/gap_closure_4b.py on t1) = cross-scale
F1 replication. jlens interpretability readout = being developed (subagent).

**READY/QUEUED (EXPERIMENTS.md portfolio, 8 tasks x 9 metrics x 6 designs):**
story-contradiction task (src/story_tasks.py, built+tested, unrun); jlens
mechanistic (B9); F1 hardening (wider ceiling, strict-RECOVERED cut);
K-vs-V independent grafting; cross-dependent chains.

**DEAD/RETIRED:** tau-banking (sessions structurally too short — all arms
0.00; F4). SWE-bench (30B floor, 0/7). Live-agent chain arms (compaction-
robust null, but champion-cures-a1.0-collapse = F3).

**INFRA:** 1 pod (t1, running 4B gap-closure). Balance ~$40. OpenRouter key
present (gitignored) for tau-style user-sims if needed. PROVENANCE: Opus
since 02:11 07-07 (PROVENANCE-CORRECTION.md); commit trailers were
mislabeled Fable in a ~6h window — noted, not rewritten.

DOC MAP: FINDINGS.md=results | EXPERIMENTS.md=portfolio | DECISIONS.md=
process/every-decision | INCIDENTS.md=26 failures+23 rules |
PROVENANCE-CORRECTION.md=model attribution | this=state.

## 08:45 07-07 — (superseded)

### 09:10 UPDATE — TWO TRACKS (don't lose either):
1. TAU VALIDATION: OpenRouter key (.openrouter_key, gitignored, user
   dropping now) → rerun banking pilot with --user-llm openrouter/<strong
   model> as customer sim → gate = sessions >12K so compaction fires.
   Banking pilot ALREADY proved integration works; only the weak user-sim
   (our 30B) failed it (sessions 300-1500 tok). tau CLI in STATE below.
2. SYNTHETIC DIAL-IN: shift plants from `evicted_fact` (too precise) to
   `sense`/`referent`/`stance` (semantic richness, Pokémon-style) — the
   regime where graft has an edge and examples stay realistic. This
   addresses the core gap: we have mechanism+honesty proof but ~no signal
   on task-outcome benefit in the operative difficulty/skill band.


MODEL: now Opus 4.8 (Fable quota out). Read INCIDENTS.md #1-26 + rules 1-23
before acting. DECISIONS.md is current (107 entries); STATE/HANDOFF/INCIDENTS
were stale until this update (incident 25).

LIVE NOW: tau2 BANKING pilot running on pod t1 (shim @ tunnel 8040), domain
banking_knowledge (EVICTABLE policy = compaction-relevant), --retrieval-config
bm25 (keyless; rank_bm25 installed in scratchpad/tau2/.venv), agent+user both
= our shim openai/sc-A. 698 docs loaded, 2 tasks in flight. Watcher (5-min
poll) checks completion + PEAK TOKENS (pilot gate: sessions must exceed 12K
for compaction to fire). Retail control ran first: reward 0, 16-msg episode —
harness+scoring work but that was NOT "pipeline proven" (control domain,
failed task; incident 24).
tau CLI: scratchpad/tau2/.venv/bin/tau2 run -d banking_knowledge
--retrieval-config bm25 --agent-llm openai/sc-A --agent-llm-args
'{"api_base":"http://localhost:8040/v1","api_key":"sc"}' (same for --user-llm).
Entry point is `tau2` console script NOT `python -m tau2`.

STANDING RESULTS (banked, clean): mechanism suite, stage-1 damage, stage-2
recovery, honesty-bf16 replication, tuning story + guards, CHAIN ARM TABLE
(champion cures a1.0 collapse=REAL; chains compaction-robust=NULL; recall
probe void — DECISIONS 08:15). SWE-bench retired (model floor, 0/7).

PENDING/QUEUED: jlens_boundary_probe/ review AFTER tau (DECISIONS 07:00-07:15,
lean-in-but-behaviorally-check stance); independent K/V tuning (next-gen);
serving-stack upgrade at phase boundary. Balance ~$43, 1 pod (t1) live.

## Older focus (02:40 07-06) notes

**NIGHT PROGRAM: E-track (agent coding) is thefocus.** LongMemEval is DEAD
(user directive; stage-1 damage table kept, 1b/H200 killed). First signal:
B 0/2 vs E 2/2 on objective pytest tasks end-to-end.
**PODS (7):** p2=honesty-bf16 (running); p4=4B block then slot-guard, then
standing TUNING/VALIDATION pod; e1=shim (old queue rounds 2-3 draining →
then deploy NEW shim + tune_configs.json + start spec1); e2/e3/e4=shims
provisioning (spec2/3/4, tunnels 8011-8013); w1=WILDCARD (wild_ideas.md).
**MATRIX (~86 runs, spec1-4.txt):** t1/t2 × seeds s1-s5 × {B, E, E:a0.5,
E:a1.0, E:a-0.5(anti), E:shuf(control), B:c6000, E:c6000, E:cfg=layers,
E:cfg=posslots} + A controls; every run ends with the from-memory recall
probe (dissociation analysis). Matrix runner skips scored runs — safe to
re-split/re-run.
**TUNING LADDER (paper exhibit):** B → E(α=.75) → E(per-layer coarse) →
E(slot mask); champion/challenger + ≥25% baseline guard after promotion
(see DECISIONS 02:30). tune_configs.json must reach every shim pod.
**MORNING REPORT:** pass-rate table by condition, dose curves (α from -0.5
to 1.0), threshold effect (6K vs 9K), shuffled control, dissociation
table, honesty-bf16 results, 4B-bf16 calibration, wildcard journal, spend
ledger (balance ~$101 at 02:00, burn ~$9.5/hr at full width).

## Older focus (22:00 07-05) notes
### (written for a possibly-different successor agent — GPT-5.5 handoff likely)

**GO-WIDE APPROVED (9 pods max, secure tier OK).** Use ONLY
scripts/launch_pod.sh to start pods (idempotent: provision→ssh→sync→launch
job.sh→register in scratchpad pods.list). Watchdog = scratchpad/podwatch.sh
run by a Monitor every 30 min; it AUTO-PULLS results from every pod in
pods.list into scratchpad/../pulled_results/ (data-loss window ≤30 min).
NEVER hand-SSH kills: kill by PID then pgrep-verify empty. After any model
load on a pod, verify nvidia-smi used-memory > weights size (CPU-offload
trap). Balance check: `uv run python src/pod.py balance` (prepaid; ~$79 +
user adding $50 soon).

**In flight right now:** pod-1 (:31918@213.173.105.10) draining stage-1
shard 0/2 (~done); pod-2 (:11989@104.255.9.187) stage-1 shard 1/2 (~30
left); pod-4 (:12618@38.128.232.177) 4B-bf16 calibration block (tune4b.log).
Local: idle. Sonnet judging of stage-1: batches 00-04 of 13 done
(results/judge_batches_lme_30b_bf16/), keep 2 concurrent, prompt pattern =
copy any prior judge Agent prompt, increment batch number.

**Queue (in order):** (1) smoke run_lme_full_hf.py with SC_LME_N=1 on freed
pod-1, then stage-1b fan-out ×4 shards via launcher (SC_SHARD=k/4, tag
30b_full); (2) slot-guard check (wrongconv through the 57 posslots — needs
small runner variant; NOT yet written); (3) stage-3 LoCoMo adapter (NOT yet
written — GitHub xiaowu0162-style download, see scouting-notes.md); (4)
Mistral-2409 pre-tune (template integration into run_tune_hf NOT done; see
arms_common.template_ops + value-steering-design-notes; run SYSTEM-LESS);
(5) Gemma hybrid code (NOT written); (6) 70B needs H200/2xA100 sizing.
Config policy: value-steering-design-notes.md is authoritative (primary =
global α=0.75 at 30B; slot mask = exploration only; evidence-grading rule:
4-bit/small results are possibility-proofs only).

**Tonight's results already banked (committed):** stage 2 complete (+0.0156,
45/75, 10% closure on real agent traces); extended-α bracket (smooth decline
past 1.0, peak 0.75); 4B factored/clamped-vs-signed resolved (clamped wins,
both below simple mid-band); 4B + 30B head profiles (no validated head
structure at 4B; 30B diffuse; posslots holdout win PENDING ITS GUARD).

## Older (17:25) notes

**FOUR PODS + LOCAL, all healthy.** Pod-1 (secure :31918@213.173.105.10)
stage-1 shard 0/2 (recovered from CPU-offload stall — ALWAYS verify
nvidia-smi VRAM > weights after model load). Pod-2 (community
:11989@104.255.9.187) shard 1/2. Pod-3 (community :11867@104.255.9.187)
stage-2 SWE-Gym. Pod-4 (secure :12618@38.128.232.177) stage-T tuning
(sweep done, layer/head phases running; early: α≥1 wins some convs at bf16).
ONE consolidated watchdog (retry-hardened, staggered). Local: scale-curve
done (generation-split curiosity, see DECISIONS); per-head playground queued
(task #17) after stage-1 preliminary judging.
QUEUE: stage-1 finish → judge → stage-1b (approved, full-haystack) on pods
1+2 → tuning holdout eval + adopt config (rule in chat 07-05: better OR
equivalent ⇒ adopt bf16-tuned; E-arm re-runs on completed items) → Mistral
2409 pre-tuning on pod-4 (template adapter DONE in arms_common.template_ops;
run system-less!) → stage 3 (LoCoMo/SCBench) → stages 4-6.
Deletions done with approval: ~/.ollama, ~/.lmstudio/models (100GiB free).

## Older focus (15:35) notes

**CLOUD PHASE RUNNING (user approved; ~$100 budget).**
- Pod-1 (secure, .pod_state.json, ssh -p 31918 root@213.173.105.10):
  stage-1 shard 0/2. Pod-2 (community, .pod2_state.json, ssh -p 11989
  root@104.255.9.187): stage-1 shard 1/2. LongMemEval ×350 all types,
  ~120 s/q, ETA ~21:30. Dual watchdog armed (stall/complete only).
- Stage 1b APPROVED (standard-protocol full-haystack LME ×100) — queued for
  pods 1+2 tonight after stage 1; runner still to write (adapt run_lme_hf:
  no subsampling, evict-oldest-sessions compaction, benchmark's judge
  prompts). Stage-1 numbers are NOT leaderboard-comparable; only 1b is.
- Stage 2 (SWE-Gym) runner BUILT (src/run_swegym_hf.py; swegym.parquet
  cached); local 0.6B smoke in flight; pod-3 after smoke passes.
- Judging: incremental Sonnet batches during runs. Balance at stage
  boundaries (opening $50; user top-up to ~$100 pending on account).
- SSH key ~/.ssh/id_ed25519_runpod; pod tool src/pod.py (SC_POD_STATE,
  SC_POD_CLOUD). ALWAYS kill pod processes by PID + pgrep-verify empty.

## Older focus (14:45) notes

**READY TO LAUNCH CLOUD PHASE — waiting ONLY on the user's explicit go.**
Read cloud-plan.md end-to-end first: Strategy P (scale evidence on
Qwen3-30B-A3B bf16), stages = ladder → LongMemEval full 500 → SWE-Gym agent
traces → 2nd standard benchmark (LoCoMo/SCBench) → [homemade probes n=50] →
[model variety: Mistral Small 3.2; Llama-70B only if budget + Meta approval]
→ [Gemma 27B hybrid profile]. One A100-80GB pod at a time, terminate between
stages, ~$40-60 core. Credentials staged in repo root (.runpod_key,
.huggingface_key — gitignored, 600, UNUSED until user approval). HF gates:
all core ungated; Gemma approved; Llama pending (dispensable).
Afternoon results: per-layer graft profile (mid-band hump L12-L22 peak L17,
late layers harmful; profile-derived rule ties tuned champion on holdout —
see per-head-alpha-speculation.md addendum + results/layer_profile_4b*).

## Earlier 07-05 focus notes

Today added: LongMemEval validation both scales (LONGMEMEVAL-RESULTS.md —
compaction damage replicates on standard data; honesty effect
frame-dependent: real at 4B, washed out at 30B QA framing), HF transformers
port core validated (kvlib_hf.py + l_hf_ladder.py PASS — cloud prerequisite
retired), cloud-plan.md ready for user review (RunPod prepaid, staged
$50-70, 10-min setup), blog-draft.md complete (benchmark section +
opaque-compaction-handle deployment framing). Incident: double-instance
swap thrash cost ~3h (fixed; kill-by-PID + progress & memory watchdogs now
standard). AWAITING USER: cloud-plan review + API key; blog draft review.

## Older focus (morning 07-05) notes

**ALL PLANNED WORK COMPLETE.** Deliverables: RESULTS.md (4B pilot),
RESULTS-30B-addendum.md, PHASE2-RESULTS.md (mitigation + tuned sweeps),
blog-draft.md (complete incl. verified related-work; needs repo LINK +
human review before publishing), scouting-notes.md (#12/#13 GO paths).
Open follow-ups if resumed: publish blog (user decision), H-pack+tail
hybrid, SWE-Gym trajectory replication, LongMemEval ≤16K subset, G/SoftGraft,
Phase-2 cloud scale-up per phase2 doc.

## Older focus (Phase-2-era) notes

**Phase 2 COMPLETE at both scales — see PHASE2-RESULTS.md.** H-pack cuts
fabrication vs production compaction (30B decoys 19:5→3:21); matched pair
shows encoding component grows with scale (10→3 at 30B). Tuned ValueGraft:
holdout-validated +0.017 (4B, midband α=.25) and +0.033/24% closure (30B,
global α=.75). All negative-control-certified. Tasks #10 #11 done.

**REMAINING:** blog write-up per writeup-guidelines.md (incl. real web
literature pass for related-work); #12 coding-trace dataset scouting;
#13 benchmark scoping (LongMemEval etc.). GPU idle.

## Older (Phase-1-era) focus notes

**Phase 1 CLOSED** (RESULTS.md + RESULTS-30B-addendum.md): honesty effect
replicates & strengthens at 30B (H-gap 1:23 fab:adm vs B 15:9); α inversion
(E-post α=1 = +29% gap closure at 30B, was harmful at 4B); C +18pp clean
sense; naturals DON'T replicate (A−B gap tiny there — effects conditional
on continuation depending on evicted content). 30B judging done (798
verdicts, results/judge_*_30b*). Remaining certification: negative controls
at 30B (supplement_arms needs outdir env patch) — queued after Phase 2.

**RUNNING: Phase 2** (run_phase2.py, 4B, results/phase2_4b/, HP-0 passed,
~4/12 as of 02:35). Then: 30B negative controls → α sweep w/ gates (#11) →
Phase 2 at 30B if signal → blog write-up (writeup-guidelines.md) →
dataset/benchmark scouting (#12, #13).

## Older focus notes

- **Phase 2 pivot is active** (user directive): mitigation-first, one clean
  contrast — see phase2-design.md. Machinery COMPLETE and validated
  (rerotate.py + LH ladder PASS, arm_h_pack_snapshot/bmin_pack_ids in
  arms.py, run_phase2.py with HP-0 identity, data/decoy_probes.json 24/24
  audited). Launch `uv run python -u src/run_phase2.py` (4B, ~1h) as soon as
  the GPU frees.
- **30B targeted chain RUNNING** (9/12 main pass as of 01:10; then naturals,
  then brief pass; scratchpad run30b.log). Interim n=7: H-gap>B-min +0.126
  (7/7); α INVERTS at scale (α=1 beats B by +0.053 ≈27% closure; was harmful
  at 4B); C≈B. When done: narrow scoring (referent/sense/evicted only, skip
  stance/ruled_out), Sonnet judging, scale addendum to RESULTS.md, commit =
  Phase 1 close.
- **Queued tasks:** #10 Phase 2 run; #11 α sweep w/ validation-holdout split
  (4B fine-low grid; 30B extended-up grid; + span-length & layer-band gates);
  #12 scout SWE-agent/OpenHands trajectory datasets (brainstorm-level).
- Write-up target: HF blog post per writeup-guidelines.md (provenance:
  Claude Fable 5 + GPT-5.5 review + user direction).

## What this project is

Testing whether KV-cache state written at generation time carries semantic
continuity that text-recomputation loses across a chat-compaction boundary.
Read in order: `semantic-continuity-experiment-brief.md` (main design),
`followup-explorations-arms-GH.md`, `phase2-scaleup-and-coding-extension.md`,
`amendments-from-external-review.md` (adds B-causal, negative controls,
leakage classes, metric hierarchy). `DECISIONS.md` = every deviation + verified
model/runtime facts (READ IT before touching cache code — it documents the
traps: think-block template instability, batched-vs-stepwise kernel mismatch,
alignment region crossing).

## Environment

- uv project; `uv run python src/...`. mlx-lm 0.31.3 + transformers pinned 5.0.0.
- Dev model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (all results so far).
- Final model `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`: ladder L0–L4
  green (2026-07-04 23:41); targeted run IN PROGRESS → results/raw_30b +
  raw_30b_brief (arms A/B/C/D/H-gap/B-min/E-post-{0.25,1.0}). Final artifact:
  HF community blog post per writeup-guidelines.md (not a paper).
- Long jobs: launch detached (`nohup ... & disown`, PID to scratchpad
  `pipeline.pid`) because harness-tracked background tasks got killed twice.
  `python -u` + `tee` to scratchpad log; monitor greps the log.

## Pipeline state (as of 2026-07-04 ~23:45)

**4B pilot COMPLETE — see RESULTS.md.** Headline: probe-accuracy mitigation
null; mechanism supported (H-gap>B-min +0.093 nats CI[.04,.14]; E-post α=.25
closure +0.08 both corpora; micro-sense positive; negative controls clean);
novel honesty effect (H-gap flips fabricate:admit from B's 19:5 to 4:20 in the
brief condition, robust across 10/12 convs). Judging: 2089 Sonnet verdicts,
all logged.

**NOW RUNNING:** L-ladder on the 30B (SC_MODEL env; scratchpad
ladder30b.log). If green → launch targeted 30B run overnight:
`SC_MODEL=...30B... SC_OUTDIR=results/raw_30b SC_E_POST=0.25,1.0 SC_E_INTER=
uv run python -u src/run_arms.py` then run_brief with
SC_OUTDIR_BRIEF=results/raw_30b_brief. Purpose: do the three live effects
(honesty, α=.25 CONT gain, H>B-min) survive scale? Negative controls at 30B
via supplement_arms.py only if 30B shows effects. Then: score both new dirs
(extend CONDITIONS in score.py or point at new dirs), judge new answers,
compare 4B vs 30B in RESULTS addendum.

## Pipeline state (older, for context)

- DONE: L0–L4 ladder (all pass; identities exact), micro sense experiment
  (results/micro_sense.json — V-swap carries sense, KV-swap ~half of oracle),
  corpus (12 synthetic in data/synthetic + 8 natural in data/natural,
  tail-contamination repaired, 115/120 plants clean), main 4B batch
  (results/raw/*.json: CONT + probes for 12 arms per conversation).
- RUNNING now (detached, ~21:30 start): supplement pass (B-causal +
  E-wrongconv + E-shuffled) merging into results/raw; then run_brief.py
  (terse-summary shadow condition) → results/raw_brief/.
- QUEUED after that (in order):
  1. `uv run python src/fix_bcausal_cont.py` — B-causal CONT was skipped by a
     template quirk during supplements (see DECISIONS); this repairs it.
  2. `uv run python src/score.py export` — builds results/judge_queue.json
     (judgments + paraphrase leakage checks).
  3. `uv run python src/judge_batches.py split` — batch files; judge each
     batch with a Sonnet subagent (user: no Haiku — use Sonnet; prompt:
     answer each item's prompt
     with the single word demanded; write verdicts_NN.json as {key: verdict});
     max 2 agents at a time. Then `judge_batches.py merge`.
  4. `uv run python src/score.py apply` — final scores (results/scores.json).
  5. `uv run python src/analyze.py` + `src/plots.py` — tables + figures.
  6. Decide 30B run scope (trim α sweep; include H/B-min/B-causal/negative
     controls); switch MODEL constant in src/run_arms.py etc., rerun L0/L3
     identities on 30B first (never report from an un-laddered config).
  7. RESULTS.md write-up structured per refocusing-and-reframing.md
     (mitigation-first claim hierarchy; E/H/negative-controls center of
     gravity; C kept for mechanism/factorization, not the practical claim;
     funny observations noted per user). Paper-style draft, Phase-2 memo,
     and 30B run are DEFERRED (user decision 2026-07-04 ~22:20) until after
     RESULTS + a careful re-review of refocusing-and-reframing.md.
     If H-gap shows judged clean-cut signal, H-pack (re-rotation + identity
     test) is the top follow-up candidate.

## Key interim findings (4B, pre-judge — do not over-claim)

- Calibration perfect: evicted-only facts A 9/9, ALL compacted arms 0/9.
- Summary leakage: 81/120 plants explicit-in-summary → clean referent/sense
  cut underpowered; hence the brief-summary shadow condition now running.
- CONT (both corpora consistently): E-post α=0.25 small reliable gain
  (closure ~+0.08, CI excludes 0 on both); heavier α hurts monotonically;
  C ~ −0.2..−0.7; D ≈ 0. Negative controls (c01): shuffled/wrong-conv graft
  crater to −2.6 — effect is content+alignment-specific.
- Probes (leaky cut, keyword-only): C/H-gap ≥ B on referent/sense; E ≤ B.
- Stance keyword numbers are junk pre-judge (anti-keyword penalizes quoting).

## File map (src/)

kvlib.py (cache primitives, GappedKVCache, teacher-forcing), arms.py (arm
builders, summary gen, alignment), run_arms.py (main driver, ArmSet,
12 variants), supplement_arms.py (B-causal + negative controls),
run_brief.py (shadow condition), fix_bcausal_cont.py (repair),
compose.py / compose_natural.py / fix_tails.py (corpus), audit_corpus.py
(contamination flags), score.py (leak classes, judge queue/apply),
judge_batches.py, analyze.py (gap closure), plots.py, micro_sense.py,
l0..l4 scripts (ladder — rerun on any new model/config).

## Deferred housekeeping (user note 07-05 eve — do NOT act mid-experiment)

Before broad sharing: repo cleanup pass — organize the accumulated docs
(RESULTS*/notes/briefs), prune scratch scripts, coherent README, verify no
credentials/large artifacts, tidy results/ layout. Explicitly deferred to the
very end; touching structure mid-flight risks disrupting running pipelines.

## Overnight mandate (user, going to sleep 07-05 ~22:30)

PRIORITY 1: E0 serving shim → smoke → E1 coding runs fanned across pods.
Design: OpenHands runs LOCALLY (docker) with NO condenser; the SHIM does
compaction internally per mode (model-name suffix selects A/B/E) — agent
sees a normal API; the model's context is silently compacted/grafted
server-side. This is the opaque-handle deployment shape and needs zero
OpenHands forking.
PRIORITY 2 (parallel, creative): try MULTIPLE cheap evaluation angles for
the coding setting, smoke-test each, follow the signal. Candidate probes to
try (add more): resume-after-compaction replay at many cut points;
mid-task file-content recall probes; repeated-failed-command with
exit-code-grounded definition; steps-to-green on small synthetic repos;
diff-quality vs gold patch. Negative results fine — be SURE, avoid
overfit; cross-model consult (AGENTS.md) when stuck.
User asleep: no check-ins; full morning report expected (what ran, what
separated, spend ledger, E-track story).

## Board snapshot (23:30 07-05) — supersedes older lines below

p1=stage-1b-mini (A/B x40 full-haystack, job_new.log); p2=honesty-bf16
(job_new.log); p4=4B-bf16 block (tune4b.log; sweep done, layer running);
g1=Gemma attempt-4 (job4.log; see DECISIONS Gemma ledger — 4 template
fixes, dry-run rule now in force); e1=shim pod loading 30B (job.log, ready
= "shim listening"; then: SSH tunnel + E0 = e1_driver.sh A t1). Stage-1
COMPLETE+judged (table in DECISIONS). Balance $112.73 (~$7/hr, 5 pods).
Stale-guard note: use pgrep -f "[b]racketed" patterns (self-match trap).
pgrep/kill discipline + charter (paced, gates=spending-not-effort,
approaches killable / mission not) all in DECISIONS.

## E-track status (22:55 07-05)

Shim (src/serve_shim.py) smoke-PASSED locally (A/B/E modes, compaction +
graft through OpenAI protocol). Pod e1 provisioning with 30B (job.log:
"shim listening" = ready; port 8000 NOT exposed externally — use SSH tunnel
`ssh -L 8000:localhost:8000 -p <port> root@<ip>`). OpenHands: uv tool
install FAILS (no entrypoints) — use venv at scratchpad/ohenv/.venv +
`python -m openhands.core.main -t "<task>"` headless; install in flight.
Driver script for SWE-Gym tasks still to write (pick instances w/ runnable
tests from swegym.parquet; conditions via model name sc-A/sc-B/sc-E;
LLM_BASE_URL=http://localhost:<tunneled>/v1). p1=1b-mini, p2=honesty,
p4=4B block(+guard,mistral queued), g1=gemma — all running.

## E1 runbook (verified 23:10 07-05)

Per task+mode: `bash scripts/e1_driver.sh <A|B|E> <t1|t2> http://localhost:<LP>/v1`
after tunneling: `ssh -f -N -L <LP>:localhost:8000 -i ~/.ssh/id_ed25519_runpod -p <pod_port> root@<pod_ip>`.
Shim pod e1 in pods.list; ready when job.log has "shim listening".
SC_COMPACT_AT=9000 default — tasks' read-everything phase crosses it.
Scoring: scratchpad/e1_runs/<task>_<mode>/score.json (tests_pass objective,
agent_steps, repeat_signals). Self-test: fail-pre/pass-post VERIFIED.
Run order: t1_A (E0 smoke) → then B/E interleaved, both tasks, ~4 rounds
each on 2-4 shim pods (launch more via scripts/launch_pod.sh eN job_shim.sh).
Judging batch 13 done → stage-1 final table = regenerate aggregate over
verdicts_00..13 (2100 verdicts).

## E0 SUCCESS (23:40 07-05)

Full pipeline works: OpenHands 1.x SDK (src/e1_agent.py) -> tunnel :8010 ->
e1 shim pod (18624@213.173.105.10) -> 30B -> t1 SOLVED, pytest green
(scratchpad/e1_runs/t1_A/score.json tests_pass=true, 40 events).
agent_steps regex is stale (SDK log format) — count "events=" line instead.
E1 next: B/E modes need compaction to trigger — CHECK max tokens from A run
vs SC_COMPACT_AT=9000; if under, restart shim with SC_COMPACT_AT lower
(kill serve_shim on e1, relaunch with env). Queue: B t1, E t1, B t2, E t2,
then repeat rounds; scores accumulate in scratchpad/e1_runs/*/score.json.

## ACTIVE FIXES (00:20 07-06) — read before touching p1/p2

p2 honesty: pod LACKED data/synthetic + decoy_probes (pre-launcher sync);
runner exited HONESTY_DONE with 0 results (silent-empty-success). FIX:
rsync -azL data root@104.255.9.187 (port 11989) :/workspace/exp/, relaunch
job.sh, then VERIFY results/honesty_30b_bf16 count == 12. RULE: every job
script must end by asserting expected output count (add `test $(ls ... | wc
-l) -ge N && echo X_DONE` pattern), podcheck greps markers.
p1 1b-mini: ALL items OOM in summary phase (bf16 weights 61G + >100K-token
KV+activations > 80G). FIX: chunked prefill in kvlib_hf.hf_prefill_ids
(feed ~4096-token chunks through DynamicCache sequentially; explicit
position_ids per chunk) + SC_MAX_FULL=85000 in job.sh. Write-up caveat:
standard-protocol subset = haystacks <=85K tokens on A100-80G.
E1 queue: running fine on e1 (B/E rounds); summary-cache deploys at drain.
UPDATE 00:35: p1 OOM root cause was snapshot CLONE in generate_summary_hf
(16GB dup at 85K) — now snapshot=("E-tuned" in ARMS); chunked prefill also
in (bit-exact verified). p1 cycled; podcheck alerts within ~5min if it
fails again.

## OVERNIGHT E-MATRIX PLAN (01:50 07-06) — the night's main thread

User: LongMemEval DEAD (stage-1 data kept; 1b/H200 killed). E-track = focus;
round-1 signal B 0/2 vs E 2/2. 64-run matrix ready: specA/specB.txt in
scratchpad (t1/t2 x s1-s5 seeds x B/E/E:a0.5/E:a1.0/B:c6000/E:c6000 + 4 A
controls; core B/E pairs FIRST in specs). Steps:
1. When e1's OLD queue prints E1_QUEUE_DONE (monitor live): kill serve_shim
   on e1 (18624@213.173.105.10), rsync src, relaunch job.sh (new shim w/
   per-request knobs + summary cache), tunnel 8010, then:
   nohup bash scratchpad/e1_matrix.sh http://localhost:8010/v1 \
     scratchpad/specA.txt scratchpad/matrixA.log &
2. When e2 pod ready ("shim listening" in job.log; launch_e2.log has
   ip/port; pods.list registered): tunnel 8011 -> :8000, run specB likewise
   into matrixB.log.
3. Analysis: per-condition pass rates; paired core B-vs-E table; alpha and
   threshold dose curves; A ceiling. Morning report leads with this.
Score dirs: scratchpad/e1_runs/<task>_<mode>/score.json (colons -> dashes).
CREATIVE ARMS (01:55): specs now 37+37 incl. E:a-0.5 (anti-graft agents),
E:shuf (shuffled-pairs control; expects NO benefit), and EVERY run ends
with a from-memory recall probe (E1_RECALL_PROBE line in agent.log; true
value = seeded constant, extract via e1_tasks prompt). Morning analysis:
pass-rate table by condition + recall-vs-compliance dissociation (predict:
E improves compliance more than recall).
PARALLEL FAN-OUT (02:10): e3+e4 launching (7 pods total). Matrix resplit
spec1-4.txt (~19 runs each, round-robin so core B/E pairs stay first).
Assignment when shims ready: e1->spec1 (tunnel 8010), e2->spec2 (8011),
e3->spec3 (8012), e4->spec4 (8013); w1 = wildcard (wild_ideas.md in
scratchpad). Matrix cmd per pod:
nohup bash scratchpad/e1_matrix.sh http://localhost:<port>/v1 \
  scratchpad/spec<n>.txt scratchpad/matrix<n>.log &
The matrix script SKIPS already-scored runs (safe across resplits).
TUNING LADDER (02:30): spec1 has E:cfg=layers rows, spec2 E:cfg=posslots.
tune_configs.json MUST be rsynced to EVERY shim pod alongside src at the
new-shim deploy step (launcher now ships it for new pods; e1 needs it
manually at queue-drain transition).

## 04:15 07-07 — PLAN OF RECORD (supersedes below)

CHAIN TIER IS THE PROGRAM. Validated: s1,s3,s4,s6 (4/6; s2/s5 pending) —
vendor-certified difficulty (Aider-Polyglot 55.1% on the model card;
RULE 18: card = difficulty certificate, consult FIRST). SWE-bench fully
retired (0/7 boundary documented; only Coder variant is certified for
it). ARM QUEUE: ~17 rows across priority lanes (B/E0.75/E1.0/champion x
validated chains); B 4/4 on s1 = only real arm datum yet; graft@1.0
"collapse" was an ARTIFACT (INCIDENTS 22; net widened; re-running).
NOTE: chain A-passes do NOT trigger the swbo gate monitor — feed arms
manually (or extend gate) on each new validation.
CONFIRM PHASE DESIGNED: tau2 banking_knowledge (evictable policy) vs core
tau (unevictable) = built-in dissociation on standard tasks
(vendor-benchmarks-scouting.md; PILOT first per rule 14).
STAGED OPTIONS: scaffold A/B (audit_ab.sh — never started, 8021-wait
issue, LOW priority now chains pass); GLM-Air supervised trial (≤2
pod-hrs, DECISIONS 00:45); Qwen3-Coder swap (template-validated);
serving upgrade at phase boundary. Balance $36.5 (~6h runway at current
burn); synthetic lane winds down at current quads' end.

## 23:15 07-06 — (superseded)

DIFFICULTY CRISIS + RESPONSE LADDER (all pre-agreed with user):
1. SWE-bench validations: pruned to ≤14-line-patch instances ONLY (16
   rows, oracle mode, 30-min caps, easiest-first incl. pytest-7521 which
   Sonnet solved in 52s/objectively verified). Per-verdict user reports
   until first PASS (standing order).
2. CHAIN TIER (easier, being built+self-tested by subagent →
   src/chain_tasks.py, "chain:<seed>"): 4 chained easy exercises/session,
   per-exercise scoring + embedded recall; RULE 14 capability smoke
   (A-mode chain:s1/s2) BEFORE any arms; if A passes → run 5-arm
   comparisons on chains (task source stratum "chain").
3. If chains ALSO fail at full capability → MODEL SWAP per DAY-PLAN
   ladder: Qwen3-Coder-30B (template-validated drop-in) first; GLM-4-9B
   serving-path smoke IN FLIGHT on r4 (glm_smoke.log) as GLM-rung
   pre-validation; user pre-authorized trying easier tasks then GLM-class
   if needed. Swap = new calibration pass + gold-patch scorer smoke +
   full 20-conv dry-run (checklist in DAY-PLAN).
EVENING'S KEY EVENTS (recorded in INCIDENTS 19-20 + DECISIONS 21:40+):
plain-Lite unmeasurable (economics); oracle setting added; Verified
merged+interleaved; 30-min caps; A-first gating + priority-queue lanes;
Sonnet probe (52s PASS) → phantom-id scorer bug found+fixed, 17 verdicts
re-scored (no flips); GLM/Coder scaffold validation done at tokenizer
level. Balance ~$60. Synthetic quads keep accruing on r4 (guaranteed
floor). All monitors + per-verdict reporting live.

## 18:40 07-06 — (superseded)

HUMANE TIER (the production-faithful stratum, post-audit) RUNNING on
r1/r2/r3: original 8 SWE-bench instances x 5 arms, compact_at 12000 (per
row suffix c12000), TAIL_KEEP 6000, SUMMARY=prod (see INCIDENTS #15-16 for
why previous settings were invalid), uniform no-cache lanes (audit C1),
alarm 5400, config self-reported in every sc_debug. r4 = synthetic
remainder on legacy calibration (internally-consistent stratum). Prior
agent strata: tier-0 real + all brief-summary synthetic rows = labeled,
non-headline. Audit fixes deployed (DECISIONS 18:20; PIPELINE-AUDIT.md).
Analysis: pre-registered rules + timeout-as-covariate + per-arm
timeout-rate + dropped_ids flagging + n_recompactions verification.
VERDICT when humane 40-row block done (~late evening) → autonomy rule.

## 15:35 07-06 — (superseded)

REAL-TASK BLOCK IN PROGRESS: 0 scored, 4 episodes in flight (15-25+ min,
logs growing, compaction firing constantly — fit criterion (a) PASSED,
see DECISIONS 15:30). Timeouts now SCORED (timeout.marker) not voided.
4 verified lanes: r1/r3/r4 stable no-cache, r2 = icache v2 CANARY
(gate-passed after tunnel mis-wire fix, INCIDENTS #14). Specs P1-P3 +
P4X (canary slice; overlaps tolerated via skip-by-score). First real
completions due ≤60 min by alarm bound; then episode economics become
measured. Verdict analysis when real block done (~18:00-19:00 est) →
autonomy rule. Balance ~$72.

## 14:00 07-06 — (superseded)

FOUR no-cache lanes (r1-r4; icache WITHDRAWN — INCIDENTS #13) running the
priority docket: 40 REAL SWE-bench rows FIRST (0 scored yet; first
episodes in flight), then 30 champion, then core synthetic, then dose
variants (NOT droppable — user; champion-improvement axis). Episode
economics UNCALIBRATED until ~5 real scores land (10-60 min/episode
range). Fit criteria + A-solvability screening + secondary process
endpoints all pre-registered (DECISIONS 10:15/13:20/13:35). Enrichment
adds steps/wall/diff-size/recall/source per row. Real-task provenance
labeling mandatory in all reports. Balance ~$78, 4 pods $5.56/hr.

## 12:15 07-06 — (superseded)

CLEAN RUN restarting after incident 12 (see INCIDENTS #11-12: ccache/icache
unbounded-VRAM leaks = the day's root pathogen; dead-shim row burn purged
102 artifact rows, 16 genuine kept). HARDENED STACK NOW: both shim caches
one-entry-bounded+ASSERTED; runner health-gates every row; driver refuses
to score invalid episodes (validity-at-source — burn class impossible);
streamcheck watches results for anomalies (A-failures, burst scoring);
stateful-change checklist MANDATORY (AGENTS.md). Lanes r1/r3/r4, all with
SC_INCR_CACHE=1 (bit-exact validated, DECISIONS 12:00).
DOCKET: ~135 synthetic + 40 standard rows re-runnable (specs unchanged,
purged rows re-run via missing-score). ETA verdict data 17:00-19:00 →
autonomy rule (DECISIONS 11:00/11:05): fit+direction validate → confirm
phase auto-proceeds; else stop high spend, small-scale exploration only,
report to user. Balance ~$80 post top-up. Canonical arm names:
writeup-guidelines.md (Original/Compacted/+graft variants). 16 clean rows
in results/agent_clean_run/; artifacts fenced in _QUARANTINE paths.

## 10:20 07-06 — (superseded)

1. CLEAN RUN (synthetic, seeds s30-39) continues on r3/r4 — keep all
   scored rows; results/agent_clean_run/ auto-syncs.
2. STANDARD-TASK PIVOT (user + pre-registered amendments, DECISIONS
   ~10:00/~10:15): SWE-bench-Lite sans Docker VERIFIED viable
   (standard-tasks-scouting.md). Adapter (src/swebench_tasks.py) being
   built+validated by subagent (gate: fail-pre/pass-post on the 2 scout
   instances). When it lands: un-run synthetic rows may be replaced by
   swb: rows, SAME five arms, task-source = analysis stratum. FIT
   CRITERIA pre-stated (compaction-pressure + dependency via sc_debug);
   pressure knobs tunable+logged; fallback ladder: other standard sets →
   synthetics.
3. Vocabulary: controlled-key-graft-reframing.md adopted; K-only arm
   queued post-clean-run.
4. Then: champion → confirm (preferring standard tasks) → sealed final.

## 09:20 07-06 — (superseded)

Lanes r3+r4 (secure, probe-gated) own ALL 150 rows (specs R3/R4 with
R0-R2 appended); spot bids s1/s2 out (SC_POD_SPOT=1 support in pod.py) —
arriving pods share specs, skip-by-score dedupes. INCIDENT 10 (see
INCIDENTS.md): cfg=layers rows 500'd until ~09:10 (sess-before-creation
bug) — artifact rows purged+requeued; shims redeployed FIXED code;
check 09:05-09:20 False rows as possible restart casualties.
FRAMEWORK: controlled-key-graft-reframing.md (2nd agent) ADOPTED — report
vocabulary = (alpha_K, alpha_V) policy space + layout/position axes;
K-only Graft arm queued post-clean-run. Honesty decomposition for
write-up: layout 83→25% decoy fab, value-source-within-layout 25→17%,
admissions 3/24→18/24.
Gate: sensitivity rule live (DECISIONS 08:05), still starving for t3-B
rows. E@0.75 clean so far 3/3 (night anomaly dissolving). Balance ~$45.

## 08:20 07-06 — (superseded)

ONLY r4 lane survives (pod ocdluygijjaisw, see pods.list; e1 QUARANTINED
mode-parse/A-mode bug + dead, p4/others terminated). r1-r3 relaunching
(500-capacity retries; launch_r*b.log). Clean run = spec_R1..R4 + spec_R0
redistributed; matrix_Rr4.log progressing. Sensitivity gate armed
(DECISIONS 08:05): B>=75% at n>=6 (2 t3) → ABORT+harden; B<=50% →
continue. Anomaly probe VOID (e1 A-mode bug). Balance $47. If r-capacity
stays dry: one lane ≈ 15h — surface interim tables every ~2h to user.
