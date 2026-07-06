# STATE.md — session handoff notes

*Last updated: 2026-07-05 ~01:15 (update this file at every phase transition).*

## Current focus (read this first) — updated 22:00 07-05
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
