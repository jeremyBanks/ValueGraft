# AGENTS.md — current orientation

## Mission and live state

This repository studies whether write-time K/V state across conversation
compaction carries useful history-specific information that identical visible
text freshly encoded loses, and whether controlled transplantation recovers it.

Start every session in this order:

1. If `.disk-space-alert.json` exists, stop new model/download work and free
   disk before continuing. The launchd monitor warns below 30 GiB and clears
   only above 35 GiB.
2. `STATE.md` — authoritative live work and gap.
3. `DECISIONS.md` — current governing decisions only.
4. `FINDINGS.md` — current observed facts and unknowns only.
5. `LOCAL-COHERENT-STATE-N48-V2.md` — frozen active study.
6. The `notes/` slug `20260712-handoff-to-new-ultra-agent`.
7. The `notes/` slug `gap-analysis-asked-vs-did-and-upper-bound`.
8. `RELIABILITY.md` and `INCIDENTS.md` for the compact current hazard index.

The mandate is to use the already-authorized remaining GPU budget to reach a
statistically and scientifically robust conclusion. Compute the gap every
cycle: independent semantic N versus the frozen CI target, and verified spend
versus the current allocation. Apparatus elegance is not the terminal goal.

Historical frozen protocols and runners remain an audit trail. They do not
authorize the successor and may fail their old hash bindings at current HEAD;
reproduce them from their recorded commits. The successor gets a new design ID,
namespace, preregistration, and release path.

The powered-v13 remote release/lifecycle route is retired. The active N48 study
runs directly on the exact cached local MLX model named in `STATE.md`; reuse of
v13 is limited to its outcome-blind CPU fixture frame and state-only placebo
constructor. Do not allocate a pod or repair the old lifecycle for this study.

## Repository working surface and archive

- Root Markdown is small and operational: current state, decisions, findings,
  reliability, methods/provenance, and active protocol.
- Completed plans, audits, ledgers, reviews, and superseded Markdown move intact
  to `notes/`. Reference notes by descriptive slug; the archive script may
  renumber date prefixes.
- Temporary investigation directories are acceptable while active. Commit
  useful work there, then move the completed record into `notes/`.
- Refresh the notes archive from repo root with
  `python3 scripts/update_notes_archive.py`; use its documented provider flags.

## Scientific hard rules

- **Build ladder first.** Every new model/config must pass the relevant
  L0/L1/L3 identity and exact production-path gates before its outcomes count.
- **Exact provenance.** Record model ID, resolved revision, parameter/cache
  dtype, backend, geometry, library versions, code commit, host GPU/driver/CUDA,
  tokenizer/template hashes, schedule/call widths, seeds, and every token’s
  origin.
- **Independent unit honesty.** Conversations/fixtures are N. Plants, layers,
  heads, schedules, regions, repeats, precisions, and hosts are not additional
  population units.
- **Applied controls.** A mathematical placebo is unavailable unless it passes
  at the actual dtype and state geometry. Missing controls stay missing.
- **Decoded validity.** Exact token geometry never substitutes for blind review
  of complete control histories. Repetitive, clipped, contradictory, padded, or
  incoherent controls are invalid.
- **Save every render and result.** Persist text, token IDs, summaries,
  checkpoints, traces, raw outcomes, config, and hashes before the next loss
  boundary. Commit scored, partial, contaminated, void, and negative results;
  quarantine rather than delete.
- **Unique outputs.** Use `<experiment>_<model-slug>_<UTC-timestamp>` and print
  resolved model plus output path at process start. Never overwrite a prior run.
- **Observed claims only.** Report past-tense observations and explicitly state
  what was not verified. `status=PASS` is not a number; read the number.
- **No local subject as collaborator.** The subject model may generate the
  experimental conversation/summary text required by a protocol. Authoring,
  judging, design review, and analysis use independent collaborators/tools.

## Compute and provider operations

- Run one local MLX/GPU job at a time. Default to one admitted remote GPU host
  per scientific batch unless a frozen plan and budget justify otherwise.
- Verify exact GPU name, memory, driver, CUDA availability, model ID/revision,
  dtype, backend, and expected VRAM after load. Containers do not pin host
  drivers.
- Use the established bounded admission/lifecycle patterns. Only retry a
  provider-confirmed no-allocation response or a positively deleted rejected
  host. Ambiguous allocation or failed cleanup is fatal until reconciled.
- Detach long work with the proven launcher pattern; verify SSH pipe EOF, child
  survival, expected GPU memory, and real log progress. Process existence is
  not progress.
- A monitor is trusted only after fault-injection and happy-path self-tests and
  must be supervised independently of the shell it guards.
- Harvest, hash-check, commit, and push wanted artifacts before termination.
  Verify deletion through provider 404 plus the active-pod inventory. Never
  carry a paid pod into analysis or paper writing.
- Timestamp long commands and measure one representative unit before scaling
  time, disk, memory, or dollar bounds.

## Git and credentials

- Integrate, publish, and push only additive reviewed history on `trunk`; never
  rebase, amend, or rewrite it. Detached temporary worktrees are allowed for
  isolated parallel shards. Review their commits before integrating them and
  remove the worktrees after their useful commits are preserved on `trunk`.
- Stage explicit paths. Never use `git add -A`. Never expose or stage key files.
- Commit and push every completed unit. Existing dirty/untracked work belongs
  to the user or another agent; preserve it and avoid overlapping edits.
- Never delete or overwrite uncommitted work. Archive important Markdown before
  removing it from the active surface.
- Keep model weights, caches, datasets, credentials, and large binaries out of
  git. Scored JSON and text renders belong in `results/` despite being data.

## Collaboration and escalation

- The main agent owns decisions and works autonomously. Shard independent
  multi-item work early with a small number of bounded agents.
- Use a fresh Fable/other-family consultation for critical scientific choices
  or after one to two failed attempts. Give the smallest sufficient,
  non-leading evidence bundle, state the review boundary, and require a serious
  assessment to be written into a `notes/` file. Advice is not authority.
- Do not offload ordinary confusion to the owner. Escalate only genuine owner
  decisions such as new money, scope, or taste. No additional GPU funding is
  assumed beyond `STATE.md`.
- Never kill a subagent from file size/mtime. Check actual process/activity and
  its last message.

## Stateful serving changes

Any commit that adds/modifies retained serving state must answer in its commit
message:

1. What state is retained, keyed by what?
2. What bounds its size, and where is that asserted?
3. Who evicts it and when?
4. What is the correctness/equivalence proof?
5. Which production-path probe gate exercises it before paid traffic?

Unanswered means do not deploy.

## Paper and publishing boundary

Data collection comes first. When it ends, terminate every pod. Write the paper
fresh from the then-current facts and methods/provenance checklist. Fable should
have a heavy role in narrative structure and initial drafting; the main agent
owns factual/methodological truth and the final revision. Run multi-angle Fable,
skeptical critic, terminology, methods/provenance, and independent Codex review
before promotion.

Do not edit `README.md` as a live draft. When a new working paper is genuinely
ready, copy it over `README.md`, commit, and push `origin/trunk`. In-repository
publication is authorized; any external blog/forum/public post remains owner-
only.
