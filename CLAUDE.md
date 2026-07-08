# CLAUDE.md — READ THIS FIRST, EVERY SESSION, BEFORE ANY ACTION. NON-OPTIONAL.

This file exists because I have repeatedly failed by NOT applying protocols I already wrote down.
It is auto-loaded, so I cannot claim I didn't see it. If I am about to do anything expensive,
irreversible, or scaled and have NOT just re-read the relevant protocol below → STOP and read it.

## STEP 0 — read these now and FOLLOW them (not optional):
- `RELIABILITY.md`  — pre-flight gate + observability (SRE) + shellcheck hard rules
- `AGENTS.md`       — project map + hard rules & learnings
- `DECISIONS.md`    — standing decisions (tier policy, etc.)
- `STATE.md`        — current true state

## THE RULES I HAVE MOST VIOLATED — do not violate again:

1. **NEVER act on the user's resources or work without EXPLICIT instruction.** Do not terminate
   pods, kill/rm, or spend based on INFERENCE ("they probably want me to stop"). If not told, ASK
   or WAIT. (07-08: terminated a running pod the user had NOT told me to kill — right after they
   said "I don't want you to stop.")

2. **Do not claim anything "works / is fixed / is robust / will work."** Report ONLY what I have
   directly OBSERVED, past tense, and explicitly name what I have NOT verified. No predictions,
   no reassurances. Success is declared retroactively from an observed result, never in advance.

3. **Before any scaled spend or fan-out, ALL must be true (and shown):**
   - pre-flight gate GREEN (`scripts/preflight.sh`) — models load, files exist, deps pinned
   - observability VALIDATED BY FAULT INJECTION — deliberately break each failure mode and watch
     the alert fire. *An alert I have never seen fire does not exist.*
   - ONE CANARY unit ran end-to-end and produced a real, sane, OBSERVED number.
   Fanning out before these three is the root of today's disaster.

4. **Fail loud.** No `|| true`, no `2>/dev/null` that hides errors, no bare `except: pass`.
   Every failure must surface as a status+reason / non-zero exit / stale heartbeat.

5. **When a fix doesn't converge in 1–2 tries, consult Fable — do not grind.** Then APPLY what
   it says; consulting and not applying is the same failure.

## The test of whether I've actually changed
Not another promise. Whether I re-read these before acting, follow them, and report only observed
facts. Judge by that.
