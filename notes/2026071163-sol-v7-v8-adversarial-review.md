# Adversarial release review: v7 NO-GO and v8 closure

**Author and reviewer:** Sol — `gpt-5.6-sol-xhigh`  
**Independent review roles:** code/validation, scientific design, and snapshot
provenance; each role was run with a fresh context before the root synthesis.  
**Review interval:** 2026-07-11, before any v8 30B technical or semantic run.

## Bottom line

Clean v7 commit `24f9f82` was a **NO-GO**. The release was not blocked merely by
the still-missing 0.6B ladder: adversarial counterexamples showed that its
independent validator could accept false donor, calibration, sensitivity,
static-provenance, semantic-aggregate, and snapshot-lineage claims. A scientific
review also found that its schedule gates did not compose the actual compacted
destination's real length, logical gap, production partition, and complete saved
summary.

Those gaps were closed additively under
`COHERENT-STATE-PREREGISTRATION-AMENDMENT-8.md`, and the exact apparatus identity
was advanced to `coherent-state-gapped-v8`. This is a code-and-design closure,
not launch authorization. v8 still requires a complete fresh 0.6B ladder and
fresh exact-commit code, science, and cross-family reviews before any spend.

## Demonstrated v7 counterexamples

The code review constructed self-consistent but false terminal evidence that v7
accepted:

1. a donor replacement containing an actual tokenizer special token while the
   producer boolean said otherwise;
2. arbitrary token changes to both purported frozen calibration prefixes with
   recomputed self-consistency hashes;
3. zero downstream sensitivity and zero tail change while retained booleans said
   the perturbation control passed; and
4. stored static hashes made self-consistent without independently reconstructing
   the launch commit's actual apparatus, amendments, input inventory, and duplicate
   fingerprint bindings.

The science review separately found that terminal semantic harvest trusted target
means, plant margins, conversation margins/outcomes, calibration outcomes, and
cross-arm plant identity rather than deriving them from the stored tokenwise data.

## Destination-schedule interaction

The v7 per-render source gate compared the actual long correct-history prefix under
ordinary and message-block chunking. Production `G_fresh`, however, omits history,
uses a large logical-position gap over compact physical storage, splits system and
request prefills, and then forces the saved summary stepwise. Separate real-length,
short-gap, and short-intervention fixtures did not test their interaction.

v8 therefore adds a per-render destination fixture after the generated summary is
durably saved but before wrong-source construction, arms, targets, calibration, or
semantic scoring. It compares identical compacted tokens and identical logical and
physical positions under the production system/request partition and ordinary
consecutive chunking, then forces every summary token stepwise through both. It
persists both full log-probability traces and every summary-row K/V maximum for all
48 layers. Independent harvest reconstructs the exact inputs, positions,
partitions, coverage, and aggregates; all differences must remain at most `5e-4`.

## Snapshot provenance and the raw-tensor waiver

The jointly agreed plan said actual incremental snapshots would be saved. v7 used
the live generation-time cache but archived only text, IDs, traces, and hashes. Raw
archival is large for the exact model: K+V is 98,304 bytes per summary token;
84.375 MiB at the 900-token cap for one source; 1012.5 MiB for twelve correct
sources; and about 2.97 GiB for correct, wrong, and fresh slices. Git LFS is absent
and the repository rejects files larger than 4 MiB.

The snapshot review concluded that raw tensor archival is not scientifically
load-bearing only under a stricter waiver than v7 supplied. Numerical equality at
`1e-4` is insufficient. v8 requires:

- bit-identical per-layer K and V hashes between actual incremental generation and
  a separate stepwise forced replay;
- explicit identification of whether scoring used the live rows or a resumed
  stepwise reconstruction that first bit-exactly matched the saved actual hashes;
- exact component-level hash lineage from correct, wrong, or fresh source channels
  to the inserted span of every gapped arm; and
- independent rejection of approximate-only replay, source/insertion tampering,
  malformed lengths or arm sets, and ambiguous materialization.

The tensor hash is SHA-256 over the dtype string, shape-tuple string, and contiguous
CPU bytes. Because raw bytes are absent, these are non-reconstructive relational
witnesses: harvest reconstructs the exact replay inputs and complete equality/
lineage graph, but cannot regenerate the original digest from repository bytes.
The paper must not claim raw snapshots are archived or reusable.

## Implemented closures and evidence observed

- Donor reconstruction and special-token negative: `7b6a99d`.
- Frozen calibration and sensitivity reconstruction: `7d48748`.
- Independent static provenance: `de05d47`.
- Independent semantic aggregates: `7b6a99d`, with additive provenance correction
  `dc9e924` after a shared-worktree staging race.
- Actual destination producer and independent validator: `273d614`, `e04e015`,
  `7240fb6`, `914fa70`.
- Strict snapshot producer, independent lineage validation, and scoring-source
  binding: `2f47f11`, `ae6fbe9`, `b6dbe8f`.
- Amendment 8 and v8 identity: `9338100`, `2f93e89`.
- Exact production-tokenizer v8 donor PASS, 12/12: `994b75f`.

On clean v8 HEAD `994b75f`, the complete unit suite was observed to pass **202
tests** with two dependency deprecation warnings. The monitor fault-injection suite
was observed to pass **59/59** and print `COHERENT MONITOR CLEARED`. The committed
v8 donor artifact independently matched all source-derived replacements for 12/12
targets. No paid pod existed; RunPod inventory was observed at zero and the balance
at `$63.3160022124`.

## Remaining release blockers

1. Complete and commit a fresh v8 0.6B bf16 eager ladder. Old v4/v6/v7 evidence is
   development history only and cannot authorize v8.
2. Re-run the complete tests, monitor self-test, strict independent donor/ladder
   validation, and preflight on the final clean commit.
3. Obtain fresh independent code, scientific, and Claude Opus cross-family reviews
   of that exact commit and ladder. Opus endorsement of Amendment 8's destination
   fixture and bit-exact hash+lineage waiver remains pending in the execution
   dialogue.
4. Only then push the checkpoint and launch one technical-only paid attempt. A
   terminal technical PASS is necessary but not sufficient for the separate
   semantic process.
