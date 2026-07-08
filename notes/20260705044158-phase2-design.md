# Phase 2 — Mitigation-first design (one clean contrast)

**Question:** can a compact, deployable cache-state method make compaction
hurt less than text-only summary compaction?

**Primary contrast: H-pack vs B-min-pack.** Both contexts are the identical
token sequence at identical positions: [4 sink tokens][summary tokens]
packed contiguously. The ONLY difference: H-pack's summary K/V entries were
written while the full conversation was attendable (keys re-rotated to packed
positions with the validated rotate_keys; values untouched), B-min-pack's are
freshly prefilled. Deployable shape: contiguous, prefix-cache-friendly, tiny
(~4+|S| entries).

**Headline metric: fabrication vs admission on unknowable items** (the 4B
honesty effect, now tested as a designed hypothesis instead of a post-hoc
find), plus referent/sense accuracy as secondary. Items per conversation:
- 2 evicted-fact plants (existing, audited clean, absent from S by
  construction in the brief-summary condition);
- 2 NEW decoy probes (plausible in-scenario specifics never discussed by
  anyone — measures pure fabrication propensity; gold = "admit");
- 2 referent + 2 sense plants (existing).

**Arms (6):**
1. A — full context (oracle)
2. B — production text compaction (reference point)
3. B-min-pack — packed summary text, fresh encode (matched control)
4. H-pack — packed summary entries, write-time encode (the intervention)
5. H-pack-wrongS — H-pack built from a DIFFERENT conversation's in-context
   summary entries (content control: does admission require matching
   content, or does any write-time state suppress fabrication?)
6. H-gap — links Phase 2 to the 4B pilot numbers

**Summary condition: brief (terse) only** — the shadow condition is where the
text channel is starved and the 4B honesty effect was strongest; it is also
the harder, fairer test for the intervention.

**Success =** H-pack beats B-min-pack on fabrication rate (and not via
leakage: items audited absent-from-S; wrongS and the existing 4B negative
controls bracket content-specificity). **Failure is reportable:** if H-pack
≈ B-min-pack, the packed/deployable form loses what H-gap had, itself a
finding about position re-stamping.

**Validation before results:** LH-0/1/2 (done, PASS — rotation exact vs mlx,
kernel shape-noise floor measured); plus HP-0: H-pack built from a
zero-length move (pack offset = original offset) must reproduce H-gap's
behavior exactly.

**Scale plan:** 4B first (fast); 30B replication only if 4B shows signal or
a clean null worth confirming.
