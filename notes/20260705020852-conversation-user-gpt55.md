_This conversation evaluated a KV-cache compaction experiment and reframed its
contribution from demonstrating cache-state non-equivalence to testing a
practical, low-cost mitigation for continuity loss._

**Participants:** User, gpt-5.5-high, and gpt-5.5-xhigh.

**Handoff State.** The project tests whether preserving or transplanting
generation-time KV states can retain conversational continuity better than
re-encoding visible summaries. Core risks are token/template stability,
positional alignment, decode-versus-prefill differences, cache geometry, and
summary leakage. L0/L1/L3/L4 validation ladders and existing implementation
notes document verified facts about RoPE, key/value handling, cache rebuilding,
gapped caches, and chat-template boundaries; GPU jobs should remain serial and
detached, and cache code changes require rereading `DECISIONS.md`.

The theory was judged plausible but insufficiently valuable if framed only as
“write-time KV states differ from re-encoded text,” since that is close to
self-evident. The stronger research question is whether a small amount of
aligned KV state can measurably reduce compaction damage without retaining the
full context. The intended contribution should therefore emphasize mitigation,
memory/compute cost, specificity, and deployability.

Methodologically, probe accuracy under leakage classes—especially
`absent-from-summary` and `evicted-only` cases—should be the primary metric;
continuation NLL is supporting evidence. Summary leakage remains the main
validity threat, requiring brief summaries and judge-based paraphrase checks.
`E` (low-alpha value grafting) is the practical mitigation path; `H` versus
`B-min` is the cleanest small-state test; `C` is informative but confounded by
gapped-cache and out-of-distribution attention geometry; a contiguous method
such as `G` may be more deployable. Shuffled and wrong-conversation grafts are
essential specificity controls. Interim evidence suggested a small, consistent
low-alpha continuation gain for `E-post-a0.25`, degradation at larger alpha,
frequent harm from `C`, and strong failure of negative controls; these results
were not publication-grade until judging and leakage-controlled scoring were
complete.

At the orientation point, the tree was clean on `trunk`, but state was
mid-transition: supplemented raw files contained newer B-causal and
negative-control arms than `results/analysis.md`; B-causal continuation remained
missing and required `src/fix_bcausal_cont.py`; `results/raw_brief/` existed but
was empty; judge queues, scores, and final `RESULTS.md` were absent. Any
takeover must re-read current state, inspect raw JSON, and confirm whether
queued or detached work has completed before interpreting results.

A new top-level Markdown memo, `refocusing-and-reframing.md`, was created to
describe this potential reframing and committed alone as
`423e8a5 Add compaction reframing memo`. Existing untracked `results/raw_brief/`
was explicitly left unmodified and uncommitted.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
