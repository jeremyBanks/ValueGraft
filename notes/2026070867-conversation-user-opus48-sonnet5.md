_This conversation covers recovery from a self-generated-summary alignment
failure, discovery that debugging used the wrong Qwen checkpoint, and
confirmation that the original graft effect is real while the augmented corpus
is currently unsuitable for the wide sweep._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

The positive-control gate initially failed because `Qwen3-30B-A3B` (thinking
variant) was tested instead of the known-good `Qwen3-30B-A3B-Instruct-2507`
checkpoint. The resulting `<think>` token divergence, alignment crashes, and
negative metrics were therefore misdiagnosed as apparatus bugs. Reverting the
shared `build_alignment` delegation to tolerant difflib alignment was committed
and pushed, but was not the numerical fix; the decisive correction was using the
exact known-good checkpoint. The corrected run reproduced the original c01–c12
effect: referent +0.1246 (known baseline approximately +0.136), sense +0.047,
and stance +0.002/null. The effect is therefore validated, and the
positive-control gate is green for the correct model.

The corpus was expanded from 27 to 54 conversations through six parallel
foundation-API subagent shards using Sonnet/Fable/Opus renderers; no local
models were used for authoring. The deleted intermediate batch files were
recovered byte-identically, merged, and verified. The frozen corpus contains 54
conversations and 626 plants, with exact inventory, valid schema, verbatim plant
preservation, correct placement, ASCII-only content, and no forbidden-name
violations. Sharding is now preferred for both latency and quality because it
increases author diversity and avoids long-context repetition. Operational rules
established: verify actual progress before killing jobs, never delete
uncommitted inputs before successful consumption and commit, commit and push
each work unit to trunk, and verify the exact model identifier before debugging
alignment or metrics. Branch work was consolidated back to trunk; no feature
branch should remain.

The augmentation is not yet validated scientifically. On c13–c54, referent
recovery was only +0.009 despite a substantial referent gap of 1.12, whereas
c01–c12 had a gap of 1.69 and recovered +0.12. Thus the new rendered
conversations appear to contain evicted meaning but fail to recover it through
grafting; adding them would dilute the signal. The leading handoff
recommendation is to run the extra-wide cross-architecture study on trusted
c01–c12 only, while treating c13–c54 rendering as a separate quality
investigation. The previously stated extra-wide scope includes more vendors
beyond the frozen 16-model set, roughly 150 plants per category, placebo and
alpha controls on every model, own-summary experiments on anchors, and a wider
alpha grid.

Wide launch was held until the positive control passed. Earlier estimates were
approximately 30–60 minutes for the null-self-graft/fix cycle, 30–50 minutes for
a fresh trusted-apparatus validation, 10–15 minutes for incremental results, and
30–40 minutes for the correct-model run; these estimates were revised by pod
instability and launcher/monitoring failures. The latest handoff state is:
effect validated on the correct checkpoint, trunk pushed, original corpus
trusted, augmented corpus questionable, and the next decision is whether to
launch wide on c01–c12 or repair and revalidate the augmented renders first.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a3206aea52f9eabd7`
- `afa22cecd486e4630`
- `a91c6f7674d0b185c`
- `a16a3dffa164d7ab1`
- `a1de9381097c7d33d`
- `aa40ced3907af5144`
- `a6b732697e956deef`
- `a99a7ab082e2d6ea9`
- `a76eba0c08b5dbac7`
- `a039abc01d52958ac`
- `af9083862e12b5076`
- `a077c5b587eb70c7d`
- `afd83b036489c7657`
- `a901c4e22664acf2a`
- `aa54e4fb5dc64a494`
