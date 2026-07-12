# Powered-v13 stimuli/tokenizer final audit

Date: 2026-07-12  
Audited commit: `2c34089a7780257c56d296c73940eb0a4bfe7092`  
Scope: independent static/mechanical audit of
`src/powered_v13_stimuli.py` and `tests/test_powered_v13_stimuli.py` against
preregistration Sections 3, 5, 6, and 7 plus the repository scientific hard
rules. No production-pool candidate/history was materialized, no entropy was
requested, and no subject forward or paid work was performed.

## Verdict

**BLOCK.** The pinned tokenizer and the out-of-pool sentinel mechanics produced
the expected frozen outputs, but the carrier validator remains fail-open in two
ways that can admit malformed token provenance or literal forbidden content.
This is a static/mechanical disposition only and authorizes no paid work,
Phase A, carrier acceptance, or treatment.

## Blocking defects

### 1. Caller-supplied token IDs and EOS are coerced instead of type-checked

`validate_carrier_attempt` converts every caller-supplied content ID with
`int(value)` at `src/powered_v13_stimuli.py:787` and similarly converts the EOS
witness at lines 798--800. This bypasses the strict integer-array rule newly
implemented in `sha256_ints` at lines 149--156: by the time the hash function is
called, the malformed original JSON types have already been erased.

Against the exact cached production tokenizer, I supplied the complete valid
49-token carrier stream as each of the following and observed a mechanical
carrier pass in both cases:

- 49 decimal strings, with the EOS ID also supplied as a decimal string;
- 49 Python floats, with the EOS ID also supplied as a float.

Both returned `production_content_token_count=49`. Thus malformed persisted
token arrays such as `['785', ...]` or `[785.0, ...]` can be relabeled as exact
integer token evidence. The direct unit test at
`tests/test_powered_v13_stimuli.py:78-85` proves that `sha256_ints` itself
rejects these types, but no test sends them through the public carrier gate.
This violates fail-closed token provenance and the hard rule requiring every
token's exact origin.

Required repair: require `type(value) is int` for every supplied content ID and
for the termination ID before any conversion, validate token-ID range, and hash
the untouched integer array. Add public-path string, float, boolean, negative,
and out-of-vocabulary failure tests.

### 2. Forbidden target and case-declared phrases can be split with unhandled characters

The surface expansion enumerates a finite joiner set at lines 533--612. The
candidate-text normalizer at lines 615--617 performs NFKC, casefolding, and
whitespace collapse but does not remove or canonicalize arbitrary Unicode
punctuation, symbol, or format characters. The token-subsequence inventory at
lines 681--697 can cover only the explicitly generated forms. Consequently the
two checks at lines 812--823 both miss forbidden words whose letters/components
are separated by an unenumerated character.

Using the threshold sentinel, whose literal focal target is `Quartz`, I
replaced `settled detail` in the otherwise-valid 44-word carrier and observed
all six of these carriers pass:

- `Q•u•a•r•t•z` (58 tokens);
- `Q·u·a·r·t·z` (58 tokens);
- `Q` plus zero-width-space-separated `u a r t z` (58 tokens);
- `Qʼuʼaʼrʼtʼz` (58 tokens);
- `Q|u|a|r|t|z` (57 tokens);
- backslash-separated `Q\u\a\r\t\z` (54 tokens).

The same defect applies to caller-declared exclusions. With
`case_declared_forbidden_phrases=['veiled checkpoint']`, all of
`veiled•checkpoint` (51 tokens), split-letter `v•e•i•l•e•d checkpoint`
(59 tokens), and `veiled` plus a zero-width space plus `checkpoint` (51 tokens)
passed. These are literal encodings of a forbidden target/phrase, not merely
semantic paraphrases reserved for independent review.

The committed tests at `tests/test_powered_v13_stimuli.py:391-446` cover the
enumerated hyphen, slash, underscore, dot, en/em dash, camel-case, and ordinary
phrase forms, but contain no unenumerated Unicode punctuation, symbol, format,
or zero-width adversary. The output's claim
`slash_camel_split_letter_expansion_applied=True` therefore overstates the
closed-world guarantee.

Required repair: compare an additional bounded canonical skeleton that removes
or normalizes all Unicode punctuation/separator/format insertions between
alphanumeric components (or reject such characters outright), for both the
full semantic inventory and case-declared phrases. Add the literal adversaries
above, including zero-width format characters, to can-it-fail tests.

## Checks that passed at this commit

### Exact tokenizer binding

The runtime binding resolved the exact model/revision and rejected an altered
backend normalization. Independent `shasum -a 256` and byte counts agreed with
all four source constants:

| File | Bytes | SHA-256 |
|---|---:|---|
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,377 | `a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

Observed binding details were:

- wrapper `transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer`;
- backend `tokenizers.Tokenizer`;
- backend serialization: 5,363,081 UTF-8 bytes, SHA-256
  `41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04`;
- vocabulary SHA-256
  `f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d`;
- chat-template SHA-256
  `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`;
- EOS `<|im_end|>` / ID `151645`;
- Transformers `5.0.0`, tokenizers `0.22.2`, huggingface-hub `1.22.0`.

The four-file aggregate binding was
`29b49d5d319daf05184345c1a61ee2411cf33bee814a6573ffdb5c535ece486e`.
Dependency versions are recorded rather than authorized here; the release and
subject loader must still bind their lock/runtime hashes.

### Canonical hashing and Section-5 attempt law

`sha256_ints` now hashes canonical UTF-8 JSON arrays without coercion. An
independent hash of `[0,1,151643,-1]` agreed at
`ce82ca475209ac9e6e15dae08672906ce0faf65a6d409b650766ec273b9791a2`.
The attempt digest is also the required canonical JSON array, followed by the
first-eight-byte big-endian 63-bit seed rule.

All six render/attempt combinations (two renders by three attempts) produced
six distinct seeds. For the first threshold sentinel they were:

`366945260647387824`, `8686567877660493374`,
`4294531769127951643`, `3736491186688189429`,
`6520104884645219423`, and `4295207695510811772`.

The spec binds the exact prompt, q=1, CPU generator, temperature `0.7`, top-p
`0.95`, disabled top-k, 80 content-token cap, and a separate final EOS witness
for at most 81 sampling calls. The frozen development carrier observed 44 words
and 49 content tokens; the committed 80-content-token boundary test also
passed with a distinct 81st EOS call. Exact decode and encode-after-decode
checks rejected both ordinary ID/text mismatch and a noncanonical ID sequence
that decoded to the same text.

This validator accepts an already-generated attempt; it does not attest the
actual model call trace, CPU generator state, per-token probability bits,
attempt-order persistence, or prior rejection receipts. Its own output is
correctly labeled `execution_ready=False` and semantic review pending. Those
items remain mandatory in the later runtime/store path and this mechanical
record must not be promoted into generation provenance.

### Target banks and out-of-pool geometry

The exhaustive compact target-bank pass observed:

- 8 focal banks and 4 nonfocal banks;
- 24 mutually distinct, pairwise nonprefix target surfaces;
- 24 mutually distinct, pairwise nonprefix contextual token sequences;
- exact decode/re-encode witnesses, answer positions, assistant-close
  positions, token arrays, and canonical token-array hashes for every target;
- qualification SHA-256
  `378c078e4ab5862e6a8895a130443dcb4e2d990d1dda69e18598114841b374ab`.

All 16 out-of-pool family-by-explicitness sentinels passed their pinned current
mechanics and had 16 unique development IDs. Observed ranges were:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| noncarrier source tokens | 1,055 | 1,113 |
| completed source tokens | 1,104 | 1,162 |
| fresh tokens | 477 | 477 |
| R1 carrier rows | 49 | 49 |
| R2 carrier-boundary rows | 74 | 74 |

C/W N geometry, C/W P geometry, and C/W fresh destinations passed for all
16/16 sentinels. An additional literal audit observed identical selected token
IDs for C/W and C/F in R1 and R2 in all 16/16 cases, and identical fresh
logical-position mappings in all 16/16 cases. The compact matrix hash was
`e5d78fa98bbf2e1725535b34aba37ee8f954c1c2319de80548e946fc49f4954e`.

The production function explicitly asserts geometry equality but does not
itself assert the cross-history selected R1/R2 ID equality just audited. Full
sampled-to-forced IDs, positions, log-probability bits, K/V rows, and EOS
identity remain runtime gates under Section 5. No claim about ranked
production candidates follows from these development sentinels.

### Forbidden-inventory breadth and randomization boundary

Across the 16 development sentinels, the expansion produced:

- 558 total base-form entries (18--106 per sentinel);
- 9,698 raw surface entries (379--1,638 per sentinel);
- 4,388 normalized entries (158--813 per sentinel);
- 19,396 production-token subsequences (758--3,276 per sentinel).

This breadth successfully rejected the committed ordinary Unicode/fullwidth,
case, number-word, time, punctuation, slash, camel, and target-token-prefix
examples, but it does not cure blocker 2's open character class.

Source and import inspection found only the explicit out-of-pool sentinel
materializer plus the compact recipe definition. There is no production ranked
materializer, pool enumerator, NumPy, `random`, `secrets`, or OS-entropy call in
the module. The aggregate pass materialized exactly 16 out-of-pool sentinels;
all candidate/rank/seed/permutation authorization fields were null. Target-bank
qualification read the compact component banks only and constructed no tuple or
history. The audit requested no entropy and did not inspect a ranked candidate.

## Test execution

- `tests/test_powered_v13_stimuli.py`: **67 collected, all passed**.
- Integrated recipe/token/stimuli selection: **170 collected, all passed**.
- Additional independent probes supplied strict-type and open-Unicode
  adversaries not present in the committed suite and reproduced both blockers.

The green committed suite therefore does not establish a fail-closed carrier
gate. Repair both blockers, add literal regression cases, rerun the pinned
tokenizer and integrated suites, and obtain another independent audit before
static freeze. Even a later PASS would remain static/mechanical evidence only,
not paid authorization.
