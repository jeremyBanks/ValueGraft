# Powered-v13 stimuli/tokenizer repair re-audit

Date: 2026-07-12

Audited HEAD: `d3809e88bb8f22edf2f9331514627c2be88b1711`

Repair commit reviewed: `20e795c2da80d483bce522c6a204dbd4df19c55e`

Prior audit-note commit reviewed: `d3809e88bb8f22edf2f9331514627c2be88b1711`

Scope: independent static/mechanical re-audit of the v13 tokenizer/stimulus
repair, including the implementation paths called by
`src/powered_v13_stimuli.py`, not only its committed tests. I reproduced every
previously reported coercion/separator adversary, added boundary and shared-BPE
probes, independently recomputed the current v2 evidence, and reran the focused
and integrated suites. I did not edit implementation, request entropy,
materialize a production-pool tuple/history, call the subject model, or perform
paid work.

## Verdict

**BLOCK.** The public carrier-array/EOS repair works and most separator
adversaries are now rejected, but the original U+02BC apostrophe bypass remains
open and token-plan helpers still erase malformed tokenizer-output types with
`int(...)`. This is a static/mechanical disposition only. It authorizes no
carrier acceptance, Phase A, treatment, entropy request, or paid work.

## Blocking defects

### 1. The original U+02BC apostrophe adversary still passes

The new matcher normalizes with NFKD and then treats every `isalnum()` character
as semantic content (`src/powered_v13_stimuli.py:650-689`). That catches the
new ASCII-apostrophe test, because U+0027 is punctuation and not alphanumeric.
It does not catch the exact apostrophe from the prior audit: U+02BC MODIFIER
LETTER APOSTROPHE has Unicode category `Lm`, `isalpha() == True`, and
`isalnum() == True`; NFKD leaves it unchanged.

Against the exact cached production tokenizer and the otherwise-valid frozen
carrier, both of these attempts returned
`MECHANICAL_CARRIER_PASS_SEMANTIC_REVIEW_PENDING`:

- focal target: `Qʼuʼaʼrʼtʼz` (U+02BC between letters), 58 content tokens;
- case-declared phrase: `veiledʼcheckpoint`, with
  `case_declared_forbidden_phrases=["veiled checkpoint"]`, 51 content tokens.

These are the same modifier-apostrophe bypass class reported before the repair.
The added regression silently substitutes ASCII U+0027 in
`tests/test_powered_v13_stimuli.py:413`; the case-phrase regression at lines
458-465 tests only bullet and zero-width space. The output claim
`separator_insensitive_matching_applied=True` is therefore not fail-closed for
the prior adversary.

The other exact prior separator probes did reject through the new matcher:

| Surface class | Focal `Quartz` | Case phrase `veiled checkpoint` |
|---|---:|---:|
| U+2022 bullet | rejected | rejected |
| U+00B7 middle dot | rejected | rejected |
| U+200B zero-width space | rejected | rejected |
| U+02BC modifier apostrophe | **passed** | **passed** |
| vertical bar | rejected | rejected |
| reverse solidus/backslash | rejected | rejected |

The prior split-letter `v•e•i•l•e•d checkpoint` case also rejected at 59
tokens. Required repair: explicitly normalize or reject apostrophe-like
modifier letters, add the literal U+02BC focal and case-phrase strings as
can-it-fail tests, and avoid replacing a reported code point with an ASCII
lookalike in regression coverage.

### 2. Token-plan helpers still coerce malformed tokenizer outputs

The repair correctly adds `_plain_int_sequence` around direct paths in
`src/powered_v13_stimuli.py`, but the implementation it calls still contains
the old type-erasing conversions in `src/powered_v13_tokens.py`:

- boundary IDs and scanned IDs: lines 27-36;
- rendered assistant-content IDs: lines 63-65;
- role-native canonical IDs: lines 142-143;
- fresh compact canonical IDs: lines 276-277;
- probe generation-prefix IDs: line 349;
- probe target IDs: lines 364-365.

I injected complete value-identical numeric-string arrays and complete
value-identical float arrays separately at each of the
`canonical_ids_any`, `generation_prefix_ids`, and
`rendered_assistant_content_ids` output boundaries used by those helpers. All
six full `validate_development_sentinel` probes passed and produced exactly the
same mechanical evidence hash as the plain-integer baseline:

`10f9ca6ed7e93d1c9157c769077bdca60f7c0502d28bcbcab83981cea6e3f7cb`.

Thus the outer validator cannot attest that those plan/probe token arrays were
plain integer outputs: the helpers erase the original types before the new
strict checks see them. Required repair: validate `type(value) is int`,
nonnegativity, and tokenizer-ID range at every helper boundary before any
conversion; remove the `int(...)` laundering; and add full-validator injection
tests for each tokenizer-output path.

## Strict-type checks that did pass

The repaired public carrier path rejected every tested malformed caller value:

- the exact old probes using all 49 IDs as decimal strings plus a decimal-
  string EOS, and all 49 IDs as floats plus a float EOS;
- generated-content IDs: decimal string, float, boolean, negative, ID equal to
  `len(tokenizer)`, and an entire string in place of an array;
- EOS witness: decimal string, float, boolean, negative, out-of-range, and
  `None`;
- render and attempt indices: decimal string, float, boolean, negative, zero,
  and values outside `1..2` / `1..3`;
- `hit_token_cap`: `True`, integer zero, integer one, and `None`;
- content cap: string, float, boolean, negative, and out-of-range;
- direct `tokenizer.encode()` re-encode output and forbidden-surface batched
  tokenizer output: string, float, boolean, negative, and out-of-range ID.

Those all failed before a mechanical PASS. This closes the prior public
carrier-array/EOS coercion, but it does not close blocker 2's helper-level
coercions.

## Matcher boundary and shared-BPE checks

The bounded matcher itself returned the intended results apart from U+02BC:

- ordinary exact `Quartz` and exact case phrase `veiled checkpoint` rejected;
- separator-obfuscated `Q•u•a•r•t•z` matched;
- it did not claim `Quartz` inside plain `Quartzite`;
- with a case-declared base `veiled`, exact `veiled` rejected while plain
  `unveiled` passed, and the private skeleton matcher likewise returned false
  for `veiled` inside `unveiled`.

Restricting production-token witnesses to semantic base forms did not reopen
the shared-BPE prefix leak. The exact cached tokenizer produced:

- `Quartz` -> `[2183, 21593]`, `Quartzite` -> `[2183, 21593, 632]`;
- whitespace-prefixed ` Quartz` -> `[65088]`, whitespace-prefixed
  ` Quartzite` -> `[65088, 632]`.

Both context-appropriate `Quartz` sequences remain in the v2 base-form witness
bank. A carrier containing plain `Quartzite` was not falsely claimed by the
bounded surface matcher, but was correctly rejected by the independent token-
subsequence gate at carrier content token 31.

## Current v2 hashes and evidence

All hashes below were read as literal values and independently recomputed from
their canonical JSON cores; a PASS label was not used as a substitute.

- forbidden expansion schema:
  `powered-v13-carrier-forbidden-surface-expansion-v2`;
- 16-sentinel compact matrix:
  `f545875979b59bdeb1efea452e2f2818420c07ec2da5d76527ee8951b566ec7f`;
- production target-bank qualification:
  `378c078e4ab5862e6a8895a130443dcb4e2d990d1dda69e18598114841b374ab`;
- threshold-explicit v2 expansion example:
  `db681b816b8d1c25826850a287bfe7bd116e16e0606ec31b54fbb47ab4bed596`.

All 16 per-sentinel v2 expansion hashes recomputed exactly. Inventory counts
were:

| Evidence inventory | Total over 16 | Per-sentinel min | Per-sentinel max |
|---|---:|---:|---:|
| semantic base forms | 558 | 18 | 106 |
| raw surface forms | 9,698 | 379 | 1,638 |
| normalized surface forms | 4,388 | 158 | 813 |
| separator-insensitive skeletons | 490 | 18 | 81 |
| base-form token subsequences | 1,116 | 36 | 212 |

The reduction from the old 19,396 expanded-surface token witnesses to 1,116
bounded base-form witnesses is recorded by the v2 schema and did not change the
production target-bank hash.

## Tokenizer binding, target bank, and geometry

The tokenizer resolved exactly:

- model `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- wrapper `transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer`;
- backend `tokenizers.Tokenizer`, 5,363,081 serialized UTF-8 bytes, SHA-256
  `41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04`;
- vocabulary SHA-256
  `f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d`;
- chat-template SHA-256
  `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`;
- EOS `<|im_end|>` / `151645`;
- Transformers `5.0.0`, tokenizers `0.22.2`, huggingface-hub `1.22.0`.

Independent byte/hash checks matched all pinned files and their aggregate:

| File | Bytes | SHA-256 |
|---|---:|---|
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,377 | `a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

Aggregate file-record SHA-256:
`29b49d5d319daf05184345c1a61ee2411cf33bee814a6573ffdb5c535ece486e`.

The production component bank contained 8 focal banks, 4 nonfocal banks, and
24 pairwise distinct/nonprefix surface and contextual-token targets with exact
decode/re-encode and answer/close witnesses.

All 16 out-of-pool family-by-subtype sentinels passed the current frozen
positive mechanics and had 16 unique development IDs. Observed geometry was:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| noncarrier source tokens | 1,055 | 1,113 |
| completed source tokens | 1,104 | 1,162 |
| fresh tokens | 477 | 477 |
| R1 carrier rows | 49 | 49 |
| R2 carrier-boundary rows | 74 | 74 |

C/W role-native geometry, C/W turn-aligned geometry, and C/W fresh destination
geometry passed 16/16. Independent literal slicing observed equal selected
token IDs for C/W and C/F in both R1 and R2 in 16/16, plus exact fresh logical
position/source-index mappings in 16/16. The frozen development carrier was 44
words / 49 tokens. All six render-by-attempt seeds were distinct:

`366945260647387824`, `8686567877660493374`,
`4294531769127951643`, `3736491186688189429`,
`6520104884645219423`, and `4295207695510811772`.

These are development-only geometry witnesses. They do not establish actual
subject sampling, forced replay, log-probability/K/V identity, attempt-order
persistence, semantic review, or production-candidate eligibility.

## Randomization boundary and tests

Source/import inspection found only the explicit out-of-pool sentinel
materializer and compact recipe-definition read in the aggregate validator. No
production ranked materializer or pool enumerator is imported/called there,
and there is no NumPy, `random`, `secrets`, OS-entropy, or permutation operation
in the stimuli/token-plan modules. The observed matrix explicitly recorded:

- `sentinel_count=16`;
- `in_pool_tuple_materialized=false`;
- `permutation_seed_present=false`;
- `literal_permutation_present=false`.

Test execution at exact HEAD:

- focused `tests/test_powered_v13_stimuli.py`: **81 collected, 81 passed**;
- integrated recipe/tokens/stimuli selection: **184 collected, 184 passed**.

The green suites do not contain the literal U+02BC regression and do not inject
malformed outputs before the remaining token-plan conversions. Repair both
blockers and obtain another independent re-audit before any static freeze.
