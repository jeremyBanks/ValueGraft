# Powered-v13 stimuli/tokenizer final-pass audit

Date: 2026-07-12

Audited HEAD: `2ba4b3650badd29e3562ce9a5fbb6831cdf99372`

Scope: independent static and mechanical review of
`src/powered_v13_stimuli.py`, `src/powered_v13_tokens.py`,
`tests/test_powered_v13_stimuli.py`, and
`tests/test_powered_v13_tokens.py`. I specifically rechecked the two defects
from the prior repair re-audit, Unicode carrier surfaces, plain-integer token
boundaries, validator failure coverage, Section-5 carrier limits/EOS, and exact
C/W/F carrier-region token/position invariants. I made no implementation edit,
requested no entropy, materialized no production-pool tuple or ranked history,
called no subject model, and performed no paid work.

## Verdict

**BLOCK.** The U+02BC modifier-apostrophe bypass and the previously observed
string/float token-ID laundering are closed, all committed tests pass, and all
16 development sentinels retain exact R1/R2 C/W/F carrier rows. However, the
replacement token-plan validator checks only that IDs are plain nonnegative
Python integers. It does not reject a positive integer equal to the tokenizer
vocabulary size. A selective out-of-vocabulary injection into 36
`generation_prefix_ids` calls passed the complete sentinel validator and
produced the exact same mechanical-evidence hash as the valid baseline.

This leaves the prior audit's explicit tokenizer-ID-range requirement
unsatisfied. The result remains static/mechanical and authorizes no entropy,
carrier acceptance, Phase A, treatment, or paid execution.

## Blocking defect: token-plan boundaries accept positive out-of-vocabulary IDs

`_plain_token_ids` at `src/powered_v13_tokens.py:27-42` requires
`type(value) is int` and `value >= 0`, but has no
`value < len(tokenizer)` (or equivalent maximum-exclusive) check. Every
token-plan boundary delegates to it:

- tokenizer boundary literals and canonical message streams, lines 45-58;
- assistant generation prefixes and contextual content IDs, lines 85-92;
- role-native canonical streams, lines 169-172;
- fresh compact canonical streams, lines 305-308;
- probe generation prefixes, lines 380-383;
- probe target content, lines 398-401.

The production tokenizer had `len(tokenizer) == 151669`, so valid token IDs
were bounded above by `151668`. The helper nevertheless returned
`[151669]` for a direct one-element input. More importantly, the full
validator accepted 36 selectively corrupted assistant-boundary prefix arrays
containing only `151669`. These arrays retain the correct lengths, which are
all `_assistant_content_bounds` consumes, so the corruption is erased from the
result and the evidence hash remains unchanged:

```text
{'vocab_size': 151669,
 'direct_helper_accepts_oov': [151669],
 'injected_calls': 36,
 'validator_status': 'MECHANICAL_SENTINEL_PASS_NO_EXECUTION_AUTHORIZATION',
 'baseline_hash': '10f9ca6ed7e93d1c9157c769077bdca60f7c0502d28bcbcab83981cea6e3f7cb',
 'injected_hash': '10f9ca6ed7e93d1c9157c769077bdca60f7c0502d28bcbcab83981cea6e3f7cb',
 'hashes_equal': True}
```

This is not the public carrier-array path: that path correctly calls
`_plain_int_sequence(..., maximum_exclusive=len(tokenizer))` at
`src/powered_v13_stimuli.py:918-922`. It is the internal tokenizer-plan layer
that remains fail-open. The distinction matters because the mechanical
sentinel currently purports to validate those exact tokenizer boundaries.

The committed direct helper test at
`tests/test_powered_v13_tokens.py:156-159` covers a decimal string, float,
boolean, and negative integer, but no positive out-of-vocabulary integer. The
full-validator injections at
`tests/test_powered_v13_stimuli.py:621-639` replace outputs only with strings
or floats. Consequently the green suite cannot catch this defect.

Required repair: make every `_plain_token_ids` call enforce the pinned
tokenizer's maximum-exclusive vocabulary bound before returning the array, and
add both a direct `len(tokenizer)` rejection and a complete-validator
out-of-vocabulary injection. The fix should cover all six boundary classes,
not rely on later equality checks that happen to reject some corruptions.

## Checks that passed

### Unicode and forbidden surfaces

The modifier-letter repair is correct for the reported bypass. The new
`_is_surface_alphanumeric` at
`src/powered_v13_stimuli.py:650-657` excludes Unicode category `Lm`, including
U+02BC MODIFIER LETTER APOSTROPHE, from semantic word characters. The
separator-insensitive skeleton matcher uses the same predicate for skeleton
construction, scanning, and both word boundaries at lines 660-702.

The literal focal adversary `Qʼuʼaʼrʼtʼz` and the case-declared adversary
`veiledʼcheckpoint` are now committed tests at
`tests/test_powered_v13_stimuli.py:417` and lines 462-471. They reject, as do
fullwidth text, bullet, middle-dot, zero-width-space, ASCII apostrophe,
vertical-bar, reverse-solidus, slash, punctuation, camel-case, numeric-word,
time, and shared-BPE-prefix probes. Bounded matching still avoids claiming the
literal `Quartz` inside `Quartzite`; the independent token-subsequence gate
rejects that shared token prefix.

### Public carrier attempt invariants

The public carrier gate now enforces all mechanical Section-5 fields reviewed:

- exact frozen prompt, render/attempt indices, digest, 63-bit seed, q=1,
  temperature `0.7`, top-p `0.95`, disabled top-k, and content cap;
- 40--60 whitespace-delimited words and 40--80 content tokens;
- plain in-range generated IDs, no special IDs or control literals, and exact
  decoded text plus exact encode-after-decode IDs;
- strict `hit_token_cap is False` and a plain integer EOS equal to the pinned
  tokenizer EOS;
- 80 content tokens plus one distinct EOS witness, for 81 maximum sampling
  calls;
- Unicode digit exclusion, normalized surface matching,
  separator-insensitive skeleton matching, production-token subsequences, and
  case-declared forbidden phrases;
- an explicitly nonauthorizing result with independent target-aware semantic
  review still pending.

The frozen 80-token carrier boundary test passed with the EOS as call 81.
String, float, boolean, negative, and out-of-range caller-supplied carrier IDs
failed; string/float/boolean EOS witnesses failed; a same-decoded-text but
noncanonical ID sequence failed.

### Exact carrier geometry and token identity

I independently rebuilt both C and W source/fresh plans for every one of the
8 strata by 2 resolution subtypes. For each of R1 and R2 I asserted:

1. C and W source intervals were equal;
2. every selected source token ID was identical across C and W;
3. each C/W fresh physical slice was byte-for-token identical to its source
   slice;
4. each fresh logical-position slice was the exact consecutive source range;
5. the complete C and W fresh destination plans were equal.

All 64 variant-by-region checks passed. The complete out-of-pool matrix and
target-bank hashes remained:

```text
sentinels: 16
exact_region_checks: 64
matrix_sha256: f545875979b59bdeb1efea452e2f2818420c07ec2da5d76527ee8951b566ec7f
target_bank_sha256: 378c078e4ab5862e6a8895a130443dcb4e2d990d1dda69e18598114841b374ab
```

These are development-only tokenizer/geometry witnesses. They do not attest
subject sampling, attempt-order persistence, forced-replay log-probability or
K/V identity, semantic review, production eligibility, or authorization.

## Commands and observed results

HEAD and initial tree state:

```sh
git rev-parse HEAD
git status --short --branch
```

```text
2ba4b3650badd29e3562ce9a5fbb6831cdf99372
## HEAD (no branch)
```

The repository lock requires Python 3.12. A bare system-Python attempt was
nonauthoritative and failed during collection under Python 3.9: first because
`src` was absent from the import path, then (with `PYTHONPATH=src`) because
Python 3.9 does not support the recipe's dataclass `slots` argument and the
system environment lacked Torch. The locked invocation was:

```sh
PYTHONPATH=src uv run --frozen pytest -q \
  tests/test_powered_v13_stimuli.py \
  tests/test_powered_v13_tokens.py
```

Result: **100 passed**, 2 SWIG deprecation warnings, in **99.85s**. The locked
runtime versions were obtained with:

```sh
PYTHONPATH=src uv run --frozen python -c \
  'import platform, transformers, tokenizers, huggingface_hub; print(platform.python_version(), transformers.__version__, tokenizers.__version__, huggingface_hub.__version__)'
```

```text
3.12.11 5.0.0 0.22.2 1.22.0
```

Focused Unicode/type/cap regressions:

```sh
PYTHONPATH=src uv run --frozen pytest -q \
  tests/test_powered_v13_stimuli.py::test_carrier_gate_rejects_expanded_semantic_leaks \
  tests/test_powered_v13_stimuli.py::test_case_declared_forbidden_phrases_are_accepted_expanded_and_hash_bound \
  tests/test_powered_v13_stimuli.py::test_full_sentinel_rejects_type_laundering_at_token_plan_boundaries \
  tests/test_powered_v13_stimuli.py::test_exactly_80_content_tokens_then_separate_eos_witness_is_accepted \
  tests/test_powered_v13_tokens.py::test_token_plan_boundary_rejects_nonplain_or_negative_ids \
  tests/test_powered_v13_tokens.py::test_dynamic_planner_rejects_boolean_middle_index
```

Result: **37 passed**, 2 SWIG deprecation warnings, in **25.23s**.

Syntax compilation:

```sh
PYTHONPATH=src uv run --frozen python -m compileall -q \
  src/powered_v13_stimuli.py src/powered_v13_tokens.py \
  tests/test_powered_v13_stimuli.py tests/test_powered_v13_tokens.py
```

Result: exit `0`, no output.

The exact-region audit was executed with
`PYTHONPATH=src uv run --frozen python -` and iterated `STRATA x (True,
False)`, using `_exact_pair_checks` and `_variant_plans`; its literal assertions
are the five numbered invariants above and its exact output is recorded in the
geometry section.

The blocking probe used the same locked stdin invocation. It computed a valid
baseline, replaced only non-probe `generation_prefix_ids` outputs with
`[len(tokenizer)] * len(ids)`, ran the complete sentinel validator, and
compared the hashes. Its exact output is reproduced in the blocker section.

No source or test file was edited during this audit.
