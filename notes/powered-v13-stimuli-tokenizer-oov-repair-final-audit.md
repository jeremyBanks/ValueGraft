# Powered-v13 stimuli/tokenizer OOV-repair final audit

Date: 2026-07-12

Audited HEAD: `a2b31eca7bff28f665a4fa01c40e09efbfb52237`

Repair commit reviewed: `edbc3f36c1242a3d22907c27901f41aa5003a533`

Scope: final independent static/mechanical recheck of
`src/powered_v13_stimuli.py`, `src/powered_v13_tokens.py`,
`tests/test_powered_v13_stimuli.py`, and
`tests/test_powered_v13_tokens.py` after the positive
out-of-vocabulary-token repair. I checked every canonical, message-marker,
generation-prefix, assistant-content, compact-render, and probe token boundary;
the exact pinned vocabulary including added tokens; the prior Unicode and
plain-integer regressions; and all 16 development sentinels' R1/R2 C/W/F token
and position invariants. I made no implementation edit, requested no entropy,
materialized no production-pool tuple or ranked history, called no subject
model, and performed no paid work.

## Verdict

**PASS.** The `len(tokenizer)`-bounded token-ID repair closes the prior OOV
defect at every production call site reviewed. ID `151669 == len(tokenizer)`
was rejected independently in all eight isolated boundary classes. The exact
pinned vocabulary is contiguous through `151668`; all 26 added-token IDs,
including the highest valid ID `151668`, remain admissible. Real canonical and
generation streams containing added structural tokens also passed the full
sentinel validator.

All committed focused tests passed, the prior Unicode and type regressions
remained closed, and all 16 development sentinels retained exact R1/R2 C/W/F
tokens and logical positions with unchanged frozen hashes. This PASS is only
for the unpaid tokenizer/stimulus mechanical layer. It does not authorize
entropy, carrier acceptance, Phase A, treatment, or paid execution.

## OOV repair correctness

### Static boundary coverage

`_plain_token_ids` at `src/powered_v13_tokens.py:27-49` still rejects a
non-sequence, empty disallowed sequence, non-plain integer, boolean, float,
string, and negative integer. It now additionally accepts an inclusive
`maximum_token_id` and rejects every value greater than that maximum.

`_maximum_token_id` at lines 52-59 obtains the exact tokenizer width, rejects
an unavailable/nonpositive width, and returns `len(tokenizer) - 1`.

AST inspection found exactly eight production `_plain_token_ids` call sites,
at lines 63, 74, 106, 111, 192, 329, 405, and 424. Every call supplies
`maximum_token_id=_maximum_token_id(tokenizer)`. They cover:

1. the encoded `<|im_start|>` boundary marker;
2. canonical message streams scanned for marker positions;
3. assistant generation prefixes used to determine content starts;
4. rendered assistant-content IDs;
5. role-native canonical streams;
6. fresh compact canonical streams;
7. probe generation prefixes;
8. probe target content.

The literal AST result was:

```text
{'call_count': 8,
 'calls': [
   {'line': 63, 'has_maximum_token_id': True},
   {'line': 74, 'has_maximum_token_id': True},
   {'line': 106, 'has_maximum_token_id': True},
   {'line': 111, 'has_maximum_token_id': True},
   {'line': 192, 'has_maximum_token_id': True},
   {'line': 329, 'has_maximum_token_id': True},
   {'line': 405, 'has_maximum_token_id': True},
   {'line': 424, 'has_maximum_token_id': True}]}
```

### Exact vocabulary and valid added tokens

Against the exact cached production tokenizer, the full vocabulary contained
151,669 unique contiguous IDs from 0 through 151,668. The base
`tokenizer.vocab_size` was 151,643, and the 26 added-token IDs occupied
151,643 through 151,668. Therefore `len(tokenizer) - 1` is the exact inclusive
maximum for this pinned tokenizer; it neither admits a gap nor excludes a
valid added token.

I passed both the lowest and highest added-token IDs directly through the new
gate with maximum 151,668. Both were accepted. ID 151,669 was rejected:

```text
{'len_tokenizer': 151669,
 'tokenizer_vocab_size': 151643,
 'unique_vocab_ids': 151669,
 'vocab_id_min': 0,
 'vocab_id_max': 151668,
 'vocab_ids_contiguous': True,
 'added_token_count': 26,
 'added_id_min': 151643,
 'added_id_max': 151668,
 'valid_added_ids_accepted': [151643, 151668],
 'len_tokenizer_id_rejected':
   'OOV witness contains an out-of-vocabulary token ID'}
```

I also wrapped the gate during one complete sentinel validation. All 130 calls
supplied maximum 151,668. Six boundary classes naturally contained real added
structural IDs above the base vocabulary: boundary markers, canonical message
streams, role-native canonical streams, fresh compact canonical streams,
assistant generation prefixes, and probe generation prefixes. Assistant and
probe content correctly excluded structural tokens. The complete sentinel
still returned its nonauthorizing mechanical PASS:

```text
{'status': 'MECHANICAL_SENTINEL_PASS_NO_EXECUTION_AUTHORIZATION',
 'bounded_calls': 130,
 'boundary_class_counts': {
   'assistant_content': 36,
   'assistant_generation_prefix': 36,
   'boundary_marker': 16,
   'canonical_message_stream': 16,
   'fresh_compact_canonical': 2,
   'probe_generation_prefix': 6,
   'probe_target_content': 12,
   'role_native_canonical': 6},
 'classes_observing_valid_added_tokens': [
   'assistant_generation_prefix',
   'boundary_marker',
   'canonical_message_stream',
   'fresh_compact_canonical',
   'probe_generation_prefix',
   'role_native_canonical'],
 'all_maxima': [151668],
 'highest_observed_valid_id': 151645}
```

### Isolated OOV failure matrix

I injected exactly `len(tokenizer) == 151669` after the real tokenizer output
but immediately before `_plain_token_ids`, isolating each of the eight boundary
classes in a fresh complete sentinel validation. All eight failed at the new
gate with the intended error; no later geometry or equality check was needed:

```text
{'oov_id': 151669,
 'boundary_classes_rejected': 8,
 'outcomes': {
   'boundary_marker':
     "boundary '<|im_start|>' contains an out-of-vocabulary token ID",
   'canonical_message_stream':
     'canonical message stream contains an out-of-vocabulary token ID',
   'role_native_canonical':
     'role-native canonical stream contains an out-of-vocabulary token ID',
   'fresh_compact_canonical':
     'fresh compact canonical stream contains an out-of-vocabulary token ID',
   'assistant_generation_prefix':
     'assistant message 2 generation prefix contains an out-of-vocabulary token ID',
   'assistant_content':
     'assistant message 2 content contains an out-of-vocabulary token ID',
   'probe_generation_prefix':
     'probe generation prefix contains an out-of-vocabulary token ID',
   'probe_target_content':
     'probe target content contains an out-of-vocabulary token ID'}}
```

This specifically closes the prior failure in which 36 corrupted generation
prefix calls survived and produced the baseline evidence hash unchanged.

## Regression and invariant results

The committed full-validator boundary test now covers string, float, and OOV
arrays for each of `canonical_ids_any`, `generation_prefix_ids`, and
`rendered_assistant_content_ids`. The direct helper test covers a value above
an explicit maximum. The older plain-integer, negative, boolean-middle-index,
U+02BC modifier-apostrophe, case-declared phrase, Unicode punctuation/format,
same-decoded-noncanonical-ID, 80-content-token-plus-EOS, and public carrier OOV
regressions remained green.

I independently rebuilt C, W, and fresh plans for every one of the 8 strata by
2 resolution subtypes. For each of R1 and R2 I reasserted equal C/W intervals,
equal selected C/W source token IDs, exact fresh-to-source token slices, exact
fresh logical source-position ranges, and complete C/W fresh-plan equality.
All 64 variant-by-region checks passed:

```text
{'sentinels': 16,
 'exact_region_checks': 64,
 'matrix_sha256':
   'f545875979b59bdeb1efea452e2f2818420c07ec2da5d76527ee8951b566ec7f',
 'target_bank_sha256':
   '378c078e4ab5862e6a8895a130443dcb4e2d990d1dda69e18598114841b374ab'}
```

These remain out-of-pool development-only geometry witnesses. They do not
attest subject sampling, attempt-order persistence, forced-replay
log-probability/K/V identity, semantic review, production eligibility, or any
execution release.

## Commands and observed results

Initial HEAD and worktree:

```sh
git rev-parse HEAD
git status --short --branch
```

```text
a2b31eca7bff28f665a4fa01c40e09efbfb52237
## HEAD (no branch)
```

Focused suites:

```sh
PYTHONPATH=src uv run --frozen pytest -q \
  tests/test_powered_v13_stimuli.py \
  tests/test_powered_v13_tokens.py
```

Result: **104 passed**, 2 SWIG deprecation warnings, in **133.16s**.

Focused Unicode/type/OOV/cap regressions:

```sh
PYTHONPATH=src uv run --frozen pytest -q \
  tests/test_powered_v13_stimuli.py::test_carrier_gate_rejects_expanded_semantic_leaks \
  tests/test_powered_v13_stimuli.py::test_case_declared_forbidden_phrases_are_accepted_expanded_and_hash_bound \
  tests/test_powered_v13_stimuli.py::test_full_sentinel_rejects_type_laundering_at_token_plan_boundaries \
  tests/test_powered_v13_stimuli.py::test_exactly_80_content_tokens_then_separate_eos_witness_is_accepted \
  tests/test_powered_v13_tokens.py::test_token_plan_boundary_rejects_nonplain_or_negative_ids \
  tests/test_powered_v13_tokens.py::test_token_plan_boundary_rejects_out_of_vocabulary_ids \
  tests/test_powered_v13_tokens.py::test_dynamic_planner_rejects_boolean_middle_index
```

Result: **41 passed**, 2 SWIG deprecation warnings, in **29.07s**.

Syntax compilation:

```sh
PYTHONPATH=src uv run --frozen python -m compileall -q \
  src/powered_v13_stimuli.py src/powered_v13_tokens.py \
  tests/test_powered_v13_stimuli.py tests/test_powered_v13_tokens.py
```

Result: exit `0`, no output.

Locked runtime versions:

```sh
PYTHONPATH=src uv run --frozen python -c \
  'import platform, transformers, tokenizers, huggingface_hub; print(platform.python_version(), transformers.__version__, tokenizers.__version__, huggingface_hub.__version__)'
```

```text
3.12.11 5.0.0 0.22.2 1.22.0
```

The AST inventory, vocabulary topology/added-token probe, 130-call runtime
instrumentation, eight-class isolated OOV matrix, and 16-sentinel R1/R2 audit
were each run with `PYTHONPATH=src uv run --frozen python -` against the exact
cached pinned tokenizer. Their literal outputs are reproduced above.

No source or test file was edited during this audit.
