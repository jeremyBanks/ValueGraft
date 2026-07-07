# Semantic Readouts Across a Compaction Boundary

This note describes a qualitative probe for the ValueGraft project: use a
Jacobian-lens readout to inspect what a model's internal state "thinks" a
summary token means when the summary is written after the full conversation,
and compare that to what the same summary token means when the summary is read
back in a clean compacted context.

The probe is not a replacement for the experimental metrics. It is a way to
look at the mechanism. It gives us concrete examples where the same visible
summary text has different internal semantics depending on whether it was
produced in the old conversation state or freshly re-encoded from text.

## The Basic Test

For each demo, we build two matched prompts:

1. Full-context summary path: the model reads the original conversation and
   then sees or emits the summary.
2. Fresh-summary path: the model sees only the compacted summary in the same
   local wrapper.

The summary tokens are aligned literally. At each aligned summary-token
position, we record the residual stream at selected model layers and decode it
through the downloaded Jacobian lens for `Qwen3.6-27B`. For each position and
layer, we ask for the top-k vocabulary readout. Then we compare the readout
sets between the full-context and fresh-summary paths.

The first demos sampled hand-selected anchor tokens such as `Ghost`,
`Vacuum`, `Maple`, and `B-410`. The later scan sampled every token in each
summary, then ranked the positions with the largest readout divergence. That
full scan is useful because it discovers examples without us deciding in
advance where the interesting boundary should be.

Raw files:

- `outputs/qwen36_pokemon_probe_v2.json`
- `outputs/qwen36_plain_probe.json`
- `outputs/qwen36_multi_demo_scan.json`

## What The Probe Shows

The readouts are noisy. Some top-ranked positions are punctuation, table
separators, or tokenization fragments. That is expected: the scan ranks
distributional difference, not human interestingness.

The useful cases are the named entities and compact labels. Those cases show a
recurring pattern:

- In the full-context path, the readout often points toward the situated role
  the label acquired in the conversation.
- In the fresh-summary path, the readout often points toward the label's
  ordinary lexical meaning, type prior, or generic association.

That is exactly the gap this project cares about. The visible summary text may
contain the right words, but re-encoding those words can lose part of the
context-conditioned state that existed when the summary was written.

## Pokemon Demo

The Pokemon demo is intentionally dense. It has nicknames, stale plans,
discarded options, rule exceptions, and labels whose plain meaning is
misleading.

`Vacuum` is a Zigzagoon with Pickup, not a household appliance. In the
full-context path, the later-layer readout around `Vacuum` contains `Zig`,
`zig`, `Pickup`, and utility-role language. In the fresh-summary path, the same
visible token leans back toward `cleaner`, `Cleaner`, `clean`, and vacuum-like
associations.

`Ghost2` is a surviving replacement, while the original `Ghost` died. In the
full-context path, the readout around the `2` token contains `survived`,
`survives`, `alive`, and `made`. In the fresh-summary path, it becomes more
like a plain number or replacement marker, with weaker access to the specific
survival contrast.

`Hariyama` is ruled out. In the full-context path, readouts include `ruled`,
`dropped`, `rejected`, `banned`, `excluded`, and `stayed`. In the fresh-summary
path, the readout is more token-form and Pokemon-association heavy. The text
still says the right thing, but the write-time state more directly exposes the
decision status.

`Dex` is not just a Pokedex-flavored label. It is tied to a pending trade. In
the full-context path, the readout includes `owes`, `owed`, `trade`, `traded`,
and `owe`. In the fresh-summary path, it leans toward `entry`, `completion`,
`count`, and generic Pokedex-progress meanings.

## Ordinary Conversation Demo

The block-party demo is deliberately mundane. This matters because we do not
want the phenomenon to depend on game jargon.

`Maple` means the library's Maple Room, not a tree, syrup, or street. In the
full-context path, the readout around `Maple` contains `Room`, `means`,
`refers`, and equality-like mapping tokens. In the fresh-summary path, it leans
toward `Street`, `Ave`, `Avenue`, `St`, and other generic place-name priors.

`Robin` is the human volunteer coordinator with van keys and vendor contacts.
In the full-context path, readouts include `volunteer`, `coordinates`, `holds`,
and role-mapping tokens. In the fresh-summary path, they drift toward generic
name associations such as `Hood`, `Robin`, and unrelated person-name priors.

`B-410` is stale, while `P-771` is current. In the full-context path, the
readout around `B-410` includes `obsolete`, `outdated`, `deprecated`,
`expired`, and `stale`. In the fresh-summary path, the readout becomes more
like a generic permit or civic identifier.

`Crane` is a stage-rental company delivering risers, not a machine. In the
full-context path, readouts include `stage`, `delivers`, `delivery`, and
company-role language. In the fresh-summary path, readouts drift toward
`crane`, `tower`, `truck`, `operator`, and equipment associations.

## Different Summary Shapes

The full scan used multiple summary formats:

- Bullet summaries for the Pokemon and block-party examples.
- Compact paragraph summaries for the same two examples.
- A Markdown table for a software incident handoff.
- A checklist for household trip planning.
- A JSON-like support-ticket summary.

The broad pattern appears across formats. The table, checklist, and JSON cases
also show a practical limitation: structural tokens can dominate raw
divergence rankings. A useful automatic version should rank or filter for
semantic targets, not merely the largest change in top-k readout.

Still, the format variation is encouraging. The effect is not limited to one
pretty summary template.

## Software Incident Table

The software-incident table is a bridge toward coding tasks. It uses compact
operational labels:

- `Mercury` is the billing service that owns invoice finalization.
- `Falcon` is the rejected rollback branch.
- `Raven` is the current rollback branch.
- `Patch 17` is stale.
- `R3` is the live hotfix label.
- `Nova` is a canary host.
- `red button` means pause webhooks.

The strongest target examples are status labels. For `Patch 17`, the
full-context path surfaces `obsolete`, `outdated`, `old`, `expired`, and
`deprecated`; the fresh-summary path mainly treats the token as a patch/version
number. For `R3`, the full-context path surfaces `latest`, `official`,
`newest`, `current`, and `live`; the fresh-summary path mostly decodes the
digit or generic identifier shape.

This suggested a natural next probe for SWE-style tasks: align on tokens in the
true next action, such as the file path, tool name, function call, branch name,
or line range, and ask whether full-context state makes the action token decode
more like its operational role.

## SWE-Gym Next Actions

We then tried the same readout on a few real SWE-Gym/OpenHands-style
trajectories. This is still qualitative: no agent loop, no generated behavior,
just teacher forcing the true next assistant action after two prompts.

The method:

1. Reload a trajectory.
2. Cut before an assistant action.
3. Generate a compact state summary from the pre-action context.
4. Build a compacted prompt from the summary plus a recent tail.
5. Teacher-force the exact next action after the full prompt and after the
   compacted prompt.
6. Lens-read every action token and aggregate important spans such as tool
   names, file paths, commands, and line ranges.

Three small examples:

- getmoto: `str_replace_editor` should `view`
  `/workspace/getmoto__moto__4.1/moto/rds/responses.py` at `[584, 600]`.
- Dask: `execute_bash` should run
  `grep -n 'normalize_token' /workspace/dask__dask__2022.6/dask/base.py`.
- MONAI: `execute_bash` should run
  `python3 /workspace/Project-MONAI__MONAI__0.8/reproduce_error.py`.

The readout does find the operational spans. In the getmoto case, span-level
divergence lands on `str_replace_editor`, the exact `responses.py` path, and
the `[584, 600]` range. In the Dask case, it lands on `execute_bash`, the
`grep -n 'normalize_token' .../base.py` command, and the `base.py` path. In the
MONAI case, it lands on the `python3` command and `reproduce_error.py` path.

This is noisier than the Pokemon and block-party examples. Tool-call syntax,
XML-ish wrappers, file paths, and subword splits create many uninteresting
high-divergence punctuation tokens. The fix is to report phrase spans rather
than raw token ranks. Once grouped by span, the examples are much easier to
read.

The other important lesson is summary control. SWE-Gym trajectories strongly
prime the model to take an action. When asked to summarize, Qwen3.6 sometimes
started writing the next tool call instead. The probe now uses a stronger
no-tool summary request, pre-fills `Summary:`, and trims generated text at
tool/chat markers. Without that guard, the compacted side can accidentally
contain the next action, which contaminates the comparison.

## Why This Is Useful

The readout gives us a concrete way to talk about the otherwise slippery
phrase "lost context-conditioned state." We can point at the same visible text
and say:

- Here is what the token looks like when the model wrote it under the original
  conversation.
- Here is what the token looks like when the model merely reads that summary
  later.
- Here is where those two internal readings separate.

That is a qualitative complement to the quantitative arms. The metrics tell us
whether grafting helps a task. The readouts help us see what kind of information
may be available to graft.

## What To Use Next

The next version should make the SWE-Gym probe span-first from the start:

- Generate or load a compact summary with safeguards against tool-call leakage.
- Extract action spans for tool names, command names, file paths, symbols, and
  line ranges.
- Report one compact comparison per span.
- Keep raw token rows as drill-down data, not as the main presentation.

That connects the qualitative method directly to the strongest use case: not
whether the model can remember a trivia label, but whether compacted state
keeps enough situated meaning to choose the next useful coding move.

## Interpretation Boundary

This probe should not be overclaimed. A top-k lens readout is an interpretive
view of a residual-stream state, not a direct dump of the model's beliefs. It
can be noisy, layer-dependent, and sensitive to tokenization.

Even with that caveat, the examples are valuable. They repeatedly show the
same qualitative shape: write-time summary tokens carry richer situated
semantics than freshly re-encoded summary tokens. That is the local mechanism
ValueGraft is trying to preserve.
