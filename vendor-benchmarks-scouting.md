# Benchmark scouting for compaction-under-context-pressure research

Investigation performed by shallow-cloning and reading source of each repo into
`/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad/vendor-scout/`
(tau-bench, tau2-bench, aider, gorilla). Read-only; nothing outside this file was
modified in the main repo.

Research question driving every verdict below: our compaction shim preserves
`messages[0]` (system prompt) intact and compacts/evicts from the rest of the
conversation. A benchmark is only useful for studying compaction if the
information an agent must obey LATE in the session was delivered EARLY via an
**evictable** channel (an early user turn or an early tool result) — not via
the system prompt, which our shim never touches.

---

## 1. TAU-bench (sierra-research/tau-bench)

**Repo**: `github.com/sierra-research/tau-bench`, MIT license (confirmed via `LICENSE`).

**Client wiring**: `tau_bench/agents/tool_calling_agent.py` and
`tau_bench/envs/user.py` call `litellm.completion(model=..., custom_llm_provider=...)`
directly — no bespoke HTTP client. litellm supports arbitrary OpenAI-compatible
endpoints (`custom_llm_provider="openai"` + `OPENAI_BASE_URL`/`OPENAI_API_KEY`
env vars, or `api_base` kwarg via a small patch). Model name is a free string,
which is exactly our "arm selector" mechanism.

**Session/context structure — verdict: LOW compaction relevance.**
`ToolCallingAgent.solve()` builds the initial messages as:
```python
messages = [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]
```
`self.wiki` is the *entire* domain policy (`tau_bench/envs/{retail,airline}/wiki.md`,
~1,000–1,050 words / ~1,300–1,500 tokens each) loaded verbatim into
**system message[0]**. There is no other place policy text appears — it is
never restated in a user or tool turn. Because our shim is defined to keep
`messages[0]` intact, TAU-bench's critical governing information would **never
be evicted by construction**, regardless of how aggressive compaction is. This
makes it a poor testbed for "does compaction silently drop policy" — the shim
structurally can't drop it here.

**Session length**: task instructions are short (1–3 sentences), and typical
solved trajectories are ~5–15 turns (few tool calls + a couple of assistant
responses). With the ~1.3–1.5K-token wiki included, total session tokens are
usually in the 2–6K range — rarely reaching the >12K token target on their own
(some very long retail tasks with many tool round-trips could get there, but
it's not the median case, and either way the vulnerable content sits in an
unevictable slot).

**User simulator**: yes — `tau_bench/envs/user.py` implements `LLMUserSimulationEnv`
(and `ReactUserSimulationEnv`, `VerifyUserSimulationEnv`, `ReflectionUserSimulationEnv`,
all litellm-based). Default user model is `gpt-4o` (`load_user(..., model="gpt-4o")`
default arg in `run.py`). Running the benchmark means **2x LLM calls per turn**
(one for the subject agent, one for the simulated customer), and the `verify`/
`reflection` strategies add a *third* judge/critic call per user turn. This is a
real cost multiplier to budget for.

**Scoring**: objective/deterministic. `env.step()` compares final DB state and
required outputs against the ground-truth `actions` list in each `Task` — no
LLM judge is involved in the reward itself (LLM judging is only used inside the
optional `verify`/`reflection` user-simulation strategies, not for scoring).

**Task/domain counts**: 2 domains — retail (`tasks_test.py`: 115 tasks, plus
`tasks_train.py`/`tasks_dev.py` splits) and airline (`tasks_test.py`: 50 tasks).
Total ~165 test-split tasks (more if train/dev splits are pooled).

**Integration effort**: low — hours. litellm already speaks OpenAI-compatible
`base_url`; wiring our shim is `export OPENAI_API_BASE=<shim-url>`, `--model-provider openai
--model <arm-name>`. No code changes needed for the agent side. Cost: budget
for the 2x (or 3x) user-simulator overhead.

---

## 2. tau2-bench (sierra-research/tau2-bench) — successor project, confirmed to exist

**Repo**: `github.com/sierra-research/tau2-bench`, MIT license (confirmed via `LICENSE`,
"Copyright 2025 Sierra Research").

**Client wiring**: same architecture, still litellm (`src/tau2/utils/llm_utils.py`
wraps `litellm.completion`/`completion_cost`). Same OpenAI-compatible base_url path
applies.

**Domains** (from `src/tau2/registry.py` and README): `mock`, `airline`, `retail`,
`telecom`, and a newer **`banking_knowledge`** domain. This is the interesting
addition over v1.

### 2a. airline / retail / telecom — same low relevance as TAU-bench

`src/tau2/agent/llm_agent.py` builds the system prompt as:
```python
SYSTEM_PROMPT = "<instructions>{agent_instruction}</instructions><policy>{domain_policy}</policy>"
```
i.e. **everything** — including, for telecom, both `main_policy.md` (~1,025 words)
*and* the ~2,822-word `tech_support_workflow.md` diagnostic flowchart-as-text —
gets concatenated and injected into system message[0] at environment
construction time (`domains/telecom/environment.py: get_environment()`). Same
verdict as TAU-bench: **critical info lives entirely in the unevictable slot**.
Telecom's `tasks.json` has 2,285 entries (large, likely combinatorially
generated; a curated "small"/"full" split exists), retail has 114, airline 50.

### 2b. banking_knowledge — the one genuinely useful discovery here

This domain (`src/tau2/domains/banking_knowledge/`) is a **RAG-style knowledge-retrieval
customer-service domain**, distinct from the policy-in-system-prompt pattern:
- 698 separate JSON documents (~130K words total) describing bank account/credit-card
  products live in `data/tau2/domains/banking_knowledge/documents/`.
- The agent does **not** get these documents in its system prompt. It gets a
  toolkit (`retrieval_toolkits.py`: `KnowledgeToolsPlain`/`WithGrep`/`WithKBSearch`/
  `WithShell`, configurable via `retrieval_variant`) and must actively search/grep/
  retrieve documents as **tool calls mid-conversation** — i.e. the actual
  governing content ("which credit card fits") arrives as a **tool-result message
  well into the transcript**, which is exactly the evictable channel we want.
- Task `user_scenario.instructions` also embed hard constraints stated only in
  the (early) user/persona turn — e.g. sample task_001: "You will not accept a
  credit card that has any annual fees unless it is the ONLY option available" —
  a constraint that must still govern a tool-call decision made many retrieval
  turns later. This is a much better structural match to "early info must
  govern late action" than any of the other tau2 domains.
- 97 tasks (`tasks.json`, 97 files under `tasks/`), officially registered and
  documented (`README.md`: "Available domains: mock · airline · retail ·
  telecom · banking_knowledge"; separate doc at `src/tau2/knowledge/README.md`).
  Installed via `uv sync --extra knowledge`.

**Session length**: with 698 candidate documents and a multi-step retrieval
workflow (search → narrow → compare → apply), sessions here can plausibly run
long — tool outputs alone can push well past 12K tokens if several documents
get pulled into context, unlike the terser retail/airline/telecom trajectories.
Would need a short pilot run to confirm actual token counts per task, but the
structure is favorable.

**Verdict**: airline/retail/telecom = low relevance (same as v1). **banking_knowledge
= the standout candidate in the tau2 family** for this research question, precisely
because it doesn't fold everything into system message[0].

**Integration effort**: low-to-medium — half a day to a day. Same litellm base_url
plumbing as tau-bench for the agent LLM calls; additionally need `uv sync --extra
knowledge` and to pick/configure a `retrieval_variant`. No changes to core client code.

**User simulator**: same litellm-based `LLMAgent`-style simulated user as tau2's other
domains (uses the tau2 "half-duplex" participant abstraction); still 2x API calls
per turn, model configurable (no hardcoded default found in registry — passed via CLI).

**Scoring**: objective — tau2 evaluates against DB-state/tool-call-sequence
checks (same family as tau-bench's reward, extended with `multi_turn_eval`
checkers), not LLM-judge based, for the core reward.

---

## 3. Aider-Polyglot benchmark (Aider-AI/aider, `benchmark/` dir)

**Repo**: `github.com/Aider-AI/aider`, Apache-2.0 (confirmed via `LICENSE.txt`).
Task set lives in a companion repo, `github.com/Aider-AI/polyglot-benchmark`
(cloned by `benchmark/README.md`'s instructions), derived from Exercism exercises.

**Task count**: 225 exercises (the hardest 225 of 697 candidate Exercism problems
across 6 languages: C++, Go, Java, JavaScript, Python, Rust — confirmed via
`aider/website/_posts/2024-12-21-polyglot.md`).

**Harness structure — verdict: poor fit, single-exercise-per-session.**
Reading `benchmark/benchmark.py::run_test_real()`: each exercise gets a **fresh,
independent** `Coder` instance seeded only with that exercise's instructions
(`instructions.md` + a file list). The harness then loops
`for i in range(tries)` (default `tries=2`): run the coder, execute unit tests,
and if they fail, feed the *test error output* back as the next user turn and
retry. So there is a tiny bit of multi-turn structure (instruction → code →
test-failure → retry), but:
- Each exercise's context resets from scratch — there is no persistent
  cross-exercise session to compact.
- Typical exercise + 1 retry is a handful of short messages; nowhere near
  12K tokens.
- There's no "policy stated early, obeyed late" structure at all — it's
  single-shot code generation against a spec, not a long agentic session with
  evictable early constraints.

**Pluggability**: yes, trivially — `aider/models.py` is built on litellm
(`from aider.llm import litellm`), so arbitrary OpenAI-compatible `base_url`
works via the standard litellm env vars, same as tau-bench/tau2-bench.

**Credibility of official harness vs. home-grown chain**: using the official
harness would buy comparability against the public Aider leaderboard numbers
Qwen's model card cites (55%), but that comparability is largely irrelevant to
*this* research question, since the benchmark's fundamental structure (isolated,
short, single-exercise sessions) doesn't exercise compaction at all. A
home-grown chain of coding tasks *deliberately designed* to carry constraints
across a long multi-turn session would be more informative here than the
official Polyglot harness, though it would sacrifice the "official/comparable
numbers" credibility. Net: Polyglot is the wrong tool for compaction research
regardless of home-grown vs. official; don't spend integration effort here for
this specific study.

---

## 4. BFCL-v3 / v4 (gorilla-llm/gorilla, `berkeley-function-call-leaderboard/`)

**Repo**: `github.com/ShishirPatil/gorilla` (the "gorilla-llm" mirror points to the
same project), Apache-2.0 (confirmed via `LICENSE`).

**Important version note**: the current `main` branch is already **BFCL-v4**
(`VERSION_PREFIX = "BFCL_v4"` in `bfcl_eval/constants/category_mapping.py`); the
v3 numbers Qwen's model card cites (~65%) predate the categories described below.
v4 is additive (adds multi-turn/memory/web-search on top of v3's categories), so
what follows describes what the successor project offers, which is the more
useful frame for scouting purposes.

**Base categories (v3-era, still present)**: `simple_*`, `multiple`, `parallel`,
`parallel_multiple`, `irrelevance`, `live_*` — these are all **single-turn**
function-calling: one user message, one or more expected function calls, done.
**Confirms the user's hypothesis: this core of BFCL is a poor fit for
compaction research** — there is no long context to compact, nothing stated
early that must govern something late.

**v4 additions that matter**:
- `MULTI_TURN_CATEGORY` (`multi_turn_base`, `multi_turn_miss_func`,
  `multi_turn_miss_param`, `multi_turn_long_context`; 200 tasks each = 800 total).
  Read actual samples: each is only **4 user turns** total (e.g.
  `multi_turn_long_context_0`: move file → grep → sort → diff-and-post). Despite
  the "long_context" name, that category's length comes from a *larger simulated
  filesystem/API state* (more distractor files/tweets in `initial_config`), not
  from many conversational turns — real session token counts here are likely in
  the low thousands, well short of our 12K target.
- `MEMORY_CATEGORY` (`memory_kv`, `memory_vector`, `memory_rec_sum`; 155 tasks
  total in `BFCL_v4_memory.json`) — this is the most relevant find. It pairs each
  scenario with a **`memory_prereq_conversation/`** transcript: a long, noisy,
  multi-turn narrative (e.g. `memory_student.json`, ~97KB / many thousands of
  words across turns, a student rambling about courses, schedule, personal
  details) in which facts are stated once, early, in conversational prose — then
  later "question" turns ask pure recall questions ("What is my first name?",
  "How old am I?"). This *is* structurally the "early info must govern late
  action" pattern we want. BUT: it is purpose-built to test **explicit
  agent-managed memory tools** (`MemoryAPI` with kv-store / vector-store /
  recursive-summarization backends — see
  `eval_checker/multi_turn_eval/func_source_code/memory_{kv,vector,rec_sum}.py`),
  not raw-context compaction. The intended design is that the agent calls
  `MemoryAPI.store(...)` proactively and later calls `MemoryAPI.retrieve(...)`;
  it isn't testing "does the raw conversation survive in context," it's testing
  whether the agent uses a memory tool well. Using it for *our* compaction study
  would require either (a) suppressing/disabling the MemoryAPI tool so the model
  is forced to rely on raw context (repurposing the harness against its intent —
  feasible but non-trivial), or (b) accepting it measures a related-but-different
  capability (agentic memory management) rather than transparent context
  compaction.
- `WEB_SEARCH_CATEGORY` (99 tasks) — single/few-turn, not relevant here.

**Integration path**: BFCL already supports arbitrary OpenAI-compatible
endpoints without touching model_handler internals —
`model_handler/local_inference/base_oss_handler.py` reads
`REMOTE_OPENAI_BASE_URL` env var and constructs `OpenAI(base_url=..., api_key=...)`
directly (not litellm here, but still literally the pattern we need). Effort:
low for base plumbing (hours), but non-trivial (low-to-mid days) if we want to
repurpose the `memory_*` categories against raw-context compaction rather than
agentic memory-tool use, since that requires forking the eval harness to strip
out MemoryAPI and feed `memory_prereq_conversation` directly into the message
history instead.

**Scoring**: objective/deterministic — AST-based function-call matching
(`ast_eval/`) and state-diff/execution checking (`multi_turn_eval/`), no
LLM-judge involved.

**License**: Apache-2.0.

---

## Ranked recommendation

1. **tau2-bench, `banking_knowledge` domain** — best fit. Critical constraints
   arrive via an early evictable user turn (persona/preferences) and via
   evictable tool-result documents pulled mid-conversation from a 698-document
   corpus, not via system message[0]. Objective DB/tool-sequence scoring, MIT
   license, litellm-based (drop-in OpenAI-compatible base_url), 97 tasks
   (a bit under our ">=20 tasks" floor is fine, comfortably above it), and the
   retrieval-heavy structure plausibly produces >12K-token sessions naturally.
   Main risk/caveat: it's a newer, less battle-tested domain than tau2's
   core three (less prior art/benchmarking to compare against), and exact
   token-length distribution needs a short pilot to confirm rather than assuming
   from static reads.

2. **tau-bench / tau2-bench (airline, retail, telecom)** — keep as a
   **negative-control / sanity-check** rather than the primary study, precisely
   because policy is baked into system message[0] and therefore structurally
   immune to our shim's eviction. Useful to run alongside banking_knowledge to
   demonstrate that compaction *doesn't* hurt these (as expected) while it *does*
   hurt banking_knowledge or a similarly-evictable setup — a nice contrast pair
   for the write-up. Cheap to stand up (same litellm plumbing), so low
   incremental cost to include.

3. **BFCL-v4 `memory_*` categories** — interesting but second-tier: the
   "early narrative fact → later recall question" shape is exactly right, but
   the benchmark is designed around an explicit MemoryAPI tool, not raw-context
   retention, so using it for straightforward compaction research means forking
   the harness to disable/bypass that tool. Worth a confirm-phase spike if
   tau2/banking_knowledge session lengths disappoint, but not the first choice.

4. **BFCL-v3/v4 core (single-turn, `multi_turn_base/miss_func/miss_param/long_context`)**
   — deprioritize. Confirmed single-turn or ~4-turn structure; no meaningful
   context-pressure story, sessions too short to matter for compaction.

5. **Aider-Polyglot** — deprioritize for this study. Confirmed via
   `benchmark.py` that each of the 225 exercises runs as an isolated,
   short (instruction + up to 1 retry-on-test-failure) session; there is no
   persistent long context and no early-constraint/late-action structure to
   compact. Fine for a general coding-capability comparison against the model
   card's cited number, but not informative for compaction research regardless
   of official-harness-vs-home-grown tradeoffs.

**Single most important caveat for the top pick (banking_knowledge)**: this
domain lives behind `uv sync --extra knowledge` and is less prominent in the
tau2 README/paper than the airline/retail/telecom trio — it reads as a newer,
possibly less-polished addition. Before committing to it for the confirm phase,
run a handful of tasks end-to-end to (a) confirm actual token lengths clear our
12K floor once tool-call overhead is included, and (b) verify the reward
function is fully deterministic in practice (not just by design) for this
domain specifically, since it's the one place across all four benchmarks where
"policy lives in an evictable slot" is also true — meaning it's the only
candidate where compaction-induced failures would actually be observable, so
it's worth an especially careful pilot before over-committing calendar time to it.
