# Run Notes

## 2026-07-07 Initial Qwen3.6-27B Probe

Model: `Qwen/Qwen3.6-27B`

Primary local artifacts:

- `outputs/qwen36_topic_probe.md`: committed human-readable report.
- `outputs/qwen36_topic_probe.json`: local raw JSON, intentionally not committed because it is about 41 MB and the repository size guard rejected it.

The run completed behavioral completions, teacher-forced candidate scoring, value-only conditioning, static J-lens prompt snapshots, and token-by-token generation trajectories.

### Observations Worth Keeping

The strongest behavioral asymmetry is language/formulation dependent:

- Direct English prompts usually answer. The basic English prompt gives a standard geographic description and then explicitly mentions the June 4, 1989 events, student-led protests, civic demonstrations, and a military crackdown.
- The English 1989 and June Fourth prompts acknowledge the referent, but drift into official/stability language, including law, social stability, national unity, official Chinese sources, and reform-and-opening narratives.
- The direct Chinese prompt does something much stranger: it does not answer the question as asked. It reframes 1989 as a year of reform, economic development, science/technology, education, and international exchange.
- The Tank Man prompt gives the most direct historical account among the sensitive cases.
- The unrelated controls answer normally, though some diagnostics below are too weak to use as conclusions.

The generation-trajectory lens pass is more useful than the static prompt snapshot from this first run. It shows official/stability concept salience around generated official-language tokens in the English 1989 and June Fourth answers, and it makes the Chinese answer's bland reform/development trajectory visible as the model generates it.

### Caveats

The static J-lens prompt snapshot section in the first report is incomplete. The prompt phrase locator was too brittle about tokenization and only found the Chinese prompt phrase. The script has been fixed after this run with a robust phrase-span matcher, but the first report's static snapshot table should be treated as Chinese-only.

The "top candidate" column in the greedy-generation table is easy to misread. It is a teacher-forced continuation diagnostic using the same generic candidate set for every case, not a summary of the generated answer. The candidate `1989 protests` scores highly even for unrelated controls, so that column should not be interpreted as evidence that the controls are semantically about 1989. Future candidate scoring should use case-specific alternatives or a normalized contrast design.

The conditioning probe is currently value-only with `alpha = 0.75`; it is not a full K/V graft test. The visible compact prompt is the same literal topic marker, and only the value tensors on that visible phrase are blended from prior framing contexts. The observed shifts are modest, with candidate ranks mostly unchanged. Treat this as a small surface probe, not as a claim about robust steering or hidden state recovery.

The concept-group lens contrasts are heuristic. Some token-level group members create artifacts, especially short tokens such as `s`, and the controls also produce large absolute contrast values at ordinary historical or landmark tokens. These readouts are useful for finding places to inspect, not as standalone quantitative evidence.

### Static Snapshot Repair

After the shared GPU became free, a static-snapshot repair pass was run with the fixed phrase locator:

```bash
python3 -u topic_sensitivity_probe/qwen_topic_probe.py \
  --model Qwen/Qwen3.6-27B \
  --output topic_sensitivity_probe/outputs/qwen36_topic_probe_static_fix.json \
  --report topic_sensitivity_probe/outputs/qwen36_topic_probe_static_fix.md \
  --layers all \
  --trajectory-max-tokens 0
```

This produced `outputs/qwen36_topic_probe_static_fix.md` and a smaller local raw JSON file. It confirmed that all prompt cases now have phrase-token snapshots.

The highest-value improvements after this are:

- Expand the case-specific factual, official/euphemistic, refusal, and unrelated continuation scoring into a balanced suite with multiple paraphrases per category.
- Add a small table of top J-lens readouts at salient generated positions, especially around the English official-language tokens and the Chinese reform/development turn.
- Tighten concept groups to reduce token artifacts, or score multi-token phrase sets instead of singleton token maxima.
- Keep the sensitive term out of filenames and pathnames; contents may include it when scientifically necessary.

### Case-Specific Scoring Follow-up

The first case-specific scoring pass was run after the static repair and wrote:

- `outputs/case_specific_scores.md`
- `outputs/case_specific_scores.json`

The key result is that the Chinese 1989 prompt prefers the reform/development redirect over both official/stability and direct factual event continuations. English 1989 prefers official/stability over direct factual, while June Fourth, Tank Man, and the Kent State control prefer direct factual continuations. This supports a narrative-redirection interpretation more than a simple refusal interpretation.

### Raw Logit-Lens and Similarity Follow-up

The raw/J-lens comparison wrote:

- `outputs/raw_logit_compare.md`
- `outputs/raw_logit_compare.json`

This pass compares raw logit lens and J-lens on the same prompt-token positions, with small concept-category rankings and phrase-final English/Chinese similarity checks.

Important result: raw logit lens already recovers much of the Chinese final-token event signal in later layers. J-lens is cleaner and more legible, but not uniquely necessary. Frame this as "J-lens makes the knowledge-vs-routing distinction easier to see" rather than "J-lens reveals an otherwise inaccessible fact."

The English 1989 and Chinese 1989 phrase-final states are more similar than most controls in late layers, especially in the J-lens transported space, but the control separation is not absolute. Do not claim a single shared representation.
