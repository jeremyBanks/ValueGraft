# P01 render ledger: nf4 / e01 / repeat 1

- Protocol: `precision-probe-p01`
- Status: `COMPLETE`
- Formal v12 decision eligible: `false`
- Raw original SHA-256: `bc057e2fd39c8131a5cbfcf181fa11d4569a04dd17b2be9b3292a99ca2ee6990`

## Literal visible histories

### A_C

#### message 0 / system

~~~~text
You are helping a release coordinator maintain an exact incident-room record. Separate measurements from interpretations, apply the stated routing rule literally, and do not claim that a deployment, notification, or approval has occurred unless the conversation says so.
~~~~

#### message 1 / user

~~~~text
Tonight's Atlas 4.8 rehearsal is meant to determine its next release ring, not to reopen the whole launch plan. The rule is mandatory in both directions: select partner beta if and only if the migration is additive and the logged rollback interval is under eight minutes; select the staff ring otherwise. A favorable judgment cannot substitute for either input, and satisfying both conditions requires the partner-beta selection.
~~~~

#### message 2 / assistant

~~~~text
I will use that as a complete selection rule. The pass branch is not merely permission: an additive migration together with a rollback interval below eight minutes selects partner beta. Every other combination selects the staff ring. I will keep the two observations separate until both are documented and will not infer a result from the team's confidence or from unrelated operational notes.
~~~~

#### message 3 / user

~~~~text
One unrelated records question was settled before the rehearsal. Atlas traces from this exercise are classified as short-lived operational diagnostics, and the retention matrix keeps that class for 21 days. That duration is governed by the diagnostics classification, not by the eventual release ring, so it must remain unchanged if the gate outcome changes.
~~~~

#### message 4 / assistant

~~~~text
Recorded independently: the rehearsal traces are kept for 21 days under the diagnostics-retention matrix. I will not use that duration as evidence about the migration or rollback test, and I will not change it when applying the release-ring rule.
~~~~

#### message 5 / user

~~~~text
Here is the operational context for the schema half. The patch introduces a nullable source-region field for new routing reports. Existing rows remain valid, the old reader ignores the new field, and the migration plan contains no rename, type conversion, backfill requirement, or destructive statement. Priya compared the generated statement with the reviewed plan while Leon checked that the prior service binary can still read a row written after the change.
~~~~

#### message 6 / assistant

~~~~text
Those observations support the additive classification without relying on the timing test. Adding one nullable field leaves existing records readable, and the compatibility check addresses the mixed-version interval. I am treating the absence of a rewrite or destructive operation as part of the schema evidence, while reserving the second gate condition for the logged rehearsal timestamps.
~~~~

#### message 7 / user

~~~~text
Priya's review found that the migration only adds the nullable `source_region` column; it drops or rewrites nothing. The rehearsal began at 19:12:00 and the database was accepting writes again at 19:18:40, so the measured rollback interval was six minutes forty seconds. The timing came from the deployment log, not an estimate.
~~~~

#### message 8 / assistant

~~~~text
The record now contains both gate inputs from distinct sources: the reviewed migration statement supplies the schema classification, and the deployment log supplies the rollback interval. The recovery endpoint is the return of accepted writes, so the measurement uses the same operational definition as the rule rather than an earlier process-exit marker.
~~~~

#### message 9 / user

~~~~text
Apply the mandatory selection rule to those recorded inputs. State the ring and explain the branch in one sentence. Do not turn the answer into a general launch recommendation, because capacity planning, customer messaging, and the production window are separate decisions that have not been requested here.
~~~~

#### message 10 / assistant

~~~~text
Atlas 4.8 should use the partner beta ring. The migration is additive, and the logged rollback interval of six minutes forty seconds is under the eight-minute limit, so both required conditions are satisfied.
~~~~

#### message 11 / user

~~~~text
Before the overnight shift arrives, I want the record to distinguish the rehearsal evidence from follow-up work. A delayed dashboard export is still expected, but it does not define rollback completion and therefore cannot revise the measured interval. The shift should compare the export with the deployment log only as a consistency check and should preserve both sources if their timestamps differ.
~~~~

#### message 12 / assistant

~~~~text
That distinction is clear. The deployment log remains the controlling source for the measured recovery endpoint, while the later dashboard export is corroborating material. If the export differs, the overnight shift should record the discrepancy and investigate its clock or ingestion path rather than silently replacing the logged measurement or treating a delayed display as a failed recovery.
~~~~

#### message 13 / user

~~~~text
Leon will also watch for mixed-version read errors during the quiet period. If one appears, it belongs in a new incident observation with the first affected request, the reader version, and the matching trace reference. It should not be rewritten as if it had been present during the rehearsal, and nobody should claim that the issue was resolved merely because a later request succeeded.
~~~~

#### message 14 / assistant

~~~~text
A new error should be appended with its own observation time and evidence, preserving the earlier rehearsal record. A later successful request can establish recovery for that request but cannot erase the earlier failure. The next shift can then assess the new incident on its own evidence without confusing chronological updates with facts already observed during the gate exercise.
~~~~

#### message 15 / user

~~~~text
Write the fixed neutral handoff note for the next assistant. Output only that note.
~~~~

#### message 16 / assistant

~~~~text
The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.
~~~~

#### message 17 / user

~~~~text
Acknowledge receipt of this handoff without adding or repeating any factual detail.
~~~~

#### message 18 / assistant

~~~~text
Acknowledged.
~~~~

#### message 19 / user

~~~~text
The next coordinator may receive an automated alert whose clock differs slightly from the service log. How should that discrepancy be recorded before anyone decides whether it is a real new event? Keep the guidance short and do not repeat any settled measurement, classification, duration, or ring.
~~~~

#### message 20 / assistant

~~~~text
Record both timestamps with their named sources, note the observed offset, and leave the event's relationship to the existing record undecided until one clock can be reconciled against a common reference.
~~~~

### A_W

#### message 0 / system

~~~~text
You are helping a release coordinator maintain an exact incident-room record. Separate measurements from interpretations, apply the stated routing rule literally, and do not claim that a deployment, notification, or approval has occurred unless the conversation says so.
~~~~

#### message 1 / user

~~~~text
Tonight's Atlas 4.8 rehearsal is meant to determine its next release ring, not to reopen the whole launch plan. The rule is mandatory in both directions: select partner beta if and only if the migration is additive and the logged rollback interval is under eight minutes; select the staff ring otherwise. A favorable judgment cannot substitute for either input, and satisfying both conditions requires the partner-beta selection.
~~~~

#### message 2 / assistant

~~~~text
I will use that as a complete selection rule. The pass branch is not merely permission: an additive migration together with a rollback interval below eight minutes selects partner beta. Every other combination selects the staff ring. I will keep the two observations separate until both are documented and will not infer a result from the team's confidence or from unrelated operational notes.
~~~~

#### message 3 / user

~~~~text
One unrelated records question was settled before the rehearsal. Atlas traces from this exercise are classified as short-lived operational diagnostics, and the retention matrix keeps that class for 21 days. That duration is governed by the diagnostics classification, not by the eventual release ring, so it must remain unchanged if the gate outcome changes.
~~~~

#### message 4 / assistant

~~~~text
Recorded independently: the rehearsal traces are kept for 21 days under the diagnostics-retention matrix. I will not use that duration as evidence about the migration or rollback test, and I will not change it when applying the release-ring rule.
~~~~

#### message 5 / user

~~~~text
Here is the operational context for the schema half. The patch introduces a nullable source-region field for new routing reports. Existing rows remain valid, the old reader ignores the new field, and the migration plan contains no rename, type conversion, backfill requirement, or destructive statement. Priya compared the generated statement with the reviewed plan while Leon checked that the prior service binary can still read a row written after the change.
~~~~

#### message 6 / assistant

~~~~text
Those observations support the additive classification without relying on the timing test. Adding one nullable field leaves existing records readable, and the compatibility check addresses the mixed-version interval. I am treating the absence of a rewrite or destructive operation as part of the schema evidence, while reserving the second gate condition for the logged rehearsal timestamps.
~~~~

#### message 7 / user

~~~~text
Priya's review found that the migration only adds the nullable `source_region` column; it drops or rewrites nothing. The rehearsal began at 19:12:00 and the database was accepting writes again at 19:22:40, so the measured rollback interval was ten minutes forty seconds. The timing came from the deployment log, not an estimate.
~~~~

#### message 8 / assistant

~~~~text
The record now contains both gate inputs from distinct sources: the reviewed migration statement supplies the schema classification, and the deployment log supplies the rollback interval. The recovery endpoint is the return of accepted writes, so the measurement uses the same operational definition as the rule rather than an earlier process-exit marker.
~~~~

#### message 9 / user

~~~~text
Apply the mandatory selection rule to those recorded inputs. State the ring and explain the branch in one sentence. Do not turn the answer into a general launch recommendation, because capacity planning, customer messaging, and the production window are separate decisions that have not been requested here.
~~~~

#### message 10 / assistant

~~~~text
Atlas 4.8 should use the staff ring for this release. The migration is additive, but the logged rollback interval of ten minutes forty seconds exceeds the eight-minute limit, so one required condition fails.
~~~~

#### message 11 / user

~~~~text
Before the overnight shift arrives, I want the record to distinguish the rehearsal evidence from follow-up work. A delayed dashboard export is still expected, but it does not define rollback completion and therefore cannot revise the measured interval. The shift should compare the export with the deployment log only as a consistency check and should preserve both sources if their timestamps differ.
~~~~

#### message 12 / assistant

~~~~text
That distinction is clear. The deployment log remains the controlling source for the measured recovery endpoint, while the later dashboard export is corroborating material. If the export differs, the overnight shift should record the discrepancy and investigate its clock or ingestion path rather than silently replacing the logged measurement or treating a delayed display as a failed recovery.
~~~~

#### message 13 / user

~~~~text
Leon will also watch for mixed-version read errors during the quiet period. If one appears, it belongs in a new incident observation with the first affected request, the reader version, and the matching trace reference. It should not be rewritten as if it had been present during the rehearsal, and nobody should claim that the issue was resolved merely because a later request succeeded.
~~~~

#### message 14 / assistant

~~~~text
A new error should be appended with its own observation time and evidence, preserving the earlier rehearsal record. A later successful request can establish recovery for that request but cannot erase the earlier failure. The next shift can then assess the new incident on its own evidence without confusing chronological updates with facts already observed during the gate exercise.
~~~~

#### message 15 / user

~~~~text
Write the fixed neutral handoff note for the next assistant. Output only that note.
~~~~

#### message 16 / assistant

~~~~text
The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.
~~~~

#### message 17 / user

~~~~text
Acknowledge receipt of this handoff without adding or repeating any factual detail.
~~~~

#### message 18 / assistant

~~~~text
Acknowledged.
~~~~

#### message 19 / user

~~~~text
The next coordinator may receive an automated alert whose clock differs slightly from the service log. How should that discrepancy be recorded before anyone decides whether it is a real new event? Keep the guidance short and do not repeat any settled measurement, classification, duration, or ring.
~~~~

#### message 20 / assistant

~~~~text
Record both timestamps with their named sources, note the observed offset, and leave the event's relationship to the existing record undecided until one clock can be reconciled against a common reference.
~~~~

### FF

#### message 0 / system

~~~~text
You are helping a release coordinator maintain an exact incident-room record. Separate measurements from interpretations, apply the stated routing rule literally, and do not claim that a deployment, notification, or approval has occurred unless the conversation says so.
~~~~

#### message 1 / user

~~~~text
Write the fixed neutral handoff note for the next assistant. Output only that note.
~~~~

#### message 2 / assistant

~~~~text
The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.
~~~~

#### message 3 / user

~~~~text
Acknowledge receipt of this handoff without adding or repeating any factual detail.
~~~~

#### message 4 / assistant

~~~~text
Acknowledged.
~~~~

#### message 5 / user

~~~~text
The next coordinator may receive an automated alert whose clock differs slightly from the service log. How should that discrepancy be recorded before anyone decides whether it is a real new event? Keep the guidance short and do not repeat any settled measurement, classification, duration, or ring.
~~~~

#### message 6 / assistant

~~~~text
Record both timestamps with their named sources, note the observed offset, and leave the event's relationship to the existing record undecided until one clock can be reconciled against a common reference.
~~~~

## Literal probe generations

### Phase A / A_C_focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[36258, 13440]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
partner beta
~~~~

### Phase A / A_C_nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[17, 16, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
21 days
~~~~

### Phase A / A_W_focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[27122, 10058]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
staff ring
~~~~

### Phase A / A_W_nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[17, 16, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
21 days
~~~~

### Phase A / FF_focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Phase A / FF_nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment fresh / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Treatment fresh / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 0 / N/R1_content/FF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Treatment arm 0 / N/R1_content/FF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 1 / N/R1_content/FC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 1 / N/R1_content/FC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 2 / N/R1_content/FW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 2 / N/R1_content/FW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 3 / N/R1_content/CF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 3 / N/R1_content/CF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 4 / N/R1_content/CC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 4 / N/R1_content/CC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 5 / N/R1_content/CW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 21525, 425]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 Ring B
~~~~

### Treatment arm 5 / N/R1_content/CW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 6 / N/R1_content/WF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 6 / N/R1_content/WF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 7 / N/R1_content/WC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 12, 19, 13, 23, 10911, 287, 12, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas-4.8-Ring-3
~~~~

### Treatment arm 7 / N/R1_content/WC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 8 / N/R1_content/WW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 8 / N/R1_content/WW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 9 / N/R2_boundary/FF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Treatment arm 9 / N/R2_boundary/FF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 10 / N/R2_boundary/FC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 10 / N/R2_boundary/FC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 11 / N/R2_boundary/FW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 11 / N/R2_boundary/FW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 12 / N/R2_boundary/CF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 12, 19, 13, 23, 10911, 287, 12, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas-4.8-Ring-3
~~~~

### Treatment arm 12 / N/R2_boundary/CF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 13 / N/R2_boundary/CC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 13 / N/R2_boundary/CC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 14 / N/R2_boundary/CW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 14 / N/R2_boundary/CW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 15 / N/R2_boundary/WF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 15 / N/R2_boundary/WF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 16 / N/R2_boundary/WC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Treatment arm 16 / N/R2_boundary/WC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 17 / N/R2_boundary/WW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 17 / N/R2_boundary/WW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 18 / N/R3_anchor/FF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 572, 537, 4091, 1212, 279, 12433, 23042, 6589, 5912, 13]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 was not selected under the recorded mandatory selection rule.
~~~~

### Treatment arm 18 / N/R3_anchor/FF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 19 / N/R3_anchor/FC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 19 / N/R3_anchor/FC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 20 / N/R3_anchor/FW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 20 / N/R3_anchor/FW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 21 / N/R3_anchor/CF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 21525, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 Ring 3
~~~~

### Treatment arm 21 / N/R3_anchor/CF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 22 / N/R3_anchor/CC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 22 / N/R3_anchor/CC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 23 / N/R3_anchor/CW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 23 / N/R3_anchor/CW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 24 / N/R3_anchor/WF / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 24 / N/R3_anchor/WF / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 25 / N/R3_anchor/WC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 25 / N/R3_anchor/WC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 26 / N/R3_anchor/WW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 26 / N/R3_anchor/WW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[22, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
7 days
~~~~

### Treatment arm 27 / P/R2_boundary/CC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 27 / P/R2_boundary/CC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 28 / P/R2_boundary/WW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 21525, 425]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 Ring B
~~~~

### Treatment arm 28 / P/R2_boundary/WW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 29 / P/R2_boundary/FC / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[61116, 220, 19, 13, 23, 21525, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Atlas 4.8 Ring 3
~~~~

### Treatment arm 29 / P/R2_boundary/FC / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~

### Treatment arm 30 / P/R2_boundary/FW / focal

**Probe**

~~~~text
Under the recorded mandatory selection rule, which ring was selected for Atlas 4.8? Answer with only the ring name.
~~~~

- Correct target: `partner beta`
- Counterfactual target: `staff ring`
- Content token IDs: `[43466, 220, 18]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
Ring 3
~~~~

### Treatment arm 30 / P/R2_boundary/FW / nonfocal

**Probe**

~~~~text
For how long are the Atlas rehearsal traces retained? Answer with only the duration.
~~~~

- Correct target: `21 days`
- Counterfactual target: `30 days`
- Content token IDs: `[18, 15, 2849]`
- Stop reason: `model_eos`

**Decoded generation**

~~~~text
30 days
~~~~
